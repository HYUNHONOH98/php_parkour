"""Keep paired course geometry aligned with a fixed reference motion.

This opt-in hook retains PHP's native motion tiling, observation wiring and
terrain importer. It replaces only the independent obstacle yaw/vertex
augmentation, which is incompatible with this fixed MinReward course.
"""

import numpy as np
import torch
import trimesh


def apply_fixed_course_preprocess(cfg, **kwargs):
    from wbt_training.config_values.terrain import apply_terrain_preprocess
    from wbt_training.utils import obstacle_helpers
    from wbt_training.utils.registry import RegistryResolver, bundle_pairs

    registries=RegistryResolver().resolve(cfg.training.registry_name)
    assert len(registries)==1,'Fixed course requires one paired registry'
    pairs=bundle_pairs(registries[0].directory)
    assert len(pairs)==1,'Fixed course requires one continuous motion'
    _,terrain_npy,terrain_obj=pairs[0]
    assert terrain_npy is not None and terrain_obj is not None
    obstacles=np.load(terrain_npy,allow_pickle=False)
    assert obstacles.ndim==3 and obstacles.shape[1]==1
    source_boxes=obstacles[:,0]
    source=trimesh.load(str(terrain_obj),force='mesh',process=False)
    components=source.split(only_watertight=False)
    # The supplied OBJ already unions the overlapping platform boxes. Keep
    # those exact surfaces, and replace its 200 m ground box with one shared
    # floor covering the native grid (rather than tiling 64 overlapping floors).
    ground=[part for part in components if min(part.extents[:2])>100 and part.extents[2]<.2]
    assert len(ground)==1,'Expected exactly one source ground component'
    obstacles_mesh=trimesh.util.concatenate([part for part in components if part is not ground[0]])

    def exact_meshes(object_state,env_origins,env_spacing=20.):
        states=np.asarray(object_state,dtype=float)
        assert states.shape[1:]==source_boxes.shape
        translations=states[:,0,:3]-source_boxes[0,:3]
        assert np.allclose(states[:,:,:3],source_boxes[None,:,:3]+translations[:,None,:],atol=2e-5)
        assert np.allclose(states[:,:,3:],source_boxes[None,:,3:],atol=2e-5)
        meshes=[];per_env=[]
        for delta,origin in zip(translations,env_origins):
            local=obstacles_mesh.copy().apply_translation(delta)
            per_env.append(local)
            meshes.append(local.copy().apply_translation(origin))
        size=env_origins.max(axis=0)-env_origins.min(axis=0)+3*env_spacing
        center=(env_origins.max(axis=0)+env_origins.min(axis=0))/2+env_spacing/2
        center[2]=-.05
        meshes.append(trimesh.creation.box(extents=[size[0],size[1],.1]).apply_translation(center))
        return per_env,trimesh.util.concatenate(meshes)

    def seeded_native_tiling(num_variants,motion_file,obstacle_file,seed=None):
        # Preprocessing must not replace each distributed rank's training RNG
        # with obstacle_seed=42. All obstacle sampling here is CPU-only, so
        # seed the CPU generator locally and preserve CUDA RNGs untouched.
        numpy_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                if seed is not None:
                    torch.random.default_generator.manual_seed(seed)
                    np.random.seed(seed)
                spacing,motion,per_env,terrain=obstacle_helpers.sample_obstacles_for_motions(num_variants,motion_file,obstacle_file,seed)
        finally:
            np.random.set_state(numpy_state)
        return spacing,motion,terrain,per_env

    # These hooks run only during single-threaded environment construction in
    # this process. Restore the stock functions even if preprocessing fails.
    original_save=obstacle_helpers.save_obstacles_to_file
    original_add=obstacle_helpers.add_onpath_obstacle_standalone
    obstacle_helpers.save_obstacles_to_file=exact_meshes
    obstacle_helpers.add_onpath_obstacle_standalone=seeded_native_tiling
    try:
        return apply_terrain_preprocess(cfg,**kwargs)
    finally:
        obstacle_helpers.save_obstacles_to_file=original_save
        obstacle_helpers.add_onpath_obstacle_standalone=original_add
