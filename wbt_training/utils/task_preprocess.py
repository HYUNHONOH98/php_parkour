"""Preserve the exact union meshes while using native ordered multi-teacher tiling."""
import numpy as np
import torch
import trimesh

def apply_task_preprocess(cfg,**kwargs):
    from wbt_training.config_values.terrain import apply_terrain_preprocess
    from wbt_training.utils import obstacle_helpers as helpers
    from wbt_training.utils.registry import RegistryResolver,bundle_pairs
    pairs=[pair for registry in RegistryResolver().resolve(cfg.training.registry_name) for pair in bundle_pairs(registry.directory)]
    meshes=[];boxes=[]
    for _,npy,obj in pairs:
        assert npy is not None and obj is not None
        data=np.load(npy,allow_pickle=False);assert data.shape[1]==1
        boxes.append(data[:,0])
        source=trimesh.load(str(obj),force='mesh',process=False)
        parts=[part for part in source.split(only_watertight=False) if not (min(part.extents[:2])>100 and part.extents[2]<.2)]
        meshes.append(trimesh.util.concatenate(parts))
    layout={}
    def exact_meshes(object_state,env_origins,env_spacing=20.):
        states=np.asarray(object_state,dtype=float);expected=layout['boxes']
        assert states.shape[1:]==expected.shape
        assert np.allclose(states,expected[None],atol=2e-5),'Terrain/motion grid alignment changed'
        local=layout['mesh'];per_env=[local.copy() for _ in env_origins]
        all_meshes=[local.copy().apply_translation(origin) for origin in env_origins]
        size=env_origins.max(axis=0)-env_origins.min(axis=0)+3*env_spacing
        center=(env_origins.max(axis=0)+env_origins.min(axis=0))/2+env_spacing/2;center[2]=-.05
        all_meshes.append(trimesh.creation.box(extents=[size[0],size[1],.1]).apply_translation(center))
        return per_env,trimesh.util.concatenate(all_meshes)
    def seeded_tiling(num_variants,motion_file,obstacle_file,seed=None):
        with np.load(motion_file) as data:
            offsets=helpers.compute_motion_grids(data)[0].numpy()
        assert len(offsets)==len(pairs)
        aligned=[];aligned_boxes=[]
        for mesh,box,offset in zip(meshes,boxes,offsets):
            delta=np.r_[offset,0.];aligned.append(mesh.copy().apply_translation(delta))
            moved=box.copy();moved[:,:3]+=delta;aligned_boxes.append(moved)
        layout['mesh']=trimesh.util.concatenate(aligned);layout['boxes']=np.concatenate(aligned_boxes)
        numpy_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                if seed is not None:torch.random.default_generator.manual_seed(seed);np.random.seed(seed)
                spacing,motion,per_env,terrain=helpers.sample_obstacles_for_motions(num_variants,motion_file,obstacle_file,seed)
        finally:np.random.set_state(numpy_state)
        return spacing,motion,terrain,per_env
    old_save,old_add=helpers.save_obstacles_to_file,helpers.add_onpath_obstacle_standalone
    helpers.save_obstacles_to_file=exact_meshes;helpers.add_onpath_obstacle_standalone=seeded_tiling
    try:return apply_terrain_preprocess(cfg,**kwargs)
    finally:helpers.save_obstacles_to_file=old_save;helpers.add_onpath_obstacle_standalone=old_add
