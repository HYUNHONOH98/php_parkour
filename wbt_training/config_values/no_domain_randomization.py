"""Explicit deterministic-domain student overrides; task/frame sampling stays on."""
from dataclasses import replace, is_dataclass, asdict


def changed(value, **kwargs):
    return replace(value, **kwargs) if is_dataclass(value) else {**value, **kwargs}


def configure(cfg):
    groups={}
    for name, group in cfg.observation.groups.items():
        terms={k:replace(v,noise=0.0,params=dict(v.params)) for k,v in group.terms.items()}
        if name=='depth_camera':
            t=terms['depth_cam']; p={**t.params,'camera_sensor_cfg':'wbt_training.utils.clean_depth.FixedG1DepthCameraConfig',
                'latency_frame':0,'buffer_len':1,'enable_holes':False,'enable_augmentation':False}
            terms['depth_cam']=replace(t,func='wbt_training.utils.clean_depth:CleanWarpDepthImageObsTerm',params=p)
        groups[name]=replace(group,enable_noise=False,terms=terms)
    r=cfg.randomization
    setup={k:replace(v,params=dict(v.params)) for k,v in r.setup_terms.items()}
    overrides={
      'push_randomizer_state':dict(enabled=False,max_push_vel=[0.0]*6),
      'actuator_randomizer_state':dict(enable_pd_gain=False,enable_rfi_lim=False,kp_range=[1.,1.],kd_range=[1.,1.],rfi_lim_range=[1.,1.]),
      'setup_action_delay_buffers':dict(enabled=False,ctrl_delay_step_range=[0,0]),
      'randomize_robot_rigid_body_material_startup':dict(enabled=True,static_friction_range=[1.,1.],dynamic_friction_range=[1.,1.],restitution_range=[0.,0.]),
      'randomize_base_com_startup':dict(enabled=False,base_com_range={a:[0.,0.]for a in 'xyz'}),
      'setup_dof_pos_bias':dict(enabled=False,dof_pos_bias_range=[0.,0.],dof_pos_bias_additive_overrides={}),
    }
    assert set(setup)==set(overrides),'Audit any newly added randomizer before using no-DR'
    for key,params in overrides.items():setup[key]=replace(setup[key],params={**setup[key].params,**params})
    reset=dict(r.reset_terms);t=reset['randomize_dof_state']
    reset['randomize_dof_state']=replace(t,params={**t.params,'joint_pos_scale_range':[1.,1.],
        'joint_vel_range':[0.,0.],'joint_pos_bias_range':[0.,0.],'randomize_dof_pos_bias':False})
    commands=dict(cfg.command.setup_terms);t=commands['motion_command'];p=dict(t.params)
    mc=p['motion_config'];noise=mc.noise_to_initial_pose if is_dataclass(mc) else mc['noise_to_initial_pose']
    ns=asdict(noise) if is_dataclass(noise) else noise
    zero={k:([0.]*len(v) if isinstance(v,(tuple,list)) else 0.)for k,v in ns.items()}
    p['motion_config']=changed(mc,noise_to_initial_pose=changed(noise,**zero))
    commands['motion_command']=replace(t,params=p)
    terrain=cfg.terrain.terrain_term
    terrain=replace(terrain,static_friction=1.,dynamic_friction=1.,restitution=0.,spawn=replace(terrain.spawn,randomize_tiles=False,xy_offset_range=0.))
    cfg=replace(cfg,observation=replace(cfg.observation,groups=groups),
        randomization=replace(r,setup_terms=setup,reset_terms=reset),
        command=replace(cfg.command,setup_terms=commands),terrain=replace(cfg.terrain,terrain_term=terrain))
    return cfg


def verify_environment(env, obs):
    import torch
    manager=env.observation_manager
    assert all(not g.enable_noise and all(t.noise==0 for t in g.terms.values()) for g in manager.cfg.groups.values())
    push=env.randomization_manager.get_state('push_randomizer_state')
    assert not push.enabled and torch.count_nonzero(push.max_push_vel)==0
    assert not env._randomize_base_com and not env._randomize_dof_pos_bias and not env._randomize_ctrl_delay
    bias=float((env.default_dof_pos-env.default_dof_pos_base).abs().max());assert bias==0
    if 'depth_camera' in manager.cfg.groups:
        from wbt_training.utils.clean_depth import CleanWarpDepthImageObsTerm
        term=manager._term_instances['depth_camera']['depth_cam']
        assert isinstance(term,CleanWarpDepthImageObsTerm) and not term.enable_holes
        assert term.latency_frame_range is None and term.latency_frame==0 and term.buffer_len==1
        assert torch.count_nonzero(term.depth_offset)==0 and not term.camera_sensor.cfg.randomize_placement
        cam=term.camera_sensor
        xyz=cam.camera_sensor_local_position
        pose_difference=float((xyz-xyz[:1]).abs().max());assert pose_difference==0
        raw=term._get_depth_images(env)[:8]
        a=term._process_depth_images(raw);b=term._process_depth_images(raw)
        assert torch.equal(a,b),'Depth preprocessing must be deterministic'
    a=manager.compute_group('actor_obs',modify_history=False);b=manager.compute_group('actor_obs',modify_history=False)
    assert torch.equal(a,b),'Observation noise must be absent'
    material=env.simulator._robot.root_physx_view.get_material_properties()
    desired=torch.tensor([1.,1.,0.],device=material.device)
    material_error=float((material-desired).abs().max());assert material_error<1e-7
    result=dict(all_domain_randomization_off=True,observation_noise=False,depth_augmentation=False,
        depth_repeated_input_exact=True if 'depth_camera' in manager.cfg.groups else None,depth_latency_frames=0,depth_holes=False,depth_offset_max=0.,
        camera_extrinsic_randomization=False,camera_pose_across_env_max_difference=pose_difference if 'depth_camera' in manager.cfg.groups else None,
        pushes=False,com_randomization=False,joint_bias_max=bias,action_delay=False,
        robot_material_max_error=material_error,robot_static_friction=1.,robot_dynamic_friction=1.,robot_restitution=0.,
        terrain_spawn_jitter=False,terrain_geometry='Original authored per-task union meshes; one fixed copy per task',
        retained='Uniform expert and reference-frame dataset sampling; PPO action exploration; deterministic termination curriculum')
    return result
