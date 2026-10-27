"""397-input student: original 108 proprio/command features then 289 height rays."""
from dataclasses import replace


def configure(cfg):
    from wbt_training.config_values.terrain import height_scan_term
    groups={k:v for k,v in cfg.observation.groups.items()if k!='depth_camera'}
    actor=groups['actor_obs'];terms={k:v for k,v in actor.terms.items()if k!='zz_height_scan'}
    assert set(terms)=={'actions','base_ang_vel','dof_pos','dof_vel','robot_anchor_projected_gravity','velocity_command'}
    # ObservationManager sorts names; zz keeps all original 108 features first.
    terms['zz_height_scan']=replace(height_scan_term,noise=0.)
    groups['actor_obs']=replace(actor,terms=terms)
    module=replace(cfg.algo.config.module,depth_backbone='',depth_output_dim=0,
        student_hidden_dims=[2048,1024,512,256,128])
    return replace(cfg,observation=replace(cfg.observation,groups=groups),
        algo=replace(cfg.algo,_target_='wbt_training.utils.heightmap_distillation.PhpHeightmapDistillationPPO',
            config=replace(cfg.algo.config,module=module)))


def verify_environment(env,obs):
    import torch
    from wbt_training.config_values.terrain import height_scan
    manager=env.observation_manager
    assert set(obs)=={'actor_obs','teacher_obs','critic_obs'}
    terms=manager.cfg.groups['actor_obs'].terms
    base=torch.cat([manager._compute_term('actor_obs',k,terms[k],modify_history=False)for k in sorted(terms)if k!='zz_height_scan'],dim=-1)
    height=height_scan(env)
    assert base.shape==(env.num_envs,108) and height.shape==(env.num_envs,289)
    assert obs['actor_obs'].shape==(env.num_envs,397)
    assert torch.equal(obs['actor_obs'][:,:108],base)and torch.equal(obs['actor_obs'][:,108:],height)
    assert all(torch.isfinite(v).all()for v in obs.values())
    sensor=env.simulator.scene.sensors['height_scanner'];sc=sensor.cfg
    assert sc.pattern_cfg.resolution==.1 and list(sc.pattern_cfg.size)==[1.6,1.6]
    assert sc.update_period==.02 and sc.ray_alignment=='yaw'
    assert not any(manager._term_instances.values()),'No camera/depth observation instance should exist'
    return dict(student_input_dim=397,proprio_command_dim=108,heightmap_dim=289,
        ordering='Original 108 flat actor features followed by 289 height_scan values',
        heightmap_grid=[17,17],heightmap_size_m=[1.6,1.6],heightmap_resolution_m=.1,
        heightmap_update_period_s=.02,heightmap_alignment='yaw',
        depth_camera_created=False,cnn_created=False,actor_input_matches_components_exactly=True)
