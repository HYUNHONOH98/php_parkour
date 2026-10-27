"""B200 entry point preserving PHP's pinned MDP and course preprocessing.

Only device/paths/render plumbing differ from the archived server_2 recipe.
Iterations are additional native optimizer iterations when resuming a policy.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['teacher','student'])
budgets=parser.add_mutually_exclusive_group(required=True)
budgets.add_argument('--iterations',type=int,help='Legacy additional native iterations, including on resume')
budgets.add_argument('--total-iterations',type=int,help='Exclusive absolute iteration target; resume from the next unfinished update')
parser.add_argument('--self-collision',choices=['on','off'],help='Explicit native articulation override for both teacher and student')
parser.add_argument('--num-envs',type=int)
parser.add_argument('--checkpoint',type=Path)
parser.add_argument('--teacher-checkpoint',type=Path,action='append',help='Repeat in the same order as comma-separated registries')
parser.add_argument('--registry',help='One local file:// registry per teacher, comma-separated')
parser.add_argument('--terrain-copies',type=int)
parser.add_argument('--save-interval',type=int,default=500)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--logger',choices=['disabled','offline','online'],default='offline')
parser.add_argument('--wandb-group',default='php-b200')
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--student-perception',choices=['depth','heightmap'],default='depth')
parser.add_argument('--no-domain-randomization',action='store_true',help='Disable physical, sensor, observation and initial-state perturbations; keep task/frame dataset sampling')
parser.add_argument('--validate',action='store_true')
args=parser.parse_args()
assert args.iterations is None or args.iterations>0
assert args.total_iterations is None or args.total_iterations>0
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','B200 launch must explicitly select physical GPU 0'
assert os.environ.get('PHP_FILESYSTEM_ISOLATED')=='1','Use tools/b200/run.sh; direct unisolated runs are disabled'
assert os.environ.get('PHP_LANDLOCK_CONFINED')=='1','Filesystem access guard must be active'
assert int(os.environ.get('WORLD_SIZE','1'))==1,'This host exposes one physical B200'
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
robot_usd=ROOT/'data/b200_robot_headless/main_mesh_arm_collision_halfspherehand.usd'
if not robot_usd.exists():robot_usd=ROOT/'data/b200_robot/main_mesh_arm_collision_halfspherehand.usd'
teacher_paths=args.teacher_checkpoint or []
for p in [args.checkpoint,*teacher_paths]:
    if p is not None and not p.is_file():raise FileNotFoundError(p)

from wbt_training.config_values.experiment import g1_wbt_terrain_ref_noscale,g1_wbt_terrain_warp_distill_finetune
from wbt_training.utils.checkpoint import load_saved_experiment_config
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.config_values.logger import disabled,wandb
from wbt_training.utils.isaacsim_clock import use_php_isaacsim
import holosoma
assert Path(holosoma.__file__).is_relative_to(ROOT/'thirdparty/holosoma'),'Wrong Holosoma checkout'

if args.checkpoint:
    cfg,_=load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint.resolve())))
    is_student='distillation' in cfg.algo._target_.lower() or 'student_teacher' in cfg.algo._target_.lower()
    assert is_student==(args.stage=='student'),'Checkpoint stage does not match requested algorithm'
else:
    cfg=g1_wbt_terrain_ref_noscale if args.stage=='teacher' else g1_wbt_terrain_warp_distill_finetune

# Absolute budgets preserve all completed optimizer updates on resume. The
# legacy --iterations path is intentionally unchanged for archived commands.
from train_support import budget,verify_resume,articulation_self_collision
checkpoint_iteration=None
if args.checkpoint:
    import torch
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    checkpoint_iteration=int(payload['iter']);assert checkpoint_iteration==payload['iteration']
    del payload
start_iteration,additional_iterations,target_iterations=budget(checkpoint_iteration,args.iterations,args.total_iterations)
args.iterations=additional_iterations
requested_self_collision=None if args.self_collision is None else args.self_collision=='on'
if requested_self_collision is not None:
    cfg=replace(cfg,robot=replace(cfg.robot,asset=replace(cfg.robot.asset,enable_self_collisions=requested_self_collision)))

if args.stage=='student' and not args.checkpoint and not args.teacher_checkpoint:
    raise ValueError('A new student requires --teacher-checkpoint; no teacher is selected implicitly')

N=args.num_envs or (4096 if args.stage=='teacher' else 2048)
copies=args.terrain_copies or (1 if args.no_domain_randomization else (64 if args.stage=='teacher' else 3))
if args.no_domain_randomization:
    assert args.stage=='student' and copies==1
    from wbt_training.config_values.no_domain_randomization import configure
    cfg=configure(cfg)
if args.student_perception=='heightmap':
    assert args.stage=='student' and args.no_domain_randomization
    if args.checkpoint:assert 'Heightmap' in cfg.algo._target_,'Cannot resume a depth checkpoint into a different input architecture'
    from wbt_training.config_values.heightmap_student import configure as configure_heightmap
    cfg=configure_heightmap(cfg)
registry=args.registry or (ROOT/'data/minreward_course_20261026').as_uri()
from wbt_training.utils.registry import RegistryResolver,bundle_pairs
import numpy as np
registries=RegistryResolver().resolve(registry)
pairs=[pair for entry in registries for pair in bundle_pairs(entry.directory)]
expected_frames=0
for motion,_,_ in pairs:
    with np.load(motion) as payload:expected_frames+=len(payload['body_pos_w'])
if args.stage=='student' and not args.checkpoint:
    assert len(teacher_paths)==len(registries),'Ordered registry/teacher count mismatch'
    if not args.validate and requested_self_collision is not None:
        import torch
        teacher_physics=[]
        for path,entry in zip(teacher_paths,registries):
            expert=torch.load(path,map_location='cpu',weights_only=False)['experiment_config']
            flag=expert['robot']['asset']['enable_self_collisions']
            assert flag is requested_self_collision,'Student/expert self-collision mismatch'
            assert expert['training']['registry_name']==entry.reference,'Student/expert registry mismatch'
            teacher_physics.append(dict(checkpoint=str(path),registry=entry.reference,self_collision=flag))
        (OUT/'teacher_physics_verification.json').write_text(json.dumps(teacher_physics,indent=2))
hook='wbt_training.utils.task_preprocess:apply_task_preprocess' if args.registry else 'wbt_training.utils.fixed_course_preprocess:apply_fixed_course_preprocess'
logger=replace(disabled,base_dir=str(OUT/'runs'),video=replace(disabled.video,enabled=False)) if args.logger=='disabled' else replace(wandb,base_dir=str(OUT/'runs'),project='php-parkour',entity='humanoid-parkour',name=f'php-b200-{args.stage}-{OUT.name}',id=None,group=args.wandb_group,mode=args.logger,headless_recording=False,video=replace(wandb.video,enabled=False))

# The frozen USD was imported from this exact URDF on server_2. Native actuator
# gains, joint/body names, collision settings, observations and rewards remain.
cfg=replace(cfg,
    robot=replace(cfg.robot,asset=replace(cfg.robot.asset,usd_file=str(robot_usd))),
    training=replace(cfg.training,project='php-parkour',num_envs=N,multigpu=False,headless=True,seed=args.seed,name=f'php-b200-{args.stage}-{OUT.name}',checkpoint=str(args.checkpoint.resolve()) if args.checkpoint else None,teacher_checkpoint=','.join(str(p.resolve()) for p in teacher_paths) if teacher_paths else None,registry_name=registry,preprocess_hook=hook,preprocess_hook_kwargs=json.dumps(dict(add_onpath_obstacle=True,num_variants=copies,obstacle_seed=42))),
    simulator=replace(cfg.simulator,config=replace(cfg.simulator.config,sim=replace(cfg.simulator.config.sim,max_episode_length_s=40.))),
    algo=replace(cfg.algo,config=replace(cfg.algo.config,num_learning_iterations=args.iterations,save_interval=args.save_interval)),logger=logger)
cfg=use_php_isaacsim(cfg)
cfg=replace(cfg,simulator=replace(cfg.simulator,_target_='wbt_training.simulators.b200.IsaacSim'))
cfg.save_config(str(OUT/'launch_config.yaml'))
(OUT/'budget.json').write_text(json.dumps(dict(start_iteration=start_iteration,additional_iterations=args.iterations,total_iterations=target_iterations,checkpoint_iteration=checkpoint_iteration,self_collision=requested_self_collision),indent=2))

if args.validate:
    from holosoma.managers.utils import resolve_callable
    checked=resolve_callable(hook,context='b200 preprocess')(cfg,add_onpath_obstacle=True,num_variants=copies,obstacle_seed=42)
    checked.save_config(str(OUT/'preprocessed_config.yaml'))
    assert checked.robot.asset.enable_self_collisions==cfg.robot.asset.enable_self_collisions
    print(json.dumps(dict(valid=True,start_iteration=start_iteration,total_iterations=target_iterations,self_collision=checked.robot.asset.enable_self_collisions,stage=args.stage,num_envs=N,copies=copies,holosoma=holosoma.__file__,physics_hz=200,control_dt=.02,headless=True,rtx_cameras=False)))
    raise SystemExit(0)

from isaaclab.app import AppLauncher
original_resolve=AppLauncher._resolve_device_settings
def device_settings(self,launcher_args):
    original_resolve(self,launcher_args)
    launcher_args.update(active_gpu=0,physics_gpu=0,multi_gpu=False,max_gpu_count=1)
    launcher_args['kit_args']=launcher_args.get('kit_args','')+f' --allow-root --/plugins/carb.tasking.plugin/threadCount=16 --/log/file={OUT}/isaac-kit.log --/app/settings/loadUserConfig=false --/app/settings/persistent=false'
AppLauncher._resolve_device_settings=device_settings

from holosoma.agents.modules.logging_utils import LoggingHelper
original_logging=LoggingHelper.post_epoch_logging
started=time.perf_counter();previous=None
def record(self,it,loss_dict,extra_log_dicts,width=80,pad=35):
    global previous
    import torch
    now=time.perf_counter()
    values={k:float(v) for k,v in loss_dict.items() if isinstance(v,(int,float)) or torch.is_tensor(v) and v.numel()==1}
    if not all(math.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite loss')
    row=dict(iteration=int(it),num_envs=self.num_envs,wall_iteration_s=None if previous is None else now-previous,collection_s=self.collection_time,learning_s=self.learn_time,seconds_since_entry=now-started,losses=values,log_dir=str(self.log_dir),cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    previous=now
    with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    original_logging(self,it,loss_dict,extra_log_dicts,width,pad)
    if args.no_domain_randomization and int(it)==3:
        current=verified_student_agent.policy.state_dict()
        changed_keys=[k for k,v in current.items() if not torch.equal(v.detach().cpu(),initial_student_state[k])]
        teacher_keys=[k for k in current if k.startswith('teachers.') or k.startswith('teacher_obs_normalizers.')]
        assert teacher_keys and not set(teacher_keys)&set(changed_keys)
        if args.student_perception=='depth':
            assert any(k.startswith('depth_backbone.') for k in changed_keys)
        else:
            assert not any('depth' in k for k in current)
            assert verified_student_agent.storage._buffers['depth_obs'].numel()==0
        assert any(k.startswith('student.') for k in changed_keys)
        proof=dict(iteration=int(it),changed_keys=changed_keys,frozen_teacher_tensor_count=len(teacher_keys),teachers_exactly_frozen=True,
                   depth_cnn_updated=args.student_perception=='depth',student_updated=True,finite_losses=True,
                   student_perception=args.student_perception,student_input_dim=verified_student_agent.actor_obs_dim,
                   cnn_present=hasattr(verified_student_agent.policy,'depth_backbone'))
        (OUT/'student_update_verification.json').write_text(json.dumps(proof,indent=2))
LoggingHelper.post_epoch_logging=record

# Capture actual environment construction once. Finite observation groups and
# the real Warp depth frames are checked before the first optimizer update.
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
original_reset=WholeBodyTrackingManager.reset_all
captured=False
def reset_all(self):
    global captured
    obs=original_reset(self)
    if not captured:
        import numpy as np,torch
        cmd=self.command_manager.get_state('motion_command')
        assert self.dt==.02 and self.num_envs==N and cmd.motion.time_step_total==copies*expected_frames
        shapes={};depth={}
        for group,values in obs.items():
            values=values if isinstance(values,dict) else {'flat':values}
            for name,value in values.items():
                shapes[group+'/'+name]=list(value.shape)
                if group!='depth_camera':assert torch.isfinite(value).all(),group
                else:
                    finite=torch.isfinite(value)
                    depth[name]=dict(finite_fraction=float(finite.float().mean()),positive_fraction=float((value>0).float().mean()),min=float(value[finite].min()) if finite.any() else None,max=float(value[finite].max()) if finite.any() else None)
                    assert finite.any() and (value[finite]>0).any(),'Empty Warp depth image'
        info=dict(num_envs=N,control_dt=self.dt,physics_dt=self.simulator.sim_dt,motion_frames=cmd.motion.time_step_total,reference_end_flags=int(cmd.motion.motion_ends.sum()),obs_shapes=shapes,depth=depth,gpu=torch.cuda.get_device_name(0),renderer_cameras=False,physics_clock_fix=bool(getattr(self.simulator.sim.step,'_php_physics_clock_fixed',False)),body_count=len(self.simulator._body_list),joint_count=len(self.dof_names),filesystem_isolated=True,landlock_confined=True,python=sys.executable,python_prefix=sys.prefix,holosoma=holosoma.__file__,torch=torch.__file__,cwd=os.getcwd())
        assert info['physics_clock_fix']
        info['gpu_max_rigid_patch_count']=int(self.simulator.sim.cfg.physx.gpu_max_rigid_patch_count)
        info['self_collision_config']=self.robot_config.asset.enable_self_collisions
        info['native_preview_surface_command']=getattr(self.simulator,'_php_preview_surface_command_name',None)
        info['native_articulation_self_collision']=articulation_self_collision(self,requested_self_collision)
        info['registries']=[entry.reference for entry in registries]
        info['teacher_checkpoints']=[str(path.resolve()) for path in teacher_paths]
        if args.stage=='student':
            from wbt_training.utils.student_teacher import routing_column_index
            route=obs['teacher_obs'][:,routing_column_index(self)]
            keys,counts=torch.unique(route.long(),return_counts=True)
            info['teacher_routing_histogram']={str(int(k)):int(v) for k,v in zip(keys,counts)}
            assert torch.all(route==route.long())
            assert all(-1<=int(k)<len(registries) for k in keys)
            if N>=128:assert set(int(k) for k in keys if k>=0)==set(range(len(registries)))
        if args.no_domain_randomization:
            from wbt_training.config_values.no_domain_randomization import verify_environment
            info['domain_randomization_verification']=verify_environment(self,obs)
        if args.student_perception=='heightmap':
            from wbt_training.config_values.heightmap_student import verify_environment as verify_heightmap
            info['heightmap_verification']=verify_heightmap(self,obs)
        (OUT/'runtime_probe.json').write_text(json.dumps(info,indent=2))
        import wandb
        if wandb.run is not None:
            wandb.run.summary['setup_validation']=info
            (OUT/'run_identity.json').write_text(json.dumps(dict(id=wandb.run.id,url=wandb.run.url,entity=wandb.run.entity,project=wandb.run.project,pid=os.getpid()),indent=2))
        np.savez_compressed(OUT/'initial_observations.npz',**{g+'_'+k:v[:8].detach().cpu().numpy() for g,value in obs.items() for k,v in (value.items() if isinstance(value,dict) else [('flat',value)])})
        captured=True
    return obs
WholeBodyTrackingManager.reset_all=reset_all

# The checkpoint is verified after native load, including both Adam states.
# Only absolute-budget resumes advance past the completed saved iteration.
if args.stage=='teacher' and args.checkpoint and args.total_iterations is not None:
    from holosoma.agents.ppo.ppo import PPO
    native_learn=PPO.learn
    def resume_learn(self):
        proof=verify_resume(self,args.checkpoint,OUT,start_iteration)
        self.current_learning_iteration=start_iteration
        self.logging_helper.num_learning_iterations=target_iterations
        import wandb
        if wandb.run is not None:wandb.run.summary['resume_lineage']=proof
        print('RESUME_VERIFIED',json.dumps(dict(checkpoint_iteration=checkpoint_iteration,next_iteration=start_iteration,target_iterations=target_iterations,weights_and_optimizer_exact=True)),flush=True)
        return native_learn(self)
    PPO.learn=resume_learn

if args.no_domain_randomization:
    from wbt_training.utils.student_teacher import PhpDistillationPPO
    original_student_learn=PhpDistillationPPO.learn
    def verified_student_learn(self):
        global verified_student_agent, initial_student_state
        verified_student_agent=self
        initial_student_state={k:v.detach().cpu().clone() for k,v in self.policy.state_dict().items()}
        return original_student_learn(self)
    PhpDistillationPPO.learn=verified_student_learn

from holosoma.utils.sim_utils import setup_simulator_imports
from holosoma.train_agent import TrainingContext
setup_simulator_imports(cfg)
# Kit terminates the interpreter in app.close(). Save completion only after
# learn() returns and before closing the app; process exit zero alone is unsafe.
with TrainingContext(cfg) as context:
    context.train()
    assert captured,'No real environment rollout was captured'
    (OUT/'finished.json').write_text(json.dumps(dict(completed=True,stage=args.stage,iterations=args.iterations,total_iterations=target_iterations,start_iteration=start_iteration,final_iteration=target_iterations-1,self_collision=cfg.robot.asset.enable_self_collisions,num_envs=N,training_wall_s=time.perf_counter()-started),indent=2))
