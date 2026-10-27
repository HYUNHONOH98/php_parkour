"""No-camera/CNN adapter retaining the native DAgger+PPO loss and expert routing."""
import copy,json,os
from pathlib import Path
import torch
from torch import nn
from holosoma.agents.distillation_ppo.distillation_ppo import DistillationPPO
from holosoma.agents.modules.student_teacher_modules import StudentTeacherCritic
from wbt_training.utils.student_teacher import PhpDistillationPPO,RoutingColumnTeacherMixin,routing_column_index


class RoutingHeightmapStudentTeacherCritic(RoutingColumnTeacherMixin,StudentTeacherCritic):
    pass


class PhpHeightmapDistillationPPO(PhpDistillationPPO):
    def _resolve_obs_dims(self):
        dims=self.env.observation_manager.get_obs_dims()
        assert set(dims)=={'actor_obs','teacher_obs','critic_obs'}
        assert dims['actor_obs']==397 and dims['teacher_obs']==450 and dims['critic_obs']==575
        self.actor_obs_dim=dims['actor_obs'];self.teacher_obs_dim=dims['teacher_obs'];self.critic_obs_dim=dims['critic_obs']
        self.num_act=self.env.robot_config.actions_dim;assert self.num_act==29
        # The shared trainer/storage transports a zero-element optional slot.
        # No sensor, image, encoder or feature is allocated/used through it.
        self.depth_shape=(0,)

    def _build_policy(self):
        m=self.config.module
        assert m.depth_backbone=='' and m.depth_output_dim==0
        self.policy=RoutingHeightmapStudentTeacherCritic(num_actor_obs=397,num_teacher_obs=450,
            num_critic_obs=575,num_actions=29,student_hidden_dims=list(m.student_hidden_dims),
            teacher_hidden_dims=list(m.teacher_hidden_dims),critic_hidden_dims=list(m.critic_hidden_dims),
            activation=m.activation,init_noise_std=self.config.init_noise_std,num_teachers=self.num_teachers,
            routing_column=routing_column_index(self.env)).to(self.device)
        assert not hasattr(self.policy,'depth_backbone')
        assert not any(isinstance(x,(nn.Conv1d,nn.Conv2d,nn.Conv3d))for x in self.policy.modules())
        assert [x.in_features for x in self.policy.student if isinstance(x,nn.Linear)]+[29]==[397,2048,1024,512,256,128,29]

    def _build_optimizer(self):
        self.optimizer=torch.optim.AdamW([
            {'params':list(self.policy.student.parameters()),'lr':self.config.learning_rate,'weight_decay':self.config.weight_decay},
            {'params':list(self.policy.critic.parameters()),'lr':self.config.learning_rate,'weight_decay':self.config.weight_decay},
            {'params':[self.policy.std],'lr':self.config.learning_rate,'weight_decay':0.}],
        )
        self._STD_GROUP_IDX=2

    def _extract_groups(self,obs_dict):
        actor=obs_dict['actor_obs']
        return actor,obs_dict['teacher_obs'],obs_dict['critic_obs'],actor.new_empty((actor.shape[0],0))

    @property
    def actor_onnx_wrapper(self):
        return copy.deepcopy(self.policy.student).cpu().eval()

    def export(self,onnx_file_path):
        from wbt_training.utils.exporter import attach_onnx_metadata,build_metadata_from_wbt_env
        target=Path(onnx_file_path);target.parent.mkdir(parents=True,exist_ok=True)
        actor=self.actor_onnx_wrapper
        torch.onnx.export(actor,torch.zeros(1,397),str(target),input_names=['obs'],output_names=['actions'],
            dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}},opset_version=13,dynamo=False)
        import onnx
        model=onnx.load(str(target));onnx.checker.check_model(model)
        assert not any(node.op_type.startswith('Conv')for node in model.graph.node)
        metadata=build_metadata_from_wbt_env(self.env)
        metadata.update(student_perception='heightmap',actor_input_dim=397,proprio_command_dim=108,heightmap_dim=289,
            actor_layout='Original108,HeightScan289',heightmap_grid=[17,17],heightmap_resolution_m=.1)
        attach_onnx_metadata(str(target),metadata)
        (target.parent/'observation_layout.json').write_text(json.dumps(metadata,indent=2,default=str))
        self.logging_helper.save_to_wandb(str(target))

    def save(self,path=None,name='last.ckpt'):
        # Bypass only the depth-specific export mixin; use the native checkpoint.
        DistillationPPO.save(self,path=path,name=name)
        target=Path(path if path is not None else os.path.join(self.log_dir,name)).with_suffix('.onnx')
        self.export(str(target))
