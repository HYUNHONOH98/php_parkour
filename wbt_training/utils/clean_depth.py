"""Clean depth sensor: identical projection/preprocessing, no sensor augmentation."""
import torch
import torch.nn.functional as F
from holosoma.managers.observation.terms.depth import WarpDepthImageObsTerm
from holosoma.sensors.warp.camera_config.d435i_depth_config import G1FlatRsD435iConfig


class FixedG1DepthCameraConfig(G1FlatRsD435iConfig):
    randomize_placement=False
    min_translation={'cam_front_depth':[0.,0.,0.]}
    max_translation={'cam_front_depth':[0.,0.,0.]}
    min_euler_rotation_deg={'cam_front_depth':[0.,0.,0.]}
    max_euler_rotation_deg={'cam_front_depth':[0.,0.,0.]}


class CleanWarpDepthImageObsTerm(WarpDepthImageObsTerm):
    def __init__(self,cfg,env):
        assert cfg.params['enable_augmentation'] is False
        assert cfg.params['enable_holes'] is False and cfg.params['latency_frame']==0
        super().__init__(cfg,env)
        self.depth_offset.zero_()

    def _process_depth_images(self,depth_images,env_ids=None):
        depth_images=self._crop_depth_images(depth_images)
        depth_images=F.interpolate(depth_images.unsqueeze(1),size=self._resize_hw,
            mode='bicubic',align_corners=False,antialias=True).squeeze(1)
        depth_images=torch.clamp(depth_images,max=self.clipping_range[1])
        depth_images[depth_images<.15]=self.clipping_range[1]
        return self._normalize_depth_images(depth_images)
