# minigpt4/processors/medical_processors.py

import numpy as np
import torch
from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf

@registry.register_processor("pancreas_3d_train")
class Pancreas3dTrainProcessor(BaseProcessor):
    def __init__(self, patch_size=96):
        self.patch_size = patch_size

    def __call__(self, volume_patch):
        # 归一化 (示例：窗宽窗位后归一化到-1, 1)
        lower, upper = -150, 250
        volume_patch = np.clip(volume_patch, lower, upper)
        volume_patch = (volume_patch - lower) / (upper - lower) * 2.0 - 1.0

        # 转换成Tensor, 并增加channel维度
        volume_tensor = torch.from_numpy(volume_patch).float().unsqueeze(0) # (1, D, H, W)
        return volume_tensor

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        patch_size = cfg.get("patch_size", 96)
        return cls(patch_size=patch_size)