# # Save this in a new file: /home/sj/MiniGPT/minigpt4/processors/pancreas_processors.py

# from minigpt4.common.registry import registry
# from minigpt4.processors.base_processor import BaseProcessor
# from torchvision import transforms
# # For real 3D data, you would likely use a specialized library like monai or torchio
# # e.g., from monai.transforms import Compose, RandCropByPosNegLabeld, etc.

# @registry.register_processor("pancreas_3d_train") # <--- This name MUST match your YAML file
# class Pancreas3dTrainProcessor(BaseProcessor):
#     def __init__(self, patch_size=96):
#         # Define your 3D transformations here.
#         # This is just an example; you'll need to use appropriate 3D transforms for your task.
#         self.transform = transforms.Compose([
#             # Example: transforms.RandomCrop(patch_size),
#             # Example: transforms.ToTensor(),
#             # ... your other 3D-specific augmentations ...
#         ])

#     def __call__(self, item):
#         # This function applies the transformations to the input item (e.g., a 3D image)
#         return self.transform(item)

#     @classmethod
#     def from_config(cls, cfg=None):
#         # This reads the 'patch_size' from your YAML configuration
#         patch_size = cfg.get("patch_size", 96)
#         return cls(patch_size=patch_size)
# /home/sj/MiniGPT/minigpt4/processors/pancreas_processors.py

import numpy as np
import torch
from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor

# 使用 @registry 装饰器将类注册到系统中
# YAML 配置文件会通过 "pancreas_3d_train" 这个名字找到这个类
@registry.register_processor("pancreas_3d_train")
class Pancreas3DTrainProcessor(BaseProcessor):
    def __init__(self, patch_size=96):
        self.patch_size = patch_size

    def __call__(self, volume_patch):
        """
        处理3D图像块，确保输出是单通道的PyTorch Tensor。
        """
        # 确保输入是numpy数组
        if not isinstance(volume_patch, np.ndarray):
            volume_patch = np.array(volume_patch)

        # 在最前面增加一个通道维度: (D, H, W) -> (1, D, H, W)
        volume_with_channel = np.expand_dims(volume_patch, axis=0)

        # 转换为PyTorch的Tensor
        volume_tensor = torch.from_numpy(volume_with_channel).float()

        return volume_tensor

    @classmethod
    def from_config(cls, cfg=None):
        """
        这个函数允许框架从YAML配置文件中读取参数来创建类的实例。
        """
        if cfg is None:
            cfg = {}
        patch_size = cfg.get("patch_size", 96)
        return cls(patch_size=patch_size)