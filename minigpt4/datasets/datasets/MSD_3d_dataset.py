# minigpt4/datasets/datasets/MSD_3d_dataset.py

import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import nibabel as nib
import numpy as np


def sample_3d_patch(volume_data, patch_size=(96, 96, 96)):
    # 这里只是一个示例，你需要根据实际情况实现
    # 比如找到胰腺的中心，然后围绕中心裁剪一个patch
    d, h, w = volume_data.shape
    pd, ph, pw = patch_size

    # 简单的中心裁剪示例
    start_d = max(0, (d - pd) // 2)
    start_h = max(0, (h - ph) // 2)
    start_w = max(0, (w - pw) // 2)

    return volume_data[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]


class ReferMSDPancreas3DDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, **kwargs):
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

        # 为了避免重复加载，我们可以按volume组织annotations
        self.volumes = {}
        for item in self.ann:
            # !! 保持不变：仍然用volume_name来组织数据 !!
            vol_name = item["volume_name"]
            if vol_name not in self.volumes:
                self.volumes[vol_name] = []
            self.volumes[vol_name].append(item)
        self.volume_keys = list(self.volumes.keys())

        self.instruction_pool = [
            "[refer] give me the location of the {}",
            # ... 其他指令 ...
        ]

    def __len__(self):
        return len(self.volume_keys)

    def __getitem__(self, index):
        volume_name = self.volume_keys[index]
        
        # ---
        # !! 关键修改 !!
        # 1. 随机选择一个与该volume关联的annotation
        ref_info = random.choice(self.volumes[volume_name])
        # 2. 直接从这个annotation中获取完整的、正确的image_path
        volume_path = ref_info["image_path"]
        # ---
        
        nifti_img = nib.load(volume_path)
        volume_array = nifti_img.get_fdata().astype(np.float32)
        
        volume_patch = sample_3d_patch(volume_array) # 采样一个 (D, H, W) 的patch
        image_volume = self.vis_processor(volume_patch) # 输出 (C, D, H, W)
        
        sample_sentence = 'pancreas' # 或者 'pancreatic tumor'
        refer_sentence = self.text_processor(sample_sentence)
        
        # 注意：这里的bbox应该是相对于整个volume的，
        # 如果你做了patch采样，可能需要对bbox进行相应的坐标变换
        bbox = ref_info.get('pancreas_bbox_3d', [0,0,0,0,0,0]) 
        # 将bbox格式化为字符串
        bbox_str = "{{<{}><{}><{}><{}><{}><{}>}}".format(*bbox)

        instruction = random.choice(self.instruction_pool).format(refer_sentence)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image_volume, # 这是关键的3D张量
            "instruction_input": instruction,
            "answer": bbox_str,
            "image_id": volume_name,
        }