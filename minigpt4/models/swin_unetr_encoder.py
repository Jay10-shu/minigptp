# minigpt4/models/swin_unetr_encoder.py
# 这个文件用于加载预训练的Swin UNETR并提供其编码器
from monai.networks.nets import SwinUNETR
import torch

def get_swin_unetr_encoder(pretrained_weights_path, img_size=(96, 96, 96), feature_size=48):
    model = SwinUNETR(
        # in_size=img_size,
        in_channels=1,
        out_channels=14, # 预训练时的输出类别数，不影响编码器使用
        feature_size=feature_size,
        use_checkpoint=True,
    )

    # 加载预训练权重
    weights = torch.load(pretrained_weights_path)
    model.load_state_dict(weights)

    encoder = model.swinViT
    return encoder