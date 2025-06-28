# -*- coding: utf-8 -*-
# @Time    : 2025/5/10 20:41
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: ViT.py
# @Project : Causal3D-Net
from monai.networks.nets import ViT
import torch.nn as nn
import torch


class ViTClassifier(nn.Module):
    def __init__(self, img_size=(40, 160, 256), num_classes=2):
        super().__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            proj_type='conv',
            pos_embed_type='learnable',
            classification=True,
            num_classes=num_classes,
            dropout_rate=0.1,
            spatial_dims=3
        )

    def forward(self, x):
        logits = self.vit(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits


if __name__ == '__main__':
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(4, 1, 50, 256, 256).to(device)

    model = ViTClassifier(img_size=(50, 256, 256), num_classes=2).to(device)

    # 设置为评估模式并执行前向传播
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    print("Output shape:", output.shape)
    print("Output logits:", output)
