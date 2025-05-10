# -*- coding: utf-8 -*-
# @Time    : 2025/5/10 20:41
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: ViT.py
# @Project : Causal3D-Net
from monai.networks.nets import ViT
import torch.nn as nn


class ViTClassifier(nn.Module):
    def __init__(self, img_size=(128, 256, 256), num_classes=2):
        super().__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=(128, 256, 256),
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
        return self.vit(x)
