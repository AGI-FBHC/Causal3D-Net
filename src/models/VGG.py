# -*- coding: utf-8 -*-
# @Time    : 2025/6/27 10:20
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: VGG.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights
import torchvision.models as models


class VGG25D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, slice_fusion='avg'):
        super().__init__()
        self.slice_fusion = slice_fusion

        # 正确方式加载预训练权重
        if pretrained:
            weights = VGG16_BN_Weights.IMAGENET1K_V1
        else:
            weights = None

        vgg = vgg16_bn(weights=weights)
        self.feature_extractor = vgg.features

        self.slice_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, C, D, H, W = x.shape
        assert C == 1, "Input channel must be 1 for grayscale CT"

        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * D, 1, H, W)
        x = x.repeat(1, 3, 1, 1)

        feats = self.feature_extractor(x)
        feats = self.slice_fc(feats)
        feats = feats.view(B, D, -1)

        if self.slice_fusion == 'avg':
            volume_feat = feats.mean(dim=1)
        elif self.slice_fusion == 'max':
            volume_feat = feats.max(dim=1)[0]
        else:
            raise ValueError("Unsupported fusion mode")

        out = self.classifier(volume_feat)
        return out




class VGG(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, slice_fusion: str = 'avg'):
        super().__init__()
        self.slice_fusion = slice_fusion

        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg16_bn(weights=weights)
        self.feature_extractor = backbone.features
        self.slice_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, D, H, W), grayscale CT slices
        Returns:
            out: Tensor of shape (B, num_classes)
        """
        B, C, D, H, W = x.shape
        assert C == 1, f"Expected 1 input channel, got {C}"

        x = x.permute(0, 2, 1, 3, 4).reshape(B * D, 1, H, W)
        x = x.repeat(1, 3, 1, 1)

        feats = self.feature_extractor(x)
        feats = self.slice_fc(feats)
        feats = feats.view(B, D, -1)

        if self.slice_fusion == 'avg':
            volume_feat = feats.mean(dim=1)
        elif self.slice_fusion == 'max':
            volume_feat = feats.max(dim=1).values
        else:
            raise ValueError(f"Unsupported fusion mode: {self.slice_fusion}")

        out = self.classifier(volume_feat)    # (B, num_classes)
        return out


if __name__ == "__main__":
    x = torch.randn(8, 1, 50, 160, 256)
    model = VGG(num_classes=2)
    out = model(x)
    print(out.shape)


