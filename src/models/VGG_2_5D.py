# -*- coding: utf-8 -*-
# @Time    : 2025/6/27 10:20
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: VGG_2_5D.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


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


if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

    model = VGG25D(num_classes=2).to(device)
    input_volume = torch.randn(4, 1, 50, 256, 256).to(device)

    output = model(input_volume)
    print(output.shape)
