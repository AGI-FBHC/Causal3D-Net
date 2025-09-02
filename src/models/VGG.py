# -*- coding: utf-8 -*-
# @Time    : 2025/6/27 10:20
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: VGG.py
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




def make_layers(cfg, in_channels=1, batch_norm=True):  # 注意 in_channels=1
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=not batch_norm)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):

    def __init__(self, cfg_name="D", num_classes=2, in_channels=1, batch_norm=True, dropout=0.5):
        super().__init__()
        self.CFG = {
            "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, "M"],
            "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, "M"],
        }
        self.features = make_layers(self.CFG[cfg_name], in_channels=in_channels, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应池化
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = VGG(cfg_name="D", num_classes=2, in_channels=1).to(device)
    x = torch.randn(8, 1, 50, 50).to(device)
    logits = model(x)
    print(logits.shape)


