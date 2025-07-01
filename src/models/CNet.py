# -*- coding: utf-8 -*-
# @Time    : 2025/6/30 21:15
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: CNet.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F


class OuterNetwork(nn.Module):
    """四组并行的VGG式特征提取器"""

    def __init__(self, in_channels=50):
        super().__init__()
        # Block 1 (64 filters)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding='same'),  # 使用参数
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Block 2 (128 filters)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Block 3 (256 filters)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Block 4 (256 filters, no pooling)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)


class MiddleNetwork(nn.Module):
    """特征融合网络（含NIN结构）"""

    def __init__(self, in_channels):
        super().__init__()
        # 论文描述的两级结构
        self.block = nn.Sequential(
            # 4个连续卷积层
            nn.Conv2d(in_channels, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),

            # NIN结构（1×1卷积）
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5)  # 论文要求的正则化
        )

    def forward(self, x):
        return self.block(x)


class CNet(nn.Module):
    """完整的C-Net架构"""

    def __init__(self, input_size=224, in_channels=50, num_classes=2):
        super().__init__()
        # Outer Networks (4组并行)
        self.outer_net1 = OuterNetwork(in_channels=in_channels)
        self.outer_net2 = OuterNetwork(in_channels=in_channels)
        self.outer_net3 = OuterNetwork(in_channels=in_channels)
        self.outer_net4 = OuterNetwork(in_channels=in_channels)

        # 计算MiddleNet输入通道数（4×256）
        self.mid_in_channels = 4 * 256

        # Middle Networks (2组)
        self.middle_net1 = MiddleNetwork(self.mid_in_channels)
        self.middle_net2 = MiddleNetwork(self.mid_in_channels)

        # Inner Network
        self.inner_net = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding='same'),  # 输入来自2个MiddleNet
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1),  # 1×1卷积
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 动态计算FC层输入尺寸
        with torch.no_grad():
            # 使用传入的in_channels参数
            dummy = torch.rand(1, in_channels, input_size, input_size)
            features = self._forward_features(dummy)
            fc_in_features = features.view(-1).shape[0]

        # 全连接层 (Sec 3.3描述)
        self.classifier = nn.Sequential(
            nn.Linear(fc_in_features, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()  # 二分类Sigmoid输出
        )

    def _forward_features(self, x):
        # Outer Networks并行处理
        out1 = self.outer_net1(x)
        out2 = self.outer_net2(x)
        out3 = self.outer_net3(x)
        out4 = self.outer_net4(x)

        # 沿通道轴拼接 (Eq.1)
        merged_outer = torch.cat([out1, out2, out3, out4], dim=1)

        # Middle Networks处理
        mid1 = self.middle_net1(merged_outer)
        mid2 = self.middle_net2(merged_outer)

        # 二次拼接
        merged_mid = torch.cat([mid1, mid2], dim=1)

        # Inner Network
        return self.inner_net(merged_mid)

    def forward(self, x):
        features = self._forward_features(x)
        features = torch.flatten(features, 1)
        return self.classifier(features)


if __name__ == "__main__":
    import torch

    batch_size = 4
    patch_size = 224
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(batch_size, 50, patch_size, patch_size).to(device)

    model = CNet(input_size=patch_size, num_classes=2).to(device)

    outputs = model(input_tensor)
    print(f"input shape: {input_tensor.shape}")
    print(f"output shape: {outputs.shape}")


