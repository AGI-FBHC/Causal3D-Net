# -*- coding: utf-8 -*-
# @Time    : 2025/5/12 21:04
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: PANDA.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="shape"):
        """
        :param conv_type: `shape` (HW downsampling), `depth` (DHW downsampling), `fixed` (no downsampling).
        """
        super().__init__()
        # 决定 stride
        if conv_type == "shape":
            stride = (1, 2, 2)
        elif conv_type == "depth":
            stride = 2  # 等价于 (2, 2, 2)
        elif conv_type == "fixed":
            stride = 1
        else:
            raise ValueError(f"Invalid conv_type: {conv_type}.")

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="fixed"):
        """
        :param conv_type:
            - 'fixed': no upsampling
            - 'depth': full DHW upsampling
            - 'shape': only H and W upsampling
        """
        super().__init__()
        if conv_type == "shape":
            kernel_size = (3, 4, 4)
            stride = (1, 2, 2)
        elif conv_type == "depth":
            kernel_size = 4  # 等价于 (4, 4, 4)
            stride = 2
        elif conv_type == "fixed":
            kernel_size = 3
            stride = 1
        else:
            raise ValueError(f"Invalid conv_type: {conv_type}")

        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpFixedConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="fixed"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpDepthConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="depth"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpShapeConv(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="shape"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)




class MultiTask3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder path
        self.enc1 = ConvBlock3D(1, 32)     # 输入为1通道 (160x256x40)
        self.enc2 = ConvBlock3D(32, 64)
        self.enc3 = ConvBlock3D(64, 128)
        self.enc4 = ConvBlock3D(128, 256)
        self.enc5 = ConvBlock3D(256, 320)
        self.enc6 = ConvBlock3D(320, 320)

        # Decoder path (symmetric)
        self.dec1 = ConvBlock3D(320, 320)
        self.dec2 = ConvBlock3D(320, 320)
        self.dec3 = ConvBlock3D(320, 256)
        self.dec4 = ConvBlock3D(256, 128)
        self.dec5 = ConvBlock3D(128, 64)
        self.dec6 = ConvBlock3D(64, 32)

        # Segmentation output head
        self.seg_head = nn.Conv3d(32, 2, kernel_size=1)  # 假设输出为2类：胰腺和病灶

        # 分类用的 Global Pooling + FC
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc_layers = nn.ModuleList([
            nn.Linear(c, 2) for c in [32, 64, 128, 256, 320, 320]  # 多层输出用于分类
        ])

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, 2))
        e3 = self.enc3(F.max_pool3d(e2, 2))
        e4 = self.enc4(F.max_pool3d(e3, 2))
        e5 = self.enc5(F.max_pool3d(e4, 2))
        e6 = self.enc6(F.max_pool3d(e5, 2))

        # Decoder
        d1 = self.dec1(F.interpolate(e6, scale_factor=2))
        d2 = self.dec2(F.interpolate(d1, scale_factor=1))  # no change in shape
        d3 = self.dec3(F.interpolate(d2, scale_factor=2))
        d4 = self.dec4(F.interpolate(d3, scale_factor=2))
        d5 = self.dec5(F.interpolate(d4, scale_factor=2))
        d6 = self.dec6(F.interpolate(d5, scale_factor=2))

        # Segmentation output
        seg_out = self.seg_head(d6)

        # Classification outputs (global pooling on encoder features)
        enc_features = [e1, e2, e3, e4, e5, e6]
        class_preds = []
        for i, feat in enumerate(enc_features):
            pooled = self.global_pool(feat).view(feat.size(0), -1)
            class_preds.append(self.fc_layers[i](pooled))

        # 平均多个层的分类预测结果（如图所示用加号）
        avg_class_pred = torch.stack(class_preds, dim=0).mean(dim=0)

        return seg_out, avg_class_pred
