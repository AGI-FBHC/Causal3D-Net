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
        :param conv_type:
            - 'fixed': no downsampling.
            - 'depth': full DHW downsampling.
            - 'shape': only H and W downsampling.
        """
        super().__init__()
        # 决定 stride
        if conv_type == "shape":
            stride = (1, 2, 2)
        elif conv_type == "depth":
            stride = 2
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
    def __init__(self, in_channels, out_channels, conv_type="shape"):
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
            kernel_size = 4
            stride = 2
        elif conv_type == "fixed":
            kernel_size = 3
            stride = 1
        else:
            raise ValueError(f"Invalid conv_type: {conv_type}")

        self.conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose3d(
                out_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class MultiTask3DCNN(nn.Module):
    def __init__(self, mask_num=2):
        super().__init__()
        self.mask_num = mask_num
        # Encoder path
        self.enc1 = DownConv(1, 32, conv_type="fixed")
        self.enc2 = DownConv(32, 64, conv_type="shape")
        self.enc3 = DownConv(64, 128, conv_type="shape")
        self.enc4 = DownConv(128, 256, conv_type="depth")
        self.enc5 = DownConv(256, 320, conv_type="depth")
        self.enc6 = DownConv(320, 320, conv_type="depth")
        self.enc7 = DownConv(320, 320, conv_type="fixed")

        # Decoder path (symmetric)
        self.dec1 = UpConv(320, 320, conv_type="depth")
        self.dec2 = UpConv(320, 256, conv_type="depth")
        self.dec3 = UpConv(256, 128, conv_type="depth")
        self.dec4 = UpConv(128,  64, conv_type="shape")
        self.dec5 = UpConv(64,  32, conv_type="shape")
        self.dec6 = UpConv(32,  self.mask_num, conv_type="fixed")

        # # 分类用的 Global Pooling + FC
        # self.global_pool = nn.AdaptiveAvgPool3d(1)
        # self.fc_layers = nn.ModuleList([
        #     nn.Linear(c, 2) for c in [32, 64, 128, 256, 320, 320]  # 多层输出用于分类
        # ])

    def forward(self, x):
        e1 = self.enc1(x)  # e1.shape = (B, 32, 40, 160, 256)
        e2 = self.enc2(e1)  # e2.shape = (B, 64, 40 80 128)
        e3 = self.enc3(e2)  # e3.shape = (B, 128, 40, 40, 64)
        e4 = self.enc4(e3)  # e4.shape = (B, 256, 20, 20, 32)
        e5 = self.enc5(e4)  # e5.shape = (B, 320, 10, 10, 16)
        e6 = self.enc6(e5)  # e6.shape = (B, 320, 5, 5, 8)

        d1 = self.enc7(e6)  # d1.shape = (B, 320, 5, 5, 8)
        d2 = self.dec1(d1)  # d2.shape = (B, 320, 10, 10, 16)
        d3 = self.dec2(d2)  # d3.shape = (B, 256, 20, 20, 32)
        d4 = self.dec3(d3)  # d4.shape = (B, 128, 40, 40, 64)
        d5 = self.dec4(d4)  # d5.shape = (B, 64, 40, 80, 128)
        d6 = self.dec5(d5)  # d6.shape = (B, 32, 40, 160, 256)
        seg = self.dec6(d6)  # seg.shape = (B, self.mask_num, 40, 160, 256)

        # # Segmentation output
        # seg_out = self.seg_head(d6)
        #
        # # Classification outputs (global pooling on encoder features)
        # enc_features = [e1, e2, e3, e4, e5, e6]
        # class_preds = []
        # for i, feat in enumerate(enc_features):
        #     pooled = self.global_pool(feat).view(feat.size(0), -1)
        #     class_preds.append(self.fc_layers[i](pooled))
        #
        # # 平均多个层的分类预测结果（如图所示用加号）
        # avg_class_pred = torch.stack(class_preds, dim=0).mean(dim=0)

        return seg
