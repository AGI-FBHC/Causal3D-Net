# -*- coding: utf-8 -*-
# @Time    : 2025/6/14 15:22
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: Causal3DNet.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Type, List, Tuple


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.InstanceNorm3d(
            out_channels,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=False
        )
        self.nonlin = nn.LeakyReLU(0.01, inplace=True)
        self.all_modules = nn.Sequential(self.conv, self.norm, self.nonlin)

    def forward(self, x):
        return self.all_modules(x)


class StackedConvBlocks(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding=1):
        super().__init__()
        self.convs = nn.Sequential(
            ConvDropoutNormReLU(in_channels, out_channels, kernel_size, stride, padding),
            ConvDropoutNormReLU(out_channels, out_channels, kernel_size, 1, padding)
        )

    def forward(self, x):
        return self.convs(x)


class PlainConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(StackedConvBlocks(1, 32, stride=1)),
            nn.Sequential(StackedConvBlocks(32, 64, stride=(1, 2, 2))),
            nn.Sequential(StackedConvBlocks(64, 128, stride=(1, 2, 2))),
            nn.Sequential(StackedConvBlocks(128, 256, stride=2)),
            nn.Sequential(StackedConvBlocks(256, 320, stride=2)),
            nn.Sequential(StackedConvBlocks(320, 320, stride=2))
        ])

    def forward(self, x):
        # 存储各阶段输出的列表
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs  # 返回包含所有层级输出的列表


class UNetDecoder(nn.Module):
    def __init__(self, mask_num: int = 2):
        super().__init__()

        self.transpconvs = nn.ModuleList([
            nn.ConvTranspose3d(320, 320, kernel_size=2, stride=2),
            nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        ])

        self.stages = nn.ModuleList([
            StackedConvBlocks(640, 320),  # 320(skip) + 320(transp)
            StackedConvBlocks(512, 256),  # 256 + 256
            StackedConvBlocks(256, 128),  # 128 + 128
            StackedConvBlocks(128, 64),  # 64 + 64
            StackedConvBlocks(64, 32)  # 32 + 32
        ])

        self.seg_layers = nn.ModuleList([
            nn.Conv3d(320, mask_num, kernel_size=1),
            nn.Conv3d(256, mask_num, kernel_size=1),
            nn.Conv3d(128, mask_num, kernel_size=1),
            nn.Conv3d(64, mask_num, kernel_size=1),
            nn.Conv3d(32, mask_num, kernel_size=1)
        ])

    def forward(self, encoder_outputs: List[torch.Tensor]):
        x = encoder_outputs[-1]
        segment_outputs = []
        classfiy_outputs = []

        for i in range(5):
            x = self.transpconvs[i](x)
            # 获取对应层级的编码器特征(索引从倒数第二层开始)
            skip = encoder_outputs[-(i + 2)]
            x = torch.cat([x, skip], dim=1)
            x = self.stages[i](x)
            segment_outputs.append(self.seg_layers[i](x))
            classfiy_outputs.append(x)

        return segment_outputs[::-1], classfiy_outputs


class ChannelAttentionDecoder(nn.Module):
    def __init__(self,
                 channel: int = 1920,
                 groups: int = 24,
                 class_num: int = 2,
                 eps=1e-5):
        super().__init__()
        self.eps = eps
        self.channel = channel
        self.groups = groups
        self.perc = channel // groups
        self.cfc1 = torch.nn.Parameter(torch.Tensor(groups, 2))
        self.cfc1.data.fill_(1e-5)
        self.cfc2 = torch.nn.Parameter(torch.Tensor(self.perc, 2))
        self.cfc2.data.fill_(1e-5)

        self.bn1 = nn.BatchNorm1d(self.groups)
        self.bn2 = nn.BatchNorm1d(self.perc)
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.classify = nn.Linear(self.channel, class_num)

    def forward(self, enc_x: List[torch.Tensor], dec_x: List[torch.Tensor]):
        all_feats = enc_x + dec_x  # List of tensors with shape [B, C, D, H, W]
        pooled = [F.adaptive_avg_pool3d(feat, output_size=1).view(feat.size(0), -1)
                  for feat in all_feats]  # -> [B, C]
        x = torch.cat(pooled, dim=1)  # -> [B, sum(C)]
        N, C = x.size()
        x = x.view(N, self.groups, self.perc)
        res = x

        channel_mean = x.mean(dim=2, keepdim=True)
        channel_std = (x.var(dim=2, keepdim=True) + self.eps).sqrt()
        t1 = torch.cat((channel_mean, channel_std), dim=2)
        z1 = t1 * self.cfc1[None, :, :]
        z1 = self.softmax1(self.bn1(torch.sum(z1, dim=2, keepdim=True)))

        channel_mean = x.permute(0, 2, 1).mean(dim=2, keepdim=True)
        channel_std = (x.permute(0, 2, 1).var(dim=2, keepdim=True) + self.eps).sqrt()
        t2 = torch.cat((channel_mean, channel_std), dim=2)
        z2 = t2 * self.cfc2[None, :, :]
        z2 = self.softmax2(self.bn2(torch.sum(z2, dim=2, keepdim=True)))

        out = res * torch.sigmoid(torch.matmul(z1, z2.permute(0, 2, 1)))
        out = out + res
        out = out.view(N, C)

        return self.classify(out)


class SegNet(nn.Module):
    def __init__(self, mask_num: int = 2):
        super().__init__()
        self.encoder = PlainConvEncoder()
        self.seg_decoder = UNetDecoder(mask_num=mask_num)

    def forward(self, x):
        skip = self.encoder(x)
        return self.seg_decoder(skip)


class Causal3DNet(nn.Module):
    def __init__(self, class_num: int = 2, groups: int = 24):
        super().__init__()
        self.encoder = PlainConvEncoder()
        self.seg_decoder = UNetDecoder(mask_num=2)
        self.cls_decoder = ChannelAttentionDecoder(channel=1920,
                                                   groups=groups,
                                                   class_num=class_num)

    def forward(self, x):
        skip = self.encoder(x)
        seg_out, cls_features = self.seg_decoder(skip)
        return seg_out, self.cls_decoder(skip, cls_features)


if __name__ == "__main__":
    x = torch.randn(4, 1, 40, 160, 256)
    model = Causal3DNet(class_num=2)
    y_seg, y_cls = model(x)
    for y in y_seg:
        print(y.shape)
    print("="*20)
    print(y_cls.shape)


