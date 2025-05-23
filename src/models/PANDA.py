# -*- coding: utf-8 -*-
# @Time    : 2025/5/12 21:04
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: PANDA.py
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
        # 使用ModuleList显式存储各阶段（更清晰的访问方式）
        self.stages = nn.ModuleList([
            nn.Sequential(StackedConvBlocks(1, 32, stride=1)),
            nn.Sequential(StackedConvBlocks(32, 64, stride=2)),
            nn.Sequential(StackedConvBlocks(64, 128, stride=2)),
            nn.Sequential(StackedConvBlocks(128, 256, stride=2)),
            nn.Sequential(StackedConvBlocks(256, 320, stride=2)),
            nn.Sequential(StackedConvBlocks(320, 320, stride=(1, 2, 2)))
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
            nn.ConvTranspose3d(320, 320, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.ConvTranspose3d(320, 256, kernel_size=2, stride=2),
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
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


class ClassifierDecoder(nn.Module):
    def __init__(self, class_num: int = 2):
        super().__init__()

        # 各层级特征处理模块
        self.stage_processors = nn.ModuleList([
            # 每个层级的处理流：全局池化 → FC
            nn.Sequential(
                nn.AdaptiveMaxPool3d(1),  # [B,320,1,1,1]
                nn.Flatten(),  # [B,320]
                nn.Linear(320, 64),  # 降维到64
                nn.LeakyReLU(0.01, inplace=True)  # 1/5.5 备选
            ),
            nn.Sequential(
                nn.AdaptiveMaxPool3d(1),  # [B,256,1,1,1]
                nn.Flatten(),
                nn.Linear(256, 64),
                nn.LeakyReLU(0.01, inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveMaxPool3d(1),  # [B,128,1,1,1]
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.01, inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveMaxPool3d(1),  # [B,64,1,1,1]
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.LeakyReLU(0.01, inplace=True)
            ),
            nn.Sequential(
                nn.AdaptiveMaxPool3d(1),  # [B,32,1,1,1]
                nn.Flatten(),
                nn.Linear(32, 64),
                nn.LeakyReLU(0.01, inplace=True)
            )
        ])

        # 最终分类层（融合所有层级特征）
        self.final_classifier = nn.Sequential(
            nn.Linear(64 * 5, 128),  # 融合特征维度64 * 5=320
            nn.LeakyReLU(0.01),
            # nn.Dropout(0.5),
            nn.Linear(128, class_num)
        )

    def forward(self, classify_outputs: List[torch.Tensor]):
        """
        :param classify_outputs: 来自UNetDecoder的5个层级特征图，形状依次为：
            [B,320,D1,H1,W1]
            [B,256,D2,H2,W2]
            [B,128,D3,H3,W3]
            [B,64,D4,H4,W4]
            [B,32,D5,H5,W5]
        """
        # 特征处理分支
        processed_features = []
        for feat, processor in zip(classify_outputs, self.stage_processors):
            processed = processor(feat)  # 每个特征输出[B,64]
            processed_features.append(processed)

        # 特征拼接
        combined = torch.cat(processed_features, dim=1)  # [B, 320]

        # 最终分类
        return self.final_classifier(combined)  # [B, class_num]


class SegNet(nn.Module):
    def __init__(self, mask_num: int = 2):
        super().__init__()
        self.encoder = PlainConvEncoder()
        self.seg_decoder = UNetDecoder(mask_num=mask_num)

    def forward(self, x):
        skip = self.encoder(x)
        return self.seg_decoder(skip)


class MultiTask3DCNN(nn.Module):
    def __init__(self, mask_num: int = 2, cls_num: int = 2):
        super().__init__()
        self.encoder = PlainConvEncoder()
        self.seg_decoder = UNetDecoder(mask_num=mask_num)
        self.cls_decoder = ClassifierDecoder(class_num=cls_num)

    def forward(self, x):
        skip = self.encoder(x)
        seg_out, cls_features = self.seg_decoder(skip)
        return seg_out, self.cls_decoder(cls_features)


# 测试代码
if __name__ == "__main__":
    stage1_model = SegNet()
    stage2_model = MultiTask3DCNN()

    # 前向传播流程
    input_tensor = torch.randn(1, 1, 80, 160, 256)  # (B, C, D, H, W)
    seg_outs, _ = stage1_model(input_tensor)
    for _, u in enumerate(seg_outs):
        print(f"{_}: {u.size()}")
    seg_outs, cls_outs = stage2_model(input_tensor)
    for _, u in enumerate(seg_outs):
        print(f"{_}: {u.size()}")
    print(cls_outs.shape)





