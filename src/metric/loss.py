# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 10:39
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: loss.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSegmentationLoss(nn.Module):
    def __init__(self, scale_weights=None,
                 dice_weight=0.7,
                 ce_weight=0.3,
                 include_background=False):
        """
        多尺度分割损失函数

        参数:
            scale_weights: 各尺度损失的权重 (list of floats)
            dice_weight: Dice损失在总损失中的权重
            ce_weight: 交叉熵损失在总损失中的权重
            include_background: 是否计算背景通道
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background

        # 设置各尺度权重
        if scale_weights is None:
            # 默认权重：高分辨率更重要
            self.scale_weights = [0.5, 0.25, 0.15, 0.07, 0.03]
        else:
            assert len(scale_weights) == 5, "scale_weights must have 5 elements"
            self.scale_weights = scale_weights

        # 初始化基础损失函数
        self.dice_loss = DiceLoss(self.include_background)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds_list, target):
        """
        计算多尺度损失

        参数:
            preds_list: 多尺度预测列表 (5个元素)
            target: 金标准标签 (B, 1, D, H, W)

        返回:
            加权后的多尺度损失
        """
        total_loss = 0.0

        # 遍历所有尺度
        for i, pred in enumerate(preds_list):
            # 调整目标尺寸以匹配当前预测
            target_resized = F.interpolate(
                target.float(),
                size=pred.shape[2:],
                mode='nearest'  # 标签必须使用最近邻插值
            )

            # 计算当前尺度的Dice损失
            dice_loss = self.dice_loss(pred, target_resized)

            # 计算当前尺度的交叉熵损失
            # CrossEntropyLoss要求目标为[B, D, H, W]形状
            target_for_ce = target_resized.squeeze(1).long()
            ce_loss = self.ce_loss(pred, target_for_ce)

            # 组合当前尺度的损失
            scale_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
            # 加权添加到总损失
            total_loss += self.scale_weights[i] * scale_loss

        return total_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, include_background=False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, pred, target):
        if target.dim() == 5:
            target = target.squeeze(1)

        probs = torch.softmax(pred, dim=1)
        num_classes = pred.shape[1]

        start_idx = 0 if self.include_background else 1

        dice_loss = 0.0
        valid_classes = num_classes - start_idx
        for class_idx in range(start_idx, num_classes):
            class_prob = probs[:, class_idx]
            class_target = (target == class_idx).float()

            intersection = (class_prob * class_target).sum(dim=[1, 2, 3])
            union = class_prob.sum(dim=[1, 2, 3]) + class_target.sum(dim=[1, 2, 3])
            dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)

            dice_loss += (1 - dice_coeff).mean()

        return dice_loss / valid_classes


def compute_dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    计算胰腺分割的Dice评分

    参数:
        pred: 模型预测输出 (B, C, D, H, W) C=2 (背景, 胰腺)
        target: 金标准标签 (B, 1, D, H, W) 或 (B, D, H, W)
        threshold: 二值化阈值
        smooth: 平滑因子避免除零

    返回:
        整个batch的平均Dice分数 (标量)
    """
    with torch.no_grad():
        # 确保目标张量维度正确
        if target.dim() == 5:
            target = target.squeeze(1)  # 移除通道维度 -> (B, D, H, W)

        # 应用softmax获取概率
        probs = torch.softmax(pred, dim=1)

        # 获取胰腺概率 (通道1)
        pancreas_prob = probs[:, 1]  # (B, D, H, W)

        # 二值化预测
        pred_binary = (pancreas_prob > threshold).float()

        # 创建胰腺目标掩码
        target_binary = (target == 1).float()  # (B, D, H, W)

        # 计算交集和并集
        intersection = (pred_binary * target_binary).sum(dim=[1, 2, 3])
        union = pred_binary.sum(dim=[1, 2, 3]) + target_binary.sum(dim=[1, 2, 3])

        # 计算每个样本的Dice分数
        dice_per_sample = (2. * intersection + smooth) / (union + smooth)

        # 返回整个batch的平均Dice分数
        return dice_per_sample.mean()


if __name__ == '__main__':
    # 模拟输入
    B, C, D, H, W = 1, 2, 4, 4, 4
    pred = torch.randn(B, C, D, H, W)
    target = torch.randint(0, 2, (B, 1, D, H, W))
    print(pred.shape)
    print(target.shape)

    # 单尺度 DiceLoss 测试
    dice_with_bg = DiceLoss(True)(pred, target)
    dice_without_bg = DiceLoss(False)(pred, target)
    print("DiceLoss (with bg):", dice_with_bg.item())
    print("DiceLoss (without bg):", dice_without_bg.item())

    # 多尺度预测（5个不同分辨率的张量）
    preds_list = [
        torch.randn(B, C, 40, 160, 256),  # 原始
        torch.randn(B, C, 40, 80, 128),
        torch.randn(B, C, 40, 40, 64),
        torch.randn(B, C, 20, 20, 32),
        torch.randn(B, C, 10, 10, 16),
    ]

    # MultiScaleLoss 测试
    multi_scale_loss_fn = MultiScaleSegmentationLoss()
    multi_loss = multi_scale_loss_fn(preds_list, target)
    print("Multi-scale Loss:", multi_loss.item())
