# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 10:29
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: compute_score.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score,
                             recall_score,
                             precision_score,
                             roc_auc_score,
                             f1_score,
                             confusion_matrix)


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


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0