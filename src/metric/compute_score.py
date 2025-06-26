# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 10:29
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: compute_score.py
# @Project : Causal3D-Net
import os

import numpy as np

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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return 0.0

    unique_classes = np.unique(y_true)

    if len(unique_classes) == 1 and unique_classes[0] == 0:
        return 1.0 if np.all(y_pred == 0) else 0.0

    if len(unique_classes) == 1 and unique_classes[0] == 1:
        return 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_multi_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = specificity_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        'acc': acc,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1
    }


def evaluate_test_result(test_result, center_groups):
    y_true = test_result['cancer']
    y_pred = test_result['y_pred']
    y_prob = test_result['y_prob']
    centers = test_result['center']

    result_summary = dict()

    # Overall
    result_summary['overall'] = compute_multi_metrics(y_true, y_pred, y_prob)

    # Per group
    for group_name, group_center_ids in center_groups.items():
        mask = np.isin(centers, group_center_ids)
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        y_prob_group = y_prob[mask]
        result_summary[group_name] = compute_multi_metrics(y_true_group, y_pred_group, y_prob_group)

    return result_summary
