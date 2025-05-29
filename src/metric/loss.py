# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 10:39
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: loss.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # 因为输入是 logits
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        dice = self.dice_loss(preds, targets)
        bce = self.bce_loss(preds, targets)
        return dice + bce


class MultiScaleSegmentationLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.criterion = SegmentationLoss()
        if weights is None:
            # 权重默认：高分辨率更重要
            self.weights = [0.5, 0.25, 0.15, 0.07, 0.03]
        else:
            self.weights = weights

    def forward(self, preds_list, target):
        loss = 0
        for i, pred in enumerate(preds_list):
            # 将 y resize 到 pred 的尺寸
            target_resized = F.interpolate(
                target, size=pred.shape[2:], mode='trilinear', align_corners=False
            )
            loss += self.weights[i] * self.criterion(pred, target_resized)
        return loss
