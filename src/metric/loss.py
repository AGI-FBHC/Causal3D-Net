# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 10:39
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: loss.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_score(preds, targets, num_classes=2, smooth=1e-6):
    """
    计算 Dice coefficient（非损失），用于评估。
    参数：
        preds: Tensor, shape = [N, C, D, H, W]，softmax 或 argmax 前的 logits
        targets: Tensor, shape = [N, 1, D, H, W] 或 [N, D, H, W]
        num_classes: 类别数
    返回：
        dice_per_class: List[float]，每个类别的 Dice 分数
    """
    if targets.dim() == 5:
        targets = targets.squeeze(1)

    preds = torch.argmax(preds, dim=1)  # 取最大类作为预测
    dice_per_class = []

    for class_idx in range(num_classes):
        pred_flat = (preds == class_idx).float().view(-1)
        target_flat = (targets == class_idx).float().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_per_class.append(dice.item())

    return dice_per_class


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # 输入形状: preds=[N,2,D,H,W], targets=[N,1,D,H,W]
        probs = torch.softmax(preds, dim=1)
        dice_loss = 0
        for class_idx in range(probs.shape[1]):  # 分别计算背景和胰腺的Dice
            pred_flat = probs[:, class_idx].contiguous().view(-1)
            target_flat = (targets == class_idx).float().view(-1)
            intersection = (pred_flat * target_flat).sum()
            dice = ((2. * intersection + self.smooth) /
                    (pred_flat.sum() + target_flat.sum() + self.smooth))
            dice_loss += 1 - dice
        return dice_loss / 2  # 平均两类损失


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()  # 自动处理softmax

    def forward(self, preds, targets):
        # 确保目标张量维度匹配 [N,D,H,W]
        if targets.dim() == 5:
            targets = targets.squeeze(1)
        dice = self.dice_loss(preds, targets)
        ce = self.ce_loss(preds, targets.long())  # 需要long类型
        return dice + ce


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
            target_resized = F.interpolate(
                target.float(),
                size=pred.shape[2:],
                mode='nearest'  # 必须使用最近邻避免插值浮点数
            ).long()
            loss += self.weights[i] * self.criterion(pred, target_resized)
        return loss


if __name__ == '__main__':
    N, C, D, H, W = 1, 2, 8, 16, 16
    preds = torch.randn(N, C, D, H, W)  # logits
    targets = torch.zeros(N, 1, D, H, W, dtype=torch.long)
    targets[:, :, D//4:3*D//4, H//4:3*H//4, W//4:3*W//4] = 1

    # 打印 Dice 分数
    scores = dice_score(preds, targets, num_classes=2)
    print("Dice Score for each class:", scores)
