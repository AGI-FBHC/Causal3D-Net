# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 10:39
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: loss.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数，结合分割损失和分类损失

    参数:
        seg_weight: 分割损失的权重
        cls_weight: 分类损失的权重
        seg_loss_params: 分割损失函数的参数
    """

    def __init__(self, alpha=0.3, seg_loss_params=None):
        super().__init__()
        self.alpha = alpha
        # 初始化分割损失函数
        seg_loss_params = seg_loss_params or {}
        self.seg_loss = MultiScaleSegmentationLoss(**seg_loss_params)

        # 初始化分类损失函数
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, seg_outputs, cls_output, seg_target, cls_target):
        """
        计算多任务损失

        参数:
            seg_outputs: 分割输出列表 (多尺度输出)
            cls_output: 分类输出 (logits)
            seg_target: 分割目标 (B, 1, D, H, W)
            cls_target: 分类目标 (B)

        返回:
            total_loss: 总损失
            seg_loss_value: 分割损失值
            cls_loss_value: 分类损失值
        """
        # 计算分割损失
        seg_loss_value = self.seg_loss(seg_outputs, seg_target)

        # 计算分类损失
        cls_loss_value = self.cls_loss(cls_output, cls_target)

        # 加权组合
        total_loss = seg_loss_value + self.alpha * cls_loss_value

        return total_loss, seg_loss_value, cls_loss_value


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


class OrthogonalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feat1, feat2):
        """
        feat1: [B, D]
        feat2: [B, D]
        """
        feat1 = F.normalize(feat1, dim=1)
        feat2 = F.normalize(feat2, dim=1)
        dot_product = (feat1 * feat2).sum(dim=1)  # [B]
        loss = torch.mean(dot_product ** 2)
        return loss



class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size]
        """
        device = features.device
        features = F.normalize(features, dim=1)  # 单位化

        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B]，正样本掩码

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)

        # 每个样本的正对比项平均
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # 损失项
        loss = -mean_log_prob_pos.mean()
        return loss


if __name__ == '__main__':
    feat1 = torch.tensor([[1., 0.], [0., 1.]])  # [2, 2]
    feat2 = torch.tensor([[0., 1.], [1., 0.]])  # 完全正交
    loss = OrthogonalLoss()(feat1, feat2)
    print(loss.item())  # 输出应接近 0
