# -*- coding: utf-8 -*-
# @Time    : 2025/9/7 22:07
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: coefficient.py
# @Project : Causal3D-Net
import math


def compute_lambdas(loss_main, loss_i, loss_c, mode='none', T=1.0):
    """
    根据三个 loss 计算 lambda_main、lambda1、lambda2，使三者之和为 1
    参数:
        loss_main: float 主任务损失
        loss_i: float 辅助任务 i 损失
        loss_c: float 辅助任务 c 损失
        mode: 'reward_good' = loss 小权重大
              'reward_bad' = loss 大权重大
        T: 温度系数，越小差异越大，越大越平滑
    返回:
        lambda_main, lambda1, lambda2
    """
    if mode == 'reward_good':
        v_main = math.exp(-loss_main / T)
        v_i = math.exp(-loss_i / T)
        v_c = math.exp(-loss_c / T)
    elif mode == 'reward_bad':
        v_main = math.exp(loss_main / T)
        v_i = math.exp(loss_i / T)
        v_c = math.exp(loss_c / T)
    elif mode == 'none':
        return 1, 1, 1
    else:
        raise ValueError("mode 必须是 'reward_good' , 'reward_bad' 或 'none'")

    s = v_main + v_i + v_c
    lambda_m = v_main / s
    lambda_i = v_i / s
    lambda_c = v_c / s
    return lambda_m, lambda_i, lambda_c
