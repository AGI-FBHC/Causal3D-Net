# -*- coding: utf-8 -*-
# @Time    : 2025/5/27 10:50
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: linear_lr.py
# @Project : Causal3D-Net
def linear_lr_lambda(total_epochs):
    """
    返回一个函数，用于LambdaLR的 lr_lambda 参数，
    使学习率从1线性衰减到0。

    参数:
        total_epochs (int): 总训练轮数

    返回:
        lr_lambda (function): 输入当前epoch，输出衰减比例
    """
    def lr_lambda(current_epoch):
        if current_epoch >= total_epochs:
            return 0.0
        return 1.0 - (current_epoch / total_epochs)
    return lr_lambda

