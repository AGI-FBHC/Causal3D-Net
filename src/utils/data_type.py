# -*- coding: utf-8 -*-
# @Time    : 2025/11/9 21:08
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: data_type.py
# @Project : Causal3D-Net
import torch


def to_number(x):
    """
    将输入安全转换为 Python 原生数值（int 或 float）
    - 如果是 tensor，则返回 x.item()
    - 如果是 int/float，则直接返回
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, (int, float)):
        return x
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

