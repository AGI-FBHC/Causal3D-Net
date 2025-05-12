# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 18:42
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: test.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import pandas as pd
from src.models.PANDA import ConvBlock3D, MultiTask3DCNN


# 实例化模块
model = ConvBlock3D(1, 32)

# 生成随机数据
x = torch.randn(1, 1, 40, 160, 256)

# 推理
with torch.no_grad():
    out = model(x)

print(out.shape)

