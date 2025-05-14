# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 18:42
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: test.py
# @Project : Causal3D-Net
import torch
import pydoc
import torch.nn as nn
import pandas as pd
from src.models.PANDA import DownConv, UpConv, MultiTask3DCNN


def inference():
    in_channels = 1
    out_channels = 2
    in_shape = (40, 160, 256)
    model = MultiTask3DCNN(mask_num=out_channels)

    # 生成随机数据
    x = torch.randn(1, in_channels, *in_shape)

    # 推理
    with torch.no_grad():
        out = model(x)

    print(f"input.shape={x.shape}")
    print(f"output.shape={out.shape}")


def test():
    s1 = "torch.nn.modules.conv.Conv2d"
    s2 = pydoc.locate(s1)
    print(s1)
    print("=" * 20)
    print(s2)


if __name__ == "__main__":
    test()
