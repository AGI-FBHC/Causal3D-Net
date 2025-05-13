# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 18:42
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: test.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn
import pandas as pd
from src.models.PANDA import DownConv, MultiTask3DCNN


def inference():
    in_channels = 320
    out_channels = 320
    in_shape = (5, 5, 8)
    model = DownConv(in_channels, out_channels, conv_type="fixed")

    # 生成随机数据
    x = torch.randn(1, in_channels, *in_shape)

    # 推理
    with torch.no_grad():
        out = model(x)

    print(f"input.shape={x.shape}")
    print(f"output.shape={out.shape}")


if __name__ == "__main__":
    # inference()
    fixed_trans_conv = nn.ConvTranspose3d(
        in_channels=128,
        out_channels=64,
        kernel_size=(3, 4, 4),
        stride=(1, 2, 2),
        padding=1,
    )
    input_tensor = torch.randn(1, 128, 40, 40, 64)
    output_tensor = fixed_trans_conv(input_tensor)
    print(f"intput_tensor.shape={input_tensor.shape}")
    print(f"output_tensor.shape={output_tensor.shape}")
