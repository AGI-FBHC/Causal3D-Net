# -*- coding: utf-8 -*-
# @Time    : 2025/5/4 19:24
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: visual3D.py
# @Project : Causal3D-Net
import numpy as np
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt


def show_volume_plotly(volume, save_name="volume_view"):
    """
    使用 Plotly 可视化 3D 体数据，并保存为 HTML 文件。

    参数:
        volume (numpy.ndarray): 3D 图像数据，shape=(D, H, W)
        save_name (str): 保存的文件名（不带扩展名）
    """
    if volume.ndim != 3:
        raise ValueError("volume 应该是一个 3D numpy 数组。")

    D, H, W = volume.shape
    x = np.linspace(0, 1, D).repeat(H * W)
    y = np.tile(np.linspace(0, 1, H).repeat(W), D)
    z = np.tile(np.linspace(0, 1, W), D * H)

    fig = go.Figure(data=go.Volume(
        x=x,
        y=y,
        z=z,
        value=volume.flatten(),
        opacity=0.1,
        surface_count=20,
        colorscale='Viridis',
    ))

    html_path = f"{save_name}.html"
    fig.write_html(html_path)
    print(f"✅ 交互式 3D 图已保存为：{os.path.abspath(html_path)}，可在浏览器中打开查看。")


def show_middle_slice(volume, save_name, title=""):
    # 确保输入是3D数据 (D, H, W)
    if volume.ndim != 3:
        print("输入必须为3D体数据")
        return

    # 获取中间的切片 (D//2)
    slice_img = volume[volume.shape[0] // 2, :, :]  # D, H, W -> 取 D 维度的中间层

    # 显示并保存中间切片
    plt.imshow(slice_img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()
