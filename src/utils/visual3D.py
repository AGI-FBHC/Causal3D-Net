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


# def show_middle_slice(volume, save_name, title=""):
#     # 确保输入是3D数据 (D, H, W)
#     if volume.ndim != 3:
#         print("输入必须为3D体数据")
#         return
#
#     # 获取中间的切片 (D//2)
#     slice_img = volume[volume.shape[0] // 2, :, :]  # D, H, W -> 取 D 维度的中间层
#
#     # 显示并保存中间切片
#     plt.imshow(slice_img, cmap='gray')
#     plt.title(title)
#     plt.axis('off')
#     plt.savefig(save_name, bbox_inches='tight', pad_inches=0, transparent=True)
#     plt.close()

def show_middle_slice(volume, save_name, title="", mask=None):
    """
    可视化中间切片，支持叠加mask
    :param volume: 3D图像，shape=(D, H, W)
    :param save_name: 保存路径
    :param title: 图像标题
    :param mask: 可选的3D掩码图像，shape=(D, H, W)，label值如0,1,2
    """
    if volume.ndim != 3:
        print("输入必须为3D体数据")
        return

    middle_slice = volume[volume.shape[0] // 2, :, :]
    plt.figure(figsize=(5, 5))
    plt.imshow(middle_slice, cmap='gray')

    if mask is not None:
        mask_slice = mask[mask.shape[0] // 2, :, :]
        # 创建颜色映射：1为红色，2为绿色
        cmap_mask = np.zeros((*mask_slice.shape, 4))  # RGBA

        # 红色表示 label 1
        cmap_mask[mask_slice == 1] = [1, 0, 0, 0.4]  # 红，透明度0.4
        # 绿色表示 label 2
        cmap_mask[mask_slice == 2] = [0, 1, 0, 0.4]  # 绿，透明度0.4
        # 黄色表示 label 3
        cmap_mask[mask_slice == 3] = [1, 1, 0, 0.4]  # 黄，透明度0.4

        plt.imshow(cmap_mask)

    plt.title(title)
    plt.axis('off')
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

