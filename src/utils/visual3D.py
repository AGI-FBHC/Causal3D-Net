# -*- coding: utf-8 -*-
# @Time    : 2025/5/4 19:24
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: visual3D.py
# @Project : Causal3D-Net
import torch
import numpy as np
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt

from src.augmentation.window import rescale_back


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



def visualize_prediction(image, gt_mask, pred_mask, slice_idx=None,
                         alpha_gt=0.4, alpha_pred=0.4,
                         save_path=None, compute_dice_fn=None):
    """
    Visualize a slice of a 3D image with GT and prediction overlays.
    Red = GT, Blue = Prediction
    Uses external Dice function if provided.
    """
    if image.ndim == 4:
        image = image.squeeze(0)  # [D, H, W]
    if gt_mask.ndim == 4:
        gt_mask = gt_mask.squeeze(0)
    if pred_mask.ndim == 4:
        pred_mask = pred_mask.squeeze(0)

    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    assert slice_idx < image.shape[0]

    # Slice
    img_slice = image[slice_idx].cpu().numpy()
    img_slice = rescale_back(img_slice, 0, 1, -100, 240)  # 恢复 CT 窗口范围

    gt_slice = gt_mask[slice_idx]  # tensor, [H, W]
    pred_slice = pred_mask[slice_idx]  # tensor, [H, W]

    # Dice computation (with compute_dice_score)
    if compute_dice_fn is not None:
        pred_tensor = torch.zeros((1, 2, 1, *pred_slice.shape), device=pred_slice.device)
        pred_tensor[0, 0, 0] = (pred_slice == 0).float()  # 背景通道
        pred_tensor[0, 1, 0] = (pred_slice > 0).float()   # 胰腺通道

        gt_tensor = gt_slice.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]

        dice = compute_dice_fn(pred_tensor, gt_tensor).item()
    else:
        # 简化 fallback
        gt_np = gt_slice.cpu().numpy()
        pred_np = pred_slice.cpu().numpy()
        intersection = np.logical_and(gt_np > 0, pred_np > 0).sum()
        union = (gt_np > 0).sum() + (pred_np > 0).sum()
        dice = (2. * intersection / union) if union > 0 else 1.0

    # Display normalization
    img_display = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())

    # Overlay RGBA masks
    red_overlay = np.zeros((*gt_slice.shape, 4), dtype=np.float32)
    blue_overlay = np.zeros((*pred_slice.shape, 4), dtype=np.float32)

    red_overlay[..., 0] = 1.0  # Red
    red_overlay[..., 3] = (gt_slice.cpu().numpy() > 0).astype(np.float32) * alpha_gt

    blue_overlay[..., 2] = 1.0  # Blue
    blue_overlay[..., 3] = (pred_slice.cpu().numpy() > 0).astype(np.float32) * alpha_pred

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(img_display, cmap='gray')
    plt.imshow(red_overlay)
    plt.imshow(blue_overlay)
    plt.title(f"Slice {slice_idx} | Dice: {dice:.4f} | Red=GT, Blue=Prediction")
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return dice


def save_slices(image3d, out_dir, prefix):
    for d in range(image3d.shape[0]):
        plt.figure(figsize=(4, 4))
        plt.imshow(image3d[d, :, :], cmap='gray', origin='lower')
        plt.axis('off')
        slice_path = os.path.join(out_dir, f"{prefix}_slice_{d:03d}.png")
        plt.savefig(slice_path, bbox_inches='tight', pad_inches=0)
        plt.close()


