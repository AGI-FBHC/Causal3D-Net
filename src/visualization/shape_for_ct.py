# -*- coding: utf-8 -*-
# @Time    : 2025/11/29 14:51
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: shape_for_ct.py
# @Project : Causal3D-Net
import os, argparse

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import shap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchio as tio

from src.dataset.PC_dataset import PCDataset
from src.augmentation.window import Windowing
from src.models.Causal3DNet import Causal3DNet


class Causal3DNetForSHAP(nn.Module):
    def __init__(self, ckpt_path="/home/huangdn/Causal3D-Net/src/results/"
                                 "2025-11-05_02-24-16/best_model.pth", cuda_id=5):
        super().__init__()
        self.device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
        self.model = Causal3DNet()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval().to(self.device)

    def forward(self, x):
        ((_, y_main, _), (_, _, _)) = self.model(x)
        return y_main


def get_shap_background(loader, num_samples=20):
    background = []
    for i, (_, X, _, _, _, _) in enumerate(loader):
        background.append(X)
        if len(background) >= num_samples:
            break
    return torch.cat(background, dim=0)  # [N, 1, D, H, W]


def build_shap_explainer(model_shap: Causal3DNetForSHAP, background):
    background = background.to(model_shap.device)
    explainer = shap.GradientExplainer(model_shap, background)
    return explainer


def show_shap_slice(volume, shap_map, slice_index=None, save_path=None):

    vol_np = volume.squeeze().cpu().numpy()  # [D,H,W]
    shap_np = shap_map.squeeze().cpu().numpy()

    D, H, W = vol_np.shape
    slice_index = D // 2 if slice_index is None else max(0, min(slice_index, D - 1))

    img = vol_np[slice_index]
    shap_img = shap_np[slice_index]

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap='gray')
    plt.imshow(shap_img, cmap='seismic', alpha=0.6)  # 正值=红色 负值=蓝色
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == "__main__":
    target_class = 1
    cuda_id = 5
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    model_dir = "/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16"
    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    save_root = os.path.join(model_dir, "shap")

    model_shap = Causal3DNetForSHAP(ckpt_path=checkpoint_path, cuda_id=cuda_id)

    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])

    dataset = PCDataset(
        excel_path="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_vis.xlsx",
        transform=pre_transform,
        return_type=5,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 1) 采样 40 个背景
    background = get_shap_background(loader, num_samples=4)

    # 2) 构建 SHAP Explainer
    explainer = build_shap_explainer(model_shap, background)

    # 3) 解释 loader 数据
    for filename, X, _, _, _, _ in loader:
        X = X.to(device)
        # nsamples: SHAP 蒙特卡洛采样次数，用于近似计算每个特征（voxel）的边际贡献。
        shap_values = explainer.shap_values(X, nsamples=8)
        shap_map = torch.tensor(shap_values[0][..., target_class])

        # 可视化
        case_name = filename[0].split(".")[0]
        fig_save_dir = os.path.join(save_root, case_name)
        os.makedirs(fig_save_dir, exist_ok=True)
        depth = X.shape[2]

        slice_idx = 20
        save_path = os.path.join(fig_save_dir, f"shap_slice_{slice_idx}.png")
        show_shap_slice(X, shap_map, slice_idx, save_path)

        # for slice_idx in range(depth):
        #     save_path = os.path.join(fig_save_dir, f"shap_slice_{slice_idx}.png")
        #     show_shap_slice(X, shap_map, slice_idx, save_path)


