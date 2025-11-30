# -*- coding: utf-8 -*-
# @Time    : 2025/11/29 14:51
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: shap_for_ct.py
# @Project : Causal3D-Net
import os, argparse

import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from skimage import measure

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


def show_slices_with_shap(volume: torch.Tensor, shap: torch.Tensor, mask: torch.Tensor = None,
                          slice_index: int = None, save_path: str = None):

    vol_np = volume.squeeze().cpu().numpy()  # [D,H,W]
    shap_np = shap.squeeze().cpu().numpy()

    D, H, W = vol_np.shape
    slice_index = D // 2 if slice_index is None else max(0, min(slice_index, D - 1))

    img = vol_np[slice_index]
    shap_img = shap_np[slice_index]

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap='gray')
    plt.imshow(shap_img, cmap='seismic', alpha=0.5)  # 正值=红色 负值=蓝色

    if mask is not None:
        mask_np = mask.squeeze().detach().cpu().numpy()  # [D,H,W]
        mask_slice = mask_np[slice_index]

        contours = measure.find_contours(mask_slice, level=0.5)

        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color='yellow')
            # 用 yellow 比 red 更容易在 seismic 上分辨

    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
    plt.close()


def run_shap_visualization(model_dir: str,
                           excel_path: str,
                           cuda_id: int = 0,
                           target_class: int = 1,
                           background_samples: int = 4):

    print(f"\n[INFO] Running SHAP visualization ...")
    print(f"model_dir     = {model_dir}")
    print(f"excel_path    = {excel_path}")
    print(f"cuda_id       = {cuda_id}")
    print(f"target_class  = {target_class}\n")

    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    # === 1. 加载模型 ===
    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    save_root = os.path.join(model_dir, "shap")

    model_shap = Causal3DNetForSHAP(
        ckpt_path=checkpoint_path,
        cuda_id=cuda_id,
    )

    # === 2. 数据预处理 ===
    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])

    dataset = PCDataset(
        excel_path=excel_path,
        transform=pre_transform,
        return_type=5,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # === 3. 采样 background ===
    background = get_shap_background(loader, num_samples=background_samples)

    # === 4. 构建 explainer ===
    explainer = build_shap_explainer(model_shap, background)

    # === 5. 对每个病例计算 SHAP ===
    for filename, X, cls_label, msk_label, center, cluster in loader:

        X = X.to(device)
        shap_values = explainer.shap_values(X, nsamples=8)
        shap_map = torch.tensor(shap_values[0][..., target_class])

        case_name = filename[0].split(".")[0]
        fig_save_dir = os.path.join(save_root, case_name)
        os.makedirs(fig_save_dir, exist_ok=True)

        depth = X.shape[2]

        # === 6. 保存所有切片 ===
        for slice_idx in range(depth):
            save_path = os.path.join(fig_save_dir, f"shap_slice_{slice_idx}.png")
            show_slices_with_shap(X, shap_map, mask=msk_label,
                                  slice_index=slice_idx, save_path=save_path)

        print(f"[OK] SHAP saved for case: {case_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16",
                        required=False)
    parser.add_argument("--excel_path", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_vis.xlsx",
                        required=False)
    parser.add_argument("--cuda_id", type=int, default=5)
    parser.add_argument("--target_class", type=int, default=1)
    args = parser.parse_args()

    run_shap_visualization(
        model_dir=args.model_dir,
        excel_path=args.excel_path,
        cuda_id=args.cuda_id,
        target_class=args.target_class,
    )


