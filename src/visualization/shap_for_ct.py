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

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import measure

import shap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchio as tio

from src.utils.seed import fix_seed
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


def show_slices_with_shap(volume: torch.Tensor,
                          shap: torch.Tensor,
                          mask: torch.Tensor = None,
                          slice_index: int = None,
                          save_path: str = None,
                          outside_color: float = 0.85,   # 淡色(0黑-1白)，0.85更“淡”
                          outside_alpha: float = 0.35,   # mask外遮罩透明度
                          shap_alpha: float = 0.5,       # mask内SHAP透明度
                          vlim: float | None = None):    # 统一色标可传固定值

    vol_np  = volume.squeeze().detach().cpu().numpy()  # [D,H,W]
    shap_np = shap.squeeze().detach().cpu().numpy()    # [D,H,W]

    D, H, W = vol_np.shape
    slice_index = D // 2 if slice_index is None else max(0, min(slice_index, D - 1))

    img = vol_np[slice_index].astype(np.float32)
    shap_img = shap_np[slice_index].astype(np.float32)

    # normalize img to [0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(10, 5))

    # 1) 画原始图（全幅）
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)

    if mask is not None:
        mask_np = mask.squeeze().detach().cpu().numpy().astype(bool)  # [D,H,W]
        mask_slice = mask_np[slice_index]  # True=mask内

        # 2) mask外叠加“淡色透明遮罩”（只影响mask外）
        overlay = np.full_like(img, outside_color, dtype=np.float32)
        overlay_alpha = (~mask_slice).astype(np.float32) * outside_alpha
        plt.imshow(overlay, cmap="gray", vmin=0, vmax=1, alpha=overlay_alpha)

        # 3) SHAP 只在 mask 内显示（mask外 alpha=0，完全不会染色）
        shap_alpha_map = mask_slice.astype(np.float32) * shap_alpha
    else:
        shap_alpha_map = shap_alpha

    # （可选）统一SHAP色标范围：跨病人/切片对比更公平
    if vlim is None:
        vlim = float(np.max(np.abs(shap_img)) + 1e-8)
    vmin, vmax = -vlim, vlim

    plt.imshow(shap_img, cmap="seismic", vmin=vmin, vmax=vmax, alpha=shap_alpha_map)

    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")
    # plt.show()
    plt.close()


def save_one_slice_png(vol_np, shap_np, mask_np, slice_idx, save_path,
                       outside_color=0.85, outside_alpha=0.35, shap_alpha=0.5):
    img = vol_np[slice_idx].astype(np.float32)
    shap_img = shap_np[slice_idx].astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.figure(figsize=(10, 5))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)

    if mask_np is not None:
        mask_slice = mask_np[slice_idx].astype(bool)

        overlay = np.full_like(img, outside_color, dtype=np.float32)
        overlay_alpha = (~mask_slice).astype(np.float32) * outside_alpha
        plt.imshow(overlay, cmap="gray", vmin=0, vmax=1, alpha=overlay_alpha)

        alpha_map = mask_slice.astype(np.float32) * shap_alpha
    else:
        alpha_map = shap_alpha

    vlim = float(np.max(np.abs(shap_img)) + 1e-8)
    plt.imshow(shap_img, cmap="seismic", vmin=-vlim, vmax=vlim, alpha=alpha_map)

    plt.axis("off")
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def save_shap_volume(model_dir: str,
                     excel_path: str,
                     cuda_id: int = 0,
                     target_class: int = 1,
                     background_samples: int = 4,
                     nsamples: int = 8):
    fix_seed()
    print(f"\n[INFO] Saving SHAP volumes ...")
    print(f"model_dir          = {model_dir}")
    print(f"excel_path         = {excel_path}")
    print(f"cuda_id            = {cuda_id}")
    print(f"target_class       = {target_class}")
    print(f"background_samples = {background_samples}")
    print(f"nsamples           = {nsamples}\n")

    checkpoint_path = os.path.join(model_dir, "best_model.pth")
    save_root = os.path.join(model_dir, "shap_volume")
    os.makedirs(save_root, exist_ok=True)

    model_shap = Causal3DNetForSHAP(ckpt_path=checkpoint_path, cuda_id=cuda_id)
    device = model_shap.device  # 用模型自己的device更稳

    pre_transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])

    dataset = PCDataset(excel_path=excel_path, transform=pre_transform, return_type=5)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    background = get_shap_background(loader, num_samples=background_samples)
    explainer = build_shap_explainer(model_shap, background)

    for filename, X, cls_label, msk_label, center, cluster in tqdm(loader, desc="Compute SHAP volume", unit="case"):
        case_name = filename[0].split(".")[0]

        X = X.to(device, non_blocking=True)
        shap_values = explainer.shap_values(X, nsamples=nsamples)
        shap_map = torch.tensor(shap_values[0][..., target_class])  # -> torch CPU

        shap_np = shap_map.squeeze().detach().cpu().numpy()  # [D,H,W]
        np.save(os.path.join(save_root, f"{case_name}_shap.npy"), shap_np)

        print(f"[OK] SHAP volume saved: {case_name}")


def run_shap_visualization(model_dir: str,
                           excel_path: str,
                           cuda_id: int = 0,
                           target_class: int = 1,
                           background_samples: int = 4,
                           MAX_WORKERS: int = 8,
                           use_process: bool = True,
                           compute_shap: bool = True,):
    fix_seed()
    print(f"\n[INFO] Running SHAP visualization ...")
    print(f"model_dir     = {model_dir}")
    print(f"excel_path    = {excel_path}")
    print(f"cuda_id       = {cuda_id}")
    print(f"target_class  = {target_class}\n")

    if compute_shap:
        save_shap_volume(model_dir=model_dir,
                         excel_path=excel_path,
                         cuda_id=cuda_id,
                         target_class=target_class,
                         background_samples=background_samples,)
        device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    else:
        print("[INFO] Skip save_shap_volume(): using existing SHAP volumes on disk.")
        device = torch.device("cpu")

    save_root = os.path.join(model_dir, "shap")                # 保存png
    save_root_volume = os.path.join(model_dir, "shap_volume")  # 读取npy

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

    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    # 进程/线程池放外面复用
    with Executor(max_workers=MAX_WORKERS) as ex:
        for filename, X, cls_label, msk_label, center, cluster in tqdm(loader, desc="Visualization", unit="case"):
            case_name = filename[0].split(".")[0]
            fig_save_dir = os.path.join(save_root, case_name)
            os.makedirs(fig_save_dir, exist_ok=True)

            shap_npy_path = os.path.join(save_root_volume, f"{case_name}_shap.npy")
            if not os.path.exists(shap_npy_path):
                print(f"[WARN] Missing SHAP volume for case {case_name}: {shap_npy_path}")
                continue

            shap_np = np.load(shap_npy_path)
            vol_np = X.squeeze().detach().cpu().numpy()
            mask_np = msk_label.squeeze().detach().cpu().numpy() if msk_label is not None else None

            depth = vol_np.shape[0]

            futures = []
            for slice_idx in range(depth):
                save_path = os.path.join(fig_save_dir, f"shap_slice_{slice_idx}.png")
                futures.append(ex.submit(save_one_slice_png, vol_np, shap_np, mask_np, slice_idx, save_path))

            for f in as_completed(futures):
                f.result()

            print(f"[OK] SHAP png saved for case: {case_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16",
                        required=False)
    parser.add_argument("--excel_path", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_vis.xlsx",
                        required=False)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target_class", type=int, default=1)
    args = parser.parse_args()

    run_shap_visualization(
        model_dir=args.model_dir,
        excel_path=args.excel_path,
        cuda_id=args.cuda_id,
        target_class=args.target_class,
        MAX_WORKERS=args.workers,
    )


