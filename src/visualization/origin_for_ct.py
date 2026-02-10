# -*- coding: utf-8 -*-
# @Time    : 2025/11/30 19:41
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: origin_for_ct.py
# @Project : Causal3D-Net
import os, argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import measure
import torchio as tio


def show_ct_slice(img_slice, mask_slice=None, save_path=None):
    """
    可视化单层 CT（经过预处理后的）。
    """
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_slice, cmap="gray")

    if mask_slice is not None:
        contours = measure.find_contours(mask_slice, level=0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], color="red", linewidth=2)

    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.close()


def run_origin_visualization(model_dir: str, excel_path: str):
    """
    根据 excel 文件，对 image_path 和 mask_path 做 Windowing+归一化+Resize 的可视化。
    """

    print(f"Loading excel: {excel_path}")
    df = pd.read_excel(excel_path)

    if "image_path" not in df.columns:
        raise ValueError("Excel 必须包含 image_path 列！")

    save_root = os.path.join(model_dir, "origin")
    os.makedirs(save_root, exist_ok=True)

    # 与你 SHAP/GCAM 一模一样的预处理：
    pre_transform = tio.Compose([
        tio.Clamp(out_min=-100, out_max=240),   # windowing 等效实现（70±170 → [-100,240]）
        tio.RescaleIntensity((0, 1)),
        tio.Resize((40, 160, 256)),
    ])

    # ===== 使用 tqdm 包裹 iterrows =====
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):

        img_path = row["image_path"]
        case_name = os.path.splitext(os.path.basename(img_path))[0]

        if not os.path.exists(img_path):
            print(f"[Warning] image not found: {img_path}")
            continue

        # ---- 1. 加载 3D volume ----
        vol_np = np.load(img_path)   # [D,H,W]
        vol_np = vol_np.astype(np.float32)

        # ---- 2. 包装为 TorchIO 输入 ----
        subject = tio.Subject(
            img=tio.ScalarImage(tensor=vol_np[None])  # [1,D,H,W]
        )
        subject = pre_transform(subject)

        vol_pp = subject["img"].data.squeeze().cpu().numpy()   # 预处理后 [D,H,W]

        # ---- 3. 读取 mask（并同步 resize）----
        mask_pp = None
        if "mask_path" in df.columns and isinstance(row["mask_path"], str):
            mask_path = row["mask_path"]
            if os.path.exists(mask_path):
                mask_np = np.load(mask_path).astype(np.float32)
                subject_mask = tio.Subject(
                    mask=tio.LabelMap(tensor=mask_np[None])
                )
                subject_mask = pre_transform(subject_mask)
                mask_pp = subject_mask["mask"].data.squeeze().cpu().numpy()

        # ---- 4. 保存 ----
        case_dir = os.path.join(save_root, case_name)
        os.makedirs(case_dir, exist_ok=True)

        D = vol_pp.shape[0]
        for slice_idx in range(D):
            img_slice = vol_pp[slice_idx]
            mask_slice = mask_pp[slice_idx] if mask_pp is not None else None

            save_path = os.path.join(case_dir, f"slice_{slice_idx:03d}.png")
            show_ct_slice(img_slice, mask_slice, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16")
    parser.add_argument("--excel_path", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_vis.xlsx")
    args = parser.parse_args()

    run_origin_visualization(
        model_dir=args.model_dir,
        excel_path=args.excel_path
    )

