# -*- coding: utf-8 -*-
# @Time    : 2025/12/26 09:14
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: Seg_dataset.py
# @Project : Causal3D-Net

import os
import pandas as pd
import torchio as tio
from typing import Optional


class SegDataset(tio.SubjectsDataset):

    def __init__(
        self,
        excel_path: str,
        transform: Optional[tio.Transform] = None,
        verify_paths: bool = True,
        auto_npy_to_nii: bool = True,
    ):
        df = pd.read_excel(excel_path)

        for col in ["image_path", "mask_path"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in {excel_path}")

        if auto_npy_to_nii:
            df = self._convert_npy_to_nii(df)

        if verify_paths:
            self._verify_files_exist(df)

        subjects = []
        for _, row in df.iterrows():
            image_path = str(row["image_path"]).strip()
            mask_path = str(row["mask_path"]).strip()
            subjects.append(
                tio.Subject(
                    image=tio.ScalarImage(image_path),
                    mask=tio.LabelMap(mask_path),
                    case_id=os.path.basename(image_path),
                )
            )

        super().__init__(subjects, transform=transform)

    @staticmethod
    def _convert_npy_to_nii(df: pd.DataFrame) -> pd.DataFrame:
        """
        If BOTH image_path and mask_path end with '.npy', replace suffix with '.nii.gz'.
        """
        def replace_suffix(p: str) -> str:
            p = str(p).strip()
            return p[:-4] + ".nii.gz"

        df = df.copy()
        for i, row in df.iterrows():
            img_path = str(row["image_path"]).strip()
            msk_path = str(row["mask_path"]).strip()
            if img_path.endswith(".npy") and msk_path.endswith(".npy"):
                df.at[i, "image_path"] = replace_suffix(img_path)
                df.at[i, "mask_path"] = replace_suffix(msk_path)
        return df

    @staticmethod
    def _verify_files_exist(df: pd.DataFrame):
        for i, row in df.iterrows():
            ip = str(row["image_path"]).strip()
            mp = str(row["mask_path"]).strip()
            if not os.path.isfile(ip):
                raise FileNotFoundError(f"[Row {i}] CT not found: {ip}")
            if not os.path.isfile(mp):
                raise FileNotFoundError(f"[Row {i}] Mask not found: {mp}")

if __name__ == "__main__":
    import torch
    import torchio as tio
    from torch.utils.data import DataLoader
    from src.augmentation.window import Windowing

    excel_path = "/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx"

    patch_size = (40, 160, 256)
    half_patch = (20, 80, 128)

    transform = tio.Compose([
        tio.ToCanonical(),
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resample((2.5, 1.0, 1.0)),
        tio.Pad(half_patch),
    ])

    dataset = SegDataset(
        excel_path=excel_path,
        transform=transform,
        verify_paths=True,
        auto_npy_to_nii=True,
    )

    print("Number of CT subjects:", len(dataset))

    sampler = tio.LabelSampler(
        patch_size=patch_size,
        label_name="mask",
    )

    queue = tio.Queue(
        subjects_dataset=dataset,
        sampler=sampler,
        samples_per_volume=4,
        max_length=16,
        num_workers=2,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    loader = tio.SubjectsLoader(
        queue,
        batch_size=2,
        num_workers=0,
        pin_memory=True,  # 这里可以保留
    )

    for batch in loader:
        x = batch["image"][tio.DATA]  # [B, 1, 40, 160, 256]
        y = batch["mask"][tio.DATA]   # [B, 1, 40, 160, 256]

        print("x shape:", x.shape)
        print("y shape:", y.shape)
        break
