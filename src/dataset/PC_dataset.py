# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 20:12
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: PC_dataset.py
# @Project : Causal3D-Net
import os
import pandas as pd
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset


class PCDataset(Dataset):
    def __init__(self, excel_path, transform, is_expand=False, use_mask=False):
        super(PCDataset, self).__init__()
        self.df = pd.read_excel(excel_path)
        self.transform = transform
        self.is_expand = is_expand
        self.use_mask = use_mask
        if not self.is_expand:
            self.df = self.df.loc[self.df["raw_data"], :]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            pass
        row = self.df.iloc[idx]
        image_path = row['image_path']
        mask_path = row['mask_path']
        label = row['cancer']

        image = sitk.ReadImage(image_path)
        image_array = sitk.GetArrayFromImage(image).astype('float32')
        mask = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask).astype('float32')

        # 找到mask中值为1的地方
        coords = (mask_array == 1).nonzero()  # 返回所有1的坐标，形状 (N, 3)，分别是 (z, y, x)
        if coords.size == 0:
            # 如果mask里没有1（极少见情况），直接返回原图或者raise
            raise ValueError(f"No positive region found in mask for index {idx}.")
        # 分别找z、y、x方向上的最小和最大索引
        z_min, y_min, x_min = coords.min(0)
        z_max, y_max, x_max = coords.max(0)
        # 裁剪image和mask
        cropped_image = image_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = mask_array[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
        if self.use_mask:
            X = cropped_mask * cropped_image
        else:
            X = cropped_image
        if self.transform:
            X = self.transform(X)
        return X, y
