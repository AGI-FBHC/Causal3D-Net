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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.window import *
import torchio as tio
from src.utils.visual3D import show_volume_plotly, show_middle_slice



class PCDataset(Dataset):
    def __init__(self, excel_path,
                 transform=None,
                 is_expand=False,
                 use_mask=False,
                 only_cls=True):
        super().__init__()
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

        image_array = np.load(image_path)
        mask_array = np.load(mask_path)
        if self.use_mask:
            X = image_array *  mask_array
        else:
            X = image_array
        show_middle_slice(image_array,
                          save_name="/home/huangdn/Causal3D-Net/src/logging_record/origin",
                          mask=mask_array)
        X = np.expand_dims(X, axis=0)
        mask_array = np.expand_dims(mask_array, axis=0)
        print(f"before: {X.shape}")
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=X),
                mask=tio.LabelMap(tensor=mask_array)
            )
            transformed = self.transform(subject)
            X = transformed['image'].data
            mask_array = transformed['mask'].data
        else:
            X = torch.from_numpy(X)
        show_middle_slice(X.squeeze().cpu().numpy(),
                          save_name="/home/huangdn/Causal3D-Net/src/logging_record/resized",
                          mask=mask_array.squeeze().cpu().numpy())
        print(f"after: {X.shape}")
        # if self.transform:
        #     X = self.transform(X)
        y = label
        return image_path, X, y


if __name__ == '__main__':
    batch_size = 1
    transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    excel_path = "/home/huangdn/Causal3D-Net/src/dataset/roi_data_finger.xlsx"
    dataset = PCDataset(excel_path=excel_path, transform=transform)
    loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)
    for _, x, y in tqdm(loader):
        break
    pass
























