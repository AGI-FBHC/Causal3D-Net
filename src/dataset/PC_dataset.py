# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 20:12
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: PC_dataset.py
# @Project : Causal3D-Net
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation.window import *
import torchio as tio


class PCDataset(Dataset):
    def __init__(self, excel_path,
                 transform=None,
                 use_mask=False,
                 return_type=0):
        """
        :param excel_path:
        :param transform:
        :param use_mask:
        :param return_type: `0` 用于仅分类, `1` 用于仅分割...
        """
        super().__init__()
        self.df = pd.read_excel(excel_path)
        self.transform = transform
        self.use_mask = use_mask
        self.return_type = return_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            pass
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        mask_path = row["mask_path"]
        cls_label = row["cancer"]
        center_type = row["center_type"]
        cluster = row["cluster"]

        image_array = np.load(image_path)
        mask_array = np.load(mask_path)
        X = image_array * mask_array if self.use_mask else image_array
        # show_middle_slice(image_array,
        #                   save_name="/home/huangdn/Causal3D-Net/src/logging_record/origin",
        #                   mask=mask_array)
        X = np.expand_dims(X, axis=0)
        mask_array = np.expand_dims(mask_array, axis=0)
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=X),
                mask=tio.LabelMap(tensor=mask_array)
            )
            transformed = self.transform(subject)
            X = transformed['image'].data
            msk_label = transformed['mask'].data
        else:
            X = torch.from_numpy(X)
            msk_label = torch.from_numpy(mask_array)
            pass
        # show_middle_slice(X.squeeze().cpu().numpy(),
        #                   save_name="/home/huangdn/Causal3D-Net/src/logging_record/resized",
        #                   mask=mask_array.squeeze().cpu().numpy())
        if self.return_type == 0:
            return X, cls_label
        elif self.return_type == 1:
            return X, msk_label
        elif self.return_type == 2:
            return X, msk_label, cls_label
        elif self.return_type == 3:
            return X, center_type, cluster
        elif self.return_type == 4:
            return X, cls_label, msk_label, center_type, cluster


if __name__ == '__main__':
    batch_size = 4
    transform = tio.Compose([
        Windowing(window_center=70, window_width=340),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((40, 160, 256)),
    ])
    excel_path = "/home/huangdn/Causal3D-Net/src/dataset/roi_data_finger.xlsx"
    dataset = PCDataset(excel_path=excel_path, transform=transform, return_type=2)
    loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)
    for x, y_msk, y_cls in tqdm(loader):
        print(x.shape, y_msk.shape, y_cls.shape)
        break
    pass
























