# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 20:12
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: PC_dataset.py
# @Project : Causal3D-Net
import os.path

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentation.window import *
import torchio as tio
import torchvision.transforms as T
from skimage.transform import resize


class PCDataset(Dataset):
    def __init__(self, excel_path,
                 transform=None,
                 use_mask=False,
                 return_type=0,
                 dimension=3):
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
        self.dimension = dimension

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
        center = row["center"]
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

        if self.dimension == 2:  # for Hybrid
            # 移除深度维度 (C=1, D, H, W) -> (C=D, H, W)
            X = X.squeeze(dim=0)
            msk_label = msk_label.squeeze(dim=0)

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
            return X, center, cluster
        elif self.return_type == 4:
            return X, cls_label, msk_label, center, cluster
        elif self.return_type == 5:
            filename = os.path.basename(image_path)
            return filename, X, cls_label, msk_label, center, cluster


class PCPatchDataset(Dataset):
    def __init__(self, excel_path, transform=None, patch_size=50):
        super().__init__()
        self.df = pd.read_excel(excel_path)
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["cancer"]

        image_array = np.load(image_path)  # (D, H, W)
        D, H, W = image_array.shape

        if H < self.patch_size or W < self.patch_size:
            resized_slices = []
            for d in range(D):
                resized_slice = resize(
                    image_array[d],
                    (max(H, self.patch_size), max(W, self.patch_size)),
                    anti_aliasing=True
                )
                resized_slices.append(resized_slice)
            image_array = np.stack(resized_slices, axis=0)
            H, W = image_array.shape[1], image_array.shape[2]

        d = np.random.randint(0, D)
        h = np.random.randint(0, H - self.patch_size + 1)
        w = np.random.randint(0, W - self.patch_size + 1)
        patch = image_array[d, h:h+self.patch_size, w:w+self.patch_size]  # (50, 50)

        patch = np.expand_dims(patch, axis=0)  # (1, 50, 50)
        patch = torch.from_numpy(patch).float()

        if self.transform:
            patch = self.transform(patch)

        return patch, label


if __name__ == '__main__':
    batch_size = 4
    transform = T.Compose([
        T.Normalize(mean=[0.5], std=[0.5]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomResizedCrop(size=(50, 50), scale=(0.8, 1.2)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    excel_path = "/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx"
    dataset = PCPatchDataset(excel_path=excel_path, transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=1,
                        pin_memory=True)
    for x, y in tqdm(loader):
        print(x.shape, y.shape)
        # break
    pass
























