# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 18:48
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: make_dataset_excel.py
# @Project : Causal3D-Net
import os
import shutil
import pandas as pd
import nibabel as nib
from tqdm import tqdm


def make_dataset_excel():
    public_dataset_dir = "/home/huangdn/dataset/public_datasets"
    private_dataset_dir = "/home/huangdn/dataset/private_datasets"
    all_data = []
    for center_index in range(1, 15):
        if center_index == 13:
            continue
        center_dir = os.path.join(public_dataset_dir, f"center_{center_index}")
        dataset_excel = os.path.join(center_dir, "dataset.xlsx")
        excel = pd.read_excel(dataset_excel)
        all_data.append(excel)
    for center_index in range(1, 6):
        center_dir = os.path.join(private_dataset_dir, f"center_{center_index}")
        dataset_excel = os.path.join(center_dir, "dataset.xlsx")
        excel = pd.read_excel(dataset_excel)
        all_data.append(excel)
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.iloc[:, :-1]
    # combined_data.to_excel("/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx", index=False)
    for path in combined_data['image_path']:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
    target_root = "/home/huangdn/Causal3D-Net/src/dataset"
    images_target_dir = os.path.join(target_root, "images")
    masks_target_dir = os.path.join(target_root, "masks")
    os.makedirs(images_target_dir, exist_ok=True)
    os.makedirs(masks_target_dir, exist_ok=True)

    all_data = []
    for idx, row in tqdm(combined_data.iterrows()):
        orig_image_path = row["image_path"]
        orig_mask_path = row["mask_path"]
        tag = "_public.nii.gz" if "public_datasets" in orig_image_path else "_private.nii.gz"
        cancer = row["cancer"]
        new_image_path = os.path.join(images_target_dir, os.path.basename(orig_image_path).replace(".nii.gz", tag))
        new_mask_path = os.path.join(masks_target_dir, os.path.basename(orig_mask_path).replace(".nii.gz", tag))
        shutil.move(orig_image_path, new_image_path)
        shutil.move(orig_mask_path, new_mask_path)
        all_data.append([new_image_path, new_mask_path, cancer])
    df_new = pd.DataFrame(all_data, columns=["image_path", "mask_path", "cancer"])
    df_new.to_excel("/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx", index=False)
    pass



if __name__ == "__main__":
    make_dataset_excel()

