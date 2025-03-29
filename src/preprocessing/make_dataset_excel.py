# -*- coding: utf-8 -*-
# @Time    : 2025/3/24 18:48
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: make_dataset_excel.py
# @Project : Causal3D-Net
import os
import pandas as pd
import nibabel as nib
from tqdm import tqdm


def make_dataset_excel():
    public_dataset_dir = "/home/huangdn/dataset/public_datasets"
    private_dataset_dir = "/home/huangdn/dataset/private_datasets"
    all_data = []
    for center_index in range(1, 11):
        center_dir = os.path.join(public_dataset_dir, f"center_{center_index}")
        dataset_excel = os.path.join(center_dir, "dataset.xlsx")
        excel = pd.read_excel(dataset_excel)
        all_data.append(excel)
    for center_index in range(1, 5):
        center_dir = os.path.join(private_dataset_dir, f"center_{center_index}")
        dataset_excel = os.path.join(center_dir, "dataset.xlsx")
        excel = pd.read_excel(dataset_excel)
        all_data.append(excel)
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_excel("/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx", index=False)
    pass



if __name__ == "__main__":
    make_dataset_excel()

