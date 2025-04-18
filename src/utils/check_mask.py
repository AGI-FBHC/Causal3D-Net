# -*- coding: utf-8 -*-
# @Time    : 2025/4/18 09:25
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: check_mask.py
# @Project : Causal3D-Net
import os
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm


def print_mask_type(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    info = f"{file_path}: mask type has {np.unique(data)}"
    logging.info(info)


def through_files():
    excel_path = "/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx"
    logging.basicConfig(
        filename="/home/huangdn/Causal3D-Net/src/logging_record/mask_type.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        filemode='w'
    )
    df = pd.read_excel(excel_path)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        mask_path = row["mask_path"]
        print_mask_type(mask_path)
    pass


if __name__ == '__main__':
    through_files()

