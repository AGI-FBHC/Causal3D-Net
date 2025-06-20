# -*- coding: utf-8 -*-
# @Time    : 2025/4/15 10:40
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: data_finger2radiomics_input.py
# @Project : Causal3D-Net
import os
import pandas as pd


def change_excel_title(is_expand=False):
    data_finger_path = "/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx"
    radiomics_read_path = "/home/huangdn/Causal3D-Net/src/dataset/dataset_for_radiomics_read.xlsx"
    df = pd.read_excel(data_finger_path)
    df = df.loc[df["raw_data"], :] if not is_expand else df
    df = df[["image_path", "mask_path"]]
    df = df.rename(columns={"image_path": "Image",
                            "mask_path": "Mask"})
    df.to_excel(radiomics_read_path, index=False)
    pass


if __name__ == '__main__':
    change_excel_title(True)
