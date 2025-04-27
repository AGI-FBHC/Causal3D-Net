# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 21:03
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: dataset_splitting.py
# @Project : Causal3D-Net
import os, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split(excel_path, output_dir):
    all_data = pd.read_excel(excel_path)
    train_data, test_data = train_test_split(
        all_data,
        test_size=0.2,
        random_state=42,  # 保证每次划分一样
        shuffle=True
    )
    train_save_path = os.path.join(output_dir, "train_dataset.xlsx")
    test_save_path = os.path.join(output_dir, "test_dataset.xlsx")

    train_data.to_excel(train_save_path, index=False)
    print(f"Train data saved to {train_save_path}")
    test_data.to_excel(test_save_path, index=False)
    print(f"Test data saved to {test_save_path}")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset splitting")
    parser.add_argument(
        "--input", type=str,
        default="/home/huangdn/Causal3D-Net/src/data/data_finger.xlsx",
        help="Origin sorted images and masks Excel file path."
    )
    parser.add_argument(
        "--outdir", type=str,
        default="/home/huangdn/Causal3D-Net/src/data/",
        help="Output folder path."
    )
    args = parser.parse_args()
    split(args.input, args.outdir)
