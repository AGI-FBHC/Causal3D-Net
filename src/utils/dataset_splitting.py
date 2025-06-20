# -*- coding: utf-8 -*-
# @Time    : 2025/4/26 21:03
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: dataset_splitting.py
# @Project : Causal3D-Net
import os, argparse
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split


def split(excel_path, output_dir):
    all_data = pd.read_excel(excel_path)
    train_data, test_data = train_test_split(
        all_data,
        test_size=0.2,
        random_state=42,  # 保证每次划分一样
        shuffle=True
    )
    train_data = train_data.sort_values(by="image_path", ascending=True)
    test_data = test_data.sort_values(by="image_path", ascending=True)

    train_save_path = os.path.join(output_dir, "train_dataset.xlsx")
    test_save_path = os.path.join(output_dir, "test_dataset.xlsx")

    train_data.to_excel(train_save_path, index=False)
    print(f"Train data saved to {train_save_path}")
    test_data.to_excel(test_save_path, index=False)
    print(f"Test data saved to {test_save_path}")
    pass


def extract_center(path):
    for i in range(1, 20):
        if f"Center{str(i).zfill(2)}" in path:
            return f"center{i}"
    return None


def extract_source(path):
    if "public" in path:
        return "public"
    else:
        return "private"


def split_with_plan(excel_path, output_dir, is_expand=False, is_print=False):
    """Divide according to the description in the paper."""
    all_data = pd.read_excel(excel_path)
    all_data = all_data.loc[all_data["raw_data"], :] if not is_expand else all_data
    # 加入 center 和 source 列
    all_data["center"] = all_data["image_path"].apply(extract_center)
    all_data["source"] = all_data["image_path"].apply(extract_source)

    # ==== Private Test ====
    private = all_data[all_data["source"] == "private"]

    # center1: cancer0 取72，cancer1 取33
    private_c1 = private[private["center"] == "center1"]
    c1_0 = private_c1[private_c1["cancer"] == 0].sample(n=72, random_state=42)
    c1_1 = private_c1[private_c1["cancer"] == 1].sample(n=32, random_state=42)
    private_c1_test = pd.concat([c1_0, c1_1])

    # center4: 全部为test
    private_c4_test = private[private["center"] == "center4"]

    # private test 总集
    private_test = pd.concat([private_c1_test, private_c4_test])
    private_train = private.drop(private_test.index)

    # ==== Public Test ====
    public = all_data[all_data["source"] == "public"]

    # center1,2,3,5,6,7,8,11全部
    public_centers = [f"center{i}" for i in [1, 2, 3, 5, 6, 7, 8, 11]]  # 测试center数量可从此处减少
    public_c_test = public[public["center"].isin(public_centers)]

    # center14: cancer 0 和 1 各取 14
    center14 = public[public["center"] == "center14"]
    c14_0 = center14[center14["cancer"] == 0].sample(n=14, random_state=42)
    c14_1 = center14[center14["cancer"] == 1].sample(n=14, random_state=42)
    public_14_test = pd.concat([c14_0, c14_1])

    # public test 总集
    public_test = pd.concat([public_c_test, public_14_test])
    public_train = public.drop(public_test.index)

    # ==== 合并 ====
    train_df = pd.concat([private_train, public_train]).reset_index(drop=True)
    test_df = pd.concat([private_test, public_test]).reset_index(drop=True)

    print(f"Train size: {len(train_df)}") if is_print else None
    print(f"Test size: {len(test_df)}") if is_print else None

    train_save_path = os.path.join(output_dir, "dataset_for_train.xlsx")
    test_save_path = os.path.join(output_dir, "dataset_for_test.xlsx")
    train_df.to_csv(train_save_path, index=False)
    test_df.to_csv(test_save_path, index=False)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset splitting")
    parser.add_argument(
        "--input", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_roi.xlsx",
        help="Origin sorted images and masks Excel file path."
    )
    parser.add_argument(
        "--outdir", type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/",
        help="Output folder path."
    )
    args = parser.parse_args()
    split_with_plan(args.input, args.outdir)
