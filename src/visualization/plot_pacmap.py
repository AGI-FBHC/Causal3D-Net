# -*- coding: utf-8 -*-
# @Time    : 2025/11/24 11:13
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: plot_pacmap.py
# @Project : Causal3D-Net
import os, argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pacmap

from src.visualization.extract_features import load_feature_and_label
from src.utils.visualization_of_experimental_results import draw_2dim_scatter


plt.rcParams['figure.dpi'] = 1200

def run_pacmap(feature_csv, label_excel, label_name, save_path_png, load_pacmap=False):
    df = load_feature_and_label(feature_csv, label_excel)

    save_pacmap_csv = save_path_png.replace(".png", ".csv")

    if load_pacmap and os.path.exists(save_pacmap_csv):
        X_2d = pd.read_csv(save_pacmap_csv).values
        print(f"Loaded PaCMAP csv: {save_pacmap_csv}")
    else:
        X = df.drop(columns=["filename", "image_path", "mask_path",
                             "cancer", "center", "cluster"], errors="ignore").values

        reducer = pacmap.PaCMAP(
            n_components=2,
            random_state=42,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
        )
        X_2d = reducer.fit_transform(X)

        pd.DataFrame(X_2d, columns=["pacmap_x", "pacmap_y"]).to_csv(save_pacmap_csv, index=False)
        print(f"Saved PaCMAP csv: {save_pacmap_csv}")

    # center 分组
    if label_name == "center":
        group_map = {
            0: 'test I', 3: 'test I',
            6: 'test II', 8: 'test II',
            15: 'test III', 16: 'test III', 17: 'test III'
        }
        y = np.array([group_map[c] for c in df["center"].values])
        is_center_group = True
    else:
        y = df[label_name].values
        is_center_group = False

    draw_2dim_scatter(
        X_2d, y,
        title=f"PaCMAP colored by {label_name}",
        xlabel="PaCMAP-1",
        ylabel="PaCMAP-2",
        save_path_png=save_path_png,
        is_center_group=is_center_group
    )


def run_all_pacmap(feature_dir, label_excel, load_pacmap=False):
    feature_files = {
        "causal": os.path.join(feature_dir, "features_causal.csv"),
        "center": os.path.join(feature_dir, "features_center.csv"),
        "individual": os.path.join(feature_dir, "features_individual.csv"),
    }

    label_names = ["cancer", "center", "cluster"]

    for feat_name, csv_path in feature_files.items():
        for label_name in label_names:
            save_png = os.path.join(feature_dir, f"pacmap_{feat_name}_by_{label_name}.png")
            run_pacmap(csv_path, label_excel, label_name, save_png, load_pacmap)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PaCMAP visualization on feature CSV files")

    parser.add_argument("--feature_dir", type=str, required=False,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16/features",
                        help="Directory containing features_causal.csv etc.")
    parser.add_argument("--label_excel", type=str, required=False,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                        help="Excel containing labels")
    parser.add_argument("--load_pacmap", type=int, choices=[0, 1], default=0,
                        help="1 to load existing PaCMAP CSV, 0 to recompute")

    args = parser.parse_args()
    load_pacmap_flag = bool(args.load_pacmap)

    run_all_pacmap(
        feature_dir=args.feature_dir,
        label_excel=args.label_excel,
        load_pacmap=load_pacmap_flag
    )