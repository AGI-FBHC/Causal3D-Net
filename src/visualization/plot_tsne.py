# -*- coding: utf-8 -*-
# @Time    : 2025/11/24 09:15
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: plot_tsne.py
# @Project : Causal3D-Net
import os, argparse

import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from src.visualization.extract_features import load_feature_and_label
from src.utils.visualization_of_experimental_results import draw_2dim_scatter


plt.rcParams['figure.dpi'] = 1200

def run_tsne(feature_csv, label_excel, label_name, save_path_png, save_path_csv, load_tsne=False):
    df = load_feature_and_label(feature_csv, label_excel)

    if load_tsne and os.path.exists(save_path_csv):
        X_2d = pd.read_csv(save_path_csv).values
        print(f"Loaded existing t-SNE: {save_path_csv}")
    else:
        X = df.drop(columns=["filename", "image_path", "mask_path",
                             "cancer", "center", "cluster"], errors="ignore").values

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
        X_2d = tsne.fit_transform(X)

        pd.DataFrame(X_2d, columns=["tsne_x", "tsne_y"]).to_csv(save_path_csv, index=False)
        print(f"Saved t-SNE csv: {save_path_csv}")

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
        title=f"t-SNE colored by {label_name}",
        xlabel="TSNE-1",
        ylabel="TSNE-2",
        save_path_png=save_path_png,
        is_center_group=is_center_group
    )


def run_all_tsne(feature_dir, label_excel, load_tsne=False):
    feature_files = {
        "causal": os.path.join(feature_dir, "features_causal.csv"),
        "center": os.path.join(feature_dir, "features_center.csv"),
        "individual": os.path.join(feature_dir, "features_individual.csv"),
    }

    label_names = ["cancer", "center", "cluster"]

    for feat_name, csv_path in feature_files.items():
        for label_name in label_names:
            save_png = os.path.join(feature_dir, f"tsne_{feat_name}_by_{label_name}.png")
            save_csv = os.path.join(feature_dir, f"tsne_{feat_name}_by_{label_name}.csv")
            run_tsne(csv_path, label_excel, label_name, save_png, save_csv, load_tsne)


if __name__ == "__main__":
    # /home/huangdn/Causal3D-Net/src/results/2025-07-04_00-37-32
    # /home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16
    # /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx
    parser = argparse.ArgumentParser(description="Run t-SNE visualization on feature CSV files")

    parser.add_argument("--feature_dir", type=str, required=False,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16/features",
                        help="Path to directory containing features_causal.csv / "
                             "features_center.csv / features_individual.csv")
    parser.add_argument("--label_excel", type=str, required=False,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                        help="Path to the label excel file (e.g., dataset_for_test.xlsx)")
    parser.add_argument("--load_tsne", type=int, choices=[0, 1], default=0,
                        help="1 to load existing t-SNE CSV files, 0 to recompute")
    args = parser.parse_args()
    load_tsne_flag = bool(args.load_tsne)

    run_all_tsne(
        feature_dir=args.feature_dir,
        label_excel=args.label_excel,
        load_tsne=load_tsne_flag
    )