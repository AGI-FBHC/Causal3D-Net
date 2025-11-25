# -*- coding: utf-8 -*-
# @Time    : 2025/11/24 10:31
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: plot_umap.py
# @Project : Causal3D-Net
import os, argparse

import numpy as np
import pandas as pd

import umap.umap_ as umap
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 1200

def run_umap(feature_csv, label_excel, label_name, save_path_png, load_umap=False):
    print(f"\nLoading Features: {feature_csv}")
    df_feat = pd.read_csv(feature_csv)

    print(f"Loading Labels: {label_excel}")
    df_label = pd.read_excel(label_excel)
    df_label["filename"] = df_label["image_path"].apply(lambda x: os.path.basename(x))

    test_centers = [0, 3, 6, 8, 15, 16, 17]
    df_label = df_label[df_label["center"].isin(test_centers)]
    print(f"Label samples after test filtering: {len(df_label)}")

    df = df_feat.merge(df_label, on="filename", how="inner")
    print(f"feature shape: {df_feat.shape}, label shape: {df_label.shape}")
    print(f"Matched samples = {len(df)}")

    # ====== 读取或重新计算 UMAP ======
    save_umap_csv = save_path_png.replace(".png", ".csv")
    if load_umap and os.path.exists(save_umap_csv):
        print(f"Loading existing UMAP results: {save_umap_csv}")
        X_umap = pd.read_csv(save_umap_csv).values
    else:
        X = df.drop(columns=["filename", "image_path", "mask_path",
                             "cancer", "center", "cluster"], errors="ignore").values

        print("Running UMAP...")
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
        )
        X_umap = reducer.fit_transform(X)

        print(f"Saving UMAP 2D coordinates to: {save_umap_csv}")
        pd.DataFrame(X_umap, columns=["umap_x", "umap_y"]).to_csv(save_umap_csv, index=False)

    if label_name == "center":
        group_map = {
            'test I': [0, 3],
            'test II': [6, 8],
            'test III': [15, 16, 17],
        }

        def map_center_to_group(center_value):
            for group, center_list in group_map.items():
                if center_value in center_list:
                    return group
            return "Unknown"

        y_raw = df["center"].values
        y = np.array([map_center_to_group(c) for c in y_raw])

        unique_groups = ['test I', 'test II', 'test III']
        color_map = {g: i for i, g in enumerate(unique_groups)}
        y_color = np.array([color_map[g] for g in y])

    else:
        y = df[label_name].values
        y_color = y

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=y_color,
        cmap="tab10",
        s=15,
        alpha=0.8
    )

    plt.title(f"UMAP ({os.path.basename(feature_csv)}) colored by {label_name}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    # legend
    if label_name == "center":
        handles = []
        labels = []
        for g in unique_groups:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=plt.cm.tab10(color_map[g] / 10),
                                      markersize=6))
            labels.append(g)
        plt.legend(handles, labels, title="Center Group")
    else:
        handles, labels = scatter.legend_elements(prop="colors")
        plt.legend(handles, [f"{label_name}={lab}" for lab in labels], title="Label")

    plt.tight_layout()
    plt.savefig(save_path_png)
    print(f"Saved PNG: {save_path_png}")
    plt.close()


def run_all_umap(feature_dir, label_excel, load_umap=False):
    feature_files = {
        "causal": os.path.join(feature_dir, "features_causal.csv"),
        "center": os.path.join(feature_dir, "features_center.csv"),
        "individual": os.path.join(feature_dir, "features_individual.csv"),
    }

    label_names = ["cancer", "center", "cluster"]

    for feat_name, csv_path in feature_files.items():
        for label_name in label_names:
            save_png = os.path.join(feature_dir, f"umap_{feat_name}_by_{label_name}.png")
            run_umap(csv_path, label_excel, label_name, save_png, load_umap)


if __name__ == "__main__":
    # /home/huangdn/Causal3D-Net/src/results/2025-07-04_00-37-32
    # /home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16
    # /home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx
    parser = argparse.ArgumentParser(description="Run UMAP visualization on feature CSV files")
    parser.add_argument("--feature_dir", type=str, required=False,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16/features",
                        help="Path to directory containing features CSV files")
    parser.add_argument("--label_excel", type=str, required=False,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                        help="Path to the label excel file (e.g., dataset_for_test.xlsx)")
    parser.add_argument("--load_umap", type=int, choices=[0, 1], default=0,
                        help="1 to load existing UMAP CSV files, 0 to recompute")

    args = parser.parse_args()
    load_umap_flag = bool(args.load_umap)

    run_all_umap(feature_dir=args.feature_dir, label_excel=args.label_excel, load_umap=load_umap_flag)
