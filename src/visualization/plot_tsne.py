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


plt.rcParams['figure.dpi'] = 300

def run_tsne(feature_csv, label_excel, label_name, save_path_png, save_path_csv, load_tsne=False):
    """
    feature_csv:   Causal / Center / Individual 特征的 CSV
    label_excel:   标签文件
    label_name:    使用哪个标签着色
    save_path_png: t-SNE 图保存路径
    save_path_csv: t-SNE 2D数据保存路径
    load_tsne:     若为 True，则优先从 CSV 读取降维结果
    """
    print(f"\nLoading Features: {feature_csv}")
    df_feat = pd.read_csv(feature_csv)

    print(f"Loading Labels: {label_excel}")
    df_label = pd.read_excel(label_excel)
    df_label["filename"] = df_label["image_path"].apply(lambda x: os.path.basename(x))

    # 过滤 test centers
    test_centers = [0, 3, 6, 8, 15, 16, 17]
    df_label = df_label[df_label["center"].isin(test_centers)]

    print(f"Label samples after test filtering: {len(df_label)}")

    df = df_feat.merge(df_label, on="filename", how="inner")
    print(f"feature shape: {df_feat.shape}, label shape: {df_label.shape}")
    print(f"Matched samples = {len(df)}")

    if load_tsne and os.path.exists(save_path_csv):
        print(f"Loading existing t-SNE results: {save_path_csv}")
        X_tsne = pd.read_csv(save_path_csv).values
    else:
        X = df.drop(columns=["filename", "image_path", "mask_path",
                             "cancer", "center", "cluster"], errors="ignore").values
        print("Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            init="pca",
            random_state=42,
            perplexity=30
        )
        X_tsne = tsne.fit_transform(X)

        print(f"Saving t-SNE 2D coordinates to: {save_path_csv}")
        pd.DataFrame(X_tsne, columns=["tsne_x", "tsne_y"]).to_csv(save_path_csv, index=False)

    y = df[label_name].values

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
                          cmap="tab10", s=15, alpha=0.8)

    plt.title(f"t-SNE ({os.path.basename(feature_csv)}) colored by {label_name}")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    handles, labels = scatter.legend_elements(prop="colors")
    plt.legend(handles, [f"{label_name}={lab}" for lab in labels], title="Label")

    plt.tight_layout()
    plt.savefig(save_path_png)
    print(f"Saved PNG: {save_path_png}")
    plt.close()


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
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16",
                        help="Path to directory containing features_causal.csv / "
                             "features_center.csv / features_individual.csv")
    parser.add_argument("--label_excel", type=str, required=False,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                        help="Path to the label excel file (e.g., dataset_for_test.xlsx)")
    parser.add_argument("--load_tsne", type=int, choices=[0, 1], default=0,
                        help="1 to load existing t-SNE CSV files, 0 to recompute")
    args = parser.parse_args()
    # Convert int flag to bool
    load_tsne_flag = bool(args.load_tsne)

    # Run
    run_all_tsne(
        feature_dir=args.feature_dir,
        label_excel=args.label_excel,
        load_tsne=load_tsne_flag
    )