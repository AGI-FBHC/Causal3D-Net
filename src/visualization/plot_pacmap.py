# -*- coding: utf-8 -*-
# @Time    : 2025/11/24 11:13
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: plot_pacmap.py
# @Project : Causal3D-Net
import os, argparse

import pandas as pd
import matplotlib.pyplot as plt

import pacmap


plt.rcParams['figure.dpi'] = 1200

def run_pacmap(feature_csv, label_excel, label_name, save_path_png, load_pacmap=False):
    """
    feature_csv:   特征 CSV（Causal / Center / Individual）
    label_excel:   标签文件
    label_name:    用于着色的标签
    save_path_png: PaCMAP 图保存路径
    load_pacmap:   若为 True，则优先从 CSV 读取降维结果
    """
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

    save_pacmap_csv = save_path_png.replace(".png", ".csv")

    if load_pacmap and os.path.exists(save_pacmap_csv):
        print(f"Loading existing PaCMAP results: {save_pacmap_csv}")
        X_p = pd.read_csv(save_pacmap_csv).values
    else:
        X = df.drop(columns=["filename", "image_path", "mask_path",
                             "cancer", "center", "cluster"], errors="ignore").values

        print("Running PaCMAP...")
        reducer = pacmap.PaCMAP(
            n_components=2,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42
        )
        X_p = reducer.fit_transform(X, init="pca")

        print(f"Saving PaCMAP 2D coordinates to: {save_pacmap_csv}")
        pd.DataFrame(X_p, columns=["pacmap_x", "pacmap_y"]).to_csv(save_pacmap_csv, index=False)

    y = df[label_name].values

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(X_p[:, 0], X_p[:, 1], c=y,
                          cmap="tab10", s=15, alpha=0.8)

    plt.title(f"PaCMAP ({os.path.basename(feature_csv)}) colored by {label_name}")
    plt.xlabel("PaCMAP-1")
    plt.ylabel("PaCMAP-2")

    handles, labels = scatter.legend_elements(prop="colors")
    plt.legend(handles, [f"{label_name}={lab}" for lab in labels], title="Label")

    plt.tight_layout()
    plt.savefig(save_path_png)
    print(f"Saved PNG: {save_path_png}")
    plt.close()



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

    parser.add_argument("--feature_dir", type=str, required=True,
                        default="/home/huangdn/Causal3D-Net/src/results/2025-11-05_02-24-16",
                        help="Directory containing features_causal.csv etc.")
    parser.add_argument("--label_excel", type=str, required=True,
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