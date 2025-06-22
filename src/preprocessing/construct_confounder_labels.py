# -*- coding: utf-8 -*-
# @Time    : 2025/6/22 19:22
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: construct_confounder_labels.py
# @Project : Causal3D-Net
import os, argparse
import re

import pandas as pd

from scipy.stats import spearmanr

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def extract_center_type(path):
    match = re.search(r'Center(\d+).*_(public|private)', path)
    if match:
        prefix = 'u' if match.group(2) == 'public' else 'r'
        return prefix + match.group(1)
    else:
        return None


def construct(dataset_file_path,
              feature_file_path,):
    dataset = pd.read_excel(dataset_file_path)
    feature = pd.read_csv(feature_file_path)

    dataset["center_type"] = dataset["image_path"].apply(extract_center_type)

    feature = feature.iloc[:, 39:]
    label = dataset[["cancer"]].values.ravel()
    spearman_results = {col: spearmanr(feature[col], label)
                        for col in feature.columns}
    spearman_df = pd.DataFrame.from_dict(
        spearman_results, orient='index', columns=['spearman_corr', 'p_value'])
    spearman_df = spearman_df.reindex(
        spearman_df['spearman_corr'].abs().sort_values(ascending=False).index)
    select_features = feature[spearman_df[:256].index]

    scaler = StandardScaler()
    select_features_standardized = scaler.fit_transform(select_features)

    k = 6  # 参考 Davies-Bouldin Index 选择
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(select_features_standardized)

    dataset["cluster"] = cluster_labels
    save_path = os.path.join(os.path.dirname(dataset_file_path), "dataset_for_confounder.xlsx")
    dataset.to_excel(save_path, index=False)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Constructing confounder labels")
    parser.add_argument("--dataset", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_roi.xlsx",
                        # required=True,
                        help="path to dataset Excel file.")
    parser.add_argument("--feature", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/radiomics_features.csv",
                        # required=True,
                        help="path to feature CSV file.")
    args = parser.parse_args()
    construct(args.dataset, args.feature)
    pass