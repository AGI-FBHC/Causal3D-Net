# -*- coding: utf-8 -*-
# @Time    : 2025/6/21 16:51
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: generate_soft_labels.py
# @Project : Causal3D-Net
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import shap

from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.stats import pointbiserialr

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def generate(features_csv_path="/home/huangdn/Causal3D-Net/src/dataset/radiomics_features.csv",
             base_info_excel_path="/home/huangdn/Causal3D-Net/src/dataset/dataset.xlsx",
             output_path="/home/huangdn/Causal3D-Net/src/dataset/soft_labels.xlsx"):
    # 读取数据
    features = pd.read_csv(features_csv_path)
    base_info = pd.read_excel(base_info_excel_path)
    # 保存 ID 列，选取特征和标签
    unique_id = features[["Image"]]
    features = features.iloc[:, 39:]
    label = base_info[["cancer"]].values.ravel()
    # 计算 Spearman 相关性
    spearman_results = {
        col: spearmanr(features[col], label)
        for col in features.columns
    }
    spearman_df = pd.DataFrame.from_dict(spearman_results, orient='index', columns=['spearman_corr', 'p_value'])
    spearman_df = spearman_df.reindex(spearman_df['spearman_corr'].abs().sort_values(ascending=False).index)
    # 选取前 256 个特征
    top_256_features = spearman_df.head(256).index.tolist()
    select_features = features[top_256_features]
    # 标准化
    scaler = StandardScaler()
    select_features_standardized = scaler.fit_transform(select_features)
    # 聚类
    k = 6
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(select_features_standardized)
    # 将标签与 Image 匹配，再与 base_info 合并
    cluster_df = pd.DataFrame({
        "Image": unique_id["Image"].values,
        "soft_label_kmeans": cluster_labels
    })
    # 将 soft label 合并进 base_info
    base_info_updated = pd.merge(base_info, cluster_df, on="Image", how="left")

    # 保存到原 Excel 文件路径（备份原数据建议先复制一份）
    base_info_updated.to_excel(output_path, index=False)
    print(f"✅ 聚类软标签已添加并保存到: {base_info_excel_path}")


if __name__ == '__main__':
    generate()
    pass
