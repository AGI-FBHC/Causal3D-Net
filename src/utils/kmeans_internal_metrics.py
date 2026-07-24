# -*- coding: utf-8 -*-
# @Time    : 2026/7/24 18:20
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: kmeans_internal_metrics.py
# @Project : Causal3D-Net
import os, argparse

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr

from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score,
                             davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import StandardScaler


def select_features(dataset_file_path,
                    feature_file_path,
                    feature_num=256):
    """Keep the feature-selection basis consistent with the confounder script."""
    dataset = pd.read_excel(dataset_file_path)
    feature = pd.read_csv(feature_file_path)

    if "cancer" not in dataset.columns:
        raise ValueError(f"Missing required column 'cancer' in {dataset_file_path}")
    if dataset.shape[0] != feature.shape[0]:
        raise ValueError(
            "The dataset and radiomics feature table must contain the same "
            f"number of rows, but got {dataset.shape[0]} and {feature.shape[0]}."
        )

    # Identical to construct_confounder_labels.py: discard the first 39 columns,
    # rank by absolute Spearman correlation, and retain the first 256 features.
    feature = feature.iloc[:, 39:]
    label = dataset[["cancer"]].values.ravel()
    spearman_results = {
        col: spearmanr(feature[col], label)
        for col in feature.columns
    }
    spearman_df = pd.DataFrame.from_dict(
        spearman_results,
        orient="index",
        columns=["spearman_corr", "p_value"],
    )
    spearman_df = spearman_df.reindex(
        spearman_df["spearman_corr"].abs().sort_values(ascending=False).index
    )

    selected_feature_names = spearman_df[:feature_num].index
    selected_features = feature[selected_feature_names]
    scaler = StandardScaler()
    selected_features_standardized = scaler.fit_transform(selected_features)

    return selected_features_standardized, spearman_df.loc[selected_feature_names]


def compute_dunn_index(features,
                       labels,
                       sample_size=1000,
                       random_state=42):
    """
    Compute the Dunn index.

    For tractable computation on high-dimensional radiomics data, a stratified
    sample is used when the sample count exceeds sample_size. Set sample_size=0
    to use all samples.
    """
    unique_labels = np.unique(labels)
    rng = np.random.default_rng(random_state)

    if sample_size > 0 and features.shape[0] > sample_size:
        sample_indices = []
        samples_per_cluster = max(2, sample_size // len(unique_labels))
        for cluster in unique_labels:
            cluster_indices = np.flatnonzero(labels == cluster)
            take_num = min(samples_per_cluster, len(cluster_indices))
            sample_indices.extend(
                rng.choice(cluster_indices, size=take_num, replace=False)
            )
        sample_indices = np.asarray(sample_indices)
        features = features[sample_indices]
        labels = labels[sample_indices]

    max_intra_cluster_distance = 0.0
    min_inter_cluster_distance = np.inf

    for cluster in unique_labels:
        cluster_features = features[labels == cluster]
        if cluster_features.shape[0] > 1:
            max_intra_cluster_distance = max(
                max_intra_cluster_distance,
                float(pdist(cluster_features).max()),
            )

    for i, cluster_i in enumerate(unique_labels[:-1]):
        features_i = features[labels == cluster_i]
        for cluster_j in unique_labels[i + 1:]:
            features_j = features[labels == cluster_j]
            min_inter_cluster_distance = min(
                min_inter_cluster_distance,
                float(cdist(features_i, features_j).min()),
            )

    if max_intra_cluster_distance == 0:
        return np.nan, features.shape[0]
    return min_inter_cluster_distance / max_intra_cluster_distance, features.shape[0]


def compute_s_dbw_index(features, labels):
    """Compute the S_Dbw scatter-density validity index."""
    unique_labels = np.unique(labels)
    cluster_features = [features[labels == cluster] for cluster in unique_labels]
    cluster_centers = [cluster.mean(axis=0) for cluster in cluster_features]

    total_std_norm = np.linalg.norm(np.std(features, axis=0, ddof=0))
    if total_std_norm == 0:
        return np.nan

    scatter = np.mean([
        np.linalg.norm(np.std(cluster, axis=0, ddof=0)) / total_std_norm
        for cluster in cluster_features
    ])

    density_radius = np.sqrt(
        np.sum([
            np.linalg.norm(np.std(cluster, axis=0, ddof=0))
            for cluster in cluster_features
        ]) / len(cluster_features)
    )

    def density(points, center):
        distances = np.linalg.norm(points - center, axis=1)
        return np.sum(distances <= density_radius)

    density_between = 0.0
    pair_count = 0
    for i in range(len(unique_labels) - 1):
        for j in range(i + 1, len(unique_labels)):
            pair_points = np.vstack([cluster_features[i], cluster_features[j]])
            midpoint = (cluster_centers[i] + cluster_centers[j]) / 2
            midpoint_density = density(pair_points, midpoint)
            center_density = max(
                density(pair_points, cluster_centers[i]),
                density(pair_points, cluster_centers[j]),
            )
            if center_density > 0:
                density_between += midpoint_density / center_density
            pair_count += 1

    density_between = density_between / pair_count if pair_count else 0.0
    return scatter + density_between


def compute_gap_statistic(features,
                          observed_inertia,
                          k,
                          reference_num=10,
                          random_state=42):
    """Compute the Gap statistic and its simulation standard error."""
    rng = np.random.default_rng(random_state + k)
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    reference_log_inertias = []

    for reference_id in range(reference_num):
        reference_features = rng.uniform(
            low=feature_min,
            high=feature_max,
            size=features.shape,
        )
        reference_model = KMeans(
            n_clusters=k,
            random_state=random_state + reference_id,
        )
        reference_model.fit(reference_features)
        reference_log_inertias.append(np.log(reference_model.inertia_))

    reference_log_inertias = np.asarray(reference_log_inertias)
    gap = reference_log_inertias.mean() - np.log(observed_inertia)
    gap_standard_error = (
        reference_log_inertias.std(ddof=1)
        * np.sqrt(1 + 1 / reference_num)
        if reference_num > 1 else 0.0
    )
    return gap, gap_standard_error


def explore_k(dataset_file_path,
              feature_file_path,
              output_path,
              min_k=2,
              max_k=15,
              feature_num=256,
              reference_num=10,
              dunn_sample_size=1000,
              random_state=42):
    features, selected_feature_info = select_features(
        dataset_file_path=dataset_file_path,
        feature_file_path=feature_file_path,
        feature_num=feature_num,
    )

    if min_k < 2 or min_k > max_k:
        raise ValueError("The k range must satisfy 2 <= min_k <= max_k.")
    if max_k >= features.shape[0]:
        raise ValueError("max_k must be smaller than the number of samples.")
    if reference_num < 1:
        raise ValueError("reference_num must be at least 1.")

    results = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=random_state,
        )
        cluster_labels = kmeans.fit_predict(features)
        cluster_sizes = np.bincount(cluster_labels, minlength=k)

        dunn_index, actual_dunn_sample_size = compute_dunn_index(
            features,
            cluster_labels,
            sample_size=dunn_sample_size,
            random_state=random_state,
        )
        gap_statistic, gap_standard_error = compute_gap_statistic(
            features,
            observed_inertia=kmeans.inertia_,
            k=k,
            reference_num=reference_num,
            random_state=random_state,
        )

        result = {
            "k": k,
            "davies_bouldin_index": davies_bouldin_score(
                features,
                cluster_labels,
            ),
            "silhouette_score": silhouette_score(features, cluster_labels),
            "calinski_harabasz_index": calinski_harabasz_score(
                features,
                cluster_labels,
            ),
            "inertia": kmeans.inertia_,
            "dunn_index": dunn_index,
            "dunn_sample_size": actual_dunn_sample_size,
            "gap_statistic": gap_statistic,
            "gap_standard_error": gap_standard_error,
            "s_dbw_index": compute_s_dbw_index(features, cluster_labels),
            "smallest_cluster_size": int(cluster_sizes.min()),
            "largest_cluster_size": int(cluster_sizes.max()),
        }
        results.append(result)

        print(
            f"k={k} | DBI={result['davies_bouldin_index']:.6f} | "
            f"Silhouette={result['silhouette_score']:.6f} | "
            f"CH={result['calinski_harabasz_index']:.6f} | "
            f"Dunn={result['dunn_index']:.6f} | "
            f"Gap={result['gap_statistic']:.6f} | "
            f"S_Dbw={result['s_dbw_index']:.6f}"
        )

    result_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    selected_feature_path = os.path.splitext(output_path)[0] + "_features.csv"
    selected_feature_info.to_csv(selected_feature_path)

    print(f"Results saved to {output_path}")
    print(f"Selected features saved to {selected_feature_path}")
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Explore internal K-Means validity indices."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_roi.xlsx",
        help="Path to dataset Excel file containing the cancer label.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/radiomics_features.csv",
        help="Path to radiomics feature CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/huangdn/Causal3D-Net/src/logging_record/"
                "kmeans_internal_metrics.csv",
        help="Path to save internal clustering metrics.",
    )
    parser.add_argument("--min_k", type=int, default=4)
    parser.add_argument("--max_k", type=int, default=16)
    parser.add_argument("--feature_num", type=int, default=256)
    parser.add_argument(
        "--reference_num",
        type=int,
        default=10,
        help="Number of reference datasets used by the Gap statistic.",
    )
    parser.add_argument(
        "--dunn_sample_size",
        type=int,
        default=1000,
        help="Stratified sample size for Dunn index; use 0 for all samples.",
    )
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    explore_k(
        dataset_file_path=args.dataset,
        feature_file_path=args.feature,
        output_path=args.output,
        min_k=args.min_k,
        max_k=args.max_k,
        feature_num=args.feature_num,
        reference_num=args.reference_num,
        dunn_sample_size=args.dunn_sample_size,
        random_state=args.random_state,
    )
