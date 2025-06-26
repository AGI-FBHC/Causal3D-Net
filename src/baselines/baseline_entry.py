# -*- coding: utf-8 -*-
# @Time    : 2025/6/25 20:54
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: baseline_entry.py
# @Project : Causal3D-Net
import logging
import os, argparse
from datetime import datetime

import numpy as np
import pandas as pd

from src.baselines.radiomics_method import radiomics_with_randomforest
from src.preprocessing.extract_radiomics import extract_radiomics_features
from src.metric.compute_score import compute_multi_metrics, evaluate_test_result
from sklearn.metrics import (accuracy_score,
                             roc_auc_score,
                             recall_score,
                             precision_score,
                             f1_score,
                             confusion_matrix)


def run_baseline(train_excel_path,
                 test_excel_path,
                 method: str,
                 cuda_id=5,
                 output_dir: str = "/home/huangdn/Causal3D-Net/src/results"):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_dir = os.path.join(output_dir, current_time)
    os.makedirs(current_dir, exist_ok=True)
    diagnose_dir = os.path.join(output_dir, current_time, "diagnose")
    os.makedirs(diagnose_dir, exist_ok=True)

    log_filename = os.path.join(current_dir, "baseline_record.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    train_excel = pd.read_excel(train_excel_path)
    test_excel = pd.read_excel(test_excel_path)
    test_center = test_excel["center"].values
    test_cancer = test_excel["cancer"].values
    train_excel['image_path'] = train_excel['image_path'].apply(
        lambda x: os.path.basename(x).replace(".npy", ".nii.gz"))
    test_excel['image_path'] = test_excel['image_path'].apply(
        lambda x: os.path.basename(x).replace(".npy", ".nii.gz"))
    train_excel = train_excel[["image_path", "cancer"]]
    test_excel = test_excel[["image_path", "cancer"]]

    center_groups = {
        'internal_test_1': [0, 3],
        'external_test_1': [6, 8],
        'external_test_2': [15, 16, 17],
        'uncertainty_test': [9, 10, 12],
    }

    logging.info("🚀 Starting model training...")
    test_result = dict()
    if method == "radiomics":
        cite = ("\n\n##############################\n"
                "Chu, L.C., Park, S., Kawamoto, S., Fouladi, D.F., Shayesteh, S., Zinreich, E.S., Graves, J.S., "
                "Horton, K.M., Hruban, R.H., Yuille, A.L. and Kinzler, K.W., 2019. Utility of CT radiomics features "
                "in differentiation of pancreatic ductal adenocarcinoma from normal pancreatic tissue. "
                "American Journal of Roentgenology, 213(2), pp.349-357.\n"
                "##############################\n")
        logging.info(cite)

        feature_file_path = "/home/huangdn/Causal3D-Net/src/dataset/radiomics_features.csv"
        features = pd.read_csv(feature_file_path) \
            if os.path.isfile(feature_file_path) \
            else extract_radiomics_features("/home/huangdn/Causal3D-Net/src/dataset/dataset_for_radiomics_read.xlsx",
                                            feature_file_path,
                                            "/home/huangdn/Causal3D-Net/src/config/Params.yaml",
                                            "/home/huangdn/Causal3D-Net/src/logging_record/extract_radiomics_features.log",
                                            8)

        test_result = radiomics_with_randomforest(train_excel, test_excel, features)
    elif method == "2.5d_vgg":

        pass
    elif method == "vit":

        pass
    elif method == "3d_cnn":

        pass
    elif method == "hybrid_transformer":

        pass
    elif method == "cnet":

        pass
    elif method == "neural_transformer":

        pass
    elif method == "mix_style":

        pass
    elif method == "big_aug":

        pass
    elif method == "rand_conv":

        pass
    elif method == "adver_conv":

        pass
    elif method == "causality_aug":

        pass
    elif method == "chen":

        pass
    elif method == "chu":

        pass
    elif method == "liu":

        pass
    elif method == "zhu":

        pass
    elif method == "xia":

        pass
    logging.info("✅ Training completed successfully.")
    test_result["center"] = test_center
    test_result["cancer"] = test_cancer

    summary = evaluate_test_result(test_result, center_groups)

    # add to log file
    for group, metrics in summary.items():
        logging.info(f"📊 Metrics for {group}:")
        for k, v in metrics.items():
            logging.info(f"  {k}: {v:.4f}")

    csv_df = pd.DataFrame.from_dict(summary)
    csv_df.to_csv(os.path.join(diagnose_dir, method + ".csv"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Reproducing baseline methods from related works")
    parser.add_argument("--train", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
                        # required=True,
                        help="path to training dataset")
    parser.add_argument("--test", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                        # required=True,
                        help="path to testing dataset")
    parser.add_argument("--method", type=str,
                        default="radiomics",
                        # required=True,
                        help="baseline method")
    parser.add_argument("--cuda_id", type=int,
                        default=5,
                        help="CUDA ID")
    parser.add_argument("--outdir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results",
                        required=False,
                        help="output directory")
    args = parser.parse_args()
    run_baseline(train_excel_path=args.train,
                 test_excel_path=args.test,
                 method=args.method,
                 cuda_id=args.cuda_id,
                 output_dir=args.outdir)
    pass
