# -*- coding: utf-8 -*-
# @Time    : 2025/6/25 20:54
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: baseline_entry.py
# @Project : Causal3D-Net
import os, argparse

import numpy as np
import pandas as pd

from src.baselines.radiomics_method import radiomics_with_randomforest
from src.preprocessing.extract_radiomics import extract_radiomics_features


def run_baseline(train_excel_path,
                 test_excel_path,
                 method: str,
                 cuda_id=5,
                 output_dir: str = "/home/huangdn/Causal3D-Net/src/results"):

    if method == "radiomics":
        feature_file_path = "/home/huangdn/Causal3D-Net/src/dataset/radiomics_features.csv"
        print(os.path.isfile(feature_file_path))
        # features = pd.read_csv(feature_file_path) \
        #     if os.path.isfile(feature_file_path) \
        #     else extract_radiomics_features("/home/huangdn/Causal3D-Net/src/dataset/dataset_for_radiomics_read.xlsx",
        #                                     feature_file_path,
        #                                     "/home/huangdn/Causal3D-Net/src/config/Params.yaml",
        #                                     "/home/huangdn/Causal3D-Net/src/logging_record/extract_radiomics_features.log",
        #                                     8)
        # radiomics_with_randomforest(train_excel_path, test_excel_path, features)
        pass
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
    pass




if __name__ == '__main__':
    cite = "Chu, L.C., Park, S., Kawamoto, S., Fouladi, D.F., Shayesteh, S., Zinreich, E.S., Graves, J.S., Horton, K.M., Hruban, R.H., Yuille, A.L. and Kinzler, K.W., 2019. Utility of CT radiomics features in differentiation of pancreatic ductal adenocarcinoma from normal pancreatic tissue. American Journal of Roentgenology, 213(2), pp.349-357."
    print(cite)
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
