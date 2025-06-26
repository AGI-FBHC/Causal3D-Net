# -*- coding: utf-8 -*-
# @Time    : 2025/6/25 20:54
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: radiomics.py
# @Project : Causal3D-Net
import os, argparse

import numpy as np
import pandas as pd


def radiomics_method(feature_file_path, train_file_path, test_file_path, output_dir):
    features = pd.read_csv(feature_file_path)
    train = pd.read_excel(train_file_path)
    test = pd.read_excel(test_file_path)
    print(features.head())
    print(train.columns)
    print(test.columns)
    pass


if __name__ == '__main__':
    cite = "Chu, L.C., Park, S., Kawamoto, S., Fouladi, D.F., Shayesteh, S., Zinreich, E.S., Graves, J.S., Horton, K.M., Hruban, R.H., Yuille, A.L. and Kinzler, K.W., 2019. Utility of CT radiomics features in differentiation of pancreatic ductal adenocarcinoma from normal pancreatic tissue. American Journal of Roentgenology, 213(2), pp.349-357."
    print(cite)
    parser = argparse.ArgumentParser(
        description="A radiomics-based method for pancreatic cancer diagnosis using random forest")
    parser.add_argument("--input", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/radiomics_features.csv",
                        # required=True,
                        help = "path to training dataset")
    parser.add_argument("--train", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_train.xlsx",
                        # required=True,
                        help="path to training dataset")
    parser.add_argument("--test", type=str,
                        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_test.xlsx",
                        # required=True,
                        help="path to testing dataset")
    parser.add_argument("--outdir", type=str,
                        default="/home/huangdn/Causal3D-Net/src/results",
                        required=False,
                        help="output directory")
    args = parser.parse_args()
    radiomics_method(feature_file_path=args.input,
                     train_file_path=args.train,
                     test_file_path=args.test,
                     output_dir=args.outdir)
    pass
