# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 16:45
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: radiomics_method.py
# @Project : Causal3D-Net
import os

import pymrmr

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler



def preprocess_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data),
                                     columns=train_data.columns,
                                     index=train_data.index)
    test_data_scaled = pd.DataFrame(scaler.transform(test_data),
                                    columns=test_data.columns,
                                    index=test_data.index)
    return train_data_scaled, test_data_scaled


def radiomics_with_randomforest(train_excel,
                                test_excel,
                                features) -> dict:
    """Chu, L.C., Park, S., Kawamoto, S., Fouladi, D.F., Shayesteh, S., Zinreich, E.S., Graves, J.S.,
    Horton, K.M., Hruban, R.H., Yuille, A.L. and Kinzler, K.W., 2019. Utility of CT radiomics features
    in differentiation of pancreatic ductal adenocarcinoma from normal pancreatic tissue.
    American Journal of Roentgenology, 213(2), pp.349-357."""

    # features = features.iloc[:, 39:]
    features_map = features.set_index('Image')
    train_features = features_map.loc[train_excel['image_path']].reset_index()
    test_features = features_map.loc[test_excel['image_path']].reset_index()
    train_label = train_excel[["cancer"]].values.ravel()
    train_features = train_features.iloc[:, 39:]
    test_features = test_features.iloc[:, 39:]

    # 1. 数据预处理
    train_features, test_features = preprocess_data(train_features, test_features)

    # 2. 执行mRMR特征选择(MID方法, k=40, 参考原文)
    train_data_mrmr = pd.concat([train_features, pd.Series(train_label, name='target')], axis=1)
    selected_features = pymrmr.mRMR(train_data_mrmr, 'MID', 40)
    train_features = train_features[selected_features]
    test_features = test_features[selected_features]

    model = RandomForestClassifier(
        n_estimators=3000,
        random_state=42,
        n_jobs=4,
    )
    model.fit(train_features, train_label)
    y_pred = model.predict(test_features)
    y_prob = model.predict_proba(test_features)[:, 1]

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


if __name__ == '__main__':

    pass
