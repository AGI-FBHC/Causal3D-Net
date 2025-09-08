# -*- coding: utf-8 -*-
# @Time    : 2025/6/26 16:45
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: radiomics_method.py
# @Project : Causal3D-Net
import os
import logging

import pymrmr

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier

from src.metric.compute_score import compute_multi_metrics



def preprocess_data(train_data, test_data):
    scaler = StandardScaler()
    train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data),
                                     columns=train_data.columns,
                                     index=train_data.index)
    test_data_scaled = pd.DataFrame(scaler.transform(test_data),
                                    columns=test_data.columns,
                                    index=test_data.index)
    return train_data_scaled, test_data_scaled


def radiomics_with_randomforest(train_excel, test_excel, features) -> dict:
    """Chu, L.C., Park, S., Kawamoto, S., Fouladi, D.F., Shayesteh, S., Zinreich, E.S., Graves, J.S.,
    Horton, K.M., Hruban, R.H., Yuille, A.L. and Kinzler, K.W., 2019. Utility of CT radiomics features
    in differentiation of pancreatic ductal adenocarcinoma from normal pancreatic tissue.
    American Journal of Roentgenology, 213(2), pp.349-357."""
    train_excel['image_path'] = train_excel['image_path'].apply(lambda x: os.path.basename(x))
    test_excel['image_path'] = test_excel['image_path'].apply(lambda x: os.path.basename(x))
    # features = features.iloc[:, 39:]
    features_map = features.set_index('Image')
    train_features = features_map.loc[train_excel['image_path']].reset_index()
    test_features = features_map.loc[test_excel['image_path']].reset_index()
    train_label = train_excel[["cancer"]].values.ravel()
    train_features = train_features.iloc[:, 39:]
    test_features = test_features.iloc[:, 39:]

    train_features, test_features = preprocess_data(train_features, test_features)

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


def radiomics_with_SVM(train_excel, test_excel, features) -> dict:
    """Mukherjee, S., Patra, A., Khasawneh, H., Korfiatis, P., Rajamohan, N., Suman, G., Majumder,
    S., Panda, A., Johnson, M.P., Larson, N.B. and Wright, D.E., 2022. Radiomics-based
    machine-learning models can detect pancreatic cancer on prediagnostic computed tomography scans
    at a substantial lead time before clinical diagnosis. Gastroenterology, 163(5), pp.1435-1446."""
    train_excel['image_path'] = train_excel['image_path'].apply(lambda x: os.path.basename(x))
    test_excel['image_path'] = test_excel['image_path'].apply(lambda x: os.path.basename(x))
    features_map = features.set_index('Image')
    train_features = features_map.loc[train_excel['image_path']].reset_index()
    test_features = features_map.loc[test_excel['image_path']].reset_index()
    train_label = train_excel[["cancer"]].values.ravel()
    train_features = train_features.iloc[:, 39:]
    test_features = test_features.iloc[:, 39:]

    # 1. 数据预处理
    train_features, test_features = preprocess_data(train_features, test_features)

    # 2. 使用LASSO模型进行特征筛选(从88个中筛选34个特征)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(train_features, train_label)
    selector = SelectFromModel(lasso, prefit=True)
    train_selected = selector.transform(train_features)
    test_selected = selector.transform(test_features)

    # 3. 使用SVM进行而分类
    svm = SVC(kernel='linear', probability=True, random_state=42)
    # svm.fit(train_features, train_label)
    svm.fit(train_selected, train_label)
    # y_pred = svm.predict(test_features)
    y_pred = svm.predict(test_selected)
    y_prob = svm.predict_proba(test_features)[:, 1]

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def radiomics_with_XGBoost(train_excel, test_excel, features) -> dict:
    train_excel['image_path'] = train_excel['image_path'].apply(lambda x: os.path.basename(x))
    test_excel['image_path'] = test_excel['image_path'].apply(lambda x: os.path.basename(x))
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
    selected_features = pymrmr.mRMR(train_data_mrmr, 'MID', 14)
    train_features = train_features[selected_features]
    test_features = test_features[selected_features]

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    xgb.fit(train_features, train_label)

    y_pred = xgb.predict(test_features)
    y_prob = xgb.predict_proba(test_features)[:, 1]

    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def cross_validate_radiomics(excel,
                             features,
                             model_func,
                             n_splits=10,
                             save_path=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(excel), 1):
        logging.info(f"Fold {fold}/{n_splits} started.")

        train_excel = excel.iloc[train_idx].copy()
        test_excel = excel.iloc[test_idx].copy()

        outputs = model_func(train_excel, test_excel, features)
        y_pred, y_prob = outputs["y_pred"], outputs["y_prob"]
        y_true = test_excel["cancer"].values.ravel()

        metrics = compute_multi_metrics(y_true, y_pred, y_prob)
        metrics["fold"] = fold
        results.append(metrics)

        logging.info(
            f"Fold {fold} finished. "
            f"Accuracy={metrics.get('Accuracy', None):.4f}, "
            f"AUC={metrics.get('AUC', None):.4f}, "
            f"F1={metrics.get('F1', None):.4f}"
        )

    df = pd.DataFrame(results)

    mean_row = df.mean(numeric_only=True)
    mean_row["fold"] = "mean"
    std_row = df.std(numeric_only=True)
    std_row["fold"] = "std"

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    if save_path:
        df.to_csv(save_path, index=False)
        logging.info(f"Cross-validation results saved to {save_path}")

    return df



if __name__ == '__main__':

    pass
