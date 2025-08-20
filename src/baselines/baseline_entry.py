# -*- coding: utf-8 -*-
# @Time    : 2025/6/25 20:54
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: baseline_entry.py
# @Project : Causal3D-Net
import logging
import os, argparse
from datetime import datetime

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from src.baselines.radiomics_method import radiomics_with_randomforest
# from src.baselines.vgg25d_method import ct_with_vgg25d
from src.baselines.end_to_end_method import ct_with_dl
from src.models.VGG_2_5D import VGG25D
from src.models.ViT import ViTClassifier
from src.models.ResNet import generate_model
from src.models.Hybrid_Transformer.Hybrid.getmodel import get_model
from src.models.CNet import CNet
from src.models.Neural_Transformer import ViTForIPMNClassification
from src.models.PANDA import MultiTask3DCNN
from src.preprocessing.extract_radiomics import extract_radiomics_features
from src.metric.compute_score import compute_multi_metrics, evaluate_test_result
from src.utils.init_weights import init_weights_kaiming, load_shared_weights
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
        lambda x: x.replace(".npy", ".nii.gz"))
    test_excel['image_path'] = test_excel['image_path'].apply(
        lambda x: x.replace(".npy", ".nii.gz"))
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
                "American Journal of Roentgenology, 213(2), pp.349-357."
                "\n##############################\n")
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
        cite = ("\n\n##############################\n"
                "Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks "
                "for large-scale image recognition. arXiv preprint arXiv:1409.1556."
                "\n##############################\n")
        logging.info(cite)
        model = VGG25D(num_classes=2)
        test_result = ct_with_dl(train_excel_path, test_excel_path, cuda_id, model, current_dir)
    elif method == "vit":
        cite = ("\n\n##############################\n"
                "Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., "
                "Dehghani, M., Minderer, M., Heigold, G., Gelly, S. and Uszkoreit, J., 2020. "
                "An image is worth 16x16 words: Transformers for image recognition at scale. "
                "arXiv preprint arXiv:2010.11929."
                "\n##############################\n")
        logging.info(cite)
        model = ViTClassifier(img_size=(50, 256, 256), num_classes=2)
        test_result = ct_with_dl(train_excel_path, test_excel_path, cuda_id, model, current_dir)
    elif method == "3d_cnn":
        cite = ("\n\n##############################\n"
                "Chen, X., Chen, Y., Ma, C., Liu, X. and Tang, X., 2018, October. "
                "Classification of pancreatic tumors based on MRI images using 3D convolutional neural networks. "
                "In Proceedings of the 2nd International Symposium on Image Computing and Digital Medicine (pp. 92-96)."
                "\n##############################\n")
        logging.info(cite)
        model = generate_model(18, n_input_channels=1, n_classes=2)
        test_result = ct_with_dl(train_excel_path, test_excel_path, cuda_id, model, current_dir)
    elif method == "hybrid_transformer":
        cite = ("\n\n##############################\n"
                "Zhang, T., Feng, Y., Zhao, Y., Fan, G., Yang, A., Lyu, S., Zhang, P., Song, F., Ma, C., Sun, "
                "Y. and Feng, Y., 2023. MSHT: Multi-stage hybrid transformer for the ROSE image analysis of "
                "pancreatic cancer. IEEE Journal of Biomedical and Health Informatics, 27(4), pp.1946-1957."
                "\n##############################\n")
        logging.info(cite)
        dimension = 2
        patch_size = 384
        model = get_model(
            num_classes=2,
            edge_size=patch_size,
            model_idx=f'Hybrid2_{patch_size}_401_test',
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            pretrained_backbone=False,  # 是否加载预训练CNN, 因channel数量改变, 故不可使用原加载预训练权重.
            use_cls_token=True,
            use_pos_embedding=True,
            use_att_module='SimAM'  # 使用 SimAM 注意力模块
        )
        test_result = ct_with_dl(train_excel_path, test_excel_path,
                                 cuda_id, model, current_dir,
                                 dimension, patch_size)
    elif method == "cnet":
        cite = ("\n\n##############################\n"
                "Barzekar, H. and Yu, Z., 2022. C-Net: A reliable convolutional neural network for "
                "biomedical image classification. Expert Systems with Applications, 187, p.116003."
                "\n##############################\n")
        logging.info(cite)
        dimension = 2
        patch_size = 224
        model = CNet(input_size=patch_size, num_classes=2)
        test_result = ct_with_dl(train_excel_path, test_excel_path, cuda_id, model, current_dir, dimension, patch_size)
    elif method == "neural_transformer":
        cite = ("\n\n##############################\n"
                "Salanitri, F.P., Bellitto, G., Palazzo, S., Irmakci, I., Wallace, M., Bolan, C., Engels, "
                "M., Hoogenboom, S., Aldinucci, M., Bagci, U. and Giordano, D., 2022, July. "
                "Neural transformers for intraductal papillary mucosal neoplasms (IPMN) classification "
                "in MRI images. In 2022 44th annual international conference of the IEEE Engineering "
                "in Medicine & Biology Society (EMBC) (pp. 475-479). IEEE."
                "\n##############################\n")
        logging.info(cite)
        dimension = 2
        patch_size = 768
        model = ViTForIPMNClassification(patch_size=16,
                                         in_channels=50,
                                         embed_dim=patch_size,
                                         depth=12,
                                         num_heads=12,
                                         mlp_dim=3072,
                                         num_classes=2)
        test_result = ct_with_dl(train_excel_path, test_excel_path, cuda_id, model, current_dir, dimension, patch_size)
    elif method == "PANDA":
        cite = ("Cao, K., Xia, Y., Yao, J., Han, X., Lambert, L., Zhang, T., Tang, W., Jin, G., "
                "Jiang, H., Fang, X. and Nogues, I., 2023. Large-scale pancreatic cancer detection "
                "via non-contrast CT and deep learning. Nature medicine, 29(12), pp.3033-3043.")
        logging.info(cite)
        weight_path = "/home/huangdn/Causal3D-Net/src/results/2025-06-21_06-49-30/best_model.pth"
        model = MultiTask3DCNN(mask_num=2, cls_num=2)
        model = load_shared_weights(model, weight_path=weight_path)
        test_result = ct_with_dl(train_excel_path,
                                 test_excel_path,
                                 cuda_id, model,
                                 current_dir,
                                 resize_shape=(40, 160, 256))
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
                        default="PANDA",
                        choices=["radiomics", "2.5d_vgg", "vit", "3d_cnn",
                                 "hybrid_transformer", "cnet", "neural_transformer",
                                 "PANDA",
                                 "mix_style", "big_aug", "rand_conv", "adver_conv",
                                 "causality_aug", "chen", "chu", "liu", "zhu", "xia"],
                        # required=True,
                        help="baseline method")
    parser.add_argument("--cuda_id", type=int,
                        default=4,
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
