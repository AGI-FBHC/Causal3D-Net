# -*- coding: utf-8 -*-
# @Time    : 2025/9/4 21:07
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: main.py
# @Project : Causal3D-Net
import argparse
from src.training.train_Causal3DNet import train_seg, train_Causal3DNet, cross_validate_Causal3DNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Causal3DNet Training Framework")

    subparsers = parser.add_subparsers(dest="mode", help="Choose training mode")

    seg_parser = subparsers.add_parser("seg", help="Segmentation training")
    seg_parser.add_argument("--train", type=str, required=True, help="path to training dataset (Excel file)")
    seg_parser.add_argument("--test", type=str, required=True, help="path to testing dataset (Excel file)")
    seg_parser.add_argument("--cuda", type=int, default=0, help="index of GPU to use")
    seg_parser.add_argument("--outdir", type=str, default="./results", help="output directory")

    cv_parser = subparsers.add_parser("cv", help="Cross-validation training")
    cv_parser.add_argument("--train", type=str, required=True, help="path to training dataset (Excel file)")
    cv_parser.add_argument("--indi", type=int, default=1, choices=[0, 1], help="whether to use the individual branch in causal methods")
    cv_parser.add_argument("--cent", type=int, default=1, choices=[0, 1], help="whether to use the center branch in causal methods")
    cv_parser.add_argument("--orth", type=int, default=1, choices=[0, 1], help="whether to use the orthogonal loss in causal methods")
    cv_parser.add_argument("--adapt", type=str, default="none", choices=["none", "reward_bad", "reward_good"], help="adaptive loss method")
    cv_parser.add_argument("--folds", type=int, default=10, help="number of cross-validation folds")
    cv_parser.add_argument("--cuda", type=int, default=0, help="index of GPU to use")
    cv_parser.add_argument("--outdir", type=str, default="./results", help="output directory for results and logs")
    cv_parser.add_argument("--weight", type=str, default="./best_model.pth", help="segmentation trained model weight")

    causal_parser = subparsers.add_parser("causal", help="Causal3DNet training")
    causal_parser.add_argument("--train", type=str, required=True, help="path to training dataset")
    causal_parser.add_argument("--test", type=str, required=True, help="path to testing dataset")
    causal_parser.add_argument("--indi", type=int, default=1, choices=[0, 1], help="whether to use the individual branch in causal methods")
    causal_parser.add_argument("--cent", type=int, default=1, choices=[0, 1], help="whether to use the center branch in causal methods")
    causal_parser.add_argument("--orth", type=int, default=1, choices=[0, 1], help="whether to use the orthogonal loss in causal methods")
    causal_parser.add_argument("--adapt", type=str, default="none", choices=["none", "reward_bad", "reward_good"], help="adaptive loss method")
    causal_parser.add_argument("--cuda", type=int, default=0, help="index of GPU to use")
    causal_parser.add_argument("--outdir", type=str, default="./results", help="output directory")
    causal_parser.add_argument("--weight", type=str, default="./best_model.pth", help="segmentation trained model weight")

    args = parser.parse_args()

    if args.mode == "seg":
        train_seg(
            train_excel=args.train,
            test_excel=args.test,
            cuda_id=args.cuda,
            output_dir=args.outdir
        )
    elif args.mode == "causal":
        train_Causal3DNet(
            train_excel=args.train,
            test_excel=args.test,
            use_indi=args.indi,
            use_cent=args.cent,
            orthogonal=args.orth,
            adaptive=args.adapt,
            cuda_id=args.cuda,
            output_dir=args.outdir,
            model_weight=args.weight,
        )
    elif args.mode == "cv":
        cross_validate_Causal3DNet(
            train_excel=args.train,
            use_indi=args.indi,
            use_cent=args.cent,
            orthogonal=args.orth,
            adaptive=args.adapt,
            output_dir=args.outdir,
            folds=args.folds,
            cuda_id=args.cuda,
            model_weight=args.weight,
        )
    else:
        parser.print_help()
