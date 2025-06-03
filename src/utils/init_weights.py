# -*- coding: utf-8 -*-
# @Time    : 2025/5/23 17:54
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: init_weights.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn


def init_weights_kaiming(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            pass
        pass
    pass


# def load_shared_weights(stage2_model, weight_path="segnet_weights.pth"):
#     segnet_weights = torch.load(weight_path)
#     model_dict = stage2_model.state_dict()
#
#     # 过滤掉分类部分
#     pretrained_dict = {k: v
#                        for k, v in segnet_weights.items()
#                        if k in model_dict and 'cls_decoder' not in k}
#
#     model_dict.update(pretrained_dict)
#     stage2_model.load_state_dict(model_dict, strict=False)
#     pass


def load_shared_weights(stage2_model, weight_path="segnet_weights.pth", verbose=False):
    """加载共享权重到新模型"""
    # 加载预训练权重
    segnet_weights = torch.load(weight_path)
    model_dict = stage2_model.state_dict()

    # 过滤共享权重
    pretrained_dict = {}
    missing_keys = []
    for k, v in segnet_weights.items():
        # 跳过分类解码器和不在新模型中的权重
        if 'cls_decoder' in k or k not in model_dict:
            if verbose:
                print(f"跳过权重: {k}")
            missing_keys.append(k)
            continue

        pretrained_dict[k] = v

    # 更新模型状态
    model_dict.update(pretrained_dict)
    stage2_model.load_state_dict(model_dict, strict=False)

    # 打印加载详情
    if verbose:
        print(f"成功加载 {len(pretrained_dict)}/{len(segnet_weights)} 个共享权重")
        if missing_keys:
            print(f"以下权重未加载: {', '.join(missing_keys)}")

    return stage2_model


def init_fc_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            pass
        pass
    pass


if __name__ == "__main__":

    pass


