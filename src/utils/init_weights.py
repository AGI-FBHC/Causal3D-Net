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


def load_shared_weights(stage2_model, path="segnet_weights.pth"):
    segnet_weights = torch.load(path)
    model_dict = stage2_model.state_dict()

    # 过滤掉分类部分
    pretrained_dict = {k: v
                       for k, v in segnet_weights.items()
                       if k in model_dict and 'cls_decoder' not in k}

    model_dict.update(pretrained_dict)
    stage2_model.load_state_dict(model_dict)
    pass


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


