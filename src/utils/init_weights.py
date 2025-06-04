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


def load_shared_weights(stage2_model, weight_path="segnet_weights.pth", verbose=False):
    """完全在CPU上安全加载共享权重，避免GPU占用"""
    cpu_model = stage2_model.to("cpu")

    segnet_weights = torch.load(weight_path, map_location="cpu")

    model_dict = cpu_model.state_dict()

    pretrained_dict = {}
    missing_keys = []
    for k, v in segnet_weights.items():
        if 'cls_decoder' in k or k not in model_dict:
            if verbose:
                print(f"跳过权重: {k} (在CPU上处理)")
            missing_keys.append(k)
            continue

        pretrained_dict[k] = v

    model_dict.update(pretrained_dict)
    cpu_model.load_state_dict(model_dict, strict=False)

    if verbose:
        print(f"成功加载 {len(pretrained_dict)}/{len(segnet_weights)} 个共享权重")
        if missing_keys:
            print(f"以下权重未加载: {', '.join(missing_keys)}")
        print(f"权重加载操作完全在CPU上完成")

    return cpu_model


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


