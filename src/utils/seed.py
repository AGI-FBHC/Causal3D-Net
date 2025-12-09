# -*- coding: utf-8 -*-
# @Time    : 2025/12/9 09:27
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: seed.py
# @Project : Causal3D-Net
import torch, random, numpy as np
import os


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    fix_seed(42)
    pass