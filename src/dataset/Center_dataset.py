# -*- coding: utf-8 -*-
# @Time    : 2025/5/11 21:58
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: Center_dataset.py
# @Project : Causal3D-Net
import os
import pandas as pd
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import torchio as tio
from src.utils.visual3D import show_volume_plotly, show_middle_slice


class CenterDataset(Dataset):

    def __init__(self,):
        pass
