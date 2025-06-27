# -*- coding: utf-8 -*-
# @Time    : 2025/6/27 14:45
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: train_baseline.py
# @Project : Causal3D-Net
import os

import numpy as np
import pandas as pd

import torch


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for X, cls_label in train_loader:
        X = X.to(device)  # shape: (B, 1, D, H, W)
        cls_label = cls_label.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, cls_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        targets = cls_label.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets)

    avg_loss = total_loss / len(train_loader)

    return avg_loss


def test_one_epoch(model, test_loader, device):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for X, cls_label in test_loader:
            X, cls_label = X.to(device), cls_label.to(device)

            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    return all_preds, all_probs
