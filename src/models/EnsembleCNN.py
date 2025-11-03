# -*- coding: utf-8 -*-
# @Time    : 2025/11/3 11:02
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: EnsembleCNN.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn

from src.models.ResNet import generate_model


class MeanEnsembleCNN(nn.Module):
    def __init__(self, model_depth=18, n_models=5, n_input_channels=1, n_classes=2, device=5):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            generate_model(model_depth,
                           n_input_channels=n_input_channels,
                           n_classes=n_classes).to(f"cuda:{device}")
            for _ in range(n_models)
        ])

    def forward(self, x):
        # 每个模型单独前向计算
        outputs = []
        for i, model in enumerate(self.models):
            out = model(x)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=0)
        mean_output = torch.mean(outputs, dim=0)
        return mean_output


if __name__ == '__main__':
    device = 0
    input_tensor = torch.randn(4, 1, 50, 256, 256).to(device)

    ensemble_model = MeanEnsembleCNN(
        model_depth=18,
        n_models=5,
        n_input_channels=1,
        n_classes=2,
        device=device
    ).to(device)

    ensemble_model.eval()
    with torch.no_grad():
        output = ensemble_model(input_tensor)

    print("Output shape:", output.shape)  # [batch, n_classes]
    print("Output logits:", output)
