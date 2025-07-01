# -*- coding: utf-8 -*-
# @Time    : 2025/7/1 09:47
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: Neural_Transformer.py
# @Project : Causal3D-Net
import torch
import torch.nn as nn


class ViTForIPMNClassification(nn.Module):
    """
    Vision Transformer for IPMN Classification based on the paper
    Input: 2D grid image of shape (2, 768, 768) [2 channels, height, width]

    Args:
        patch_size (int): Size of image patches (default=16)
        in_channels (int): Number of input channels (2 for T1+T2)
        embed_dim (int): Transformer embedding dimension (default=768)
        depth (int): Number of transformer blocks (default=12)
        num_heads (int): Number of attention heads (default=12)
        mlp_dim (int): MLP hidden dimension (default=3072)
        num_classes (int): Number of output classes (3 for IPMN classification)
    """

    def __init__(self, patch_size=16, in_channels=2, embed_dim=768,
                 depth=12, num_heads=12, mlp_dim=3072, num_classes=3):
        super().__init__()
        self.patch_size = patch_size

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Position embeddings + class token
        num_patches = (768 // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(embed_dim),
                "attn": nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                "norm2": nn.LayerNorm(embed_dim),
                "mlp": nn.Sequential(
                    nn.Linear(embed_dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, embed_dim)
                )
            }) for _ in range(depth)
        ])

        # Layer normalization and classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, 2, 768, 768)
        Returns: class logits (batch_size, num_classes)
        """
        # Create patch embeddings
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add class token and position embeddings
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # Process through transformer blocks
        for block in self.blocks:
            # Multi-head self-attention
            attn_output, _ = block["attn"](
                block["norm1"](x),
                block["norm1"](x),
                block["norm1"](x)
            )
            x = x + attn_output

            # MLP block
            mlp_output = block["mlp"](block["norm2"](x))
            x = x + mlp_output

        # Extract class token
        cls_token_final = self.norm(x[:, 0, :])

        # Classification head
        return self.classifier(cls_token_final)


if __name__ == "__main__":
    _size = 768

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # 1. 创建模型实例 (论文参数)
    model = ViTForIPMNClassification(
        patch_size=16,
        in_channels=50,
        embed_dim=_size,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        num_classes=2
    ).to(device)

    # 2. 创建模拟输入数据 (4张假图像)，并放到 cuda:4 上
    dummy_images = torch.randn(4, 50, 768, 768).to(device)
    print("Dummy input shape:", dummy_images.shape)

    # 3. 执行推理
    with torch.no_grad():
        outputs = model(dummy_images)

    # 4. 打印输出结果
    print("\nOutput logits shape:", outputs.shape)


