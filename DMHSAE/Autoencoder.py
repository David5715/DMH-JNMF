import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer  # <-- 改进 A: 导入 TransformerEncoderLayer


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)


# --- 改进 A: 已删除 MultiHeadSelfAttention 类 ---


# --- 修改过的 Autoencoder_Reconstructor ---
# 它现在包含 4 个独立的 (Encoder -> TransformerLayer -> Decoder) 路径
# 并且包含了可学习的损失权重
class Autoencoder_Reconstructor(nn.Module):
    def __init__(self, input_dims, hidden_dims=[512, 128, 32]):
        super().__init__()

        # --- 改进 C: 添加可学习的损失权重 ---
        self.log_vars = nn.Parameter(torch.zeros(4))

        # 1. 编码器 (4个)
        self.encoders = nn.ModuleList([
            self._build_encoder(dim, hidden_dims) for dim in input_dims
        ])

        # --- 改进 A: 使用 TransformerEncoderLayer 替换 ---
        # 2. 独立的 Transformer Encoder Layer (4个)
        self.attentions = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dims[-1],  # 潜（latent）维度
                nhead=4,  # 头的数量
                dim_feedforward=hidden_dims[-1] * 4,  # FF层维度
                dropout=0.2,
                activation='relu',
                batch_first=True  # 确保 (B, 1, D)
            ) for _ in range(len(input_dims))
        ])

        # 3. 解码器 (4个)
        self.decoders = nn.ModuleList([
            self._build_decoder(hidden_dims[-1], dim, hidden_dims[::-1])
            for dim in input_dims
        ])

    def _build_encoder(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                ResidualBlock(dim),
                nn.Tanh(),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        return nn.Sequential(*layers)

    def _build_decoder(self, latent_dim, output_dim, hidden_dims):
        layers = []
        prev_dim = latent_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                ResidualBlock(dim),
                nn.Tanh()
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))

        # --- 改进 B: 添加 Sigmoid 激活函数 ---
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, *modalities):
        # 1. 分别编码
        latents = [enc(mod) for enc, mod in zip(self.encoders, modalities)]

        # --- 改进 A: 适配 TransformerEncoderLayer 的输入输出 ---
        # (B, D) -> (B, 1, D)
        latents_unsqueezed = [lat.unsqueeze(1) for lat in latents]

        # 2. 分别应用模态内部的自注意力 (Transformer Layer)
        attended_latents_unsqueezed = [
            att(lat_u) for att, lat_u in zip(self.attentions, latents_unsqueezed)
        ]

        # (B, 1, D) -> (B, D)
        attended_latents = [att_lat_u.squeeze(1) for att_lat_u in attended_latents_unsqueezed]

        # 3. 从各自的潜（latent）空间分别解码
        recons = [
            dec(att_lat) for dec, att_lat in zip(self.decoders, attended_latents)
        ]

        return tuple(recons)