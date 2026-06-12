import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

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
class Autoencoder_Reconstructor(nn.Module):
    def __init__(self, input_dims, hidden_dims=[512, 128, 32]):
        super().__init__()


        self.log_vars = nn.Parameter(torch.zeros(4))


        self.encoders = nn.ModuleList([
            self._build_encoder(dim, hidden_dims) for dim in input_dims
        ])


        self.attentions = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dims[-1],
                nhead=4,
                dim_feedforward=hidden_dims[-1] * 4,
                dropout=0.2,
                activation='relu',
                batch_first=True
            ) for _ in range(len(input_dims))
        ])


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


        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, *modalities):

        latents = [enc(mod) for enc, mod in zip(self.encoders, modalities)]
        latents_unsqueezed = [lat.unsqueeze(1) for lat in latents]

        attended_latents_unsqueezed = [
            att(lat_u) for att, lat_u in zip(self.attentions, latents_unsqueezed)
        ]

        attended_latents = [att_lat_u.squeeze(1) for att_lat_u in attended_latents_unsqueezed]

        recons = [
            dec(att_lat) for dec, att_lat in zip(self.decoders, attended_latents)
        ]

        return tuple(recons)