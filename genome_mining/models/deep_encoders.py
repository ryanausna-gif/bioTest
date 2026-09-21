"""Deep sequence encoders for V12.

The goal is to make the project more explicitly deep-learning oriented:
- CNN: local carrier traces
- DilatedCNN: multi-scale local/medium-range traces
- Transformer: global protocol/checksum/distributed watermark dependencies
- HybridCNNTransformer: local CNN front-end + global transformer encoder
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: [B, L, D]
        return x + self.pe[:, : x.size(1), :]


class CNNEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.out_dim = emb_dim
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 11, padding=5), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, emb_dim, 13, padding=6), nn.BatchNorm1d(emb_dim), nn.ReLU(),
        )

    def forward(self, x):
        feat = self.net(x)
        h = F.adaptive_max_pool1d(feat, 1).squeeze(-1)
        return feat, h


class DilatedCNNEncoder(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        self.out_dim = emb_dim
        layers = []
        in_ch = 4
        channels = [64, 96, 128, emb_dim]
        dilations = [1, 2, 4, 8]
        for out_ch, dil in zip(channels, dilations):
            pad = dil * 4
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=9, padding=pad, dilation=dil),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.net(x)
        h = F.adaptive_max_pool1d(feat, 1).squeeze(-1)
        return feat, h


class TransformerEncoder(nn.Module):
    def __init__(self, length=1000, emb_dim=256, nhead=8, layers=3, patch_size=8, dropout=0.1):
        super().__init__()
        self.out_dim = emb_dim
        self.patch_size = int(patch_size)
        self.patch = nn.Conv1d(4, emb_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos = SinusoidalPositionalEncoding(emb_dim, max_len=max(16, length // self.patch_size + 8))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # x [B,4,L] -> tokens [B,T,D]
        tok = self.patch(x).transpose(1, 2)
        tok = self.pos(tok)
        tok = self.encoder(tok)
        tok = self.norm(tok)
        h = tok.mean(dim=1)
        # Upsample token features to approximate per-base feature map for localization heads.
        feat = tok.transpose(1, 2)
        feat = F.interpolate(feat, size=x.size(-1), mode="linear", align_corners=False)
        return feat, h


class HybridCNNTransformerEncoder(nn.Module):
    def __init__(self, length=1000, emb_dim=256, nhead=8, layers=2, pool_stride=4, dropout=0.1):
        super().__init__()
        self.out_dim = emb_dim
        self.local = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, emb_dim, 9, padding=4), nn.BatchNorm1d(emb_dim), nn.ReLU(),
        )
        self.pool_stride = int(pool_stride)
        self.pos = SinusoidalPositionalEncoding(emb_dim, max_len=max(16, length // self.pool_stride + 8))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        local_feat = self.local(x)  # [B,D,L]
        pooled = F.avg_pool1d(local_feat, kernel_size=self.pool_stride, stride=self.pool_stride, ceil_mode=True)
        tok = pooled.transpose(1, 2)
        tok = self.pos(tok)
        tok = self.encoder(tok)
        tok = self.norm(tok)
        h = tok.mean(dim=1)
        global_feat = tok.transpose(1, 2)
        global_feat = F.interpolate(global_feat, size=x.size(-1), mode="linear", align_corners=False)
        feat = local_feat + global_feat
        return feat, h


def build_encoder(name: str, length=1000, emb_dim=256):
    name = name.lower()
    if name == "cnn":
        return CNNEncoder(emb_dim=emb_dim)
    if name in ["dilated", "dilated_cnn", "tcn"]:
        return DilatedCNNEncoder(emb_dim=emb_dim)
    if name in ["transformer", "vit"]:
        return TransformerEncoder(length=length, emb_dim=emb_dim)
    if name in ["hybrid", "cnn_transformer", "hybrid_cnn_transformer"]:
        return HybridCNNTransformerEncoder(length=length, emb_dim=emb_dim)
    raise ValueError(f"Unknown encoder: {name}")
