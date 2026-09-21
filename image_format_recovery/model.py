from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
    raise ImportError(
        "PyTorch is required for image_format_recovery.model. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : tokens.shape[1], :]


class ByteCNNTransformer(nn.Module):
    """Byte-level multi-task model for image format and payload offset prediction."""

    def __init__(
        self,
        num_formats: int,
        max_prefix: int,
        seq_len: int = 4096,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_formats = num_formats
        self.max_prefix = max_prefix
        self.seq_len = seq_len
        self.pad_token = 256

        self.embedding = nn.Embedding(257, dim, padding_idx=self.pad_token)
        self.position = SinusoidalPositionEncoding(dim, seq_len)
        self.local = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.format_head = nn.Linear(dim, num_formats)
        self.start_head = nn.Linear(dim, max_prefix + 1)

    def forward(self, byte_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        padding_mask = byte_tokens.eq(self.pad_token)
        x = self.embedding(byte_tokens) + self.position(byte_tokens)
        x = self.local(x.transpose(1, 2)).transpose(1, 2)
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        valid = (~padding_mask).unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        pooled = self.norm(pooled)
        return {
            "format_logits": self.format_head(pooled),
            "start_logits": self.start_head(pooled),
        }
