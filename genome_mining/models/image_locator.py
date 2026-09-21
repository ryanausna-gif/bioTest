from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class LocatorTaskHeads(nn.Module):
    """Presence, exact start/end, and coarse segmentation heads shared by all backbones."""

    def __init__(self, d_model: int, stride: int, dropout: float) -> None:
        super().__init__()
        self.presence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.start_bin_head = nn.Conv1d(d_model, 1, 1)
        self.start_offset_head = nn.Conv1d(d_model, stride, 1)
        self.end_bin_head = nn.Conv1d(d_model, 1, 1)
        self.end_offset_head = nn.Conv1d(d_model, stride, 1)
        self.segment_head = nn.Conv1d(d_model, 1, 1)

    def forward(self, contextual):
        pooled = contextual.mean(dim=-1)
        return {
            "presence_logit": self.presence_head(pooled).squeeze(-1),
            "start_bin_logits": self.start_bin_head(contextual).squeeze(1),
            "offset_logits": self.start_offset_head(contextual).transpose(1, 2),
            "end_bin_logits": self.end_bin_head(contextual).squeeze(1),
            "end_offset_logits": self.end_offset_head(contextual).transpose(1, 2),
            "segment_logits": self.segment_head(contextual).squeeze(1),
            "embedding": pooled,
        }


class HierarchicalImageLocator(nn.Module):
    """Detect and localize a fixed raw-image payload in a long DNA window.

    Four stride-4 stages turn a 131072-base window into 512 bins. The model
    predicts a coarse start bin and an exact 0..255 offset inside that bin.
    """

    def __init__(
        self,
        *,
        d_model: int = 128,
        downsample_stride: int = 256,
        context: str = "dilated_cnn",
        context_layers: int = 6,
        dropout: float = 0.1,
        max_bins: int = 4096,
    ) -> None:
        super().__init__()
        stages = round(math.log(downsample_stride, 4))
        if downsample_stride < 4 or 4**stages != downsample_stride:
            raise ValueError("downsample_stride must be a power of four and at least four.")
        if d_model % 8:
            raise ValueError("d_model must be divisible by eight.")
        widths = []
        for index in range(stages):
            width = d_model if index == stages - 1 else max(32, d_model * (index + 1) // stages)
            widths.append(max(8, (width // 8) * 8))
        layers = []
        in_channels = 4
        for out_channels in widths:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=4, padding=4),
                    nn.GroupNorm(8, out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.downsample_stride = downsample_stride
        self.context_name = context
        self.max_bins = max_bins

        if context == "dilated_cnn":
            self.context = nn.Sequential(
                *[
                    DilatedResidualBlock(d_model, 2 ** (index % 5), dropout)
                    for index in range(context_layers)
                ]
            )
            self.position_embedding = None
        elif context == "transformer":
            self.position_embedding = nn.Parameter(torch.zeros(1, max_bins, d_model))
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context = nn.TransformerEncoder(layer, num_layers=context_layers)
        else:
            raise ValueError("context must be 'dilated_cnn' or 'transformer'.")

        self.task_heads = LocatorTaskHeads(d_model, downsample_stride, dropout)

    def forward(self, tokens):
        # N/unknown bases become an all-zero channel vector rather than a fifth
        # learnable base, preventing ambiguous assembly gaps from looking encoded.
        one_hot = torch.stack([(tokens == base) for base in range(4)], dim=1).to(torch.float32)
        encoded = self.encoder(one_hot)
        if self.context_name == "transformer":
            sequence = encoded.transpose(1, 2)
            if sequence.shape[1] > self.max_bins:
                raise ValueError(f"Encoded length {sequence.shape[1]} exceeds max_bins={self.max_bins}.")
            sequence = sequence + self.position_embedding[:, : sequence.shape[1]]
            contextual = self.context(sequence).transpose(1, 2)
        else:
            contextual = self.context(encoded)
        return self.task_heads(contextual)


def image_locator_loss(
    output,
    labels,
    starts,
    ends=None,
    *,
    payload_bases: int | None = None,
    stride: int,
    presence_weight: float = 1.0,
    start_weight: float = 1.0,
    offset_weight: float = 1.0,
    segment_weight: float = 0.5,
    rc_consistency_weight: float = 0.0,
):
    presence = F.binary_cross_entropy_with_logits(output["presence_logit"], labels)
    bins = output["start_bin_logits"].shape[1]
    positive = labels > 0.5
    zero = output["presence_logit"].sum() * 0.0
    start_loss = offset_loss = end_loss = end_offset_loss = zero
    if ends is None:
        if payload_bases is None:
            raise ValueError("Provide ends or payload_bases to image_locator_loss.")
        ends = starts + int(payload_bases)
    if positive.any():
        positive_starts = starts[positive]
        target_bins = torch.clamp(positive_starts // stride, max=bins - 1)
        target_offsets = positive_starts % stride
        start_loss = F.cross_entropy(output["start_bin_logits"][positive], target_bins)
        selected_offsets = output["offset_logits"][positive]
        selected_offsets = selected_offsets[torch.arange(len(target_bins), device=labels.device), target_bins]
        offset_loss = F.cross_entropy(selected_offsets, target_offsets)
        end_indices = torch.clamp(ends[positive] - 1, min=0)
        target_end_bins = torch.clamp(end_indices // stride, max=bins - 1)
        target_end_offsets = end_indices % stride
        end_loss = F.cross_entropy(output["end_bin_logits"][positive], target_end_bins)
        selected_end_offsets = output["end_offset_logits"][positive]
        selected_end_offsets = selected_end_offsets[
            torch.arange(len(target_end_bins), device=labels.device), target_end_bins
        ]
        end_offset_loss = F.cross_entropy(selected_end_offsets, target_end_offsets)

    bin_starts = torch.arange(bins, device=labels.device) * stride
    bin_ends = bin_starts + stride
    true_starts = starts[:, None]
    true_ends = ends[:, None]
    segment_target = (
        positive[:, None]
        & (bin_ends[None, :] > true_starts)
        & (bin_starts[None, :] < true_ends)
    ).to(torch.float32)
    segment_loss = F.binary_cross_entropy_with_logits(output["segment_logits"], segment_target)
    rc_consistency = output.get("rc_consistency_loss", zero)
    total = (
        presence_weight * presence
        + start_weight * start_loss
        + offset_weight * offset_loss
        + start_weight * end_loss
        + offset_weight * end_offset_loss
        + segment_weight * segment_loss
        + rc_consistency_weight * rc_consistency
    )
    return {
        "loss": total,
        "presence_loss": presence,
        "start_loss": start_loss,
        "offset_loss": offset_loss,
        "end_loss": end_loss,
        "end_offset_loss": end_offset_loss,
        "segment_loss": segment_loss,
        "rc_consistency_loss": rc_consistency,
    }


def decode_locator_output(
    output,
    *,
    stride: int,
    window_length: int,
    payload_bases: int | None = None,
):
    presence_prob = torch.sigmoid(output["presence_logit"])
    start_bin = output["start_bin_logits"].argmax(dim=1)
    selected_offsets = output["offset_logits"][torch.arange(len(start_bin), device=start_bin.device), start_bin]
    offset = selected_offsets.argmax(dim=1)
    start = torch.clamp(start_bin * stride + offset, max=max(0, window_length - 1))
    if "end_bin_logits" in output and "end_offset_logits" in output:
        end_bin = output["end_bin_logits"].argmax(dim=1)
        selected_end_offsets = output["end_offset_logits"][
            torch.arange(len(end_bin), device=end_bin.device), end_bin
        ]
        end_offset = selected_end_offsets.argmax(dim=1)
        end = torch.clamp(end_bin * stride + end_offset + 1, min=1, max=window_length)
        end = torch.maximum(end, start + 1)
    elif payload_bases is not None:
        end_bin = torch.zeros_like(start_bin)
        end_offset = torch.zeros_like(offset)
        end = torch.clamp(start + int(payload_bases), max=window_length)
    else:
        raise ValueError("This checkpoint has no end boundary head and no payload_bases fallback.")
    return {
        "presence_prob": presence_prob,
        "start_bin": start_bin,
        "offset": offset,
        "start": start,
        "end_bin": end_bin,
        "end_offset": end_offset,
        "end": end,
        "segment_prob": torch.sigmoid(output["segment_logits"]),
    }
