from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .image_locator import LocatorTaskHeads


DEFAULT_HYENADNA_MODEL = "LongSafari/hyenadna-medium-160k-seqlen-hf"


def reverse_complement_tokens(tokens):
    """Reverse-complement A=0,C=1,G=2,T=3,N=4 token tensors."""
    lookup = torch.tensor([3, 2, 1, 0, 4], device=tokens.device, dtype=torch.long)
    safe = torch.clamp(tokens.to(torch.long), min=0, max=4)
    return lookup[safe].flip(dims=(-1,))


def to_hyenadna_token_ids(tokens):
    """Map the project vocabulary to the official tokenizer IDs A/C/G/T/N=7..11."""
    safe = torch.clamp(tokens.to(torch.long), min=0, max=4)
    return safe + 7


class HyenaDNALocator(nn.Module):
    """HyenaDNA backbone with bidirectional RC fusion and exact boundary heads."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_HYENADNA_MODEL,
        model_revision: str | None = None,
        local_files_only: bool = False,
        downsample_stride: int = 256,
        head_dim: int = 128,
        dropout: float = 0.1,
        reverse_complement: bool = True,
        fine_tune_mode: str = "frozen",
        last_n_layers: int = 2,
        pretrained: bool = True,
        gradient_checkpointing: bool = False,
        backbone_module=None,
    ) -> None:
        super().__init__()
        if downsample_stride <= 0:
            raise ValueError("downsample_stride must be positive.")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive.")
        if fine_tune_mode not in {"frozen", "last_n", "full"}:
            raise ValueError("fine_tune_mode must be frozen, last_n, or full.")
        self.model_name = model_name
        self.model_revision = model_revision
        self.downsample_stride = int(downsample_stride)
        self.reverse_complement = bool(reverse_complement)
        self.fine_tune_mode = fine_tune_mode
        self.last_n_layers = int(last_n_layers)
        self.pretrained = bool(pretrained)

        if backbone_module is None:
            try:
                from transformers import AutoConfig, AutoModel
            except ImportError as exc:
                raise ImportError(
                    "HyenaDNA requires transformers. Install requirements-hyenadna-stego.txt."
                ) from exc
            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": bool(local_files_only),
            }
            if model_revision:
                load_kwargs["revision"] = model_revision
            if pretrained:
                self.backbone = AutoModel.from_pretrained(model_name, **load_kwargs)
            else:
                config = AutoConfig.from_pretrained(model_name, **load_kwargs)
                self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        else:
            self.backbone = backbone_module

        config = getattr(self.backbone, "config", None)
        backbone_dim = int(getattr(config, "d_model", getattr(self.backbone, "d_model", head_dim)))
        self.max_seq_len = int(getattr(config, "max_seq_len", 0) or 0)
        fused_dim = backbone_dim * (2 if self.reverse_complement else 1)
        self.fused_projection = nn.Sequential(
            nn.Conv1d(fused_dim, head_dim, 1),
            nn.GroupNorm(8 if head_dim % 8 == 0 else 1, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.single_projection = nn.Sequential(
            nn.Conv1d(backbone_dim, head_dim, 1),
            nn.GroupNorm(8 if head_dim % 8 == 0 else 1, head_dim),
            nn.GELU(),
        )
        self.task_heads = LocatorTaskHeads(head_dim, self.downsample_stride, dropout)
        self._configure_fine_tuning(fine_tune_mode, self.last_n_layers)
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def _layers(self):
        candidate = getattr(self.backbone, "backbone", None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
        hyena = getattr(self.backbone, "hyena", None)
        candidate = getattr(hyena, "backbone", None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
        return None

    def _configure_fine_tuning(self, mode: str, last_n_layers: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = mode == "full"
        if mode != "last_n":
            return
        layers = self._layers()
        if layers is None:
            raise ValueError("Could not find HyenaDNA backbone layers for last_n fine-tuning.")
        if last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive in last_n mode.")
        for layer in list(layers)[-last_n_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        backbone = getattr(self.backbone, "backbone", None)
        final_norm = getattr(backbone, "ln_f", None)
        if final_norm is not None:
            for parameter in final_norm.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.fine_tune_mode == "frozen":
            self.backbone.eval()
        return self

    def _backbone_features(self, tokens):
        if self.max_seq_len and tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Input length {tokens.shape[1]} exceeds HyenaDNA max_seq_len={self.max_seq_len}."
            )
        input_ids = to_hyenadna_token_ids(tokens)
        context = torch.no_grad() if self.fine_tune_mode == "frozen" else nullcontext()
        with context:
            output = self.backbone(input_ids=input_ids, return_dict=True)
            hidden = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
        features = hidden.transpose(1, 2)
        return F.avg_pool1d(
            features,
            kernel_size=self.downsample_stride,
            stride=self.downsample_stride,
            ceil_mode=True,
        )

    def forward(self, tokens):
        forward_features = self._backbone_features(tokens)
        reverse_features = None
        if self.reverse_complement:
            reverse_features = self._backbone_features(reverse_complement_tokens(tokens)).flip(dims=(-1,))
            contextual = self.fused_projection(torch.cat([forward_features, reverse_features], dim=1))
        else:
            contextual = self.fused_projection(forward_features)
        output = self.task_heads(contextual)
        if self.training and reverse_features is not None:
            forward_output = self.task_heads(self.single_projection(forward_features))
            reverse_output = self.task_heads(self.single_projection(reverse_features))
            output["rc_consistency_loss"] = F.mse_loss(
                torch.sigmoid(forward_output["presence_logit"]),
                torch.sigmoid(reverse_output["presence_logit"]),
            ) + F.mse_loss(
                torch.sigmoid(forward_output["segment_logits"]),
                torch.sigmoid(reverse_output["segment_logits"]),
            )
        return output


def trainable_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
    }
