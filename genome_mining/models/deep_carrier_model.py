import torch
import torch.nn as nn
import torch.nn.functional as F

from .deep_encoders import build_encoder
from .hierarchical_contrastive import hierarchical_contrastive_loss
from .mixture_prototype import mixture_proto_logits
from .segmentation_decoder import UNet1DDecoder


class DeepOpenCarrierModel(nn.Module):
    """V12 deep open-set model with selectable encoder.

    It keeps the PE-NAC++ heads but makes the deep representation the main object.
    """
    def __init__(self, n_classes, length=1000, encoder="hybrid", emb_dim=256, proj_dim=128, prototypes_per_class=4):
        super().__init__()
        self.encoder_name = encoder
        self.encoder = build_encoder(encoder, length=length, emb_dim=emb_dim)
        self.drop = nn.Dropout(0.25)
        self.carrier = nn.Linear(emb_dim, 1)
        self.family = nn.Linear(emb_dim, n_classes)
        self.naturalness = nn.Linear(emb_dim, 1)
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, proj_dim))
        self.decoder = UNet1DDecoder(in_dim=emb_dim, hidden=128)
        self.prototypes = nn.Parameter(torch.randn(n_classes, prototypes_per_class, proj_dim) * 0.02)

    def forward(self, x):
        feat, h = self.encoder(x)
        h = self.drop(h)
        z = F.normalize(self.proj(h), dim=1)
        proto = F.normalize(self.prototypes, dim=-1)
        mix_logits = mixture_proto_logits(z, proto, temperature=0.2)
        dense_logits, sparse_logits = self.decoder(feat)
        return {
            "carrier_logit": self.carrier(h).squeeze(-1),
            "family_logits": self.family(h),
            "naturalness_logit": self.naturalness(h).squeeze(-1),
            "projection": z,
            "embedding": h,
            "mixture_proto_logits": mix_logits,
            "localization_logits": dense_logits,
            "sparse_site_logits": sparse_logits,
        }


def deep_carrier_loss(out, y, family_labels, naturalness, loc_mask, rule_labels=None, pseudo_unknown_mask=None,
                      family_weight=0.35, naturalness_weight=0.3, prototype_weight=0.3,
                      localization_weight=0.25, sparse_weight=0.1, hier_contrast_weight=0.15, episodic_weight=0.2):
    loss = F.binary_cross_entropy_with_logits(out["carrier_logit"], y)
    loss = loss + family_weight * F.cross_entropy(out["family_logits"], family_labels)
    loss = loss + naturalness_weight * F.binary_cross_entropy_with_logits(out["naturalness_logit"], naturalness)
    loss = loss + prototype_weight * F.cross_entropy(out["mixture_proto_logits"], family_labels)
    if hier_contrast_weight > 0:
        loss = loss + hier_contrast_weight * hierarchical_contrastive_loss(out["projection"], y.long(), family_labels, rule_labels)
    if localization_weight > 0 and loc_mask is not None:
        pos_weight = torch.tensor(4.0, device=loc_mask.device)
        loss = loss + localization_weight * F.binary_cross_entropy_with_logits(out["localization_logits"], loc_mask, pos_weight=pos_weight)
        loss = loss + sparse_weight * F.binary_cross_entropy_with_logits(out["sparse_site_logits"], loc_mask, pos_weight=pos_weight)
    if episodic_weight > 0 and pseudo_unknown_mask is not None and pseudo_unknown_mask.any():
        prob = F.softmax(out["mixture_proto_logits"], dim=1)
        loss = loss + episodic_weight * prob.max(dim=1).values[pseudo_unknown_mask].mean()
    return loss
