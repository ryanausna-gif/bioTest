import numpy as np
import torch
import torch.nn.functional as F


def mixture_pairwise_sq_dist(x, prototypes):
    x = x[:, None, None, :]
    p = prototypes[None, :, :, :]
    return ((x - p) ** 2).sum(dim=-1)


def mixture_proto_logits(x, prototypes, temperature=0.2):
    dist = mixture_pairwise_sq_dist(x, prototypes)
    return torch.logsumexp(-dist / temperature, dim=-1)


def mixture_unknown_score_np(emb, prototypes):
    emb = np.asarray(emb, dtype=float)
    proto = np.asarray(prototypes, dtype=float)
    d = ((emb[:, None, None, :] - proto[None, :, :, :]) ** 2).sum(axis=-1)
    min_d = d.min(axis=(1, 2))
    med = np.median(min_d)
    mad = np.median(np.abs(min_d - med)) + 1e-8
    dist_score = 1.0 / (1.0 + np.exp(-((min_d - med) / (1.4826 * mad + 1e-8))))
    logits = -d.reshape(len(emb), -1)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    ent = -(p * np.log(p + 1e-8)).sum(axis=1) / np.log(p.shape[1])
    return 0.65 * dist_score + 0.35 * ent
