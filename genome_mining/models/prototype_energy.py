import torch
import torch.nn.functional as F


def pairwise_sq_dist(x, prototypes):
    """
    x: [B, D]
    prototypes: [C, D]
    returns [B, C]
    """
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    p2 = (prototypes ** 2).sum(dim=1).unsqueeze(0)
    xp = x @ prototypes.t()
    return x2 + p2 - 2 * xp


def prototype_logits(x, prototypes, temperature=1.0):
    return -pairwise_sq_dist(x, prototypes) / temperature


def prototype_ce_loss(x, labels, prototypes, temperature=1.0):
    logits = prototype_logits(x, prototypes, temperature=temperature)
    return F.cross_entropy(logits, labels)


def prototype_unknown_score_np(emb, prototypes, temperature=1.0):
    import numpy as np
    emb = np.asarray(emb, dtype=float)
    prototypes = np.asarray(prototypes, dtype=float)
    d = ((emb[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    min_d = d.min(axis=1)
    # map distance to 0-1 score using robust normalization
    med = np.median(min_d)
    mad = np.median(np.abs(min_d - med)) + 1e-8
    z = (min_d - med) / (1.4826 * mad + 1e-8)
    return 1 / (1 + np.exp(-z))


def energy_unknown_score_np(logits):
    import numpy as np
    x = np.asarray(logits, dtype=float)
    m = x.max(axis=1, keepdims=True)
    energy_known = (m.squeeze(1) + np.log(np.exp(x - m).sum(axis=1)))
    # unknown = low known energy
    u = -energy_known
    u = (u - u.mean()) / (u.std() + 1e-8)
    return 1 / (1 + np.exp(-u))
