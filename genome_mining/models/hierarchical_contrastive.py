import torch
import torch.nn.functional as F


def supervised_contrastive_loss(z, labels, temperature=0.1):
    z = F.normalize(z, dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.t()).float().to(z.device)
    logits = torch.div(torch.matmul(z, z.t()), temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=z.device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    return -mean_log_prob_pos.mean()


def hierarchical_contrastive_loss(z, carrier_labels, codec_labels, rule_labels=None, carrier_weight=0.5, codec_weight=1.0, rule_weight=0.25):
    """Contrast at multiple semantic levels.

    Level 1: carrier vs non-carrier
    Level 2: codec family
    Level 3: codec rule / embedding type if available
    """
    loss = carrier_weight * supervised_contrastive_loss(z, carrier_labels)
    loss = loss + codec_weight * supervised_contrastive_loss(z, codec_labels)
    if rule_labels is not None and rule_weight > 0:
        loss = loss + rule_weight * supervised_contrastive_loss(z, rule_labels)
    return loss
