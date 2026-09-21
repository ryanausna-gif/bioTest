import random
import torch


class EpisodicCodecMasker:
    """
    Randomly selects one known family label per batch as pseudo-unknown.
    Samples from that label are optimized to have lower known-family confidence.
    """

    def __init__(self, p=0.5):
        self.p = p

    def make_mask(self, family_labels):
        if random.random() > self.p:
            return torch.zeros_like(family_labels, dtype=torch.bool)
        labels = family_labels.detach().cpu().unique().tolist()
        if len(labels) <= 1:
            return torch.zeros_like(family_labels, dtype=torch.bool)
        chosen = random.choice(labels)
        return family_labels == int(chosen)
