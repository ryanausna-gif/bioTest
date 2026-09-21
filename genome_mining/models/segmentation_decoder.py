import torch.nn as nn


class UNet1DDecoder(nn.Module):
    """Lightweight 1D decoder for dense region and sparse-site localization."""
    def __init__(self, in_dim=256, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv1d(in_dim, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.dense = nn.Conv1d(hidden, 1, 1)
        self.sparse = nn.Conv1d(hidden, 1, 1)

    def forward(self, feat):
        h = self.trunk(feat)
        return self.dense(h).squeeze(1), self.sparse(h).squeeze(1)
