"""Sequence-only independent detector baselines for RRC experiments."""
from __future__ import annotations

import random
from pathlib import Path


def initialize_torch(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch


def cnn_model():
    from torch import nn

    class SequenceCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(4, 32, 7, padding=3), nn.GroupNorm(4, 32), nn.GELU(),
                nn.Conv1d(32, 64, 5, padding=2), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv1d(64, 64, 3, padding=1), nn.GELU(),
            )
            self.head = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

        def forward(self, tokens):
            import torch
            x = torch.nn.functional.one_hot(tokens, 4).float().transpose(1, 2)
            x = self.features(x)
            return self.head(torch.cat([x.mean(-1), x.amax(-1)], dim=1)).squeeze(-1)

    return SequenceCNN()


def token_tensor(sequences):
    import torch
    ids = {base: index for index, base in enumerate("ACGT")}
    return torch.tensor([[ids[b] for b in s] for s in sequences], dtype=torch.long)


def predict_cnn(model, sequences, device, batch_size):
    import numpy as np
    import torch
    model.eval()
    result = []
    with torch.inference_mode():
        for offset in range(0, len(sequences), batch_size):
            tokens = token_tensor(sequences[offset:offset + batch_size]).to(device)
            result.append(torch.sigmoid(model(tokens)).cpu().numpy())
    return np.concatenate(result)


def fit_cnn(train, val, *, epochs, batch_size, device, seed, out):
    import copy
    torch = initialize_torch(seed)
    model = cnn_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    tokens = token_tensor(train[0])
    targets = torch.tensor(train[1], dtype=torch.float32)
    val_tokens = token_tensor(val[0])
    val_targets = torch.tensor(val[1], dtype=torch.float32)
    best, best_state, history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tokens))
        training_loss = 0.0
        for offset in range(0, len(tokens), batch_size):
            indices = order[offset:offset + batch_size]
            x, y = tokens[indices].to(device), targets[indices].to(device)
            # Reverse-complement augmentation uses the same rule for both labels.
            mask = torch.rand(len(x), device=device) < 0.5
            x[mask] = 3 - x[mask].flip(-1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += float(loss.detach()) * len(indices)
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for offset in range(0, len(val_tokens), batch_size):
                x = val_tokens[offset:offset + batch_size].to(device)
                y = val_targets[offset:offset + batch_size].to(device)
                val_loss += float(criterion(model(x), y)) * len(x)
        val_loss /= len(val_tokens)
        history.append({"epoch": epoch + 1, "train_loss": training_loss / len(tokens), "val_loss": val_loss})
        if val_loss < best:
            best, best_state = val_loss, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    torch.save(best_state, Path(out) / "weights.pt")
    return model, history


class HyenaFeatures:
    """Frozen pooled embeddings plus logistic head, not full Hyena fine-tuning."""
    def __init__(self, path, device, length, trust_remote_code):
        from .lm_stego.rrc_models import LocalCausalDNA
        self.adapter = LocalCausalDNA(path, device=device, context_limit=length,
                                      trust_remote_code=trust_remote_code)
        self.encoder = self.adapter.model.base_model

    def transform(self, sequences):
        import numpy as np
        torch = self.adapter.torch
        features = []
        with torch.inference_mode():
            for sequence in sequences:
                ids = torch.tensor([[self.adapter.ids[b] for b in sequence]], device=self.adapter.device)
                output = self.encoder(input_ids=ids, return_dict=True)
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None:
                    raise ValueError("Backbone lacks last_hidden_state for frozen probe")
                features.append(torch.cat([hidden.mean(1), hidden.amax(1)], dim=1)[0].cpu().numpy())
        return np.asarray(features)
