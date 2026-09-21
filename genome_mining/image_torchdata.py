from __future__ import annotations

import csv
from pathlib import Path


class ImageGenomeDataset:
    """Memory-mapped long DNA windows with carrier and exact-start labels."""

    def __init__(self, dataset_dir: str | Path, split: str) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("ImageGenomeDataset requires numpy.") from exc

        root = Path(dataset_dir)
        with (root / "metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle) if row["split"] == split]
        if not rows:
            raise ValueError(f"Dataset has no samples in split={split!r}.")
        self.root = root
        self.split = split
        self.rows = rows
        self.sequences = np.load(root / "sequences.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import numpy as np
        import torch

        row = self.rows[index]
        array_index = int(row["array_index"])
        # Copy a single memory-mapped row so PyTorch receives writable storage.
        tokens = torch.from_numpy(np.array(self.sequences[array_index], dtype=np.int64, copy=True))
        label = torch.tensor(float(row["y"]), dtype=torch.float32)
        start = torch.tensor(max(0, int(row["payload_start"])), dtype=torch.long)
        end = torch.tensor(max(0, int(row["payload_end"])), dtype=torch.long)
        return tokens, label, start, end, torch.tensor(array_index, dtype=torch.long)
