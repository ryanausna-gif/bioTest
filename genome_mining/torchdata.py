from __future__ import annotations


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(sequence: str, length: int):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for genome_mining model training. Install the V12/torch environment first.") from exc

    x = torch.zeros((4, int(length)), dtype=torch.float32)
    for idx, base in enumerate(str(sequence).upper()[: int(length)]):
        channel = BASE_TO_IDX.get(base)
        if channel is not None:
            x[channel, idx] = 1.0
    return x


class CarrierLocalizationDataset:
    def __init__(self, df, length: int, label_map: dict[str, int]):
        try:
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise ImportError("torch is required for CarrierLocalizationDataset.") from exc

        class _Dataset(Dataset):
            def __init__(self, frame, seq_length, labels):
                self.df = frame.reset_index(drop=True)
                self.length = int(seq_length)
                self.label_map = labels

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                import torch

                from .localization import region_to_mask

                row = self.df.iloc[idx]
                x = one_hot(row.sequence, self.length)
                y = torch.tensor(float(row.y), dtype=torch.float32)
                fam = torch.tensor(self.label_map.get(str(row.family), 0), dtype=torch.long)
                nat = torch.tensor(float(getattr(row, "naturalness_label", 1 - row.y)), dtype=torch.float32)
                start = int(getattr(row, "payload_start", getattr(row, "start", -1)))
                end = int(getattr(row, "payload_end", getattr(row, "end", -1)))
                loc = torch.tensor(region_to_mask(start, end, self.length), dtype=torch.float32)
                return x, y, fam, nat, loc

        self._dataset = _Dataset(df, length, label_map)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


class V12CarrierDataset:
    def __init__(self, df, length: int, family_map: dict[str, int], rule_map: dict[str, int] | None = None):
        try:
            from torch.utils.data import Dataset
        except ImportError as exc:
            raise ImportError("torch is required for V12CarrierDataset.") from exc

        class _Dataset(Dataset):
            def __init__(self, frame, seq_length, families, rules):
                self.df = frame.reset_index(drop=True)
                self.length = int(seq_length)
                self.family_map = families
                self.rule_map = rules or {"unknown": 0}

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                import torch

                from .localization import region_to_mask

                row = self.df.iloc[idx]
                x = one_hot(str(row.sequence), self.length)
                y = torch.tensor(float(row.y), dtype=torch.float32)
                fam = torch.tensor(self.family_map.get(str(row.family), 0), dtype=torch.long)
                rule_val = str(getattr(row, "codec_rule", getattr(row, "rule_id", "unknown")))
                rule = torch.tensor(self.rule_map.get(rule_val, 0), dtype=torch.long)
                nat = torch.tensor(float(getattr(row, "naturalness_label", 1 - row.y)), dtype=torch.float32)
                start = int(getattr(row, "payload_start", getattr(row, "start", -1)))
                end = int(getattr(row, "payload_end", getattr(row, "end", -1)))
                loc = torch.tensor(region_to_mask(start, end, self.length), dtype=torch.float32)
                return x, y, fam, rule, nat, loc

        self._dataset = _Dataset(df, length, family_map, rule_map)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]
