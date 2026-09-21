from __future__ import annotations

import hashlib
from pathlib import Path


def _require_analysis_deps():
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Benchmark analysis requires numpy and pandas.") from exc
    return np, pd


def _counts(frame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts(dropna=False).items()}


def analyze_benchmark(csv_path: str | Path, out_path: str | Path | None = None) -> dict[str, object]:
    np, pd = _require_analysis_deps()
    from .features import compute_sequence_features
    from .model_metrics import save_json

    frame = pd.read_csv(csv_path)
    report: dict[str, object] = {
        "n_rows": int(len(frame)),
        "columns": frame.columns.astype(str).tolist(),
        "missing_values": {str(key): int(value) for key, value in frame.isna().sum().items() if value},
    }
    for column in [
        "split",
        "y",
        "open_target",
        "carrier_type",
        "codec_family",
        "codec_rule",
        "message_type",
        "embedding_type",
        "detectability",
        "background_id",
    ]:
        counts = _counts(frame, column)
        if counts:
            report[column] = counts

    warnings: list[str] = []
    required = {"split", "sequence", "y", "family", "is_unknown", "open_target"}
    missing_required = sorted(required.difference(frame.columns))
    if missing_required:
        warnings.append("Missing required model fields: " + ", ".join(missing_required))

    if "sequence" in frame.columns:
        sequence_features = frame["sequence"].astype(str).map(compute_sequence_features)
        stats = pd.DataFrame(sequence_features.tolist())
        stats["carrier_type"] = frame.get("carrier_type", pd.Series(["unknown"] * len(frame))).astype(str)
        by_carrier: dict[str, object] = {}
        for name, group in stats.groupby("carrier_type"):
            by_carrier[str(name)] = {
                "n": int(len(group)),
                "length_mean": float(group["length"].mean()),
                "gc_mean": float(group["gc_content"].mean()),
                "gc_std": float(group["gc_content"].std(ddof=0)),
                "entropy_mean": float(group["entropy"].mean()),
                "max_run_mean": float(group["longest_homopolymer"].mean()),
            }
        report["stats_by_carrier_type"] = by_carrier

        hashes = frame["sequence"].astype(str).map(
            lambda sequence: hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        )
        duplicate_count = int(hashes.duplicated().sum())
        report["duplicate_sequence_count"] = duplicate_count
        overlaps: dict[str, int] = {}
        if "split" in frame.columns:
            split_hashes = {
                str(split): set(hashes.loc[index].tolist())
                for split, index in frame.groupby("split").groups.items()
            }
            split_names = sorted(split_hashes)
            for idx, left in enumerate(split_names):
                for right in split_names[idx + 1 :]:
                    overlaps[f"{left}_vs_{right}"] = int(len(split_hashes[left] & split_hashes[right]))
        report["duplicate_hash_overlap"] = overlaps
        if any(value > 0 for value in overlaps.values()):
            warnings.append("Duplicate sequence hashes occur across splits; possible data leakage.")

    if "split" in frame.columns:
        split_names = set(frame["split"].astype(str))
        missing_splits = sorted({"train", "val", "test"}.difference(split_names))
        if missing_splits:
            warnings.append("Missing benchmark splits: " + ", ".join(missing_splits))
    if {"split", "is_unknown"}.issubset(frame.columns):
        train_unknown = frame.loc[frame["split"] == "train", "is_unknown"].astype(int).sum()
        if train_unknown:
            warnings.append("Training split contains unknown-carrier labels; verify the open-set protocol.")

    report["warnings"] = warnings
    report["interpretation_notes"] = [
        "Positive controls are simulated artificial information carriers.",
        "Unknown anomalies are held-out codec families or configured distribution shifts.",
        "This benchmark is a detector validation resource, not evidence of artificial information in nature.",
    ]
    if out_path is not None:
        save_json(report, out_path)
    return report


def make_benchmark_card(csv_path: str | Path, out_path: str | Path) -> dict[str, object]:
    _np, pd = _require_analysis_deps()
    from .model_metrics import save_json

    frame = pd.read_csv(csv_path)
    card = {
        "name": "genome_mining V12 artificial DNA information carrier benchmark",
        "task": "Open-set detection of artificially embedded information carriers in nucleotide sequences",
        "num_samples": int(len(frame)),
        "splits": _counts(frame, "split"),
        "open_targets": _counts(frame, "open_target"),
        "carrier_types": _counts(frame, "carrier_type"),
        "codec_families": _counts(frame, "codec_family"),
        "codec_rules": _counts(frame, "codec_rule"),
        "message_types": _counts(frame, "message_type"),
        "embedding_types": _counts(frame, "embedding_type"),
        "detectability": _counts(frame, "detectability"),
        "recommended_metrics": [
            "carrier AUROC",
            "carrier AUPR",
            "unknown-family AUROC",
            "AUPR-IN/OUT",
            "FPR95",
            "hard-negative FPR",
            "ECE",
            "Brier",
            "localization IoU/token F1",
        ],
        "limitations": [
            "Synthetic simulation benchmark.",
            "Performance on controls does not establish that artificial information exists in real genomes.",
        ],
    }
    save_json(card, out_path)
    return card


def analyze_failures(
    predictions_path: str | Path,
    out_path: str | Path | None = None,
    *,
    score_column: str = "special_prob",
    threshold: float = 0.5,
) -> dict[str, object]:
    _np, pd = _require_analysis_deps()
    from .model_metrics import save_json

    frame = pd.read_csv(predictions_path)
    if "y" not in frame.columns:
        raise ValueError("Failure analysis requires a labeled predictions CSV with column 'y'.")
    if score_column not in frame.columns:
        raise ValueError(f"Failure analysis score column not found: {score_column}")
    frame["pred_label"] = (frame[score_column].astype(float) >= threshold).astype(int)
    false_positive = frame[(frame["y"] == 0) & (frame["pred_label"] == 1)]
    false_negative = frame[(frame["y"] == 1) & (frame["pred_label"] == 0)]
    report = {
        "n": int(len(frame)),
        "score_column": score_column,
        "threshold": float(threshold),
        "false_positive_count": int(len(false_positive)),
        "false_negative_count": int(len(false_negative)),
        "false_positive_rate": float(len(false_positive) / max(1, int((frame["y"] == 0).sum()))),
        "false_negative_rate": float(len(false_negative) / max(1, int((frame["y"] == 1).sum()))),
        "false_positive_by_family": _counts(false_positive, "codec_family"),
        "false_negative_by_family": _counts(false_negative, "codec_family"),
        "false_positive_by_embedding": _counts(false_positive, "embedding_type"),
        "false_negative_by_embedding": _counts(false_negative, "embedding_type"),
        "false_positive_by_detectability": _counts(false_positive, "detectability"),
        "false_negative_by_detectability": _counts(false_negative, "detectability"),
    }
    if out_path is not None:
        save_json(report, out_path)
    return report
