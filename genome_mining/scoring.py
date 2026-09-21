from __future__ import annotations

import statistics
from typing import Iterable

NUMERIC_FEATURES = [
    "gc_content",
    "entropy_norm",
    "dinuc_entropy",
    "trinuc_entropy",
    "compression_ratio",
    "homopolymer_fraction",
    "kmer3_max_fraction",
    "periodicity_score",
    "n_fraction",
]

EVIDENCE_CAP = 25.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mad(values: list[float], center: float) -> float:
    deviations = [abs(value - center) for value in values]
    mad = statistics.median(deviations) if deviations else 0.0
    return mad if mad > 1.0e-12 else 1.0e-12


def build_baseline(rows: Iterable[dict[str, object]]) -> dict[str, dict[str, float]]:
    materialized = list(rows)
    baseline: dict[str, dict[str, float]] = {}
    for feature in NUMERIC_FEATURES:
        values = [float(row.get(feature, 0.0)) for row in materialized]
        center = _median(values)
        baseline[feature] = {"median": center, "mad": _mad(values, center)}
    return baseline


def robust_z(value: float, center: float, mad: float) -> float:
    return 0.6745 * (value - center) / mad


def score_feature_row(row: dict[str, object], baseline: dict[str, dict[str, float]]) -> dict[str, object]:
    z: dict[str, float] = {}
    for feature in NUMERIC_FEATURES:
        stats = baseline[feature]
        z[feature] = robust_z(float(row.get(feature, 0.0)), stats["median"], stats["mad"])

    evidence_parts = {
        "gc_deviation": min(EVIDENCE_CAP, abs(z["gc_content"])),
        "entropy_low": min(EVIDENCE_CAP, max(0.0, -z["entropy_norm"])),
        "dinuc_entropy_low": min(EVIDENCE_CAP, max(0.0, -z["dinuc_entropy"])),
        "trinuc_entropy_low": min(EVIDENCE_CAP, max(0.0, -z["trinuc_entropy"])),
        "compression_low": min(EVIDENCE_CAP, max(0.0, -z["compression_ratio"])),
        "homopolymer_high": min(EVIDENCE_CAP, max(0.0, z["homopolymer_fraction"])),
        "kmer3_repeat_high": min(EVIDENCE_CAP, max(0.0, z["kmer3_max_fraction"])),
        "periodicity_high": min(EVIDENCE_CAP, max(0.0, z["periodicity_score"])),
    }
    weights = {
        "gc_deviation": 0.18,
        "entropy_low": 0.12,
        "dinuc_entropy_low": 0.08,
        "trinuc_entropy_low": 0.08,
        "compression_low": 0.12,
        "homopolymer_high": 0.14,
        "kmer3_repeat_high": 0.14,
        "periodicity_high": 0.14,
    }
    score = sum(weights[name] * value for name, value in evidence_parts.items())
    top_evidence = sorted(evidence_parts.items(), key=lambda item: item[1], reverse=True)[:3]

    scored = dict(row)
    scored["anomaly_score"] = score
    scored["evidence"] = ";".join(f"{name}={value:.3f}" for name, value in top_evidence)
    for feature, value in z.items():
        scored[f"z_{feature}"] = value
    return scored


def score_feature_rows(rows: Iterable[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    materialized = list(rows)
    baseline = build_baseline(materialized)
    scored = [score_feature_row(row, baseline) for row in materialized]
    scored.sort(key=lambda row: float(row["anomaly_score"]), reverse=True)
    return scored, baseline
