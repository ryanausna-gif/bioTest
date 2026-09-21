from __future__ import annotations

import json
from pathlib import Path


def _require_numpy_sklearn():
    try:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            brier_score_loss,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError("numpy and scikit-learn are required for genome_mining model metrics.") from exc
    return np, accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score


def save_json(obj, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def softmax(logits):
    np, *_ = _require_numpy_sklearn()
    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def fpr_at_tpr(y_true, score, target_tpr: float = 0.95):
    np, *_ = _require_numpy_sklearn()
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(score)[::-1]
    y = y_true[order]
    pos = max(1, int((y_true == 1).sum()))
    neg = max(1, int((y_true == 0).sum()))
    tp = fp = 0
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if tp / pos >= target_tpr:
            return float(fp / neg)
    return 1.0


def ece(y_true, prob, n_bins: int = 15) -> float:
    np, *_ = _require_numpy_sklearn()
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if mask.any():
            out += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(out)


def binary_metrics(y_true, score, threshold: float = 0.5) -> dict[str, object]:
    (
        np,
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    ) = _require_numpy_sklearn()
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    out: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "fpr95": fpr_at_tpr(y_true, score),
        "ece": ece(y_true, np.clip(score, 0, 1)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, score))
    except Exception:
        out["auroc"] = None
    try:
        out["aupr"] = float(average_precision_score(y_true, score))
    except Exception:
        out["aupr"] = None
    try:
        out["brier"] = float(brier_score_loss(y_true, np.clip(score, 0, 1)))
    except Exception:
        out["brier"] = None
    return out


def evaluate_open_set(df, carrier_prob, known_conf, naturalness_prob=None):
    np, *_ = _require_numpy_sklearn()
    frame = df.copy()
    frame["special_prob"] = np.asarray(carrier_prob)
    frame["carrier_prob"] = frame["special_prob"]
    frame["known_conf"] = np.asarray(known_conf)
    if naturalness_prob is None:
        # A missing naturalness head should be neutral in the multiplicative
        # unknown score, so (1 - naturalness_prob) must equal one.
        naturalness_prob = np.zeros(len(frame))
    frame["naturalness_prob"] = np.asarray(naturalness_prob)
    frame["unknown_score"] = frame["carrier_prob"] * (1 - frame["known_conf"]) * (1 - frame["naturalness_prob"])
    y = frame["y"].values.astype(int)
    unknown = ((frame["y"] == 1) & (frame["is_unknown"] == 1)).astype(int).values
    known = ((frame["y"] == 1) & (frame["is_unknown"] == 0)).astype(int).values
    hardneg = None
    if "carrier_type" in frame:
        hardneg = ((frame["y"] == 0) & (frame["carrier_type"].astype(str).str.contains("hard", na=False))).astype(int).values
    out = {
        "carrier_detection": binary_metrics(y, frame["carrier_prob"].values),
        "natural_vs_special": binary_metrics(y, frame["carrier_prob"].values),
        "unknown_codec_detection": binary_metrics(unknown, frame["unknown_score"].values) if unknown.sum() > 0 and unknown.sum() < len(unknown) else None,
        "unknown_family_detection": binary_metrics(unknown, frame["unknown_score"].values) if unknown.sum() > 0 and unknown.sum() < len(unknown) else None,
        "known_carrier_detection": binary_metrics(known, frame["carrier_prob"].values) if known.sum() > 0 and known.sum() < len(known) else None,
        "counts": {
            "n": int(len(frame)),
            "positive": int(y.sum()),
            "unknown_positive": int(unknown.sum()),
            "negative": int((y == 0).sum()),
        },
    }
    if hardneg is not None and hardneg.sum() > 0:
        out["hard_negative_fpr_at_0.5"] = float(((frame["carrier_prob"] >= 0.5) & (hardneg == 1)).sum() / hardneg.sum())
    return out, frame


def detection_error_at_tpr(y_true, score, target_tpr: float = 0.95):
    np, *_ = _require_numpy_sklearn()
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(score)[::-1]
    y = y_true[order]
    pos = max(1, int((y_true == 1).sum()))
    neg = max(1, int((y_true == 0).sum()))
    tp = fp = 0
    for label in y:
        if label == 1:
            tp += 1
        else:
            fp += 1
        if tp / pos >= target_tpr:
            return float(0.5 * (fp / neg + 1 - tp / pos))
    return 0.5


def risk_coverage_auc(y_true, prob) -> float:
    np, *_ = _require_numpy_sklearn()
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    conf = np.maximum(prob, 1 - prob)
    order = np.argsort(conf)[::-1]
    risks = []
    coverage = []
    for k in range(1, len(order) + 1):
        idx = order[:k]
        risks.append((pred[idx] != y_true[idx]).mean())
        coverage.append(k / len(order))
    return float(np.trapz(risks, coverage))


def openood_report(y_true, score) -> dict[str, object]:
    (
        np,
        _accuracy_score,
        average_precision_score,
        brier_score_loss,
        _f1_score,
        _precision_score,
        _recall_score,
        roc_auc_score,
    ) = _require_numpy_sklearn()
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"error": "Need both classes"}
    clipped = np.clip(score, 0, 1)
    yin = 1 - y_true
    return {
        "auroc": float(roc_auc_score(y_true, score)),
        "aupr_out": float(average_precision_score(y_true, score)),
        "aupr_in": float(average_precision_score(yin, -score)),
        "fpr95": fpr_at_tpr(y_true, score),
        "detection_error_at_95tpr": detection_error_at_tpr(y_true, score),
        "ece": ece(y_true, clipped),
        "brier": float(brier_score_loss(y_true, clipped)),
        "risk_coverage_auc": risk_coverage_auc(y_true, clipped),
    }
