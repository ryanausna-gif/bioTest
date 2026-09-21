from __future__ import annotations


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Localization evaluation requires numpy.") from exc
    return np


def _row_length(row, explicit_length: int | None = None) -> int:
    if explicit_length:
        return int(explicit_length)
    if "sequence" in row and str(row.get("sequence", "")):
        return len(str(row["sequence"]))
    if "sequence_length" in row:
        return int(row["sequence_length"])
    ends = [int(row.get(name, -1)) for name in ("payload_end", "end", "pred_region_end")]
    return max(1, max(ends))


def token_localization_metrics(df, length: int | None = None) -> dict[str, object]:
    np = _require_numpy()
    from .localization import region_to_mask, token_f1

    f1s: list[float] = []
    ious: list[float] = []
    boundary_errors: list[float] = []
    positives = df[df["y"] == 1] if "y" in df.columns else df
    for _, row in positives.iterrows():
        seq_length = _row_length(row, length)
        true_start = int(row.get("payload_start", row.get("start", -1)))
        true_end = int(row.get("payload_end", row.get("end", -1)))
        pred_start = int(row.get("pred_region_start", -1))
        pred_end = int(row.get("pred_region_end", -1))
        if true_start < 0 or true_end <= true_start:
            continue
        true_mask = region_to_mask(true_start, true_end, seq_length)
        pred_mask = region_to_mask(pred_start, pred_end, seq_length)
        f1 = token_f1(true_mask, pred_mask)
        if f1 is not None:
            f1s.append(float(f1))
        inter = int(((true_mask == 1) & (pred_mask == 1)).sum())
        union = int(((true_mask == 1) | (pred_mask == 1)).sum())
        ious.append(float(inter / max(1, union)))
        if pred_start >= 0 and pred_end > pred_start:
            boundary_errors.append((abs(pred_start - true_start) + abs(pred_end - true_end)) / 2.0)
    return {
        "token_f1_mean": None if not f1s else float(np.mean(f1s)),
        "token_f1_median": None if not f1s else float(np.median(f1s)),
        "token_iou_mean": None if not ious else float(np.mean(ious)),
        "boundary_error_mean": None if not boundary_errors else float(np.mean(boundary_errors)),
        "n": int(len(ious)),
    }


def multiscale_localization_metrics(df, length: int | None = None) -> dict[str, object]:
    np = _require_numpy()
    from .localization import region_iou

    token_report = token_localization_metrics(df, length=length)
    ious: list[float] = []
    by_embedding: dict[str, list[float]] = {}
    positives = df[df["y"] == 1] if "y" in df.columns else df
    for _, row in positives.iterrows():
        true_start = int(row.get("payload_start", row.get("start", -1)))
        true_end = int(row.get("payload_end", row.get("end", -1)))
        pred_start = int(row.get("pred_region_start", -1))
        pred_end = int(row.get("pred_region_end", -1))
        iou = region_iou(true_start, true_end, pred_start, pred_end)
        if iou is None:
            continue
        value = float(iou)
        ious.append(value)
        embedding_type = str(row.get("embedding_type", "unknown"))
        by_embedding.setdefault(embedding_type, []).append(value)
    return {
        "dense_region_iou_mean": None if not ious else float(np.mean(ious)),
        "dense_region_iou_median": None if not ious else float(np.median(ious)),
        "token_f1_mean": token_report["token_f1_mean"],
        "token_f1_median": token_report["token_f1_median"],
        "token_iou_mean": token_report["token_iou_mean"],
        "boundary_error_mean": token_report["boundary_error_mean"],
        "by_embedding_iou_mean": {
            name: float(np.mean(values)) for name, values in sorted(by_embedding.items())
        },
        "n": int(len(ious)),
    }
