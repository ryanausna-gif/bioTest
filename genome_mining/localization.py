import numpy as np


def region_to_mask(start: int, end: int, length: int):
    mask = np.zeros(length, dtype=np.float32)
    if start is None or end is None:
        return mask
    start = int(start)
    end = int(end)
    if start < 0 or end <= start:
        return mask
    start = max(0, min(length, start))
    end = max(0, min(length, end))
    mask[start:end] = 1.0
    return mask


def mask_to_region(score, threshold=0.5):
    score = np.asarray(score, dtype=float)
    idx = np.where(score >= threshold)[0]
    if len(idx) == 0:
        return -1, -1
    return int(idx.min()), int(idx.max() + 1)


def region_iou(true_start, true_end, pred_start, pred_end):
    if true_start < 0 or true_end <= true_start:
        return None
    if pred_start < 0 or pred_end <= pred_start:
        return 0.0
    inter = max(0, min(true_end, pred_end) - max(true_start, pred_start))
    union = max(true_end, pred_end) - min(true_start, pred_start)
    return float(inter / max(1, union))


def token_f1(true_mask, pred_mask):
    true_mask = np.asarray(true_mask).astype(int)
    pred_mask = np.asarray(pred_mask).astype(int)
    tp = int(((true_mask == 1) & (pred_mask == 1)).sum())
    fp = int(((true_mask == 0) & (pred_mask == 1)).sum())
    fn = int(((true_mask == 1) & (pred_mask == 0)).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return None
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def localization_metrics_from_regions(df):
    rows = []
    for _, r in df.iterrows():
        ts = int(r.get("payload_start", r.get("start", -1)))
        te = int(r.get("payload_end", r.get("end", -1)))
        ps = int(r.get("pred_region_start", -1))
        pe = int(r.get("pred_region_end", -1))
        iou = region_iou(ts, te, ps, pe)
        if iou is not None:
            rows.append(iou)
    return {
        "mean_iou": None if not rows else float(np.mean(rows)),
        "median_iou": None if not rows else float(np.median(rows)),
        "n_localized": int(len(rows)),
    }
