from __future__ import annotations

import csv
from pathlib import Path


def _require_inference_deps():
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError(
            "score-candidates requires numpy and torch. Activate your existing V12/PyTorch environment first."
        ) from exc
    return np, torch


def score_candidates(
    candidates_csv: str | Path,
    model_path: str | Path,
    out_csv: str | Path,
    *,
    batch_size: int = 64,
    device: str | None = None,
    include_sequence: bool = False,
    unknown_calibrator: str | Path | None = None,
    evidence_fusion_model: str | Path | None = None,
) -> list[dict[str, object]]:
    model_file = _resolve_model_path(model_path)
    if model_file.suffix.lower() == ".joblib":
        scored_rows = _score_baseline_candidates(candidates_csv, model_file)
        scored_rows = _apply_postprocessors(
            scored_rows,
            unknown_calibrator=unknown_calibrator,
            evidence_fusion_model=evidence_fusion_model,
        )
        if not include_sequence:
            for row in scored_rows:
                row.pop("sequence", None)
        _write_rows(Path(out_csv), scored_rows)
        return scored_rows

    np, torch = _require_inference_deps()
    from .models.mixture_prototype import mixture_unknown_score_np
    from .models.prototype_energy import energy_unknown_score_np
    from .torchdata import one_hot

    try:
        checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_file, map_location="cpu")
    model_type = str(checkpoint.get("model_type", ""))
    length = int(checkpoint.get("length") or _infer_length_from_csv(candidates_csv))
    model, classes = _load_model_from_checkpoint(checkpoint, model_type, length)
    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(run_device)
    model.eval()

    rows = _read_candidate_rows(candidates_csv)
    scored_rows: list[dict[str, object]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        x = torch.stack([one_hot(str(row["sequence"]), length) for row in batch]).to(run_device)
        with torch.no_grad():
            out = model(x)
        carrier = torch.sigmoid(out["carrier_logit"]).cpu().numpy()
        naturalness = torch.sigmoid(out["naturalness_logit"]).cpu().numpy()
        family_logits = out["family_logits"].cpu().numpy()
        mixture_logits = out["mixture_proto_logits"].cpu().numpy()
        projection = out["projection"].cpu().numpy()
        family_prob = _softmax_np(family_logits)
        mixture_prob = _softmax_np(mixture_logits)
        known_conf = 0.45 * family_prob.max(axis=1) + 0.55 * mixture_prob.max(axis=1)
        prototypes = model.prototypes.detach().cpu().numpy()
        proto_unknown = mixture_unknown_score_np(projection, prototypes)
        energy_unknown = energy_unknown_score_np(mixture_logits)
        unknown_score = carrier * proto_unknown * energy_unknown * (1.0 - naturalness)
        loc_score = torch.sigmoid(out["localization_logits"]).cpu().numpy()
        sparse_score = torch.sigmoid(out["sparse_site_logits"]).cpu().numpy() if "sparse_site_logits" in out else None

        for idx, row in enumerate(batch):
            local_start, local_end = _mask_to_region(loc_score[idx])
            scored = dict(row)
            scored["special_prob"] = float(carrier[idx])
            scored["carrier_prob"] = float(carrier[idx])
            scored["naturalness_prob"] = float(naturalness[idx])
            scored["known_conf"] = float(known_conf[idx])
            scored["unknown_score"] = float(unknown_score[idx])
            scored["prototype_unknown_score"] = float(proto_unknown[idx])
            scored["mixture_prototype_unknown_score"] = float(proto_unknown[idx])
            scored["energy_unknown_score"] = float(energy_unknown[idx])
            scored["pred_family"] = classes[int(family_prob[idx].argmax())] if classes else str(int(family_prob[idx].argmax()))
            scored["pred_region_start_local"] = local_start
            scored["pred_region_end_local"] = local_end
            abs_start, abs_end = _local_to_genome(row, local_start, local_end)
            scored["pred_region_start"] = abs_start
            scored["pred_region_end"] = abs_end
            scored["localization_max_score"] = float(np.max(loc_score[idx]))
            if sparse_score is not None:
                scored["sparse_site_max_score"] = float(np.max(sparse_score[idx]))
            if not include_sequence:
                scored.pop("sequence", None)
            scored_rows.append(scored)

    scored_rows = _apply_postprocessors(
        scored_rows,
        unknown_calibrator=unknown_calibrator,
        evidence_fusion_model=evidence_fusion_model,
    )
    _write_rows(Path(out_csv), scored_rows)
    return scored_rows


def _load_model_from_checkpoint(checkpoint: dict, model_type: str, length: int):
    classes = checkpoint.get("classes")
    if classes is None and "label_map" in checkpoint:
        label_map = checkpoint["label_map"]
        classes = [label for label, _idx in sorted(label_map.items(), key=lambda item: item[1])]
    if classes is None and "family_map" in checkpoint:
        family_map = checkpoint["family_map"]
        classes = [label for label, _idx in sorted(family_map.items(), key=lambda item: item[1])]
    classes = list(classes or [])
    n_classes = len(classes)
    prototypes_per_class = int(checkpoint.get("prototypes_per_class", 4))
    if model_type == "pe_nac_plus":
        from .models.pe_nac_plus import PENACPlusDetector

        model = PENACPlusDetector(n_classes, length=length, prototypes_per_class=prototypes_per_class)
    elif model_type in {"ef_pe_nac_plus", "ef_pe_nac_plus_v11"}:
        from .models.pe_nac_v11 import EF_PENACPlusDetector

        model = EF_PENACPlusDetector(n_classes, length=length, prototypes_per_class=prototypes_per_class)
    elif model_type == "v12_deep_open_carrier":
        from .models.deep_carrier_model import DeepOpenCarrierModel

        encoder = str(checkpoint.get("encoder", "hybrid"))
        model = DeepOpenCarrierModel(n_classes, length=length, encoder=encoder, prototypes_per_class=prototypes_per_class)
    else:
        raise ValueError(f"Unsupported model_type in checkpoint: {model_type}")
    model.load_state_dict(checkpoint["state_dict"])
    return model, classes


def _resolve_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.is_dir():
        for name in ("model.pt", "model.joblib"):
            candidate = path / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No model.pt or model.joblib found under {path}")
    return path


def _score_baseline_candidates(candidates_csv: str | Path, model_path: Path) -> list[dict[str, object]]:
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Baseline candidate scoring requires joblib and scikit-learn.") from exc
    from .baseline_model import featurize

    artifact = joblib.load(model_path)
    rows = _read_candidate_rows(candidates_csv)
    if not rows:
        return []
    features, _names = featurize(
        [str(row.get("sequence", "")) for row in rows],
        k=int(artifact.get("k", 4)),
        use_extra=bool(artifact.get("use_extra", True)),
    )
    binary_prob = artifact["binary_model"].predict_proba(features)[:, 1]
    family_prob = artifact["family_model"].predict_proba(features)
    known_conf = family_prob.max(axis=1)
    family_indices = family_prob.argmax(axis=1)
    predicted_families = artifact["label_encoder"].inverse_transform(family_indices)
    scored_rows = []
    for idx, row in enumerate(rows):
        scored = dict(row)
        carrier = float(binary_prob[idx])
        confidence = float(known_conf[idx])
        scored.update(
            {
                "special_prob": carrier,
                "carrier_prob": carrier,
                "naturalness_prob": 0.0,
                "known_conf": confidence,
                "unknown_score": carrier * (1.0 - confidence),
                "pred_family": str(predicted_families[idx]),
                "pred_region_start_local": -1,
                "pred_region_end_local": -1,
                "pred_region_start": -1,
                "pred_region_end": -1,
            }
        )
        scored_rows.append(scored)
    return scored_rows


def _apply_postprocessors(
    rows: list[dict[str, object]],
    *,
    unknown_calibrator: str | Path | None,
    evidence_fusion_model: str | Path | None,
) -> list[dict[str, object]]:
    if not rows or (unknown_calibrator is None and evidence_fusion_model is None):
        return rows
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Calibrated candidate scoring requires pandas.") from exc

    frame = pd.DataFrame(rows)
    if evidence_fusion_model is not None:
        from .evidence_fusion import apply_evidence_fusion, load_artifact

        model, feature_names, target = load_artifact(evidence_fusion_model)
        column = "evidence_fused_unknown_prob" if target == "unknown" else "evidence_fused_carrier_prob"
        frame[column] = apply_evidence_fusion(frame, model, feature_names)
        if target == "unknown":
            frame["unknown_score_raw"] = frame["unknown_score"]
            frame["unknown_score"] = frame[column]
        else:
            frame["special_prob_raw"] = frame["special_prob"]
            frame["special_prob"] = frame[column]
            frame["carrier_prob"] = frame[column]
    if unknown_calibrator is not None:
        from .calibration import apply_unknown_calibrator, load_unknown_calibrator

        artifact = load_unknown_calibrator(unknown_calibrator)
        frame["unknown_score_raw"] = frame["unknown_score"]
        frame["calibrated_unknown_prob"] = apply_unknown_calibrator(frame, artifact)
        frame["unknown_score"] = frame["calibrated_unknown_prob"]
    return frame.to_dict(orient="records")


def _softmax_np(logits):
    import numpy as np

    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def _read_candidate_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _infer_length_from_csv(path: str | Path) -> int:
    rows = _read_candidate_rows(path)
    if not rows:
        raise ValueError("Cannot infer model length from an empty candidate CSV.")
    return len(str(rows[0].get("sequence", "")))


def _mask_to_region(score, threshold: float = 0.5) -> tuple[int, int]:
    import numpy as np

    idx = np.where(np.asarray(score, dtype=float) >= threshold)[0]
    if len(idx) == 0:
        return -1, -1
    return int(idx.min()), int(idx.max() + 1)


def _local_to_genome(row: dict[str, object], local_start: int, local_end: int) -> tuple[int, int]:
    if local_start < 0 or local_end <= local_start:
        return -1, -1
    extract_start = int(float(row.get("extract_start", row.get("start", 0))))
    extract_end = int(float(row.get("extract_end", row.get("end", extract_start))))
    left_pad = int(float(row.get("left_pad", 0)))
    if str(row.get("strand", "+")) == "-":
        right_pad = int(float(row.get("right_pad", 0)))
        start = extract_end - max(0, local_end - right_pad)
        end = extract_end - max(0, local_start - right_pad)
    else:
        start = extract_start + max(0, local_start - left_pad)
        end = extract_start + max(0, local_end - left_pad)
    return max(extract_start, min(start, extract_end)), max(extract_start, min(end, extract_end))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
