from __future__ import annotations

import json
from pathlib import Path


def _require_deep_deps():
    try:
        import numpy as np
        import pandas as pd
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "V12 model training requires numpy, pandas, torch, and tqdm. "
            "Activate your existing V12/PyTorch environment before running this command."
        ) from exc
    return np, pd, torch, DataLoader, tqdm


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _score_model_predictions(frame, pred, model, inverse_family_map):
    np, _pd, _torch, _DataLoader, _tqdm = _require_deep_deps()
    from .localization import region_iou
    from .model_metrics import evaluate_open_set, softmax
    from .models.mixture_prototype import mixture_unknown_score_np
    from .models.prototype_energy import energy_unknown_score_np

    family_prob = softmax(pred["family_logits"])
    mixture_prob = softmax(pred["mixture_proto_logits"])
    known_conf = 0.45 * family_prob.max(axis=1) + 0.55 * mixture_prob.max(axis=1)
    prototypes = model.prototypes.detach().cpu().numpy()
    prototype_unknown = mixture_unknown_score_np(pred["projection"], prototypes)
    energy_unknown = energy_unknown_score_np(pred["mixture_proto_logits"])
    carrier = pred["carrier_prob"]
    naturalness = pred["naturalness_prob"]
    unknown_score = carrier * prototype_unknown * energy_unknown * (1.0 - naturalness)

    metrics, predictions = evaluate_open_set(frame, carrier, known_conf, naturalness_prob=naturalness)
    predictions["unknown_score"] = unknown_score
    predictions["mixture_prototype_unknown_score"] = prototype_unknown
    predictions["prototype_unknown_score"] = prototype_unknown
    predictions["energy_unknown_score"] = energy_unknown
    predictions["pred_family"] = [inverse_family_map[int(idx)] for idx in family_prob.argmax(axis=1)]
    predictions["pred_region_start"] = [region[0] for region in pred["loc_regions"]]
    predictions["pred_region_end"] = [region[1] for region in pred["loc_regions"]]
    predictions["localization_max_score"] = pred["loc_max_scores"]
    if "sparse_max_scores" in pred:
        predictions["sparse_site_max_score"] = pred["sparse_max_scores"]

    ious = []
    for _, row in predictions[predictions.y == 1].iterrows():
        iou = region_iou(
            int(row.payload_start),
            int(row.payload_end),
            int(row.pred_region_start),
            int(row.pred_region_end),
        )
        if iou is not None:
            ious.append(iou)
    metrics["localization"] = {
        "mean_iou": None if not ious else float(np.mean(ious)),
        "median_iou": None if not ious else float(np.median(ious)),
        "n": int(len(ious)),
    }
    arrays = {
        "carrier_prob": carrier,
        "naturalness_prob": naturalness,
        "unknown_score": unknown_score,
    }
    return metrics, predictions, arrays


def _save_embedding_bundle(path: Path, pred, frame, arrays) -> None:
    np, _pd, _torch, _DataLoader, _tqdm = _require_deep_deps()
    np.savez_compressed(
        path,
        embedding=pred["embedding"],
        projection=pred["projection"],
        family_logits=pred["family_logits"],
        mixture_proto_logits=pred["mixture_proto_logits"],
        carrier_prob=arrays["carrier_prob"],
        naturalness_prob=arrays["naturalness_prob"],
        unknown_score=arrays["unknown_score"],
        y=frame.y.values,
        is_unknown=frame.is_unknown.values,
    )


def train_pe_nac_plus(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    variant: str = "pe_nac_plus",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1.0e-3,
    length: int | None = None,
    prototypes_per_class: int = 4,
    family_weight: float = 0.35,
    contrast_weight: float = 0.1,
    naturalness_weight: float = 0.3,
    prototype_weight: float = 0.3,
    localization_weight: float = 0.25,
    episodic_weight: float = 0.2,
    episodic_prob: float = 0.5,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, object]:
    np, pd, torch, DataLoader, tqdm = _require_deep_deps()
    _set_seed(seed)

    from .localization import mask_to_region, region_iou
    from .model_metrics import evaluate_open_set, save_json, softmax
    from .models.episodic import EpisodicCodecMasker
    from .models.mixture_prototype import mixture_unknown_score_np
    from .models.prototype_energy import energy_unknown_score_np
    from .torchdata import CarrierLocalizationDataset

    if variant == "ef_pe_nac_plus":
        from .models.pe_nac_v11 import EF_PENACPlusDetector as Model
        from .models.pe_nac_v11 import ef_pe_nac_loss as loss_fn

        method = "EF-PE-NAC++"
    elif variant == "pe_nac_plus":
        from .models.pe_nac_plus import PENACPlusDetector as Model
        from .models.pe_nac_plus import pe_nac_plus_loss as loss_fn

        method = "PE-NAC++"
    else:
        raise ValueError(f"Unknown PE-NAC variant: {variant}")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    seq_length = int(length or df.sequence.astype(str).str.len().mode().iloc[0])
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    classes = sorted(train.family.astype(str).unique().tolist())
    label_map = {label: idx for idx, label in enumerate(classes)}
    inv = {idx: label for label, idx in label_map.items()}

    train_loader = DataLoader(CarrierLocalizationDataset(train, seq_length, label_map), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CarrierLocalizationDataset(val, seq_length, label_map), batch_size=batch_size)
    test_loader = DataLoader(CarrierLocalizationDataset(test, seq_length, label_map), batch_size=batch_size)

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(len(classes), length=seq_length, prototypes_per_class=prototypes_per_class).to(run_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    masker = EpisodicCodecMasker(p=episodic_prob)

    best = None
    best_score = float("inf")
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y, fam, nat, loc in tqdm(train_loader, desc=f"{method} epoch {epoch}"):
            x = x.to(run_device)
            y = y.to(run_device)
            fam = fam.to(run_device)
            nat = nat.to(run_device)
            loc = loc.to(run_device)
            out = model(x)
            pseudo = masker.make_mask(fam).to(run_device)
            loss = loss_fn(
                out,
                y,
                fam,
                nat,
                loc,
                family_weight=family_weight,
                naturalness_weight=naturalness_weight,
                contrast_weight=contrast_weight,
                prototype_weight=prototype_weight,
                localization_weight=localization_weight,
                episodic_weight=episodic_weight,
                pseudo_unknown_mask=pseudo,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_pred = _predict_pe_model(model, val_loader, run_device, has_sparse=(variant == "ef_pe_nac_plus"))
        val_score = float(np.mean((val_pred["carrier_prob"] - val.y.values) ** 2))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_brier_like": val_score}
        logs.append(row)
        print(json.dumps(row, indent=2))
        if val_score < best_score:
            best_score = val_score
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best is not None:
        model.load_state_dict(best)

    val_pred = _predict_pe_model(model, val_loader, run_device, has_sparse=(variant == "ef_pe_nac_plus"))
    _val_metrics, val_pred_df, val_arrays = _score_model_predictions(val, val_pred, model, inv)
    pred = _predict_pe_model(model, test_loader, run_device, has_sparse=(variant == "ef_pe_nac_plus"))
    metrics, pred_df, test_arrays = _score_model_predictions(test, pred, model, inv)
    metrics["method"] = method
    metrics["v12_scores"] = {
        "unknown_score_formula": "carrier_prob * mixture_prototype_unknown * energy_unknown * (1-naturalness_prob)",
        "prototypes_per_class": prototypes_per_class,
    }

    pred_df.to_csv(output / "predictions.csv", index=False)
    val_pred_df.to_csv(output / "validation_predictions.csv", index=False)
    save_json(metrics, output / "metrics.json")
    save_json(logs, output / "train_log.json")
    torch.save(
        {
            "model_type": variant,
            "state_dict": model.state_dict(),
            "length": seq_length,
            "label_map": label_map,
            "classes": classes,
            "prototypes_per_class": prototypes_per_class,
        },
        output / "model.pt",
    )
    _save_embedding_bundle(output / "embeddings.npz", pred, test, test_arrays)
    _save_embedding_bundle(output / "validation_embeddings.npz", val_pred, val, val_arrays)
    print(json.dumps(metrics.get("natural_vs_special", {}), indent=2))
    print("Saved:", output)
    return metrics


def _predict_pe_model(model, loader, device: str, *, has_sparse: bool):
    np, _pd, torch, _DataLoader, _tqdm = _require_deep_deps()
    from .localization import mask_to_region

    model.eval()
    carrier_prob = []
    family_logits = []
    naturalness_prob = []
    embeddings = []
    projections = []
    mixture_logits = []
    loc_regions = []
    loc_max_scores = []
    sparse_max_scores = []
    with torch.no_grad():
        for x, _y, _fam, _nat, _loc in loader:
            out = model(x.to(device))
            carrier_prob.extend(torch.sigmoid(out["carrier_logit"]).cpu().numpy().tolist())
            naturalness_prob.extend(torch.sigmoid(out["naturalness_logit"]).cpu().numpy().tolist())
            family_logits.append(out["family_logits"].cpu().numpy())
            embeddings.append(out["embedding"].cpu().numpy())
            projections.append(out["projection"].cpu().numpy())
            mixture_logits.append(out["mixture_proto_logits"].cpu().numpy())
            loc_score = torch.sigmoid(out["localization_logits"]).cpu().numpy()
            sparse_score = torch.sigmoid(out["sparse_site_logits"]).cpu().numpy() if has_sparse and "sparse_site_logits" in out else None
            for idx, score in enumerate(loc_score):
                loc_regions.append(mask_to_region(score, threshold=0.5))
                loc_max_scores.append(float(np.max(score)))
                if sparse_score is not None:
                    sparse_max_scores.append(float(np.max(sparse_score[idx])))
    result = {
        "carrier_prob": np.asarray(carrier_prob),
        "family_logits": np.vstack(family_logits),
        "naturalness_prob": np.asarray(naturalness_prob),
        "embedding": np.vstack(embeddings),
        "projection": np.vstack(projections),
        "mixture_proto_logits": np.vstack(mixture_logits),
        "loc_regions": loc_regions,
        "loc_max_scores": np.asarray(loc_max_scores),
    }
    if sparse_max_scores:
        result["sparse_max_scores"] = np.asarray(sparse_max_scores)
    return result


def train_deep_carrier_model(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    encoder: str = "hybrid",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1.0e-3,
    length: int | None = None,
    prototypes_per_class: int = 4,
    hier_contrast_weight: float = 0.15,
    prototype_weight: float = 0.3,
    localization_weight: float = 0.25,
    episodic_weight: float = 0.2,
    episodic_prob: float = 0.5,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, object]:
    np, pd, torch, DataLoader, tqdm = _require_deep_deps()
    _set_seed(seed)

    from .localization import mask_to_region, region_iou
    from .model_metrics import evaluate_open_set, save_json, softmax
    from .models.deep_carrier_model import DeepOpenCarrierModel, deep_carrier_loss
    from .models.episodic import EpisodicCodecMasker
    from .models.mixture_prototype import mixture_unknown_score_np
    from .models.prototype_energy import energy_unknown_score_np
    from .torchdata import V12CarrierDataset

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    seq_length = int(length or df.sequence.astype(str).str.len().mode().iloc[0])
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    classes = sorted(train.family.astype(str).unique().tolist())
    family_map = {label: idx for idx, label in enumerate(classes)}
    inv = {idx: label for label, idx in family_map.items()}
    rule_values = sorted(train.get("codec_rule", train.get("rule_id", pd.Series(["unknown"]))).astype(str).unique().tolist())
    rule_map = {rule: idx for idx, rule in enumerate(rule_values)}

    train_loader = DataLoader(V12CarrierDataset(train, seq_length, family_map, rule_map), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(V12CarrierDataset(val, seq_length, family_map, rule_map), batch_size=batch_size)
    test_loader = DataLoader(V12CarrierDataset(test, seq_length, family_map, rule_map), batch_size=batch_size)

    run_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepOpenCarrierModel(len(classes), length=seq_length, encoder=encoder, prototypes_per_class=prototypes_per_class).to(run_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    masker = EpisodicCodecMasker(p=episodic_prob)

    best = None
    best_score = float("inf")
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y, fam, rule, nat, loc in tqdm(train_loader, desc=f"V12 {encoder} epoch {epoch}"):
            x = x.to(run_device)
            y = y.to(run_device)
            fam = fam.to(run_device)
            rule = rule.to(run_device)
            nat = nat.to(run_device)
            loc = loc.to(run_device)
            out = model(x)
            pseudo = masker.make_mask(fam).to(run_device)
            loss = deep_carrier_loss(
                out,
                y,
                fam,
                nat,
                loc,
                rule_labels=rule,
                pseudo_unknown_mask=pseudo,
                prototype_weight=prototype_weight,
                localization_weight=localization_weight,
                hier_contrast_weight=hier_contrast_weight,
                episodic_weight=episodic_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        val_pred = _predict_deep_model(model, val_loader, run_device)
        val_score = float(np.mean((val_pred["carrier_prob"] - val.y.values) ** 2))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_brier_like": val_score}
        logs.append(row)
        print(json.dumps(row, indent=2))
        if val_score < best_score:
            best_score = val_score
            best = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best is not None:
        model.load_state_dict(best)

    val_pred = _predict_deep_model(model, val_loader, run_device)
    _val_metrics, val_pred_df, val_arrays = _score_model_predictions(val, val_pred, model, inv)
    pred = _predict_deep_model(model, test_loader, run_device)
    metrics, pred_df, test_arrays = _score_model_predictions(test, pred, model, inv)
    metrics["method"] = f"V12-DeepOpenCarrier-{encoder}"
    metrics["encoder"] = encoder

    pred_df.to_csv(output / "predictions.csv", index=False)
    val_pred_df.to_csv(output / "validation_predictions.csv", index=False)
    save_json(metrics, output / "metrics.json")
    save_json(logs, output / "train_log.json")
    torch.save(
        {
            "model_type": "v12_deep_open_carrier",
            "state_dict": model.state_dict(),
            "length": seq_length,
            "family_map": family_map,
            "rule_map": rule_map,
            "encoder": encoder,
            "prototypes_per_class": prototypes_per_class,
        },
        output / "model.pt",
    )
    _save_embedding_bundle(output / "embeddings.npz", pred, test, test_arrays)
    _save_embedding_bundle(output / "validation_embeddings.npz", val_pred, val, val_arrays)
    print(json.dumps(metrics.get("natural_vs_special", {}), indent=2))
    print("Saved:", output)
    return metrics


def _predict_deep_model(model, loader, device: str):
    np, _pd, torch, _DataLoader, _tqdm = _require_deep_deps()
    from .localization import mask_to_region

    model.eval()
    carrier_prob = []
    family_logits = []
    naturalness_prob = []
    embeddings = []
    projections = []
    mixture_logits = []
    loc_regions = []
    loc_max_scores = []
    sparse_max_scores = []
    with torch.no_grad():
        for x, _y, _fam, _rule, _nat, _loc in loader:
            out = model(x.to(device))
            carrier_prob.extend(torch.sigmoid(out["carrier_logit"]).cpu().numpy().tolist())
            naturalness_prob.extend(torch.sigmoid(out["naturalness_logit"]).cpu().numpy().tolist())
            family_logits.append(out["family_logits"].cpu().numpy())
            embeddings.append(out["embedding"].cpu().numpy())
            projections.append(out["projection"].cpu().numpy())
            mixture_logits.append(out["mixture_proto_logits"].cpu().numpy())
            dense = torch.sigmoid(out["localization_logits"]).cpu().numpy()
            sparse = torch.sigmoid(out["sparse_site_logits"]).cpu().numpy()
            for dense_score, sparse_score in zip(dense, sparse):
                loc_regions.append(mask_to_region(dense_score, threshold=0.5))
                loc_max_scores.append(float(np.max(dense_score)))
                sparse_max_scores.append(float(np.max(sparse_score)))
    return {
        "carrier_prob": np.asarray(carrier_prob),
        "family_logits": np.vstack(family_logits),
        "naturalness_prob": np.asarray(naturalness_prob),
        "embedding": np.vstack(embeddings),
        "projection": np.vstack(projections),
        "mixture_proto_logits": np.vstack(mixture_logits),
        "loc_regions": loc_regions,
        "loc_max_scores": np.asarray(loc_max_scores),
        "sparse_max_scores": np.asarray(sparse_max_scores),
    }


def evaluate_openood_metrics(pred_path: str | Path, out_path: str | Path, *, target: str = "unknown", score_col: str | None = None) -> dict[str, object]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for evaluate-openood.") from exc
    from .model_metrics import openood_report, save_json

    df = pd.read_csv(pred_path)
    if target == "carrier":
        y = df.y.astype(int).values
        column = score_col or "special_prob"
    else:
        y = ((df.y == 1) & (df.is_unknown == 1)).astype(int).values
        column = score_col or ("unknown_score" if "unknown_score" in df else "special_prob")
    report = openood_report(y, df[column].values)
    save_json(report, out_path)
    print(json.dumps(report, indent=2))
    return report


def run_ood_method_suite(
    pred_path: str | Path,
    embeddings_path: str | Path,
    out_path: str | Path,
    *,
    target: str = "unknown",
    reference_pred_path: str | Path | None = None,
    reference_embeddings_path: str | Path | None = None,
) -> dict[str, object]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("numpy and pandas are required for run-ood-suite.") from exc
    from .model_metrics import openood_report, save_json
    from .ood_baselines import energy_score, feature_norm_score, knn_ood_score, mahalanobis_score, msp_score

    df = pd.read_csv(pred_path)
    data = np.load(embeddings_path)
    emb = data["embedding"] if "embedding" in data else data["projection"]
    reference_df = pd.read_csv(reference_pred_path) if reference_pred_path else df
    reference_data = np.load(reference_embeddings_path) if reference_embeddings_path else data
    reference_emb = (
        reference_data["embedding"] if "embedding" in reference_data else reference_data["projection"]
    )
    if len(reference_df) != len(reference_emb):
        raise ValueError("Reference predictions and embeddings have different row counts.")
    logits = None
    for key in ["family_logits", "mixture_proto_logits", "prototype_logits"]:
        if key in data:
            logits = data[key]
            break
    if target == "unknown":
        y = ((df.y == 1) & (df.is_unknown == 1)).astype(int).values
        reference_mask = ~((reference_df.y == 1) & (reference_df.is_unknown == 1)).values
    else:
        y = df.y.astype(int).values
        reference_mask = (reference_df.y == 0).values
    if not reference_mask.any():
        raise ValueError("OOD reference set has no in-distribution samples for the selected target.")
    scores: dict[str, object] = {}
    if "unknown_score" in df.columns:
        scores["model_unknown_score"] = df["unknown_score"].values
    if "special_prob" in df.columns:
        scores["carrier_prob"] = df["special_prob"].values
    if logits is not None:
        scores["msp"] = msp_score(logits)
        scores["energy"] = energy_score(logits)
    scores["feature_norm"] = feature_norm_score(emb)
    scores["knn_distance"] = knn_ood_score(
        reference_emb[reference_mask],
        emb,
        k=min(10, max(1, int(reference_mask.sum()))),
    )
    try:
        labels = reference_df.loc[reference_mask, "family"].astype(str).factorize()[0]
        scores["mahalanobis"] = mahalanobis_score(reference_emb[reference_mask], labels, emb)
    except Exception as exc:
        scores["mahalanobis_error"] = str(exc)

    report = {"reference": {
        "predictions": str(reference_pred_path or pred_path),
        "embeddings": str(reference_embeddings_path or embeddings_path),
        "n": int(len(reference_emb)),
        "uses_independent_reference": bool(reference_pred_path and reference_embeddings_path),
    }}
    for name, score in scores.items():
        if isinstance(score, str):
            report[name] = {"error": score}
        else:
            report[name] = openood_report(y, score)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, output)
    rows = []
    for name, metrics in report.items():
        if name != "reference" and isinstance(metrics, dict) and "auroc" in metrics:
            rows.append({"method": name, **metrics})
    if rows:
        pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps(report, indent=2))
    return report


def train_evidence_fusion(
    pred_path: str | Path,
    out_path: str | Path,
    model_out: str | Path,
    *,
    target: str = "unknown",
    model_type: str = "logreg",
    calibration_pred_path: str | Path | None = None,
) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas and scikit-learn are required for evidence fusion.") from exc
    from .evidence_fusion import apply_evidence_fusion, fit_evidence_fusion, save_model
    from .model_metrics import openood_report, save_json

    df = pd.read_csv(pred_path)
    calibration = pd.read_csv(calibration_pred_path) if calibration_pred_path else df
    model, names = fit_evidence_fusion(calibration, target=target, model_type=model_type)
    column = "evidence_fused_unknown_prob" if target == "unknown" else "evidence_fused_carrier_prob"
    df[column] = apply_evidence_fusion(df, model, names)
    if target == "unknown":
        df["unknown_score_raw"] = df.get("unknown_score", df[column])
        df["unknown_score"] = df[column]
    else:
        df["special_prob_raw"] = df.get("special_prob", df[column])
        df["special_prob"] = df[column]
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    save_model(model, names, model_out, target=target)
    labels = (
        ((df.y == 1) & (df.is_unknown == 1)).astype(int).to_numpy()
        if target == "unknown"
        else df.y.astype(int).to_numpy()
    )
    score = df["unknown_score"].to_numpy() if target == "unknown" else df["special_prob"].to_numpy()
    save_json(
        {
            "target": target,
            "calibration_predictions": str(calibration_pred_path or pred_path),
            "evaluation_predictions": str(pred_path),
            "features": names,
            "metrics": openood_report(labels, score),
        },
        output.with_suffix(".metrics.json"),
    )
    print(f"Saved calibrated predictions: {output}")
    print(f"Saved evidence fusion model: {model_out}")


def calibrate_unknown_score(
    pred_path: str | Path,
    calibration_pred_path: str | Path,
    out_path: str | Path,
    model_out: str | Path,
) -> dict[str, object]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Unknown-score calibration requires pandas and scikit-learn.") from exc
    from .calibration import apply_unknown_calibrator, fit_unknown_calibrator
    from .model_metrics import openood_report, save_json

    calibration = pd.read_csv(calibration_pred_path)
    predictions = pd.read_csv(pred_path)
    artifact = fit_unknown_calibrator(calibration, model_out)
    predictions["unknown_score_raw"] = predictions.get("unknown_score", 0.0)
    predictions["calibrated_unknown_prob"] = apply_unknown_calibrator(predictions, artifact)
    predictions["unknown_score"] = predictions["calibrated_unknown_prob"]
    labels = ((predictions.y == 1) & (predictions.is_unknown == 1)).astype(int).to_numpy()
    metrics = {
        "calibration_predictions": str(calibration_pred_path),
        "evaluation_predictions": str(pred_path),
        "calibrated_unknown_detection": openood_report(labels, predictions["unknown_score"].to_numpy()),
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    save_json(metrics, output.with_suffix(".metrics.json"))
    return metrics
