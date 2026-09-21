from __future__ import annotations

import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path


def _require_training_deps():
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "Image locator training requires numpy, PyTorch, and tqdm. "
            "Activate the genome-mining-v12 Conda environment."
        ) from exc
    return np, torch, DataLoader, tqdm


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_manifest(dataset_dir: str | Path) -> dict[str, object]:
    return json.loads((Path(dataset_dir) / "dataset_manifest.json").read_text(encoding="utf-8"))


def _resolve_device(torch, device: str | None) -> str:
    selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(selected).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return str(selected)


def _make_loader(dataset, DataLoader, *, batch_size: int, shuffle: bool, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _build_model(model_type: str, model_config: dict[str, object]):
    if model_type == "hierarchical_image_locator_v2":
        from .models.image_locator import HierarchicalImageLocator

        return HierarchicalImageLocator(**model_config)
    if model_type == "hyenadna_image_locator_v1":
        from .models.hyenadna_locator import HyenaDNALocator

        return HyenaDNALocator(**model_config)
    if model_type == "hierarchical_image_locator_v1":
        from .models.image_locator import HierarchicalImageLocator

        return HierarchicalImageLocator(**model_config)
    raise ValueError(f"Unsupported image locator model_type={model_type!r}.")


def _load_checkpoint_model(torch, model_path: str | Path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    model_type = str(checkpoint.get("model_type", "hierarchical_image_locator_v1"))
    model = _build_model(model_type, dict(checkpoint["model_config"]))
    if model_type == "hierarchical_image_locator_v1":
        state = {}
        prefix_map = {
            "presence_head.": "task_heads.presence_head.",
            "start_bin_head.": "task_heads.start_bin_head.",
            "offset_head.": "task_heads.start_offset_head.",
            "segment_head.": "task_heads.segment_head.",
        }
        for name, value in checkpoint["state_dict"].items():
            for old, new in prefix_map.items():
                if name.startswith(old):
                    name = new + name[len(old) :]
                    break
            state[name] = value
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model


def _predict(model, loader, *, device: str, stride: int, window_length: int, legacy_payload_bases=None):
    np, torch, _DataLoader, _tqdm = _require_training_deps()
    from .models.image_locator import decode_locator_output

    model.eval()
    values: dict[str, list] = {
        "y": [],
        "true_start": [],
        "true_end": [],
        "array_index": [],
        "presence_prob": [],
        "pred_start": [],
        "pred_end": [],
        "segment_iou": [],
        "embedding": [],
    }
    with torch.no_grad():
        for tokens, y, start, end, array_index in loader:
            tokens = tokens.to(device, non_blocking=True)
            output = model(tokens)
            decoded = decode_locator_output(
                output,
                stride=stride,
                window_length=window_length,
                payload_bases=legacy_payload_bases,
            )
            if legacy_payload_bases is not None:
                decoded["start"] = torch.clamp(
                    decoded["start"], max=max(0, window_length - int(legacy_payload_bases))
                )
                decoded["end"] = decoded["start"] + int(legacy_payload_bases)
            segment_pred = decoded["segment_prob"] >= 0.5
            bins = segment_pred.shape[1]
            bin_starts = torch.arange(bins, device=device) * stride
            bin_ends = bin_starts + stride
            true_segment = (
                (y.to(device)[:, None] > 0.5)
                & (bin_ends[None, :] > start.to(device)[:, None])
                & (bin_starts[None, :] < end.to(device)[:, None])
            )
            intersection = (segment_pred & true_segment).sum(dim=1).to(torch.float32)
            union = (segment_pred | true_segment).sum(dim=1).clamp(min=1).to(torch.float32)
            values["segment_iou"].extend((intersection / union).cpu().tolist())
            values["y"].extend(y.to(torch.int64).tolist())
            values["true_start"].extend(start.tolist())
            values["true_end"].extend(end.tolist())
            values["array_index"].extend(array_index.tolist())
            values["presence_prob"].extend(decoded["presence_prob"].cpu().tolist())
            values["pred_start"].extend(decoded["start"].cpu().tolist())
            values["pred_end"].extend(decoded["end"].cpu().tolist())
            values["embedding"].extend(output["embedding"].detach().float().cpu().tolist())
    return {
        "y": np.asarray(values["y"], dtype=int),
        "true_start": np.asarray(values["true_start"], dtype=int),
        "true_end": np.asarray(values["true_end"], dtype=int),
        "array_index": np.asarray(values["array_index"], dtype=int),
        "presence_prob": np.asarray(values["presence_prob"], dtype=float),
        "pred_start": np.asarray(values["pred_start"], dtype=int),
        "pred_end": np.asarray(values["pred_end"], dtype=int),
        "segment_iou": np.asarray(values["segment_iou"], dtype=float),
        "embedding": np.asarray(values["embedding"], dtype=float),
    }


def _localization_metrics(prediction, mask=None) -> dict[str, object]:
    np, _torch, _DataLoader, _tqdm = _require_training_deps()
    positive = prediction["y"] == 1
    if mask is not None:
        positive &= np.asarray(mask, dtype=bool)
    if not positive.any():
        return {"n_positive": 0}
    true_start = prediction["true_start"][positive]
    true_end = prediction["true_end"][positive]
    pred_start = prediction["pred_start"][positive]
    pred_end = prediction["pred_end"][positive]
    start_error = np.abs(pred_start - true_start)
    end_error = np.abs(pred_end - true_end)
    intersection = np.maximum(0, np.minimum(true_end, pred_end) - np.maximum(true_start, pred_start))
    union = np.maximum(true_end, pred_end) - np.minimum(true_start, pred_start)
    iou = intersection / np.maximum(1, union)
    detected = prediction["presence_prob"][positive] >= 0.5
    exact_span = (start_error == 0) & (end_error == 0)
    return {
        "n_positive": int(positive.sum()),
        "start_mae_bases": float(start_error.mean()),
        "end_mae_bases": float(end_error.mean()),
        "boundary_mae_bases": float(np.mean(np.concatenate([start_error, end_error]))),
        "start_median_ae_bases": float(np.median(start_error)),
        "start_exact_rate": float((start_error == 0).mean()),
        "end_exact_rate": float((end_error == 0).mean()),
        "exact_span_rate": float(exact_span.mean()),
        "start_within_4_rate": float((start_error <= 4).mean()),
        "both_boundaries_within_16_rate": float(((start_error <= 16) & (end_error <= 16)).mean()),
        "both_boundaries_within_256_rate": float(((start_error <= 256) & (end_error <= 256)).mean()),
        "base_phase_accuracy": float(((pred_start % 4) == (true_start % 4)).mean()),
        "mean_span_iou": float(iou.mean()),
        "median_span_iou": float(np.median(iou)),
        "mean_coarse_segment_iou": float(prediction["segment_iou"][positive].mean()),
        "joint_detect_and_exact_span_rate": float((detected & exact_span).mean()),
    }


def _prediction_rows(dataset, prediction) -> list[dict[str, object]]:
    by_array_index = {int(row["array_index"]): row for row in dataset.rows}
    rows = []
    for index, array_index in enumerate(prediction["array_index"]):
        source = by_array_index[int(array_index)]
        positive = bool(int(prediction["y"][index]))
        true_start = int(prediction["true_start"][index]) if positive else -1
        true_end = int(prediction["true_end"][index]) if positive else -1
        pred_start = int(prediction["pred_start"][index])
        pred_end = int(prediction["pred_end"][index])
        intersection = max(0, min(true_end, pred_end) - max(true_start, pred_start)) if positive else 0
        union = max(true_end, pred_end) - min(true_start, pred_start) if positive else 0
        rows.append(
            {
                "sample_id": source["sample_id"],
                "array_index": int(array_index),
                "split": source["split"],
                "y": int(positive),
                "presence_prob": float(prediction["presence_prob"][index]),
                "pred_y": int(prediction["presence_prob"][index] >= 0.5),
                "true_start": true_start,
                "pred_start": pred_start,
                "start_abs_error": abs(pred_start - true_start) if positive else "",
                "true_end": true_end,
                "pred_end": pred_end,
                "end_abs_error": abs(pred_end - true_end) if positive else "",
                "span_iou": intersection / max(1, union) if positive else "",
                "coarse_segment_iou": float(prediction["segment_iou"][index]),
                "codec": source.get("codec", ""),
                "codec_level": source.get("codec_level", ""),
                "is_unknown": int(source.get("is_unknown", 0)),
                "open_target": source.get("open_target", ""),
                "carrier_type": source.get("carrier_type", ""),
                "donor_id": source["donor_id"],
                "assembly_id": source["assembly_id"],
                "chrom": source["chrom"],
                "image_id": source["image_id"],
                "class_id": source["class_id"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metrics_by_codec(dataset, prediction, binary_metrics) -> dict[str, object]:
    np, _torch, _DataLoader, _tqdm = _require_training_deps()
    by_index = {int(row["array_index"]): row for row in dataset.rows}
    codecs = np.asarray([by_index[int(index)].get("codec", "none") for index in prediction["array_index"]])
    output = {}
    manifest = _load_manifest(dataset.root)
    train_codecs = set(manifest.get("codecs_by_split", {}).get("train", []))
    for codec in sorted(set(codecs) - {"none"}):
        relevant = (prediction["y"] == 0) | (codecs == codec)
        detection = binary_metrics(
            prediction["y"][relevant], prediction["presence_prob"][relevant]
        )
        auroc = detection.get("auroc")
        detection["detector_advantage"] = None if auroc is None else float(2.0 * auroc - 1.0)
        output[codec] = {
            "seen_during_training": codec in train_codecs,
            "presence_detection": detection,
            "localization": _localization_metrics(prediction, codecs == codec),
            "n_codec_positive": int(((prediction["y"] == 1) & (codecs == codec)).sum()),
        }
    return output


def evaluate_image_locator_model(
    dataset_dir: str | Path,
    model_path: str | Path,
    out_dir: str | Path,
    *,
    split: str = "test",
    batch_size: int = 4,
    workers: int = 2,
    device: str | None = None,
    save_images: bool = False,
    max_saved_images: int = 25,
    codec_key_file: str | Path | None = None,
) -> dict[str, object]:
    np, torch, DataLoader, _tqdm = _require_training_deps()
    from .image_payload import (
        RawImageDNASpec,
        compare_pixel_bytes,
        preprocess_image,
        token_values_to_dna,
    )
    from .image_torchdata import ImageGenomeDataset
    from .model_metrics import binary_metrics, save_json
    from .stego_codecs import codec_key_fingerprint, load_codec_key, make_stego_codec

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(dataset_dir)
    checkpoint, model = _load_checkpoint_model(torch, model_path)
    run_device = _resolve_device(torch, device)
    model.to(run_device)
    dataset = ImageGenomeDataset(dataset_dir, split)
    loader = _make_loader(dataset, DataLoader, batch_size=batch_size, shuffle=False, workers=workers)
    spec = manifest["image_spec"]
    window_length = int(manifest["window_length"])
    legacy_payload = int(spec["payload_bases"]) if checkpoint.get("model_type") == "hierarchical_image_locator_v1" else None
    prediction = _predict(
        model,
        loader,
        device=run_device,
        stride=int(checkpoint["model_config"]["downsample_stride"]),
        window_length=window_length,
        legacy_payload_bases=legacy_payload,
    )
    metrics = {
        "split": split,
        "n_samples": int(len(prediction["y"])),
        "model_type": checkpoint.get("model_type"),
        "presence_detection": binary_metrics(prediction["y"], prediction["presence_prob"]),
        "localization": _localization_metrics(prediction),
        "by_codec": _metrics_by_codec(dataset, prediction, binary_metrics),
    }

    codec_key = load_codec_key(codec_key_file)
    expected_fingerprint = str(manifest.get("codec_key_fingerprint", ""))
    if codec_key is not None and expected_fingerprint:
        actual_fingerprint = codec_key_fingerprint(codec_key)
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Codec key fingerprint mismatch: dataset={expected_fingerprint}, "
                f"provided={actual_fingerprint}."
            )
    recovery_rows = []
    saved = 0
    image_dir = output_dir / "recovered_images"
    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    index_to_row = {int(row["array_index"]): row for row in dataset.rows}
    codec_cache = {}
    lm_config = manifest.get("lm_stego", {})
    for index in np.flatnonzero(prediction["y"] == 1):
        array_index = int(prediction["array_index"][index])
        row = index_to_row[array_index]
        tokens = np.asarray(dataset.sequences[array_index])
        true_start = int(prediction["true_start"][index])
        true_end = int(prediction["true_end"][index])
        pred_start = int(prediction["pred_start"][index])
        pred_end = int(prediction["pred_end"][index])
        codec_name = row.get("codec", "raw_gray_2bit_v1")
        reference = preprocess_image(
            row["image_path"],
            RawImageDNASpec(
                width=int(spec["width"]),
                height=int(spec["height"]),
                resize_short_side=(
                    None
                    if spec.get("resize_short_side") is None
                    else int(spec["resize_short_side"])
                ),
            ),
        ).tobytes()
        try:
            if codec_name not in codec_cache:
                codec_cache[codec_name] = make_stego_codec(
                    codec_name,
                    key=codec_key,
                    kmer_order=int(manifest.get("kmer_order", 3)),
                    ecc_symbols=int(
                        manifest.get("ecc", {}).get("symbols_per_255_base_byte_block", 0)
                    ),
                    lm_model_path=lm_config.get("model_file"),
                    lm_block_bytes=int(lm_config.get("block_bytes", 8)),
                    lm_max_symbols_per_block=int(
                        lm_config.get("max_symbols_per_block", 256)
                    ),
                    lm_context_bases=int(lm_config.get("context_bases", 4096)),
                    lm_cdf_precision_bits=int(lm_config.get("cdf_precision_bits", 12)),
                    lm_uniform_mix=float(lm_config.get("uniform_mix", 0.20)),
                )
            predicted_dna = token_values_to_dna(tokens[pred_start:pred_end])
            context_bases = int(getattr(codec_cache[codec_name], "context_bases", 0))
            left_context = token_values_to_dna(
                tokens[max(0, pred_start - context_bases) : pred_start]
            )
            recovered = codec_cache[codec_name].decode(
                predicted_dna,
                expected_bytes=int(spec["payload_bytes"]),
                sample_id=row["sample_id"],
                context=left_context,
            )
            pixel_metrics = compare_pixel_bytes(reference, recovered)
            decode_success = True
            error = ""
        except (ValueError, ImportError) as exc:
            recovered = None
            pixel_metrics = {}
            decode_success = False
            error = str(exc)
        record = {
            "sample_id": row["sample_id"],
            "image_id": row["image_id"],
            "codec": codec_name,
            "true_start": true_start,
            "true_end": true_end,
            "pred_start": pred_start,
            "pred_end": pred_end,
            "presence_prob": float(prediction["presence_prob"][index]),
            "decode_success": decode_success,
            "decode_error": error,
            **pixel_metrics,
        }
        recovery_rows.append(record)
        if save_images and saved < max_saved_images:
            from PIL import Image

            size = (int(spec["width"]), int(spec["height"]))
            stem = f"{saved:04d}_{row['sample_id']}"
            Image.frombytes("L", size, reference).save(image_dir / f"{stem}_reference.png")
            if recovered is not None:
                Image.frombytes("L", size, recovered).save(image_dir / f"{stem}_recovered.png")
            saved += 1

    decoded = [row for row in recovery_rows if row["decode_success"]]
    metrics["image_recovery"] = {
        "n_positive": len(recovery_rows),
        "decode_success_rate": sum(row["decode_success"] for row in recovery_rows) / len(recovery_rows)
        if recovery_rows
        else None,
        "exact_image_rate": sum(bool(row.get("exact", False)) for row in recovery_rows) / len(recovery_rows)
        if recovery_rows
        else None,
        "mean_pixel_exact_fraction": float(np.mean([row["pixel_exact_fraction"] for row in decoded]))
        if decoded
        else None,
        "mean_pixel_mae": float(np.mean([row["pixel_mae"] for row in decoded])) if decoded else None,
        "mean_psnr": _finite_mean(np, [row["psnr"] for row in decoded]),
        "mean_ssim_global": float(np.mean([row["ssim_global"] for row in decoded])) if decoded else None,
    }
    _write_csv(output_dir / "predictions.csv", _prediction_rows(dataset, prediction))
    _write_csv(output_dir / "image_recovery.csv", recovery_rows)
    np.save(output_dir / "embeddings.npy", prediction["embedding"])
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def _finite_mean(np, values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def train_image_locator(
    dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 2.0e-4,
    weight_decay: float = 1.0e-4,
    backbone: str = "cnn",
    d_model: int = 128,
    downsample_stride: int = 256,
    context: str = "dilated_cnn",
    context_layers: int = 6,
    dropout: float = 0.1,
    hyena_model_name: str = "LongSafari/hyenadna-medium-160k-seqlen-hf",
    hyena_revision: str | None = None,
    hyena_local_files_only: bool = False,
    hyena_fine_tune: str = "frozen",
    hyena_last_n_layers: int = 2,
    hyena_reverse_complement: bool = True,
    hyena_pretrained: bool = True,
    gradient_checkpointing: bool = False,
    gradient_accumulation: int = 1,
    grad_clip: float = 1.0,
    presence_weight: float = 1.0,
    start_weight: float = 1.0,
    offset_weight: float = 1.0,
    segment_weight: float = 0.5,
    rc_consistency_weight: float = 0.1,
    workers: int = 2,
    device: str | None = None,
    amp: bool = False,
    codec_key_file: str | Path | None = None,
    seed: int = 42,
) -> dict[str, object]:
    np, torch, DataLoader, tqdm = _require_training_deps()
    from .image_torchdata import ImageGenomeDataset
    from .model_metrics import save_json
    from .models.hyenadna_locator import trainable_parameter_summary
    from .models.image_locator import image_locator_loss

    _set_seed(seed)
    if epochs <= 0 or batch_size <= 0 or gradient_accumulation <= 0:
        raise ValueError("epochs, batch_size, and gradient_accumulation must be positive.")
    if backbone not in {"cnn", "hyenadna"}:
        raise ValueError("backbone must be cnn or hyenadna.")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(dataset_dir)
    window_length = int(manifest["window_length"])
    train_dataset = ImageGenomeDataset(dataset_dir, "train")
    val_dataset = ImageGenomeDataset(dataset_dir, "val")
    test_dataset = ImageGenomeDataset(dataset_dir, "test")
    for split_dataset in (train_dataset, val_dataset, test_dataset):
        labels = {int(row["y"]) for row in split_dataset.rows}
        if labels != {0, 1}:
            raise ValueError(
                f"split={split_dataset.split!r} must contain both positive and negative samples; "
                f"found labels={sorted(labels)}."
            )
    train_loader = _make_loader(train_dataset, DataLoader, batch_size=batch_size, shuffle=True, workers=workers)
    val_loader = _make_loader(val_dataset, DataLoader, batch_size=batch_size, shuffle=False, workers=workers)
    if backbone == "cnn":
        model_type = "hierarchical_image_locator_v2"
        model_config = {
            "d_model": d_model,
            "downsample_stride": downsample_stride,
            "context": context,
            "context_layers": context_layers,
            "dropout": dropout,
            "max_bins": max(4096, math.ceil(window_length / downsample_stride) + 8),
        }
    else:
        model_type = "hyenadna_image_locator_v1"
        model_config = {
            "model_name": hyena_model_name,
            "model_revision": hyena_revision,
            "local_files_only": hyena_local_files_only,
            "downsample_stride": downsample_stride,
            "head_dim": d_model,
            "dropout": dropout,
            "reverse_complement": hyena_reverse_complement,
            "fine_tune_mode": hyena_fine_tune,
            "last_n_layers": hyena_last_n_layers,
            "pretrained": hyena_pretrained,
            "gradient_checkpointing": gradient_checkpointing,
        }
    run_device = _resolve_device(torch, device)
    model = _build_model(model_type, model_config).to(run_device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(amp and run_device.startswith("cuda"))
    best_state = None
    best_score = float("inf")
    logs = []
    component_names = (
        "presence_loss",
        "start_loss",
        "offset_loss",
        "end_loss",
        "end_offset_loss",
        "segment_loss",
        "rc_consistency_loss",
    )

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_losses = []
        component_totals = {name: [] for name in component_names}
        progress = tqdm(train_loader, desc=f"{backbone}-image-locator epoch {epoch}")
        for step, (tokens, labels, starts, ends, _indices) in enumerate(progress, start=1):
            tokens = tokens.to(run_device, non_blocking=True)
            labels = labels.to(run_device, non_blocking=True)
            starts = starts.to(run_device, non_blocking=True)
            ends = ends.to(run_device, non_blocking=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_enabled
                else nullcontext()
            )
            with autocast:
                prediction = model(tokens)
                losses = image_locator_loss(
                    prediction,
                    labels,
                    starts,
                    ends,
                    stride=downsample_stride,
                    presence_weight=presence_weight,
                    start_weight=start_weight,
                    offset_weight=offset_weight,
                    segment_weight=segment_weight,
                    rc_consistency_weight=rc_consistency_weight,
                )
                scaled_loss = losses["loss"] / gradient_accumulation
            scaled_loss.backward()
            if step % gradient_accumulation == 0 or step == len(train_loader):
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            value = float(losses["loss"].detach().item())
            epoch_losses.append(value)
            for name in component_totals:
                component_totals[name].append(float(losses[name].detach().item()))
            progress.set_postfix(loss=f"{value:.4f}")

        val_prediction = _predict(
            model,
            val_loader,
            device=run_device,
            stride=downsample_stride,
            window_length=window_length,
        )
        positive = val_prediction["y"] == 1
        presence_brier = float(np.mean((val_prediction["presence_prob"] - val_prediction["y"]) ** 2))
        boundary_mae = (
            float(
                np.mean(
                    np.concatenate(
                        [
                            np.abs(val_prediction["pred_start"][positive] - val_prediction["true_start"][positive]),
                            np.abs(val_prediction["pred_end"][positive] - val_prediction["true_end"][positive]),
                        ]
                    )
                )
                / window_length
            )
            if positive.any()
            else 0.0
        )
        exact_span_rate = (
            float(
                (
                    (val_prediction["pred_start"][positive] == val_prediction["true_start"][positive])
                    & (val_prediction["pred_end"][positive] == val_prediction["true_end"][positive])
                ).mean()
            )
            if positive.any()
            else 0.0
        )
        validation_score = presence_brier + boundary_mae + 0.25 * (1.0 - exact_span_rate)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            **{name: float(np.mean(values)) for name, values in component_totals.items()},
            "val_presence_brier": presence_brier,
            "val_normalized_boundary_mae": boundary_mae,
            "val_exact_span_rate": exact_span_rate,
            "val_selection_score": validation_score,
        }
        logs.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if validation_score < best_score:
            best_score = validation_score
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training completed without a model state.")
    checkpoint_path = output / "model.pt"
    parameter_summary = trainable_parameter_summary(model)
    torch.save(
        {
            "model_type": model_type,
            "state_dict": best_state,
            "model_config": model_config,
            "dataset_manifest": manifest,
            "best_validation_score": best_score,
            "parameter_summary": parameter_summary,
            "seed": seed,
        },
        checkpoint_path,
    )
    save_json(logs, output / "train_log.json")
    metrics = evaluate_image_locator_model(
        dataset_dir,
        checkpoint_path,
        output / "test_evaluation",
        split="test",
        batch_size=batch_size,
        workers=workers,
        device=run_device,
        save_images=True,
        codec_key_file=codec_key_file,
    )
    save_json(
        {
            "model_type": model_type,
            "best_validation_score": best_score,
            "parameter_summary": parameter_summary,
            "test_metrics": metrics,
        },
        output / "run_summary.json",
    )
    return metrics
