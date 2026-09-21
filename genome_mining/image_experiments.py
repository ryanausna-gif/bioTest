from __future__ import annotations

import json
from pathlib import Path


def run_hyenadna_comparison(
    dataset_dir: str | Path,
    out_root: str | Path,
    *,
    epochs: int = 10,
    batch_size: int = 1,
    lr: float = 2.0e-4,
    d_model: int = 128,
    downsample_stride: int = 256,
    hyena_model_name: str = "LongSafari/hyenadna-medium-160k-seqlen-hf",
    hyena_revision: str | None = None,
    hyena_local_files_only: bool = False,
    workers: int = 2,
    device: str | None = None,
    amp: bool = True,
    gradient_accumulation: int = 8,
    codec_key_file: str | Path | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> list[dict[str, object]]:
    """Run the minimum comparison needed to attribute gains to HyenaDNA pretraining and RC fusion."""
    from .image_training import train_image_locator

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    variants = [
        {"name": "cnn", "backbone": "cnn"},
        {
            "name": "hyena_random_forward",
            "backbone": "hyenadna",
            "hyena_pretrained": False,
            "hyena_reverse_complement": False,
        },
        {
            "name": "hyena_pretrained_forward",
            "backbone": "hyenadna",
            "hyena_pretrained": True,
            "hyena_reverse_complement": False,
        },
        {
            "name": "hyena_pretrained_rc",
            "backbone": "hyenadna",
            "hyena_pretrained": True,
            "hyena_reverse_complement": True,
        },
    ]
    results = []
    for variant in variants:
        record = {**variant, "out": str((root / str(variant["name"])).resolve())}
        if dry_run:
            results.append(record)
            continue
        metrics = train_image_locator(
            dataset_dir,
            root / str(variant["name"]),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            backbone=str(variant["backbone"]),
            d_model=d_model,
            downsample_stride=downsample_stride,
            hyena_fine_tune="frozen",
            hyena_model_name=hyena_model_name,
            hyena_revision=hyena_revision,
            hyena_local_files_only=hyena_local_files_only,
            hyena_pretrained=bool(variant.get("hyena_pretrained", True)),
            hyena_reverse_complement=bool(variant.get("hyena_reverse_complement", True)),
            gradient_accumulation=gradient_accumulation,
            workers=workers,
            device=device,
            amp=amp,
            codec_key_file=codec_key_file,
            seed=seed,
        )
        results.append({**record, "test_metrics": metrics})
    (root / "comparison_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return results
