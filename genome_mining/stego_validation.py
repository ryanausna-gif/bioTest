from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .image_dataset import read_dataset_metadata
from .image_payload import RawImageDNASpec, compare_pixel_bytes, preprocess_image, token_values_to_dna
from .stego_codecs import codec_key_fingerprint, make_stego_codec


def _image_spec(manifest: dict[str, object]) -> RawImageDNASpec:
    values = manifest["image_spec"]
    resize_short_side = values.get("resize_short_side")
    return RawImageDNASpec(
        width=int(values["width"]),
        height=int(values["height"]),
        resize_short_side=None if resize_short_side is None else int(resize_short_side),
    )


def verify_stego_roundtrip_dataset(
    dataset_dir: str | Path,
    out_dir: str | Path,
    *,
    codec_key: bytes | None = None,
    split: str = "all",
    max_samples: int | None = None,
) -> dict[str, object]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Stego round-trip validation requires numpy.") from exc

    if split not in {"all", "train", "val", "test"}:
        raise ValueError("split must be all, train, val, or test.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")

    dataset = Path(dataset_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in read_dataset_metadata(dataset)
        if int(row["y"]) == 1 and (split == "all" or row["split"] == split)
    ]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No positive stego samples found for split={split}.")

    expected_fingerprint = str(manifest.get("codec_key_fingerprint", ""))
    if expected_fingerprint:
        actual_fingerprint = codec_key_fingerprint(codec_key)
        if actual_fingerprint != expected_fingerprint:
            raise ValueError(
                f"Codec key fingerprint mismatch: dataset={expected_fingerprint}, "
                f"provided={actual_fingerprint or '<none>'}."
            )

    sequences = np.load(dataset / "sequences.npy", mmap_mode="r")
    spec = _image_spec(manifest)
    expected_bytes = spec.payload_bytes
    ecc_symbols = int(manifest.get("ecc", {}).get("symbols_per_255_base_byte_block", 0))
    lm_config = manifest.get("lm_stego", {})
    codec_cache = {}
    failures: list[dict[str, object]] = []
    summaries: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "n_samples": 0,
            "decode_successes": 0,
            "exact_images": 0,
            "pixel_exact_fraction_sum": 0.0,
        }
    )

    for row in rows:
        codec_name = row["codec"]
        summary = summaries[codec_name]
        summary["n_samples"] += 1
        start = int(row["payload_start"])
        end = int(row["payload_end"])
        array_index = int(row["array_index"])
        tokens = np.asarray(sequences[array_index])
        dna = token_values_to_dna(tokens[start:end])
        context_bases = int(lm_config.get("context_bases", 0))
        left_context = token_values_to_dna(tokens[max(0, start - context_bases) : start])
        try:
            if codec_name not in codec_cache:
                codec_cache[codec_name] = make_stego_codec(
                    codec_name,
                    key=codec_key,
                    kmer_order=int(manifest.get("kmer_order", 3)),
                    ecc_symbols=ecc_symbols,
                    lm_model_path=lm_config.get("model_file"),
                    lm_block_bytes=int(lm_config.get("block_bytes", 8)),
                    lm_max_symbols_per_block=int(lm_config.get("max_symbols_per_block", 256)),
                    lm_context_bases=context_bases or 4096,
                    lm_cdf_precision_bits=int(lm_config.get("cdf_precision_bits", 12)),
                    lm_uniform_mix=float(lm_config.get("uniform_mix", 0.20)),
                )
            recovered = codec_cache[codec_name].decode(
                dna,
                expected_bytes=expected_bytes,
                sample_id=row["sample_id"],
                context=left_context,
            )
            reference = preprocess_image(row["image_path"], spec).tobytes()
            reference_hash = hashlib.sha256(reference).hexdigest()
            expected_pixel_hash = row.get("pixel_sha256", "")
            if expected_pixel_hash and reference_hash != expected_pixel_hash:
                raise ValueError("Preprocessed image hash no longer matches dataset metadata.")
            pixel_metrics = compare_pixel_bytes(reference, recovered)
            summary["decode_successes"] += 1
            summary["exact_images"] += int(bool(pixel_metrics["exact"]))
            summary["pixel_exact_fraction_sum"] += float(pixel_metrics["pixel_exact_fraction"])
            if not pixel_metrics["exact"]:
                failures.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "codec": codec_name,
                        "error": "Decoded image bytes are not exact.",
                        **pixel_metrics,
                    }
                )
        except (ImportError, ValueError) as exc:
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "split": row["split"],
                    "codec": codec_name,
                    "error": str(exc),
                }
            )

    by_codec = {}
    for codec_name, values in sorted(summaries.items()):
        count = int(values["n_samples"])
        by_codec[codec_name] = {
            "n_samples": count,
            "decode_success_rate": int(values["decode_successes"]) / count,
            "exact_image_rate": int(values["exact_images"]) / count,
            "mean_pixel_exact_fraction": float(values["pixel_exact_fraction_sum"]) / count,
        }
    report = {
        "dataset": str(dataset.resolve()),
        "split": split,
        "n_checked": len(rows),
        "n_failures": len(failures),
        "passed": not failures,
        "by_codec": by_codec,
        "note": "Uses oracle payload boundaries; this validates codecs and data, not model localization.",
    }
    (output / "roundtrip_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        fieldnames = sorted({key for row in failures for key in row})
        with (output / "roundtrip_failures.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(failures)
    return report
