from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .fasta import acgt_fraction, iter_fasta_records
from .image_payload import RawImageDNASpec, preprocess_image
from .stego_codecs import (
    available_stego_codecs,
    codec_key_fingerprint,
    make_stego_codec,
    normalize_codec_name,
)
from .stego_metrics import payload_stealth_metrics


IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VALID_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class GenomeSource:
    donor_id: str
    assembly_id: str
    fasta: Path
    split: str


@dataclass(frozen=True)
class ImageSource:
    image_id: str
    path: Path
    class_id: str
    split: str = ""


@dataclass(frozen=True)
class GenomeWindow:
    donor_id: str
    assembly_id: str
    fasta: str
    chrom: str
    start: int
    end: int
    sequence: str


def read_genome_manifest(path: str | Path) -> list[GenomeSource]:
    manifest = Path(path)
    rows: list[GenomeSource] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            donor_id = str(row.get("donor_id", "")).strip()
            split = str(row.get("split", "")).strip().lower()
            fasta_value = str(row.get("fasta", row.get("fasta_path", ""))).strip()
            if not donor_id or not fasta_value or split not in VALID_SPLITS:
                raise ValueError("Genome manifest requires donor_id,fasta,split with split=train/val/test.")
            fasta = Path(fasta_value)
            if not fasta.is_absolute():
                fasta = (manifest.parent / fasta).resolve()
            if not fasta.exists():
                raise FileNotFoundError(f"Genome FASTA not found: {fasta}")
            rows.append(
                GenomeSource(
                    donor_id=donor_id,
                    assembly_id=str(row.get("assembly_id") or fasta.stem),
                    fasta=fasta,
                    split=split,
                )
            )
    _validate_donor_isolation(rows)
    return rows


def _validate_donor_isolation(sources: list[GenomeSource]) -> None:
    donor_splits: dict[str, set[str]] = {}
    fasta_splits: dict[str, set[str]] = {}
    for source in sources:
        donor_splits.setdefault(source.donor_id, set()).add(source.split)
        fasta_splits.setdefault(str(source.fasta.resolve()), set()).add(source.split)
    leaked_donors = sorted(donor for donor, splits in donor_splits.items() if len(splits) > 1)
    leaked_fastas = sorted(fasta for fasta, splits in fasta_splits.items() if len(splits) > 1)
    if leaked_donors:
        raise ValueError("Donor IDs occur in multiple splits: " + ", ".join(leaked_donors))
    if leaked_fastas:
        raise ValueError("Genome FASTA files occur in multiple splits: " + ", ".join(leaked_fastas))


def discover_images(image_root: str | Path) -> list[ImageSource]:
    root = Path(image_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        rows.append(
            ImageSource(
                image_id=relative.as_posix(),
                path=path,
                class_id=path.parent.name,
            )
        )
    if not rows:
        raise ValueError(f"No supported image files found under {root}")
    return rows


def read_image_manifest(path: str | Path) -> list[ImageSource]:
    manifest = Path(path)
    rows = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            value = str(row.get("path", row.get("image_path", ""))).strip()
            if not value:
                raise ValueError("Image manifest requires path or image_path.")
            image_path = Path(value)
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            split = str(row.get("split", "")).strip().lower()
            if split and split not in VALID_SPLITS:
                raise ValueError(f"Invalid image split: {split}")
            rows.append(
                ImageSource(
                    image_id=str(row.get("image_id") or f"image_{index:08d}"),
                    path=image_path,
                    class_id=str(row.get("class_id") or image_path.parent.name),
                    split=split,
                )
            )
    if len({row.image_id for row in rows}) != len(rows):
        raise ValueError("Image manifest contains duplicate image_id values.")
    return rows


def _positive_count(n_samples: int, positive_fraction: float) -> int:
    count = int(round(n_samples * positive_fraction))
    if n_samples >= 2 and 0.0 < positive_fraction < 1.0:
        count = min(n_samples - 1, max(1, count))
    return count


def allocate_images(
    images: list[ImageSource],
    positive_counts: dict[str, int],
    *,
    seed: int,
) -> dict[str, list[ImageSource]]:
    rng = random.Random(seed)
    explicit = any(image.split for image in images)
    allocated: dict[str, list[ImageSource]] = {}
    if explicit:
        for split in VALID_SPLITS:
            pool = [image for image in images if image.split == split]
            rng.shuffle(pool)
            if len(pool) < positive_counts[split]:
                raise ValueError(f"Image split {split} has {len(pool)} images; needs {positive_counts[split]}.")
            allocated[split] = pool[: positive_counts[split]]
    else:
        pool = list(images)
        rng.shuffle(pool)
        needed = sum(positive_counts.values())
        if len(pool) < needed:
            raise ValueError(f"Found {len(pool)} unique image paths; need {needed} positive images.")
        offset = 0
        for split in VALID_SPLITS:
            count = positive_counts[split]
            allocated[split] = pool[offset : offset + count]
            offset += count
    selected_ids = [image.image_id for split in VALID_SPLITS for image in allocated[split]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("The same image_id was allocated more than once.")
    return allocated


def _iter_windows(
    source: GenomeSource,
    *,
    window_length: int,
    min_acgt_fraction: float,
) -> Iterator[GenomeWindow]:
    for record in iter_fasta_records(source.fasta, min_length=window_length):
        for start in range(0, len(record.sequence) - window_length + 1, window_length):
            sequence = record.sequence[start : start + window_length]
            if acgt_fraction(sequence) < min_acgt_fraction:
                continue
            yield GenomeWindow(
                donor_id=source.donor_id,
                assembly_id=source.assembly_id,
                fasta=str(source.fasta),
                chrom=record.name,
                start=start,
                end=start + window_length,
                sequence=sequence,
            )


def _quota_by_source(sources: list[GenomeSource], total: int) -> list[int]:
    if not sources:
        return []
    base, remainder = divmod(total, len(sources))
    return [base + (1 if index < remainder else 0) for index in range(len(sources))]


def _selected_windows(
    sources: list[GenomeSource],
    total: int,
    *,
    window_length: int,
    min_acgt_fraction: float,
    seed: int,
) -> Iterable[GenomeWindow]:
    produced = 0
    for source_index, (source, quota) in enumerate(zip(sources, _quota_by_source(sources, total))):
        if quota == 0:
            continue
        source_rng = random.Random(seed + source_index * 1_000_003)
        reservoir: list[GenomeWindow] = []
        seen = 0
        for window in _iter_windows(
            source,
            window_length=window_length,
            min_acgt_fraction=min_acgt_fraction,
        ):
            seen += 1
            if len(reservoir) < quota:
                reservoir.append(window)
            else:
                replacement = source_rng.randrange(seen)
                if replacement < quota:
                    reservoir[replacement] = window
        if len(reservoir) < quota:
            raise ValueError(
                f"Donor {source.donor_id} provides {len(reservoir)} non-overlapping valid windows; "
                f"needs {quota}."
            )
        source_rng.shuffle(reservoir)
        for window in reservoir:
            produced += 1
            yield window
    if produced != total:
        raise ValueError(f"Generated {produced} genome windows; expected {total}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mutate_dna_payload(
    sequence: str,
    *,
    substitution_rate: float,
    indel_rate: float,
    rng: random.Random,
) -> tuple[str, int, int, int]:
    if not 0.0 <= substitution_rate <= 1.0 or not 0.0 <= indel_rate <= 1.0:
        raise ValueError("substitution_rate and indel_rate must be between zero and one.")
    output: list[str] = []
    substitutions = 0
    deleted_positions = {
        index for index in range(len(sequence)) if indel_rate and rng.random() < indel_rate / 2.0
    }
    for index, original in enumerate(sequence):
        if index in deleted_positions:
            continue
        base = original
        if substitution_rate and rng.random() < substitution_rate:
            base = rng.choice([candidate for candidate in "ACGT" if candidate != original])
            substitutions += 1
        output.append(base)
    # Pair every deletion with an insertion at another random coordinate. This
    # creates local frame shifts while preserving the fixed observation length.
    insertions_by_offset: dict[int, list[str]] = {}
    for _ in range(len(deleted_positions)):
        offset = rng.randint(0, len(output))
        insertions_by_offset.setdefault(offset, []).append(rng.choice("ACGT"))
    balanced: list[str] = []
    for offset in range(len(output) + 1):
        balanced.extend(insertions_by_offset.get(offset, ()))
        if offset < len(output):
            balanced.append(output[offset])
    return "".join(balanced), substitutions, len(deleted_positions), len(deleted_positions)


def build_image_genome_dataset(
    genome_manifest: str | Path,
    out_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    image_manifest: str | Path | None = None,
    n_train: int = 1000,
    n_val: int = 200,
    n_test: int = 300,
    positive_fraction: float = 0.5,
    image_size: int = 128,
    resize_short_side: int | None = None,
    window_length: int = 131072,
    min_acgt_fraction: float = 0.95,
    codecs: tuple[str, ...] | list[str] = ("direct",),
    train_codecs: tuple[str, ...] | list[str] | None = None,
    val_codecs: tuple[str, ...] | list[str] | None = None,
    test_codecs: tuple[str, ...] | list[str] | None = None,
    codec_key: bytes | str | None = None,
    kmer_order: int = 3,
    ecc_symbols: int = 0,
    lm_model_path: str | Path | None = None,
    lm_block_bytes: int = 8,
    lm_max_symbols_per_block: int = 256,
    lm_context_bases: int = 4096,
    lm_cdf_precision_bits: int = 12,
    lm_uniform_mix: float = 0.20,
    substitution_rate: float = 0.0,
    indel_rate: float = 0.0,
    seed: int = 42,
) -> dict[str, object]:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Image-genome dataset building requires numpy.") from exc

    if not 0.0 <= positive_fraction <= 1.0:
        raise ValueError("positive_fraction must be between zero and one.")
    spec = RawImageDNASpec(
        width=image_size,
        height=image_size,
        resize_short_side=resize_short_side,
    )
    if isinstance(codec_key, str):
        codec_key = codec_key.encode("utf-8")
    base_codecs = [normalize_codec_name(name) for name in codecs]
    if not base_codecs:
        raise ValueError("At least one stego codec is required.")
    codec_names_by_split = {
        "train": [normalize_codec_name(name) for name in (train_codecs or base_codecs)],
        "val": [normalize_codec_name(name) for name in (val_codecs or base_codecs)],
        "test": [normalize_codec_name(name) for name in (test_codecs or base_codecs)],
    }
    if any(not names for names in codec_names_by_split.values()):
        raise ValueError("Every split must have at least one configured codec.")
    codec_instances = {
        name: make_stego_codec(
            name,
            key=codec_key,
            kmer_order=kmer_order,
            ecc_symbols=ecc_symbols,
            lm_model_path=lm_model_path,
            lm_block_bytes=lm_block_bytes,
            lm_max_symbols_per_block=lm_max_symbols_per_block,
            lm_context_bases=lm_context_bases,
            lm_cdf_precision_bits=lm_cdf_precision_bits,
            lm_uniform_mix=lm_uniform_mix,
        )
        for name in sorted({name for names in codec_names_by_split.values() for name in names})
    }
    sample_counts = {"train": n_train, "val": n_val, "test": n_test}
    if any(count < 0 for count in sample_counts.values()) or sum(sample_counts.values()) == 0:
        raise ValueError("Split sample counts must be non-negative and contain at least one sample.")
    positive_counts = {
        split: _positive_count(sample_counts[split], positive_fraction) for split in VALID_SPLITS
    }
    genomes = read_genome_manifest(genome_manifest)
    genomes_by_split = {split: [row for row in genomes if row.split == split] for split in VALID_SPLITS}
    for split, count in sample_counts.items():
        if count and not genomes_by_split[split]:
            raise ValueError(f"No genome donor assigned to required split: {split}")

    if image_manifest is not None:
        images = read_image_manifest(image_manifest)
    elif image_root is not None:
        images = discover_images(image_root)
    else:
        raise ValueError("Provide image_root or image_manifest.")
    images_by_split = allocate_images(images, positive_counts, seed=seed)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    total_samples = sum(sample_counts.values())
    sequence_path = output / "sequences.npy"
    sequence_array = np.lib.format.open_memmap(
        sequence_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_samples, window_length),
    )
    token_lookup = np.full(256, 4, dtype=np.uint8)
    for base, value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
        token_lookup[ord(base)] = value
    metadata_rows: list[dict[str, object]] = []
    rng = random.Random(seed)
    array_index = 0
    seen_image_hashes: set[str] = set()
    seen_pixel_hashes: set[str] = set()
    max_payload_bases = 0

    for split_index, split in enumerate(VALID_SPLITS):
        n_samples = sample_counts[split]
        labels = [1] * positive_counts[split] + [0] * (n_samples - positive_counts[split])
        rng.shuffle(labels)
        image_iterator = iter(images_by_split[split])
        windows = _selected_windows(
            genomes_by_split[split],
            n_samples,
            window_length=window_length,
            min_acgt_fraction=min_acgt_fraction,
            seed=seed + split_index * 10_000_019,
        )
        positive_index = 0
        for sample_index, (label, window) in enumerate(zip(labels, windows)):
            payload_start = payload_end = -1
            image_id = image_path = class_id = image_hash = pixel_hash = ""
            codec_name = "none"
            codec_level = "none"
            embedding_mode = "none"
            encoded_bases_before_channel = 0
            channel_substitutions = channel_insertions = channel_deletions = 0
            stealth = {
                "effective_bits_per_base": "",
                "gc_abs_delta": "",
                "cpg_abs_delta": "",
                "cover_change_fraction": "",
                "kmer_jsd": "",
                "encoded_max_homopolymer": "",
            }
            if label:
                image = next(image_iterator)
                image_hash = _sha256_file(image.path)
                if image_hash in seen_image_hashes:
                    raise ValueError(f"Duplicate image content selected: {image.path}")
                seen_image_hashes.add(image_hash)
                pixels = preprocess_image(image.path, spec).tobytes()
                pixel_hash = hashlib.sha256(pixels).hexdigest()
                if pixel_hash in seen_pixel_hashes:
                    raise ValueError(
                        f"Duplicate preprocessed pixel content selected: {image.path}"
                    )
                seen_pixel_hashes.add(pixel_hash)
                codec_name = codec_names_by_split[split][positive_index % len(codec_names_by_split[split])]
                positive_index += 1
                codec = codec_instances[codec_name]
                maximum = codec.max_encoded_bases(len(pixels))
                if maximum > window_length:
                    raise ValueError(
                        f"Codec {codec_name} may need {maximum} bases for a {image_size}x{image_size} "
                        f"image, exceeding window_length={window_length}. Increase --window-length or "
                        "reduce --image-size."
                    )
                minimum_start = min(int(getattr(codec, "context_bases", 0)), window_length - maximum)
                payload_start = rng.randint(minimum_start, window_length - maximum)
                cover = window.sequence[payload_start : payload_start + maximum]
                context_bases = int(getattr(codec, "context_bases", 0))
                left_context = window.sequence[max(0, payload_start - context_bases) : payload_start]
                sample_id = f"{split}_{sample_index:08d}"
                pristine_payload = codec.encode(
                    pixels,
                    cover=cover,
                    sample_id=sample_id,
                    context=left_context,
                )
                encoded_bases_before_channel = len(pristine_payload)
                natural_cover = cover[:encoded_bases_before_channel]
                stealth_values = payload_stealth_metrics(
                    pristine_payload,
                    natural_cover,
                    source_bits=len(pixels) * 8,
                    kmer_order=kmer_order,
                )
                stealth = {
                    "effective_bits_per_base": stealth_values["effective_bits_per_base"],
                    "gc_abs_delta": stealth_values["gc_abs_delta"],
                    "cpg_abs_delta": stealth_values["cpg_abs_delta"],
                    "cover_change_fraction": stealth_values["cover_change_fraction"],
                    "kmer_jsd": stealth_values[f"kmer_{kmer_order}_jsd"],
                    "encoded_max_homopolymer": stealth_values["encoded_max_homopolymer"],
                }
                payload_dna, channel_substitutions, channel_insertions, channel_deletions = (
                    _mutate_dna_payload(
                        pristine_payload,
                        substitution_rate=substitution_rate,
                        indel_rate=indel_rate,
                        rng=rng,
                    )
                )
                natural_length = window_length - len(payload_dna)
                payload_end = payload_start + len(payload_dna)
                embedding_mode = codec.embedding_mode
                if embedding_mode == "replacement":
                    sequence = (
                        window.sequence[:payload_start]
                        + payload_dna
                        + window.sequence[payload_end:]
                    )
                else:
                    sequence = (
                        window.sequence[:payload_start]
                        + payload_dna
                        + window.sequence[payload_start:natural_length]
                    )
                image_id = image.image_id
                image_path = str(image.path)
                class_id = image.class_id
                codec_level = codec.description.level
                max_payload_bases = max(max_payload_bases, len(payload_dna))
            else:
                sequence = window.sequence
            if len(sequence) != window_length:
                raise AssertionError("Constructed sequence length does not match window_length.")
            ascii_sequence = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
            sequence_array[array_index, :] = token_lookup[ascii_sequence]
            metadata_rows.append(
                {
                    "array_index": array_index,
                    "sample_id": f"{split}_{sample_index:08d}",
                    "split": split,
                    "y": label,
                    "donor_id": window.donor_id,
                    "assembly_id": window.assembly_id,
                    "fasta": window.fasta,
                    "chrom": window.chrom,
                    "source_start": window.start,
                    "source_end": window.end,
                    "sequence_length": window_length,
                    "payload_start": payload_start,
                    "payload_end": payload_end,
                    "payload_bases": payload_end - payload_start if label else 0,
                    "encoded_bases_before_channel": encoded_bases_before_channel,
                    "image_id": image_id,
                    "image_path": image_path,
                    "image_sha256": image_hash,
                    "pixel_sha256": pixel_hash,
                    "class_id": class_id,
                    "codec": codec_name,
                    "codec_level": codec_level,
                    "is_unknown": int(bool(label) and codec_name not in codec_names_by_split["train"]),
                    "open_target": (
                        "natural"
                        if not label
                        else (
                            "unknown_anomaly"
                            if codec_name not in codec_names_by_split["train"]
                            else "known_special"
                        )
                    ),
                    "carrier_type": "natural" if not label else codec_name,
                    "embedding_mode": embedding_mode,
                    "channel_substitutions": channel_substitutions,
                    "channel_insertions": channel_insertions,
                    "channel_deletions": channel_deletions,
                    **stealth,
                    "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                }
            )
            array_index += 1
    sequence_array.flush()

    metadata_path = output / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)
    lm_runtime = [
        codec.runtime_metadata()
        for codec in codec_instances.values()
        if hasattr(codec, "runtime_metadata")
    ]
    manifest = {
        "name": "Image-in-genome localization MVP",
        "task": "Open-codec image-carrier detection, variable-span localization, and reversible recovery",
        "genome_manifest": str(Path(genome_manifest).resolve()),
        "image_root": None if image_root is None else str(Path(image_root).resolve()),
        "image_manifest": None if image_manifest is None else str(Path(image_manifest).resolve()),
        "sequence_file": str(sequence_path.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "sample_counts": sample_counts,
        "positive_counts": positive_counts,
        "positive_fraction": positive_fraction,
        "window_length": window_length,
        "min_acgt_fraction": min_acgt_fraction,
        "image_spec": {**spec.to_dict(), "codec_name": "variable_stego_pipeline_v2"},
        "codecs_by_split": codec_names_by_split,
        "codec_catalog": available_stego_codecs(),
        "codec_key_fingerprint": codec_key_fingerprint(codec_key),
        "kmer_order": kmer_order,
        "lm_stego": {
            "enabled": bool(lm_runtime),
            "model_file": None if lm_model_path is None else str(Path(lm_model_path).resolve()),
            "block_bytes": lm_block_bytes,
            "max_symbols_per_block": lm_max_symbols_per_block,
            "context_bases": lm_context_bases,
            "cdf_precision_bits": lm_cdf_precision_bits,
            "uniform_mix": lm_uniform_mix,
            "runtime": lm_runtime,
        },
        "ecc": {
            "mode": "reed_solomon" if ecc_symbols else "none",
            "symbols_per_255_base_byte_block": ecc_symbols,
            "note": "Byte-level RS targets substitutions/erasures; it does not resynchronize DNA indels.",
        },
        "max_payload_bases": max_payload_bases,
        "channel": {
            "substitution_rate": substitution_rate,
            "indel_rate": indel_rate,
            "scope": "payload_only",
            "indel_protocol": "paired deletion/insertion with zero net payload-length change",
        },
        "donors_by_split": {
            split: sorted({source.donor_id for source in genomes_by_split[split]}) for split in VALID_SPLITS
        },
        "seed": seed,
        "leakage_controls": {
            "donor_disjoint": True,
            "fasta_disjoint": True,
            "image_id_unique": True,
            "image_content_hash_unique": True,
            "preprocessed_pixel_hash_unique": True,
            "genome_windows_non_overlapping_within_source": True,
            "genome_windows_sampled_across_assembly": True,
        },
        "limitations": [
            "One-channel 8-bit square images and a fixed observation length are retained.",
            "The encrypted codecs need the authorized key for recovery; the key is never stored in the dataset.",
            "Reed-Solomon is optional; HEDGES-style DNA indel synchronization is not implemented.",
            "lm_arithmetic_v1 is an exact no-noise L5 protocol prototype; substitution/indel resynchronization remains future work.",
        ],
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def read_dataset_metadata(dataset_dir: str | Path) -> list[dict[str, str]]:
    path = Path(dataset_dir) / "metadata.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def audit_image_genome_dataset(dataset_dir: str | Path) -> dict[str, object]:
    dataset = Path(dataset_dir)
    rows = read_dataset_metadata(dataset)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    donor_splits: dict[str, set[str]] = {}
    image_splits: dict[str, set[str]] = {}
    image_hash_splits: dict[str, set[str]] = {}
    pixel_hash_splits: dict[str, set[str]] = {}
    image_id_counts: dict[str, int] = {}
    image_hash_counts: dict[str, int] = {}
    pixel_hash_counts: dict[str, int] = {}
    sequence_hashes: dict[str, set[str]] = {}
    intervals: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    codec_counts: dict[str, dict[str, int]] = {split: {} for split in VALID_SPLITS}
    invalid_payload_spans = 0
    for row in rows:
        split = row["split"]
        donor_splits.setdefault(row["donor_id"], set()).add(split)
        if row["image_id"]:
            image_splits.setdefault(row["image_id"], set()).add(split)
            image_id_counts[row["image_id"]] = image_id_counts.get(row["image_id"], 0) + 1
        if row["image_sha256"]:
            image_hash_splits.setdefault(row["image_sha256"], set()).add(split)
            image_hash_counts[row["image_sha256"]] = image_hash_counts.get(row["image_sha256"], 0) + 1
        if row.get("pixel_sha256"):
            pixel_hash = row["pixel_sha256"]
            pixel_hash_splits.setdefault(pixel_hash, set()).add(split)
            pixel_hash_counts[pixel_hash] = pixel_hash_counts.get(pixel_hash, 0) + 1
        sequence_hashes.setdefault(row["sequence_sha256"], set()).add(split)
        key = (row["donor_id"], row["assembly_id"], row["chrom"])
        intervals.setdefault(key, []).append((int(row["source_start"]), int(row["source_end"])))
        codec = row.get("codec", "none")
        codec_counts[split][codec] = codec_counts[split].get(codec, 0) + 1
        if int(row["y"]):
            start = int(row["payload_start"])
            end = int(row["payload_end"])
            length = int(row["payload_bases"])
            if start < 0 or end <= start or end - start != length or end > int(row["sequence_length"]):
                invalid_payload_spans += 1
    overlapping_intervals = 0
    for values in intervals.values():
        values.sort()
        overlapping_intervals += sum(left[1] > right[0] for left, right in zip(values, values[1:]))
    actual_sample_counts = {
        split: sum(row["split"] == split for row in rows) for split in VALID_SPLITS
    }
    count_mismatches = {
        split: {"expected": int(manifest["sample_counts"][split]), "actual": actual_sample_counts[split]}
        for split in VALID_SPLITS
        if int(manifest["sample_counts"][split]) != actual_sample_counts[split]
    }
    report = {
        "n_samples": len(rows),
        "sample_counts": actual_sample_counts,
        "positive_counts": {
            split: sum(row["split"] == split and int(row["y"]) == 1 for row in rows)
            for split in VALID_SPLITS
        },
        "donor_leakage": sorted(donor for donor, splits in donor_splits.items() if len(splits) > 1),
        "image_leakage": sorted(image for image, splits in image_splits.items() if len(splits) > 1),
        "image_hash_leakage": sorted(
            image_hash for image_hash, splits in image_hash_splits.items() if len(splits) > 1
        ),
        "pixel_hash_leakage": sorted(
            pixel_hash for pixel_hash, splits in pixel_hash_splits.items() if len(splits) > 1
        ),
        "reused_image_ids": sorted(image for image, count in image_id_counts.items() if count > 1),
        "reused_image_hashes": sorted(
            image_hash for image_hash, count in image_hash_counts.items() if count > 1
        ),
        "reused_pixel_hashes": sorted(
            pixel_hash for pixel_hash, count in pixel_hash_counts.items() if count > 1
        ),
        "cross_split_sequence_duplicates": sum(len(splits) > 1 for splits in sequence_hashes.values()),
        "overlapping_source_intervals": overlapping_intervals,
        "sample_count_mismatches": count_mismatches,
        "codec_counts": codec_counts,
        "invalid_payload_spans": invalid_payload_spans,
        "sequence_file_exists": (dataset / "sequences.npy").exists(),
        "image_spec": manifest["image_spec"],
        "passed": False,
    }
    report["passed"] = not any(
        [
            report["donor_leakage"],
            report["image_leakage"],
            report["image_hash_leakage"],
            report["pixel_hash_leakage"],
            report["reused_image_ids"],
            report["reused_image_hashes"],
            report["reused_pixel_hashes"],
            report["cross_split_sequence_duplicates"],
            report["overlapping_source_intervals"],
            report["sample_count_mismatches"],
            report["invalid_payload_spans"],
            not report["sequence_file_exists"],
        ]
    )
    return report
