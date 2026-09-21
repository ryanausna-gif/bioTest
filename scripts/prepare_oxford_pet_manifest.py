from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PetImage:
    image_id: str
    path: Path
    class_id: str
    class_name: str
    species_id: str
    breed_id: str
    source_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_name(image_id: str) -> str:
    head, separator, tail = image_id.rpartition("_")
    return head if separator and tail.isdigit() else image_id


def read_oxford_pet_annotations(
    images_dir: str | Path,
    annotations_file: str | Path,
    *,
    verify_images: bool = False,
) -> tuple[list[PetImage], list[dict[str, str]]]:
    image_root = Path(images_dir).resolve()
    annotations = Path(annotations_file).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Oxford Pets image directory not found: {image_root}")
    if not annotations.is_file():
        raise FileNotFoundError(f"Oxford Pets annotation list not found: {annotations}")

    if verify_images:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("--verify-images requires Pillow.") from exc

    rows: list[PetImage] = []
    skipped_duplicates: list[dict[str, str]] = []
    hashes: dict[str, Path] = {}
    image_ids: set[str] = set()
    for line_number, raw_line in enumerate(
        annotations.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"Malformed Oxford Pets annotation at line {line_number}: {raw_line}")
        image_id, class_id, species_id, breed_id = fields[:4]
        if image_id in image_ids:
            raise ValueError(f"Duplicate image ID in annotations: {image_id}")
        image_ids.add(image_id)
        image_path = image_root / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Annotated image is missing: {image_path}")
        if verify_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"Unreadable image: {image_path}: {exc}") from exc
        source_hash = _sha256_file(image_path)
        if source_hash in hashes:
            skipped_duplicates.append(
                {
                    "duplicate": str(image_path),
                    "kept": str(hashes[source_hash]),
                    "source_sha256": source_hash,
                }
            )
            continue
        hashes[source_hash] = image_path
        rows.append(
            PetImage(
                image_id=image_id,
                path=image_path,
                class_id=class_id,
                class_name=_class_name(image_id),
                species_id=species_id,
                breed_id=breed_id,
                source_sha256=source_hash,
            )
        )
    if not rows:
        raise ValueError("No Oxford Pets images were loaded from the annotation list.")
    return rows, skipped_duplicates


def stratified_order(images: list[PetImage], *, seed: int) -> list[PetImage]:
    groups: dict[str, list[PetImage]] = defaultdict(list)
    for image in images:
        groups[image.class_id].append(image)
    for class_id, values in groups.items():
        random.Random(f"{seed}:{class_id}").shuffle(values)

    rng = random.Random(seed)
    active = sorted(groups)
    ordered: list[PetImage] = []
    offsets = {class_id: 0 for class_id in groups}
    while active:
        cycle = list(active)
        rng.shuffle(cycle)
        next_active = []
        for class_id in cycle:
            offset = offsets[class_id]
            values = groups[class_id]
            if offset < len(values):
                ordered.append(values[offset])
                offsets[class_id] = offset + 1
            if offsets[class_id] < len(values):
                next_active.append(class_id)
        active = sorted(next_active)
    return ordered


def prepare_manifest(
    images_dir: str | Path,
    annotations_file: str | Path,
    out_path: str | Path,
    *,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    verify_images: bool = False,
) -> dict[str, object]:
    counts = {"train": int(n_train), "val": int(n_val), "test": int(n_test)}
    if any(value < 0 for value in counts.values()) or sum(counts.values()) <= 0:
        raise ValueError("Split counts must be non-negative and contain at least one image.")
    images, skipped_duplicates = read_oxford_pet_annotations(
        images_dir,
        annotations_file,
        verify_images=verify_images,
    )
    required = sum(counts.values())
    if len(images) < required:
        raise ValueError(f"Found {len(images)} unique images; requested {required}.")

    ordered = stratified_order(images, seed=seed)
    assignments: list[tuple[str, PetImage]] = []
    offset = 0
    for split in ("train", "val", "test"):
        split_images = ordered[offset : offset + counts[split]]
        assignments.extend((split, image) for image in split_images)
        offset += counts[split]

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "path",
        "class_id",
        "class_name",
        "species_id",
        "breed_id",
        "split",
        "source_sha256",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split, image in assignments:
            writer.writerow(
                {
                    "image_id": image.image_id,
                    "path": str(image.path),
                    "class_id": image.class_id,
                    "class_name": image.class_name,
                    "species_id": image.species_id,
                    "breed_id": image.breed_id,
                    "split": split,
                    "source_sha256": image.source_sha256,
                }
            )

    class_counts = {
        split: dict(sorted(Counter(image.class_id for assigned, image in assignments if assigned == split).items()))
        for split in ("train", "val", "test")
    }
    report = {
        "dataset": "Oxford-IIIT Pet",
        "manifest": str(output.resolve()),
        "images_dir": str(Path(images_dir).resolve()),
        "annotations_file": str(Path(annotations_file).resolve()),
        "available_unique_images": len(images),
        "selected_images": len(assignments),
        "split_counts": counts,
        "class_counts": class_counts,
        "n_classes": len({image.class_id for _, image in assignments}),
        "duplicate_files_skipped": skipped_duplicates,
        "seed": seed,
        "manifest_sha256": _sha256_file(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, class-stratified Oxford-IIIT Pet image manifest."
    )
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=750)
    parser.add_argument("--n-test", type=int, default=1250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    report = prepare_manifest(
        args.images_dir,
        args.annotations,
        args.out,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        verify_images=args.verify_images,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
