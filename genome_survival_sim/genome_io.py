from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .genome import Interval


GENE_TYPES = {"gene", "pseudogene", "ncRNA_gene", "lnc_RNA", "miRNA_gene", "rRNA_gene", "snRNA_gene", "snoRNA_gene"}
EXON_TYPES = {"exon", "CDS", "five_prime_UTR", "three_prime_UTR", "UTR"}


@dataclass(frozen=True)
class Feature:
    chrom: str
    source: str
    type: str
    start: int
    end: int
    strand: str
    attrs: dict[str, str]

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def load_chrom_sizes_dict(path: str | Path) -> dict[str, int]:
    """Read a UCSC chrom.sizes file or a FASTA .fai file.

    Both formats begin with chromosome name and chromosome length, so the same
    parser handles either.
    """

    sizes: dict[str, int] = {}
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            sizes[parts[0]] = int(parts[1])
    if not sizes:
        raise ValueError(f"No chromosome sizes found in {source}")
    return sizes


def load_gff_features(path: str | Path, include_types: set[str] | None = None) -> list[Feature]:
    """Load GFF3/GTF features using 0-based half-open coordinates."""

    features: list[Feature] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) != 9:
                continue
            feature_type = parts[2]
            if include_types is not None and feature_type not in include_types:
                continue
            start = int(parts[3]) - 1
            end = int(parts[4])
            if end <= start:
                continue
            features.append(
                Feature(
                    chrom=parts[0],
                    source=parts[1],
                    type=feature_type,
                    start=start,
                    end=end,
                    strand=parts[6],
                    attrs=parse_attributes(parts[8]),
                )
            )
    return features


def parse_attributes(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if "=" in raw:
        for item in raw.split(";"):
            if not item.strip() or "=" not in item:
                continue
            key, value = item.split("=", 1)
            attrs[key.strip()] = value.strip().strip('"')
    else:
        for item in raw.split(";"):
            item = item.strip()
            if not item or " " not in item:
                continue
            key, value = item.split(" ", 1)
            attrs[key.strip()] = value.strip().strip('"')
    return attrs


def merge_intervals(intervals: list[Interval], label: str | None = None) -> list[Interval]:
    by_chrom: dict[str, list[Interval]] = defaultdict(list)
    for interval in intervals:
        if interval.end > interval.start:
            by_chrom[interval.chrom].append(interval)

    merged: list[Interval] = []
    for chrom, chrom_intervals in by_chrom.items():
        chrom_intervals.sort(key=lambda item: (item.start, item.end))
        current = chrom_intervals[0]
        for interval in chrom_intervals[1:]:
            if interval.start <= current.end:
                current = Interval(
                    chrom=current.chrom,
                    start=current.start,
                    end=max(current.end, interval.end),
                    label=label or current.label,
                )
            else:
                merged.append(Interval(current.chrom, current.start, current.end, label or current.label))
                current = interval
        merged.append(Interval(current.chrom, current.start, current.end, label or current.label))
    return merged


def subtract_intervals(base: list[Interval], masks: list[Interval], label: str) -> list[Interval]:
    masks_by_chrom: dict[str, list[Interval]] = defaultdict(list)
    for mask in merge_intervals(masks):
        masks_by_chrom[mask.chrom].append(mask)

    output: list[Interval] = []
    for interval in base:
        cursor = interval.start
        for mask in masks_by_chrom.get(interval.chrom, []):
            if mask.end <= cursor:
                continue
            if mask.start >= interval.end:
                break
            if mask.start > cursor:
                output.append(Interval(interval.chrom, cursor, min(mask.start, interval.end), label))
            cursor = max(cursor, mask.end)
            if cursor >= interval.end:
                break
        if cursor < interval.end:
            output.append(Interval(interval.chrom, cursor, interval.end, label))
    return [item for item in output if item.end > item.start]


def build_candidate_intervals(
    chrom_sizes: dict[str, int],
    features: list[Feature],
    mode: str = "intergenic",
    min_length: int = 200,
    flank: int = 1_000,
) -> list[Interval]:
    """Build simple candidate intervals from annotations.

    This is intentionally conservative and approximate. It is meant to produce
    computational screening regions, not wet-lab-ready safe-harbor calls.
    """

    normalized = mode.lower().replace("-", "_")
    whole_genome = [Interval(chrom, 0, length, "genome") for chrom, length in chrom_sizes.items()]
    gene_like = [
        Interval(item.chrom, max(0, item.start - flank), min(chrom_sizes.get(item.chrom, item.end), item.end + flank), "gene_mask")
        for item in features
        if item.type in GENE_TYPES and item.chrom in chrom_sizes
    ]
    exon_like = [
        Interval(item.chrom, item.start, item.end, "exon_mask")
        for item in features
        if item.type in EXON_TYPES and item.chrom in chrom_sizes
    ]

    if normalized in {"intergenic", "noncoding", "random_noncoding"}:
        candidates = subtract_intervals(whole_genome, merge_intervals(gene_like), "intergenic")
    elif normalized in {"intronic", "intron"}:
        genes = [
            Interval(item.chrom, item.start, item.end, "gene")
            for item in features
            if item.type in GENE_TYPES and item.type != "pseudogene" and item.chrom in chrom_sizes
        ]
        candidates = subtract_intervals(merge_intervals(genes), merge_intervals(exon_like), "intronic")
    elif normalized in {"pseudogene", "pseudogene_like"}:
        candidates = [
            Interval(item.chrom, item.start, item.end, "pseudogene")
            for item in features
            if ("pseudogene" in item.type.lower() or "pseudogene" in " ".join(item.attrs.values()).lower())
            and item.chrom in chrom_sizes
        ]
    else:
        raise ValueError(f"Unsupported candidate mode: {mode}")

    return [item for item in merge_intervals(candidates, normalized) if item.length >= min_length]


def write_bed(intervals: list[Interval], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        for interval in intervals:
            handle.write(f"{interval.chrom}\t{interval.start}\t{interval.end}\t{interval.label}\n")
