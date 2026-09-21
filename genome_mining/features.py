from __future__ import annotations

import math
import zlib
from collections import Counter

from .records import SequenceWindow

BASES = "ACGT"


def _base_counts(sequence: str) -> Counter[str]:
    return Counter(base for base in sequence.upper() if base in BASES)


def shannon_entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def gc_content(sequence: str) -> float:
    counts = _base_counts(sequence)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return (counts["G"] + counts["C"]) / total


def longest_homopolymer(sequence: str) -> int:
    best = 0
    current = 0
    previous = ""
    for base in sequence.upper():
        if base not in BASES:
            current = 0
            previous = ""
            continue
        if base == previous:
            current += 1
        else:
            current = 1
            previous = base
        best = max(best, current)
    return best


def kmer_max_fraction(sequence: str, k: int) -> float:
    cleaned = "".join(base for base in sequence.upper() if base in BASES)
    if len(cleaned) < k:
        return 0.0
    counts = Counter(cleaned[i : i + k] for i in range(0, len(cleaned) - k + 1))
    return max(counts.values()) / sum(counts.values())


def kmer_entropy(sequence: str, k: int) -> float:
    cleaned = "".join(base for base in sequence.upper() if base in BASES)
    if len(cleaned) < k:
        return 0.0
    counts = Counter(cleaned[i : i + k] for i in range(0, len(cleaned) - k + 1))
    return shannon_entropy_from_counts(counts)


def compression_ratio(sequence: str) -> float:
    if not sequence:
        return 0.0
    compressed = zlib.compress(sequence.upper().encode("ascii", errors="ignore"), level=9)
    return len(compressed) / max(1, len(sequence))


def periodicity_score(sequence: str, max_period: int = 12) -> float:
    cleaned = "".join(base for base in sequence.upper() if base in BASES)
    if len(cleaned) < 4:
        return 0.0
    best = 0.0
    for period in range(1, min(max_period, len(cleaned) // 2) + 1):
        matches = sum(1 for idx in range(period, len(cleaned)) if cleaned[idx] == cleaned[idx - period])
        best = max(best, matches / (len(cleaned) - period))
    return best


def compute_sequence_features(sequence: str) -> dict[str, float | int]:
    upper = sequence.upper()
    counts = _base_counts(upper)
    acgt_total = sum(counts.values())
    length = len(upper)
    n_count = sum(1 for base in upper if base not in BASES)
    hp = longest_homopolymer(upper)
    entropy = shannon_entropy_from_counts(counts)
    return {
        "length": length,
        "acgt_bases": acgt_total,
        "n_fraction": n_count / length if length else 0.0,
        "gc_content": gc_content(upper),
        "entropy": entropy,
        "entropy_norm": entropy / 2.0 if entropy else 0.0,
        "dinuc_entropy": kmer_entropy(upper, 2),
        "trinuc_entropy": kmer_entropy(upper, 3),
        "compression_ratio": compression_ratio(upper),
        "longest_homopolymer": hp,
        "homopolymer_fraction": hp / length if length else 0.0,
        "kmer3_max_fraction": kmer_max_fraction(upper, 3),
        "periodicity_score": periodicity_score(upper),
    }


def compute_window_features(window: SequenceWindow, *, include_sequence: bool = False) -> dict[str, object]:
    row: dict[str, object] = {
        "window_id": window.window_id,
        "chrom": window.chrom,
        "start": window.start,
        "end": window.end,
        "strand": window.strand,
        "source": window.source,
        "species": window.species,
    }
    if include_sequence:
        row["sequence"] = window.sequence
    row.update(compute_sequence_features(window.sequence))
    return row
