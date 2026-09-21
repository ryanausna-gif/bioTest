from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def gc_fraction(sequence: str) -> float:
    dna = str(sequence).upper()
    return (dna.count("G") + dna.count("C")) / len(dna) if dna else 0.0


def cpg_fraction(sequence: str) -> float:
    dna = str(sequence).upper()
    return sum(dna[index : index + 2] == "CG" for index in range(len(dna) - 1)) / max(
        1, len(dna) - 1
    )


def max_homopolymer(sequence: str) -> int:
    longest = current = 0
    previous = ""
    for base in str(sequence).upper():
        current = current + 1 if base == previous else 1
        longest = max(longest, current)
        previous = base
    return longest


def _kmer_distribution(sequence: str, k: int) -> dict[str, float]:
    dna = str(sequence).upper()
    counts = Counter(dna[index : index + k] for index in range(len(dna) - k + 1))
    total = sum(counts.values())
    return {key: value / total for key, value in counts.items()} if total else {}


def kmer_js_divergence(left: str, right: str, *, k: int = 3) -> float:
    p = _kmer_distribution(left, k)
    q = _kmer_distribution(right, k)
    keys = set(p) | set(q)
    value = 0.0
    for key in keys:
        pv = p.get(key, 0.0)
        qv = q.get(key, 0.0)
        midpoint = 0.5 * (pv + qv)
        if pv:
            value += 0.5 * pv * math.log2(pv / midpoint)
        if qv:
            value += 0.5 * qv * math.log2(qv / midpoint)
    return float(value)


def payload_stealth_metrics(
    encoded: str,
    natural_cover: str,
    *,
    source_bits: int,
    kmer_order: int = 3,
) -> dict[str, float | int]:
    dna = str(encoded).upper()
    cover = str(natural_cover).upper()[: len(dna)]
    compared = min(len(dna), len(cover))
    substitutions = sum(dna[index] != cover[index] for index in range(compared))
    return {
        "encoded_bases": len(dna),
        "source_bits": int(source_bits),
        "effective_bits_per_base": source_bits / len(dna) if dna else 0.0,
        "encoded_gc": gc_fraction(dna),
        "cover_gc": gc_fraction(cover),
        "gc_abs_delta": abs(gc_fraction(dna) - gc_fraction(cover)),
        "encoded_cpg": cpg_fraction(dna),
        "cover_cpg": cpg_fraction(cover),
        "cpg_abs_delta": abs(cpg_fraction(dna) - cpg_fraction(cover)),
        "encoded_max_homopolymer": max_homopolymer(dna),
        "cover_max_homopolymer": max_homopolymer(cover),
        "cover_change_fraction": substitutions / compared if compared else 0.0,
        f"kmer_{kmer_order}_jsd": kmer_js_divergence(dna, cover, k=kmer_order),
    }


def summarize_stego_dataset(dataset_dir: str | Path, out_dir: str | Path) -> dict[str, object]:
    dataset = Path(dataset_dir)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (dataset / "metadata.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["y"]) == 1]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["codec"]].append(row)
    metric_names = (
        "effective_bits_per_base",
        "gc_abs_delta",
        "cpg_abs_delta",
        "cover_change_fraction",
        "kmer_jsd",
        "encoded_max_homopolymer",
    )
    summaries = []
    for codec, codec_rows in sorted(groups.items()):
        summary: dict[str, object] = {"codec": codec, "n": len(codec_rows)}
        for name in metric_names:
            values = [float(row[name]) for row in codec_rows if row.get(name, "") != ""]
            summary[f"mean_{name}"] = sum(values) / len(values) if values else None
        summaries.append(summary)
    report = {"dataset": str(dataset.resolve()), "n_positive": len(rows), "by_codec": summaries}
    (output / "stego_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summaries:
        with (output / "stego_report.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    return report
