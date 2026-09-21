from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .fasta import acgt_fraction, iter_fasta_records


@dataclass(frozen=True)
class CandidateRegion:
    chrom: str
    start: int
    end: int
    strand: str = "+"
    candidate_id: str = ""
    score: str = ""
    evidence: str = ""
    source: str = ""


def load_fasta_dict(path: str | Path) -> dict[str, str]:
    return {record.name: record.sequence for record in iter_fasta_records(path)}


def reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(table)[::-1].upper()


def read_candidate_regions(path: str | Path) -> list[CandidateRegion]:
    candidate_path = Path(path)
    suffix = candidate_path.suffix.lower()
    if suffix in {".bed", ".bedgraph"}:
        return list(_read_bed(candidate_path))
    return list(_read_csv_candidates(candidate_path))


def _read_bed(path: Path) -> Iterable[CandidateRegion]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3] if len(fields) >= 4 and fields[3] else f"candidate_{idx}"
            score = fields[4] if len(fields) >= 5 else ""
            strand = fields[5] if len(fields) >= 6 and fields[5] in {"+", "-"} else "+"
            yield CandidateRegion(chrom=chrom, start=start, end=end, strand=strand, candidate_id=name, score=score, source=str(path))


def _read_csv_candidates(path: Path) -> Iterable[CandidateRegion]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            chrom = str(row.get("chrom") or row.get("seqid") or row.get("contig") or "").lstrip("\ufeff")
            if not chrom:
                continue
            start = int(float(row.get("start", 0)))
            end = int(float(row.get("end", start)))
            candidate_id = str(row.get("candidate_id") or row.get("window_id") or row.get("name") or f"candidate_{idx}")
            strand = str(row.get("strand") or "+")
            if strand not in {"+", "-"}:
                strand = "+"
            score = str(row.get("anomaly_score") or row.get("score") or "")
            evidence = str(row.get("evidence") or "")
            yield CandidateRegion(
                chrom=chrom,
                start=start,
                end=end,
                strand=strand,
                candidate_id=candidate_id,
                score=score,
                evidence=evidence,
                source=str(path),
            )


def extract_candidate_sequences(
    fasta: str | Path,
    candidates: str | Path,
    out_csv: str | Path,
    *,
    out_fasta: str | Path | None = None,
    flank: int = 0,
    target_length: int | None = None,
    min_acgt_fraction: float = 0.0,
    max_candidates: int | None = None,
    split_label: str = "real",
) -> list[dict[str, object]]:
    regions = read_candidate_regions(candidates)
    if max_candidates is not None:
        regions = regions[:max_candidates]
    by_chrom: dict[str, list[tuple[int, CandidateRegion]]] = {}
    for index, region in enumerate(regions):
        by_chrom.setdefault(region.chrom, []).append((index, region))

    indexed_rows: list[tuple[int, dict[str, object]]] = []
    for record in iter_fasta_records(fasta):
        chromosome_regions = by_chrom.get(record.name)
        if not chromosome_regions:
            continue
        chrom_seq = record.sequence
        for index, region in chromosome_regions:
            row = _extract_region(
                region,
                chrom_seq,
                flank=flank,
                target_length=target_length,
                min_acgt_fraction=min_acgt_fraction,
                split_label=split_label,
            )
            if row is not None:
                indexed_rows.append((index, row))
    indexed_rows.sort(key=lambda item: item[0])
    rows = [row for _index, row in indexed_rows]

    _write_csv(Path(out_csv), rows)
    if out_fasta is not None:
        _write_fasta(Path(out_fasta), rows)
    return rows


def _extract_region(
    region: CandidateRegion,
    chrom_seq: str,
    *,
    flank: int,
    target_length: int | None,
    min_acgt_fraction: float,
    split_label: str,
) -> dict[str, object] | None:
    desired_start, desired_end = _desired_bounds(region, len(chrom_seq), flank=flank, target_length=target_length)
    extract_start = max(0, desired_start)
    extract_end = min(len(chrom_seq), desired_end)
    left_pad = max(0, -desired_start)
    right_pad = max(0, desired_end - len(chrom_seq))
    sequence = ("N" * left_pad) + chrom_seq[extract_start:extract_end] + ("N" * right_pad)
    if region.strand == "-":
        sequence = reverse_complement(sequence)
    if acgt_fraction(sequence) < min_acgt_fraction:
        return None
    return {
        "split": split_label,
        "candidate_id": region.candidate_id or f"{region.chrom}:{region.start}-{region.end}:{region.strand}",
        "chrom": region.chrom,
        "start": region.start,
        "end": region.end,
        "strand": region.strand,
        "score": region.score,
        "evidence": region.evidence,
        "source_candidates": region.source,
        "extract_start": extract_start,
        "extract_end": extract_end,
        "left_pad": left_pad,
        "right_pad": right_pad,
        "sequence_length": len(sequence),
        "sequence": sequence,
    }


def _desired_bounds(region: CandidateRegion, chrom_length: int, *, flank: int, target_length: int | None) -> tuple[int, int]:
    if target_length is not None and target_length > 0:
        center = (region.start + region.end) // 2
        desired_start = center - target_length // 2
        desired_end = desired_start + target_length
        return desired_start, desired_end
    return region.start - flank, region.end + flank


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_fasta(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            header = (
                f">{row['candidate_id']} chrom={row['chrom']} start={row['start']} end={row['end']} "
                f"strand={row['strand']} extract={row['extract_start']}-{row['extract_end']}"
            )
            handle.write(header + "\n")
            seq = str(row["sequence"])
            for idx in range(0, len(seq), 80):
                handle.write(seq[idx : idx + 80] + "\n")
