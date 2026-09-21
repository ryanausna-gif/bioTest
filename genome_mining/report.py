from __future__ import annotations

import csv
import json
from pathlib import Path

from .records import MiningHit


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_to_hit(row: dict[str, object]) -> MiningHit:
    return MiningHit(
        chrom=str(row["chrom"]),
        start=int(row["start"]),
        end=int(row["end"]),
        strand=str(row.get("strand", "+")),
        score=float(row["anomaly_score"]),
        evidence=str(row.get("evidence", "")),
        window_id=str(row.get("window_id", "")),
        sequence=str(row.get("sequence", "")),
    )


def write_bed(path: str | Path, hits: list[MiningHit]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, hit in enumerate(hits, start=1):
            name = hit.window_id or f"mining_hit_{index}"
            handle.write(
                f"{hit.chrom}\t{hit.start}\t{hit.end}\t{name}\t{hit.bed_score}\t{hit.strand}\t"
                f"{hit.start}\t{hit.end}\t0\t1\t{hit.end - hit.start}\t0\n"
            )


def write_summary_json(
    path: str | Path,
    *,
    fasta: str,
    window_size: int,
    step: int,
    total_windows: int,
    candidates: list[MiningHit],
    threshold: float,
    baseline: dict[str, dict[str, float]],
) -> None:
    payload = {
        "fasta": fasta,
        "window_size": window_size,
        "step": step,
        "total_windows": total_windows,
        "candidate_count": len(candidates),
        "threshold": threshold,
        "top_score": candidates[0].score if candidates else 0.0,
        "baseline": baseline,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(
    path: str | Path,
    *,
    fasta: str,
    window_size: int,
    step: int,
    total_windows: int,
    candidates: list[MiningHit],
    threshold: float,
) -> None:
    lines = [
        "# Genome Mining MVP Report",
        "",
        "This report is an exploratory anomaly screen, not evidence of artificial origin by itself.",
        "",
        "## Run",
        "",
        f"- FASTA: `{fasta}`",
        f"- Window size: `{window_size}`",
        f"- Step: `{step}`",
        f"- Scored windows: `{total_windows}`",
        f"- Candidate threshold: `{threshold:.3f}`",
        f"- Candidate count: `{len(candidates)}`",
        "",
        "## Top Candidates",
        "",
    ]
    if not candidates:
        lines.append("No windows exceeded the configured candidate threshold.")
    else:
        lines.append("| rank | region | score | evidence |")
        lines.append("|---:|---|---:|---|")
        for rank, hit in enumerate(candidates[:20], start=1):
            region = f"{hit.chrom}:{hit.start}-{hit.end}"
            lines.append(f"| {rank} | `{region}` | {hit.score:.3f} | `{hit.evidence}` |")
    lines.extend(
        [
            "",
            "## Recommended Next Checks",
            "",
            "1. Intersect candidates with repeat, transposon, gene, CpG island, and low-complexity annotations.",
            "2. Re-run with multiple window sizes and keep only candidates stable across scales.",
            "3. Compare against shuffled and chromosome-held-out controls before training deep models.",
            "4. Send high-confidence candidates to decoder and survival-simulation layers.",
        ]
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
