from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def load_tree_sequence(path: str | Path):
    """Load a tskit tree sequence.

    tskit/pyslim are optional dependencies because the pure-Python lineage
    simulator and stdout SLiM bridge should remain usable on minimal machines.
    """

    try:
        import tskit
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "tskit is required to analyze .trees files. Install optional dependencies with: "
            "conda install -c conda-forge tskit pyslim msprime stdpopsim"
        ) from exc
    return tskit.load(str(path))


def summarize_tree_sequence(ts: Any, mutation_type: int | None = None) -> dict[str, Any]:
    mutation_type_counts = count_mutation_types(ts)
    filtered_mutations = None
    if mutation_type is not None:
        filtered_mutations = mutation_type_counts.get(str(mutation_type), 0)

    summary = {
        "sequence_length": _attr(ts, "sequence_length"),
        "num_trees": _attr(ts, "num_trees"),
        "num_nodes": _attr(ts, "num_nodes"),
        "num_edges": _attr(ts, "num_edges"),
        "num_sites": _attr(ts, "num_sites"),
        "num_mutations": _attr(ts, "num_mutations"),
        "num_individuals": _attr(ts, "num_individuals"),
        "num_populations": _attr(ts, "num_populations"),
        "num_samples": _num_samples(ts),
        "mutation_type_filter": mutation_type,
        "filtered_mutations": filtered_mutations,
        "mutation_type_counts": dict(mutation_type_counts),
    }
    return summary


def count_mutation_types(ts: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    mutations = _iter_mutations(ts)
    for mutation in mutations:
        metadata = getattr(mutation, "metadata", None)
        types = extract_slim_mutation_types(metadata)
        if not types:
            counts["unknown"] += 1
        else:
            for mutation_type in types:
                counts[str(mutation_type)] += 1
    return counts


def extract_slim_mutation_types(metadata: Any) -> list[int]:
    """Extract SLiM mutation types from common pyslim metadata layouts."""

    if metadata is None:
        return []
    if isinstance(metadata, bytes):
        return []
    if isinstance(metadata, dict):
        if "mutation_type" in metadata:
            return [_safe_int(metadata["mutation_type"])]
        if "mutation_type_id" in metadata:
            return [_safe_int(metadata["mutation_type_id"])]
        if "mutation_list" in metadata and isinstance(metadata["mutation_list"], list):
            result: list[int] = []
            for item in metadata["mutation_list"]:
                if isinstance(item, dict):
                    if "mutation_type" in item:
                        result.append(_safe_int(item["mutation_type"]))
                    elif "mutation_type_id" in item:
                        result.append(_safe_int(item["mutation_type_id"]))
            return result
    return []


def analyze_tree_sequence_file(
    tree_path: str | Path,
    out_dir: str | Path,
    mutation_type: int | None = None,
) -> tuple[Path, Path]:
    ts = load_tree_sequence(tree_path)
    summary = summarize_tree_sequence(ts, mutation_type=mutation_type)
    return write_tree_sequence_analysis(summary, out_dir)


def write_tree_sequence_analysis(summary: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "tree_sequence_summary.json"
    report_path = target / "tree_sequence_report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    report_path.write_text(make_tree_sequence_report(summary), encoding="utf-8")
    return summary_path, report_path


def write_mutation_type_counts(summary: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mutation_type", "count"])
        writer.writeheader()
        for mutation_type, count in sorted(summary.get("mutation_type_counts", {}).items()):
            writer.writerow({"mutation_type": mutation_type, "count": count})


def make_tree_sequence_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Tree Sequence Report",
        "",
        f"- Sequence length: `{summary.get('sequence_length')}`",
        f"- Trees: `{summary.get('num_trees')}`",
        f"- Nodes: `{summary.get('num_nodes')}`",
        f"- Edges: `{summary.get('num_edges')}`",
        f"- Sites: `{summary.get('num_sites')}`",
        f"- Mutations: `{summary.get('num_mutations')}`",
        f"- Individuals: `{summary.get('num_individuals')}`",
        f"- Samples: `{summary.get('num_samples')}`",
        f"- Mutation type filter: `{summary.get('mutation_type_filter')}`",
        f"- Filtered mutations: `{summary.get('filtered_mutations')}`",
        "",
        "## Mutation Types",
        "",
        "| mutation_type | count |",
        "|---|---:|",
    ]
    counts = summary.get("mutation_type_counts", {})
    if counts:
        for mutation_type, count in sorted(counts.items()):
            lines.append(f"| {mutation_type} | {count} |")
    else:
        lines.append("| NA | 0 |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a tree-sequence level summary. It does not decode payload blocks by itself.",
            "- Combine it with the Python payload decoder and SLiM marker report to separate population loss from sequence corruption.",
        ]
    )
    return "\n".join(lines)


def _attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
    return value


def _num_samples(ts: Any) -> int | None:
    if hasattr(ts, "num_samples"):
        return getattr(ts, "num_samples")
    if hasattr(ts, "samples"):
        try:
            return len(list(ts.samples()))
        except TypeError:
            return None
    return None


def _iter_mutations(ts: Any) -> Iterable[Any]:
    mutations = getattr(ts, "mutations", None)
    if callable(mutations):
        return mutations()
    return []


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
