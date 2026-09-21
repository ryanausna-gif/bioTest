from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .tskit_backend import summarize_tree_sequence, write_tree_sequence_analysis


@dataclass(frozen=True)
class MsprimeConfig:
    samples: int = 100
    sequence_length: int = 1_000_000
    population_size: int = 10_000
    recombination_rate: float = 1.0e-8
    mutation_rate: float = 1.0e-8
    seed: int = 17
    ploidy: int = 2
    payload_bp: int = 1_000


@dataclass(frozen=True)
class MsprimeRunResult:
    tree_path: Path
    summary_path: Path
    report_path: Path


def run_msprime_baseline(config: MsprimeConfig, out_dir: str | Path) -> MsprimeRunResult:
    """Run a neutral msprime ancestry/mutation baseline and summarize it.

    This backend is intentionally neutral: it estimates background ancestry and
    mutation behavior for a region, while the Python payload simulator remains
    responsible for block/auth decoding.
    """

    try:
        import msprime
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "msprime is required for run-msprime. Install optional dependencies with: "
            "conda install -c conda-forge msprime tskit"
        ) from exc

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    tree_path = target / "msprime_baseline.trees"

    ancestry = msprime.sim_ancestry(
        samples=config.samples,
        sequence_length=config.sequence_length,
        recombination_rate=config.recombination_rate,
        population_size=config.population_size,
        ploidy=config.ploidy,
        random_seed=config.seed,
    )
    mutated = msprime.sim_mutations(
        ancestry,
        rate=config.mutation_rate,
        random_seed=config.seed + 1,
    )
    mutated.dump(str(tree_path))

    summary = summarize_msprime_tree_sequence(mutated, config)
    summary_path, report_path = write_msprime_analysis(summary, target)
    return MsprimeRunResult(tree_path=tree_path, summary_path=summary_path, report_path=report_path)


def summarize_msprime_tree_sequence(ts: Any, config: MsprimeConfig) -> dict[str, Any]:
    summary = summarize_tree_sequence(ts)
    summary["backend"] = "msprime"
    summary["config"] = asdict(config)
    summary["expected_payload_mutations_per_lineage"] = expected_payload_mutations(
        payload_bp=config.payload_bp,
        generations=1,
        mutation_rate=config.mutation_rate,
    )
    summary["expected_region_mutations_per_generation"] = expected_payload_mutations(
        payload_bp=config.sequence_length,
        generations=1,
        mutation_rate=config.mutation_rate,
    )
    return summary


def write_msprime_analysis(summary: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = target / "msprime_summary.json"
    report_path = target / "msprime_report.md"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    report_path.write_text(make_msprime_report(summary), encoding="utf-8")
    return summary_path, report_path


def make_msprime_report(summary: dict[str, Any]) -> str:
    config = summary.get("config", {})
    lines = [
        "# msprime Neutral Baseline Report",
        "",
        f"- Samples: `{config.get('samples')}`",
        f"- Ploidy: `{config.get('ploidy')}`",
        f"- Sequence length: `{config.get('sequence_length')}`",
        f"- Population size: `{config.get('population_size')}`",
        f"- Recombination rate: `{config.get('recombination_rate')}`",
        f"- Mutation rate: `{config.get('mutation_rate')}`",
        f"- Payload bp for expectation: `{config.get('payload_bp')}`",
        "",
        "## Tree Sequence",
        "",
        f"- Trees: `{summary.get('num_trees')}`",
        f"- Sites: `{summary.get('num_sites')}`",
        f"- Mutations: `{summary.get('num_mutations')}`",
        f"- Individuals: `{summary.get('num_individuals')}`",
        f"- Samples in tree sequence: `{summary.get('num_samples')}`",
        "",
        "## Payload Mutation Expectation",
        "",
        f"- Expected payload mutations per lineage per generation: `{summary.get('expected_payload_mutations_per_lineage')}`",
        f"- Expected region mutations per generation: `{summary.get('expected_region_mutations_per_generation')}`",
        "",
        "## Interpretation",
        "",
        "- msprime provides a neutral baseline for ancestry, recombination, and mutation behavior.",
        "- It does not model encrypted payload decoding by itself.",
        "- Compare this baseline with the Python payload simulator and SLiM marker survival results.",
        "",
    ]
    return "\n".join(lines)


def expected_payload_mutations(payload_bp: int, generations: int, mutation_rate: float) -> float:
    return max(0, payload_bp) * max(0, generations) * max(0.0, mutation_rate)
