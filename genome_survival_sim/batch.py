from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

from .analysis import build_batch_report, write_design_tradeoff_artifacts
from .simulate import SimulationConfig, make_recommendations, run_simulation, write_result


def load_batch_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_batch(config: dict[str, Any], out_dir: str | Path) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = _payload_from_config(config)

    species_values = _as_list(config.get("species", "toy"))
    generation_values = _as_list(config.get("generations", 30))
    strategy_values = _as_list(config.get("strategies", config.get("strategy", "single_heterozygous_copy")))
    multiplier_values = _as_list(config.get("mutation_rate_multipliers", config.get("mutation_rate_multiplier", 1.0)))
    copies_values = _as_list(config.get("copies_per_block", None))
    replicate_values = _as_list(config.get("replicates", 100))
    encryption_values = _as_list(config.get("encryption_modes", config.get("encryption_mode", "hmac_stream")))
    ecc_values = _as_list(config.get("ecc_modes", config.get("ecc_mode", "none")))
    ecc_symbol_values = _as_list(config.get("ecc_symbols", 16))
    erasure_values = _as_list(config.get("erasure_modes", config.get("erasure_mode", "none")))
    erasure_group_values = _as_list(config.get("erasure_group_sizes", config.get("erasure_group_size", 8)))
    erasure_repair_values = _as_list(config.get("erasure_repair_blocks", 4))
    resync_values = _as_list(config.get("resync_modes", config.get("resync_mode", "none")))
    resync_chunk_values = _as_list(config.get("resync_chunk_bytes", 32))
    resync_marker_values = _as_list(config.get("resync_marker_bp", 24))
    resync_mismatch_values = _as_list(config.get("resync_mismatches", 0))

    combined_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    recommendations: list[str] = []

    for (
        species,
        generations,
        strategy,
        multiplier,
        copies,
        replicates,
        encryption_mode,
        ecc_mode,
        ecc_symbols,
        erasure_mode,
        erasure_group_size,
        erasure_repair_blocks,
        resync_mode,
        resync_chunk_bytes,
        resync_marker_bp,
        resync_mismatches,
    ) in product(
        species_values,
        generation_values,
        strategy_values,
        multiplier_values,
        copies_values,
        replicate_values,
        encryption_values,
        ecc_values,
        ecc_symbol_values,
        erasure_values,
        erasure_group_values,
        erasure_repair_values,
        resync_values,
        resync_chunk_values,
        resync_marker_values,
        resync_mismatch_values,
    ):
        sim_config = SimulationConfig(
            species=str(species),
            generations=int(generations),
            replicates=int(replicates),
            strategy=str(strategy),
            placement=str(config.get("placement", "intergenic_random")),
            mutation_rate_multiplier=float(multiplier),
            snv_rate=_optional_float(config.get("snv_rate")),
            indel_rate=_optional_float(config.get("indel_rate")),
            copy_loss_rate=float(config.get("copy_loss_rate", 0.0)),
            seed=int(config.get("seed", 17)),
            block_size_bytes=int(config.get("block_size_bytes", 128)),
            copies_per_block=None if copies is None else int(copies),
            sync_marker_bp=int(config.get("sync_marker_bp", 32)),
            sync_mismatches=int(config.get("sync_mismatches", 0)),
            secret=str(config.get("secret", "replace-this-project-secret")),
            encryption_mode=str(encryption_mode),
            ecc_mode=str(ecc_mode),
            ecc_symbols=int(ecc_symbols),
            erasure_mode=str(erasure_mode),
            erasure_group_size=int(erasure_group_size),
            erasure_repair_blocks=int(erasure_repair_blocks),
            resync_mode=str(resync_mode),
            resync_chunk_bytes=int(resync_chunk_bytes),
            resync_marker_bp=int(resync_marker_bp),
            resync_mismatches=int(resync_mismatches),
            chrom_sizes=Path(config["chrom_sizes"]) if config.get("chrom_sizes") else None,
            candidate_bed=Path(config["candidate_bed"]) if config.get("candidate_bed") else None,
        )
        result = run_simulation(payload, sim_config)
        run_name = (
            f"{sim_config.species}_{sim_config.strategy}_gen{sim_config.generations}"
            f"_rep{sim_config.replicates}_mutx{_format_float(sim_config.mutation_rate_multiplier)}"
            f"_copies{sim_config.copies_per_block or 'default'}"
            f"_enc{sim_config.encryption_mode}"
            f"_ecc{sim_config.ecc_mode}{sim_config.ecc_symbols}"
            f"_era{sim_config.erasure_mode}{sim_config.erasure_group_size}"
            f"_repair{sim_config.erasure_repair_blocks}"
            f"_res{sim_config.resync_mode}{sim_config.resync_chunk_bytes}"
            f"_rmark{sim_config.resync_marker_bp}_rmis{sim_config.resync_mismatches}"
        )
        write_result(result, target / run_name)
        combined_rows.extend(result.rows)
        summaries.append(result.summary)
        recommendations.append(make_recommendations(result))

    if combined_rows:
        with (target / "batch_generation_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_rows)
    with (target / "batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    (target / "batch_recommendations.md").write_text("\n\n---\n\n".join(recommendations), encoding="utf-8")
    write_design_tradeoff_artifacts(summaries, combined_rows, target)
    (target / "batch_report.md").write_text(build_batch_report(summaries, combined_rows), encoding="utf-8")


def _payload_from_config(config: dict[str, Any]) -> bytes:
    if config.get("payload"):
        return Path(config["payload"]).read_bytes()
    if config.get("payload_text") is not None:
        return str(config["payload_text"]).encode("utf-8")
    raise ValueError("Batch config requires either 'payload' or 'payload_text'.")


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")
