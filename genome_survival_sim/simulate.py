from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

from .codec import EncodedFragment, PayloadCodec, PayloadCodecConfig
from .genome import GenomeModel, Interval, Locus, default_genome, load_candidate_bed, load_chrom_sizes, select_random_locus
from .mutation import MutationStats, mutate_dna
from .recombination import crossover_positions, haplotype_source_at


STRATEGIES = ("single_heterozygous_copy", "homozygous_same_locus", "multi_locus_redundant")


@dataclass(frozen=True)
class PayloadCopy:
    block_id: int
    copy_id: int
    locus: Locus
    haplotype: int
    dna: str


@dataclass(frozen=True)
class Individual:
    copies: tuple[PayloadCopy, ...]


@dataclass(frozen=True)
class SimulationConfig:
    species: str = "toy"
    generations: int = 30
    replicates: int = 100
    strategy: str = "single_heterozygous_copy"
    placement: str = "intergenic_random"
    mutation_rate_multiplier: float = 1.0
    snv_rate: float | None = None
    indel_rate: float | None = None
    copy_loss_rate: float = 0.0
    seed: int = 17
    block_size_bytes: int = 128
    copies_per_block: int | None = None
    sync_marker_bp: int = 32
    sync_mismatches: int = 0
    secret: str = "replace-this-project-secret"
    encryption_mode: str = "hmac_stream"
    ecc_mode: str = "none"
    ecc_symbols: int = 16
    erasure_mode: str = "none"
    erasure_group_size: int = 8
    erasure_repair_blocks: int = 4
    resync_mode: str = "none"
    resync_chunk_bytes: int = 32
    resync_marker_bp: int = 24
    resync_mismatches: int = 0
    chrom_sizes: Path | None = None
    candidate_bed: Path | None = None


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    genome: GenomeModel
    rows: list[dict[str, object]]
    summary: dict[str, object]


def run_simulation(payload: bytes, config: SimulationConfig) -> SimulationResult:
    if config.strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {config.strategy!r}; expected one of {STRATEGIES}.")
    genome = load_chrom_sizes(config.chrom_sizes, species=config.species) if config.chrom_sizes else default_genome(config.species)
    intervals = load_candidate_bed(config.candidate_bed) if config.candidate_bed else None
    copies_per_block = _copies_for_strategy(config)
    codec = PayloadCodec(
        PayloadCodecConfig(
            block_size_bytes=config.block_size_bytes,
            copies_per_block=copies_per_block,
            sync_marker_bp=config.sync_marker_bp,
            sync_mismatches=config.sync_mismatches,
            secret=config.secret,
            encryption_mode=config.encryption_mode,
            ecc_mode=config.ecc_mode,
            ecc_symbols=config.ecc_symbols,
            erasure_mode=config.erasure_mode,
            erasure_group_size=config.erasure_group_size,
            erasure_repair_blocks=config.erasure_repair_blocks,
            resync_mode=config.resync_mode,
            resync_chunk_bytes=config.resync_chunk_bytes,
            resync_marker_bp=config.resync_marker_bp,
            resync_mismatches=config.resync_mismatches,
        )
    )
    expected_codec = codec.with_expected_payload_marker(payload)
    fragments = codec.encode(payload)
    encoding_stats = _encoding_stats(payload, fragments)
    snv_rate = genome.snv_rate if config.snv_rate is None else config.snv_rate
    indel_rate = genome.indel_rate if config.indel_rate is None else config.indel_rate

    aggregate: dict[int, Counter[str]] = defaultdict(Counter)
    numeric: dict[int, Counter[str]] = defaultdict(Counter)
    reason_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for replicate in range(config.replicates):
        rng = Random(config.seed + replicate * 104_729)
        founder = _build_founder(
            fragments=fragments,
            genome=genome,
            strategy=config.strategy,
            placement=config.placement,
            rng=rng,
            intervals=intervals,
        )
        individual = founder

        for generation in range(config.generations + 1):
            decoded = expected_codec.decode_fragments(copy.dna for copy in individual.copies)
            success = decoded.success and decoded.payload == payload
            reason = _loss_reason(individual, decoded, success)

            aggregate[generation]["replicates"] += 1
            aggregate[generation]["has_any_copy"] += int(bool(individual.copies))
            aggregate[generation]["identifier_detected"] += int(decoded.marker_seen > 0)
            aggregate[generation]["decode_success"] += int(success)
            numeric[generation]["copy_count"] += len(individual.copies)
            numeric[generation]["valid_fragments"] += decoded.valid_fragments
            numeric[generation]["recovered_blocks"] += decoded.recovered_blocks
            reason_counts[generation][reason] += 1

            if generation == config.generations:
                break

            individual, mutation_stats = _make_next_generation(
                individual=individual,
                genome=genome,
                rng=rng,
                snv_rate=snv_rate,
                indel_rate=indel_rate,
                mutation_rate_multiplier=config.mutation_rate_multiplier,
                copy_loss_rate=config.copy_loss_rate,
            )
            numeric[generation + 1]["new_mutations"] += mutation_stats.total

    rows = _aggregate_rows(config, aggregate, numeric, reason_counts, encoding_stats)
    summary = _summary(config, genome, rows, encoding_stats)
    return SimulationResult(config=config, genome=genome, rows=rows, summary=summary)


def write_result(result: SimulationResult, out_dir: str | Path) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    metrics_path = target / "generation_metrics.csv"
    if result.rows:
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.rows[0].keys()))
            writer.writeheader()
            writer.writerows(result.rows)

    summary_path = target / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, ensure_ascii=False, indent=2)

    recommendations_path = target / "recommendations.md"
    recommendations_path.write_text(make_recommendations(result), encoding="utf-8")


def make_recommendations(result: SimulationResult) -> str:
    final = result.rows[-1] if result.rows else {}
    success_rate = float(final.get("decode_success_rate", 0.0))
    inherited_rate = float(final.get("has_any_copy_rate", 0.0))
    marker_rate = float(final.get("identifier_detected_rate", 0.0))

    lines = [
        "# 仿真建议",
        "",
        f"- 物种/基因组模型：`{result.genome.species}`。",
        f"- 插入策略：`{result.config.strategy}`。",
        f"- 最后一代完整解码率：`{success_rate:.4f}`。",
        f"- 最后一代仍携带至少一个片段的比例：`{inherited_rate:.4f}`。",
        f"- 最后一代可检测到标识符的比例：`{marker_rate:.4f}`。",
        "",
    ]

    if inherited_rate < 0.5:
        lines.append("- 主要风险首先是遗传分离导致的丢失；不要只放单个位点单拷贝。")
    if marker_rate < inherited_rate:
        lines.append("- 标识符本身需要冗余或允许少量错配，否则扫描入口会先失效。")
    if success_rate < inherited_rate:
        lines.append("- 已继承但无法完整解码，说明 block 冗余和纠错强度需要提高。")
    if result.config.strategy == "single_heterozygous_copy":
        lines.append("- 单杂合拷贝适合作为负对照，不建议作为长期存活设计。")
    if result.config.strategy == "multi_locus_redundant":
        lines.append("- 多位点冗余更符合长期存活目标，下一步应扫描不同候选 BED 区域。")

    lines.extend(
        [
            "- 当前 MVP 使用稀疏 payload 仿真，不等价于完整湿实验或真实释放场景。",
            "- 下一步建议接入真实 `.fai` 染色体长度、非编码候选 BED，以及独立的 SLiM 群体模型作为交叉验证。",
            "",
        ]
    )
    return "\n".join(lines)


def _copies_for_strategy(config: SimulationConfig) -> int:
    if config.copies_per_block is not None:
        return config.copies_per_block
    if config.strategy == "single_heterozygous_copy":
        return 1
    if config.strategy == "homozygous_same_locus":
        return 2
    return 4


def _build_founder(
    fragments: list[EncodedFragment],
    genome: GenomeModel,
    strategy: str,
    placement: str,
    rng: Random,
    intervals: list[Interval] | None,
) -> Individual:
    if strategy == "single_heterozygous_copy":
        return Individual(tuple(_place_package(fragments, genome, rng, placement, intervals, haplotype=0)))
    if strategy == "homozygous_same_locus":
        return Individual(tuple(_place_homozygous(fragments, genome, rng, placement, intervals)))
    return Individual(tuple(_place_multilocus(fragments, genome, rng, placement, intervals)))


def _place_package(
    fragments: list[EncodedFragment],
    genome: GenomeModel,
    rng: Random,
    placement: str,
    intervals: list[Interval] | None,
    haplotype: int,
) -> list[PayloadCopy]:
    spacer = 16
    total_length = sum(fragment.length for fragment in fragments) + spacer * max(0, len(fragments) - 1)
    package_locus = select_random_locus(genome, rng, total_length, placement, intervals)
    copies: list[PayloadCopy] = []
    cursor = package_locus.start
    for fragment in fragments:
        locus = Locus(package_locus.chrom, cursor, cursor + fragment.length, package_locus.label)
        copies.append(PayloadCopy(fragment.block_id, fragment.copy_id, locus, haplotype, fragment.dna))
        cursor += fragment.length + spacer
    return copies


def _place_homozygous(
    fragments: list[EncodedFragment],
    genome: GenomeModel,
    rng: Random,
    placement: str,
    intervals: list[Interval] | None,
) -> list[PayloadCopy]:
    by_block: dict[int, list[EncodedFragment]] = defaultdict(list)
    for fragment in fragments:
        by_block[fragment.block_id].append(fragment)

    copies: list[PayloadCopy] = []
    for block_id in sorted(by_block):
        fragment_group = by_block[block_id]
        locus = select_random_locus(genome, rng, max(item.length for item in fragment_group), placement, intervals)
        for index, fragment in enumerate(fragment_group[:2]):
            copies.append(PayloadCopy(fragment.block_id, fragment.copy_id, locus, index % 2, fragment.dna))
    return copies


def _place_multilocus(
    fragments: list[EncodedFragment],
    genome: GenomeModel,
    rng: Random,
    placement: str,
    intervals: list[Interval] | None,
) -> list[PayloadCopy]:
    copies: list[PayloadCopy] = []
    for fragment in fragments:
        locus = select_random_locus(genome, rng, fragment.length, placement, intervals)
        haplotype = rng.randrange(2)
        copies.append(PayloadCopy(fragment.block_id, fragment.copy_id, locus, haplotype, fragment.dna))
    return copies


def _make_next_generation(
    individual: Individual,
    genome: GenomeModel,
    rng: Random,
    snv_rate: float,
    indel_rate: float,
    mutation_rate_multiplier: float,
    copy_loss_rate: float,
) -> tuple[Individual, MutationStats]:
    if not individual.copies:
        return individual, MutationStats()

    crossovers_by_chrom = {chrom.name: crossover_positions(chrom, rng) for chrom in genome.chromosomes}
    initial_source_by_chrom = {chrom.name: rng.randrange(2) for chrom in genome.chromosomes}

    next_copies: list[PayloadCopy] = []
    total_stats = MutationStats()
    for copy in individual.copies:
        source = haplotype_source_at(
            copy.locus.start,
            initial_source_by_chrom.get(copy.locus.chrom, 0),
            crossovers_by_chrom.get(copy.locus.chrom, []),
        )
        if source != copy.haplotype:
            continue
        if copy_loss_rate > 0 and rng.random() < copy_loss_rate:
            continue

        mutated_dna, stats = mutate_dna(copy.dna, rng, snv_rate, indel_rate, mutation_rate_multiplier)
        total_stats = MutationStats(
            snv=total_stats.snv + stats.snv,
            insertion=total_stats.insertion + stats.insertion,
            deletion=total_stats.deletion + stats.deletion,
        )
        # The target descendant receives the simulated carrier gamete as haplotype 0.
        next_copies.append(
            PayloadCopy(copy.block_id, copy.copy_id, copy.locus, 0, mutated_dna)
        )
    return Individual(tuple(next_copies)), total_stats


def _loss_reason(individual: Individual, decoded, success: bool) -> str:
    if success:
        return "success"
    if not individual.copies:
        return "not_inherited"
    return decoded.reason


def _aggregate_rows(
    config: SimulationConfig,
    aggregate: dict[int, Counter[str]],
    numeric: dict[int, Counter[str]],
    reason_counts: dict[int, Counter[str]],
    encoding_stats: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for generation in range(config.generations + 1):
        reps = max(1, aggregate[generation]["replicates"])
        reasons = reason_counts[generation]
        top_reason = reasons.most_common(1)[0][0] if reasons else "none"
        rows.append(
            {
                "species": config.species,
                "strategy": config.strategy,
                "copies_per_block": config.copies_per_block if config.copies_per_block is not None else "default",
                "block_size_bytes": config.block_size_bytes,
                **encoding_stats,
                "encryption_mode": config.encryption_mode,
                "ecc_mode": config.ecc_mode,
                "ecc_symbols": config.ecc_symbols,
                "erasure_mode": config.erasure_mode,
                "erasure_group_size": config.erasure_group_size,
                "erasure_repair_blocks": config.erasure_repair_blocks,
                "resync_mode": config.resync_mode,
                "resync_chunk_bytes": config.resync_chunk_bytes,
                "resync_marker_bp": config.resync_marker_bp,
                "resync_mismatches": config.resync_mismatches,
                "mutation_rate_multiplier": config.mutation_rate_multiplier,
                "generation": generation,
                "replicates": reps,
                "has_any_copy_rate": aggregate[generation]["has_any_copy"] / reps,
                "identifier_detected_rate": aggregate[generation]["identifier_detected"] / reps,
                "decode_success_rate": aggregate[generation]["decode_success"] / reps,
                "mean_copy_count": numeric[generation]["copy_count"] / reps,
                "mean_valid_fragments": numeric[generation]["valid_fragments"] / reps,
                "mean_recovered_blocks": numeric[generation]["recovered_blocks"] / reps,
                "mean_new_mutations": numeric[generation]["new_mutations"] / reps,
                "top_loss_reason": top_reason,
                "not_inherited": reasons["not_inherited"],
                "sync_marker_destroyed": reasons["sync_marker_destroyed"],
                "block_missing_or_mutated": reasons["block_missing_or_mutated"],
                "auth_failed": reasons["auth_failed"],
                "payload_mismatch": reasons["payload_mismatch"],
                "success": reasons["success"],
            }
        )
    return rows


def _summary(
    config: SimulationConfig,
    genome: GenomeModel,
    rows: list[dict[str, object]],
    encoding_stats: dict[str, object],
) -> dict[str, object]:
    final = rows[-1] if rows else {}
    return {
        "species": genome.species,
        "genome_notes": genome.notes,
        "strategy": config.strategy,
        "copies_per_block": config.copies_per_block if config.copies_per_block is not None else "default",
        "block_size_bytes": config.block_size_bytes,
        **encoding_stats,
        "encryption_mode": config.encryption_mode,
        "ecc_mode": config.ecc_mode,
        "ecc_symbols": config.ecc_symbols,
        "erasure_mode": config.erasure_mode,
        "erasure_group_size": config.erasure_group_size,
        "erasure_repair_blocks": config.erasure_repair_blocks,
        "resync_mode": config.resync_mode,
        "resync_chunk_bytes": config.resync_chunk_bytes,
        "resync_marker_bp": config.resync_marker_bp,
        "resync_mismatches": config.resync_mismatches,
        "generations": config.generations,
        "replicates": config.replicates,
        "mutation_rate_multiplier": config.mutation_rate_multiplier,
        "final_decode_success_rate": final.get("decode_success_rate", 0.0),
        "final_has_any_copy_rate": final.get("has_any_copy_rate", 0.0),
        "final_identifier_detected_rate": final.get("identifier_detected_rate", 0.0),
        "top_final_loss_reason": final.get("top_loss_reason", "none"),
    }


def _encoding_stats(payload: bytes, fragments: list[EncodedFragment]) -> dict[str, object]:
    payload_bytes = len(payload)
    encoded_total_bp = sum(fragment.length for fragment in fragments)
    return {
        "payload_bytes": payload_bytes,
        "encoded_fragment_count": len(fragments),
        "encoded_total_bp": encoded_total_bp,
        "encoded_bp_per_payload_byte": encoded_total_bp / max(1, payload_bytes),
    }
