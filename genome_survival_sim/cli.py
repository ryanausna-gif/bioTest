from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .batch import load_batch_config, run_batch
from .genome_io import build_candidate_intervals, load_chrom_sizes_dict, load_gff_features, write_bed
from .msprime_backend import MsprimeConfig, run_msprime_baseline
from .simulate import STRATEGIES, SimulationConfig, make_recommendations, run_simulation, write_result
from .slim_backend import SlimBridgeConfig, parse_slim_stdout_file, run_slim_script, write_slim_script
from .stdpopsim_backend import list_stdpopsim_species, write_stdpopsim_species
from .tskit_backend import analyze_tree_sequence_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulate DNA payload survival in a single descendant lineage.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser("simulate", help="Run lineage payload-survival simulations.")
    payload_group = simulate_parser.add_mutually_exclusive_group(required=True)
    payload_group.add_argument("--payload", type=Path, help="Input bytes to encode as a DNA payload.")
    payload_group.add_argument("--payload-text", help="Small text payload for demos.")
    simulate_parser.add_argument("--species", default="human_t2t", choices=["human_t2t", "mouse", "fly", "toy"])
    simulate_parser.add_argument("--chrom-sizes", type=Path, default=None, help="Optional chrom.sizes/.fai-like file.")
    simulate_parser.add_argument("--candidate-bed", type=Path, default=None, help="Optional BED file of candidate loci.")
    simulate_parser.add_argument("--placement", default="intergenic_random")
    simulate_parser.add_argument("--generations", type=int, default=30)
    simulate_parser.add_argument("--replicates", type=int, default=100)
    simulate_parser.add_argument("--strategy", nargs="+", default=["single_heterozygous_copy"], choices=STRATEGIES)
    simulate_parser.add_argument("--mutation-rate-multiplier", nargs="+", type=float, default=[1.0])
    simulate_parser.add_argument("--snv-rate", type=float, default=None)
    simulate_parser.add_argument("--indel-rate", type=float, default=None)
    simulate_parser.add_argument("--copy-loss-rate", type=float, default=0.0)
    simulate_parser.add_argument("--block-size-bytes", type=int, default=128)
    simulate_parser.add_argument("--copies-per-block", type=int, default=None)
    simulate_parser.add_argument("--sync-marker-bp", type=int, default=32)
    simulate_parser.add_argument("--sync-mismatches", type=int, default=0)
    simulate_parser.add_argument("--secret", default="replace-this-project-secret")
    simulate_parser.add_argument(
        "--encryption-mode",
        default="hmac_stream",
        choices=["hmac_stream", "chacha20_poly1305"],
        help="Use chacha20_poly1305 for real AEAD if cryptography is installed.",
    )
    simulate_parser.add_argument(
        "--ecc-mode",
        default="none",
        choices=["none", "reed_solomon"],
        help="Use reed_solomon for byte-level block correction if reedsolo is installed.",
    )
    simulate_parser.add_argument(
        "--ecc-symbols",
        type=int,
        default=16,
        help="Number of Reed-Solomon parity symbols per encoded fragment.",
    )
    simulate_parser.add_argument(
        "--erasure-mode",
        default="none",
        choices=["none", "xor_parity", "fountain"],
        help="Use xor_parity or fountain for cross-block erasure recovery.",
    )
    simulate_parser.add_argument(
        "--erasure-group-size",
        type=int,
        default=8,
        help="Number of data blocks protected by each XOR parity block.",
    )
    simulate_parser.add_argument(
        "--erasure-repair-blocks",
        type=int,
        default=4,
        help="Number of fountain repair blocks per erasure group.",
    )
    simulate_parser.add_argument(
        "--resync-mode",
        default="none",
        choices=["none", "chunked"],
        help="Use chunked to insert internal sync markers for indel resynchronization.",
    )
    simulate_parser.add_argument(
        "--resync-chunk-bytes",
        type=int,
        default=32,
        help="Bytes per internal resync chunk before DNA serialization.",
    )
    simulate_parser.add_argument(
        "--resync-marker-bp",
        type=int,
        default=24,
        help="Length of each internal chunk sync marker in bases.",
    )
    simulate_parser.add_argument(
        "--resync-mismatches",
        type=int,
        default=0,
        help="Allowed mismatches when scanning internal chunk sync markers.",
    )
    simulate_parser.add_argument("--seed", type=int, default=17)
    simulate_parser.add_argument("--out", type=Path, default=Path("artifacts/genome_survival"))

    batch_parser = subparsers.add_parser("batch", help="Run a JSON-defined simulation matrix.")
    batch_parser.add_argument("--config", type=Path, required=True)
    batch_parser.add_argument("--out", type=Path, default=Path("artifacts/genome_survival_batch"))

    candidates_parser = subparsers.add_parser("make-candidates", help="Build candidate BED intervals from GFF/GTF.")
    candidates_parser.add_argument("--chrom-sizes", type=Path, required=True, help="chrom.sizes or FASTA .fai file.")
    candidates_parser.add_argument("--annotation", type=Path, required=True, help="GFF3 or GTF annotation file.")
    candidates_parser.add_argument("--mode", default="intergenic", choices=["intergenic", "intronic", "pseudogene"])
    candidates_parser.add_argument("--min-length", type=int, default=200)
    candidates_parser.add_argument("--flank", type=int, default=1000, help="Gene flank to exclude for intergenic mode.")
    candidates_parser.add_argument("--out", type=Path, required=True)

    slim_parser = subparsers.add_parser("generate-slim", help="Generate a SLiM marker-survival bridge script.")
    slim_parser.add_argument("--out", type=Path, required=True)
    slim_parser.add_argument("--genome-length", type=int, default=1_000_000)
    slim_parser.add_argument("--population-size", type=int, default=1_000)
    slim_parser.add_argument("--generations", type=int, default=50)
    slim_parser.add_argument("--payload-copy-count", type=int, default=16)
    slim_parser.add_argument("--mutation-rate", type=float, default=1.0e-8)
    slim_parser.add_argument("--recombination-rate", type=float, default=1.0e-8)
    slim_parser.add_argument("--selection-coefficient", type=float, default=0.0)
    slim_parser.add_argument("--dominance", type=float, default=0.5)
    slim_parser.add_argument("--seed", type=int, default=17)
    slim_parser.add_argument(
        "--tree-seq-output",
        default=None,
        help="Optional .trees path to emit from SLiM via sim.treeSeqOutput().",
    )

    run_slim_parser = subparsers.add_parser("run-slim", help="Run a SLiM bridge script and parse its output.")
    run_slim_parser.add_argument("--script", type=Path, required=True)
    run_slim_parser.add_argument("--out", type=Path, default=Path("artifacts/slim_run"))
    run_slim_parser.add_argument("--slim-bin", default="slim")
    run_slim_parser.add_argument("--population-size", type=int, default=None)
    run_slim_parser.add_argument("--timeout-seconds", type=int, default=None)
    run_slim_parser.add_argument(
        "--tree-sequence",
        type=Path,
        default=None,
        help="Optional .trees file to analyze after SLiM completes.",
    )
    run_slim_parser.add_argument("--mutation-type", type=int, default=None)

    parse_slim_parser = subparsers.add_parser("parse-slim", help="Parse stdout emitted by a SLiM bridge run.")
    parse_slim_parser.add_argument("--stdout", type=Path, required=True)
    parse_slim_parser.add_argument("--out", type=Path, default=Path("artifacts/slim_parse"))
    parse_slim_parser.add_argument("--population-size", type=int, default=None)

    trees_parser = subparsers.add_parser("analyze-trees", help="Analyze a tskit/pyslim .trees file.")
    trees_parser.add_argument("--tree-sequence", type=Path, required=True)
    trees_parser.add_argument("--out", type=Path, default=Path("artifacts/tree_sequence_analysis"))
    trees_parser.add_argument("--mutation-type", type=int, default=None)

    msprime_parser = subparsers.add_parser("run-msprime", help="Run a neutral msprime baseline.")
    msprime_parser.add_argument("--out", type=Path, default=Path("artifacts/msprime_baseline"))
    msprime_parser.add_argument("--samples", type=int, default=100)
    msprime_parser.add_argument("--sequence-length", type=int, default=1_000_000)
    msprime_parser.add_argument("--population-size", type=int, default=10_000)
    msprime_parser.add_argument("--recombination-rate", type=float, default=1.0e-8)
    msprime_parser.add_argument("--mutation-rate", type=float, default=1.0e-8)
    msprime_parser.add_argument("--seed", type=int, default=17)
    msprime_parser.add_argument("--ploidy", type=int, default=2)
    msprime_parser.add_argument("--payload-bp", type=int, default=1_000)

    stdpopsim_parser = subparsers.add_parser("list-stdpopsim", help="List stdpopsim species catalog metadata.")
    stdpopsim_parser.add_argument("--out", type=Path, default=Path("artifacts/stdpopsim_catalog"))

    args = parser.parse_args(argv)
    if args.command == "simulate":
        _simulate(args)
    elif args.command == "batch":
        _batch(args)
    elif args.command == "make-candidates":
        _make_candidates(args)
    elif args.command == "generate-slim":
        _generate_slim(args)
    elif args.command == "run-slim":
        _run_slim(args)
    elif args.command == "parse-slim":
        _parse_slim(args)
    elif args.command == "analyze-trees":
        _analyze_trees(args)
    elif args.command == "run-msprime":
        _run_msprime(args)
    elif args.command == "list-stdpopsim":
        _list_stdpopsim(args)


def _simulate(args: argparse.Namespace) -> None:
    payload = args.payload.read_bytes() if args.payload else args.payload_text.encode("utf-8")
    args.out.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, object]] = []
    combined_summary: list[dict[str, object]] = []
    recommendation_blocks: list[str] = []

    for strategy in args.strategy:
        for multiplier in args.mutation_rate_multiplier:
            run_dir = (
                args.out
                / f"{args.species}_{strategy}_mutx{_format_multiplier(multiplier)}"
                f"_enc{args.encryption_mode}_ecc{args.ecc_mode}{args.ecc_symbols}"
                f"_era{args.erasure_mode}{args.erasure_group_size}"
                f"_repair{args.erasure_repair_blocks}"
                f"_res{args.resync_mode}{args.resync_chunk_bytes}"
                f"_rmark{args.resync_marker_bp}_rmis{args.resync_mismatches}"
            )
            config = SimulationConfig(
                species=args.species,
                generations=args.generations,
                replicates=args.replicates,
                strategy=strategy,
                placement=args.placement,
                mutation_rate_multiplier=multiplier,
                snv_rate=args.snv_rate,
                indel_rate=args.indel_rate,
                copy_loss_rate=args.copy_loss_rate,
                seed=args.seed,
                block_size_bytes=args.block_size_bytes,
                copies_per_block=args.copies_per_block,
                sync_marker_bp=args.sync_marker_bp,
                sync_mismatches=args.sync_mismatches,
                secret=args.secret,
                encryption_mode=args.encryption_mode,
                ecc_mode=args.ecc_mode,
                ecc_symbols=args.ecc_symbols,
                erasure_mode=args.erasure_mode,
                erasure_group_size=args.erasure_group_size,
                erasure_repair_blocks=args.erasure_repair_blocks,
                resync_mode=args.resync_mode,
                resync_chunk_bytes=args.resync_chunk_bytes,
                resync_marker_bp=args.resync_marker_bp,
                resync_mismatches=args.resync_mismatches,
                chrom_sizes=args.chrom_sizes,
                candidate_bed=args.candidate_bed,
            )
            result = run_simulation(payload, config)
            write_result(result, run_dir)
            combined_rows.extend(result.rows)
            combined_summary.append(result.summary)
            recommendation_blocks.append(make_recommendations(result))
            print(
                f"{strategy} mutx={multiplier:g} enc={args.encryption_mode} "
                f"ecc={args.ecc_mode}/{args.ecc_symbols} "
                f"erasure={args.erasure_mode}/{args.erasure_group_size}/{args.erasure_repair_blocks} "
                f"resync={args.resync_mode}/{args.resync_chunk_bytes}: "
                f"final_decode_success_rate={result.summary['final_decode_success_rate']:.4f} "
                f"top_loss_reason={result.summary['top_final_loss_reason']}"
            )

    if combined_rows:
        metrics_path = args.out / "all_generation_metrics.csv"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_rows)

    with (args.out / "all_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(combined_summary, handle, ensure_ascii=False, indent=2)
    (args.out / "recommendations.md").write_text("\n\n---\n\n".join(recommendation_blocks), encoding="utf-8")


def _format_multiplier(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _batch(args: argparse.Namespace) -> None:
    config = load_batch_config(args.config)
    run_batch(config, args.out)
    print(f"wrote batch results to {args.out}")


def _make_candidates(args: argparse.Namespace) -> None:
    chrom_sizes = load_chrom_sizes_dict(args.chrom_sizes)
    features = load_gff_features(args.annotation)
    intervals = build_candidate_intervals(
        chrom_sizes=chrom_sizes,
        features=features,
        mode=args.mode,
        min_length=args.min_length,
        flank=args.flank,
    )
    write_bed(intervals, args.out)
    print(f"wrote {len(intervals)} {args.mode} candidate intervals to {args.out}")


def _generate_slim(args: argparse.Namespace) -> None:
    config = SlimBridgeConfig(
        genome_length=args.genome_length,
        population_size=args.population_size,
        generations=args.generations,
        payload_copy_count=args.payload_copy_count,
        mutation_rate=args.mutation_rate,
        recombination_rate=args.recombination_rate,
        selection_coefficient=args.selection_coefficient,
        dominance=args.dominance,
        seed=args.seed,
        tree_seq_output=args.tree_seq_output,
    )
    write_slim_script(config, args.out)
    print(f"wrote SLiM bridge script to {args.out}")


def _run_slim(args: argparse.Namespace) -> None:
    result = run_slim_script(
        script_path=args.script,
        out_dir=args.out,
        slim_bin=args.slim_bin,
        population_size=args.population_size,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"wrote SLiM stdout to {result.stdout_path}")
    print(f"wrote SLiM metrics to {result.csv_path}")
    print(f"wrote SLiM summary to {result.summary_path}")
    print(f"wrote SLiM report to {result.report_path}")
    if args.tree_sequence is not None:
        summary_path, report_path = analyze_tree_sequence_file(
            tree_path=args.tree_sequence,
            out_dir=args.out / "tree_sequence",
            mutation_type=args.mutation_type,
        )
        print(f"wrote tree-sequence summary to {summary_path}")
        print(f"wrote tree-sequence report to {report_path}")


def _parse_slim(args: argparse.Namespace) -> None:
    csv_path, summary_path, report_path = parse_slim_stdout_file(
        stdout_path=args.stdout,
        out_dir=args.out,
        population_size=args.population_size,
    )
    print(f"wrote SLiM metrics to {csv_path}")
    print(f"wrote SLiM summary to {summary_path}")
    print(f"wrote SLiM report to {report_path}")


def _analyze_trees(args: argparse.Namespace) -> None:
    summary_path, report_path = analyze_tree_sequence_file(
        tree_path=args.tree_sequence,
        out_dir=args.out,
        mutation_type=args.mutation_type,
    )
    print(f"wrote tree-sequence summary to {summary_path}")
    print(f"wrote tree-sequence report to {report_path}")


def _run_msprime(args: argparse.Namespace) -> None:
    result = run_msprime_baseline(
        MsprimeConfig(
            samples=args.samples,
            sequence_length=args.sequence_length,
            population_size=args.population_size,
            recombination_rate=args.recombination_rate,
            mutation_rate=args.mutation_rate,
            seed=args.seed,
            ploidy=args.ploidy,
            payload_bp=args.payload_bp,
        ),
        args.out,
    )
    print(f"wrote msprime tree sequence to {result.tree_path}")
    print(f"wrote msprime summary to {result.summary_path}")
    print(f"wrote msprime report to {result.report_path}")


def _list_stdpopsim(args: argparse.Namespace) -> None:
    rows = list_stdpopsim_species()
    csv_path, json_path, report_path = write_stdpopsim_species(rows, args.out)
    print(f"wrote stdpopsim catalog CSV to {csv_path}")
    print(f"wrote stdpopsim catalog JSON to {json_path}")
    print(f"wrote stdpopsim catalog report to {report_path}")
