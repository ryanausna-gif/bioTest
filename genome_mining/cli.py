from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze_benchmark, analyze_failures, make_benchmark_card
from .carrier_controls import build_carrier_benchmark
from .embedding_viz import plot_embeddings
from .experiments import (
    compare_predictions,
    make_leaderboard,
    run_deep_encoder_comparison,
    run_holdout_matrix,
    run_v12_ablations,
    run_v12_suite,
)
from .fasta import iter_fasta_windows
from .features import compute_window_features
from .model_training import (
    calibrate_unknown_score,
    evaluate_openood_metrics,
    run_ood_method_suite,
    train_deep_carrier_model,
    train_evidence_fusion,
    train_pe_nac_plus,
)
from .localization_metrics import multiscale_localization_metrics, token_localization_metrics
from .model_metrics import save_json
from .baseline_model import train_baseline_model
from .candidates import extract_candidate_sequences
from .model_inference import score_candidates
from .report import row_to_hit, write_bed, write_csv, write_markdown_report, write_summary_json
from .scoring import score_feature_rows
from .image_dataset import audit_image_genome_dataset, build_image_genome_dataset
from .image_training import evaluate_image_locator_model, train_image_locator
from .image_experiments import run_hyenadna_comparison
from .stego_codecs import available_stego_codecs, load_codec_key, make_stego_codec
from .stego_metrics import summarize_stego_dataset
from .stego_validation import verify_stego_roundtrip_dataset
from .lm_stego.model_adapter import KmerProbabilityModel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mine coordinate-aware anomaly candidates from genome FASTA files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Run deterministic feature extraction and anomaly scoring.")
    scan_parser.add_argument("--fasta", type=Path, required=True)
    scan_parser.add_argument("--out", type=Path, default=Path("artifacts/genome_mining_scan"))
    scan_parser.add_argument("--window", type=int, default=1000)
    scan_parser.add_argument("--step", type=int, default=500)
    scan_parser.add_argument("--threshold", type=float, default=2.5)
    scan_parser.add_argument("--top-n", type=int, default=50)
    scan_parser.add_argument("--min-acgt-fraction", type=float, default=0.85)
    scan_parser.add_argument("--max-windows", type=int, default=None)
    scan_parser.add_argument("--species", default="")
    scan_parser.add_argument("--source", default="")
    scan_parser.add_argument(
        "--include-sequence",
        action="store_true",
        help="Include raw window sequence in CSV output. Disabled by default for whole-genome scans.",
    )

    controls_parser = subparsers.add_parser(
        "build-controls",
        help="Build V12-style synthetic carrier controls for open-set genome mining benchmarks.",
    )
    controls_parser.add_argument("--out", type=Path, required=True)
    controls_parser.add_argument("--background-fasta", type=Path, default=None)
    controls_parser.add_argument("--generator-kind", default="realistic", choices=["base", "literature", "realistic"])
    controls_parser.add_argument("--protocol", default="standard", choices=["standard", "background_shift", "detectability_shift"])
    controls_parser.add_argument("--length", type=int, default=1000)
    controls_parser.add_argument("--n-train", type=int, default=6000)
    controls_parser.add_argument("--n-val", type=int, default=1500)
    controls_parser.add_argument("--n-test", type=int, default=2500)
    controls_parser.add_argument(
        "--holdout-codecs",
        nargs="+",
        default=["CodonStegoCodec", "AdversarialMimicCodec"],
        help="Codec families reserved as unknown carriers in the test split.",
    )
    controls_parser.add_argument(
        "--calibration-codecs",
        nargs="+",
        default=["StegoCodec"],
        help="Codec families reserved for validation-time unknown-score calibration.",
    )
    controls_parser.add_argument("--negative-rate", type=float, default=0.35)
    controls_parser.add_argument("--train-detectability", nargs="+", default=["easy", "medium"])
    controls_parser.add_argument("--test-detectability", nargs="+", default=["hard", "adversarial"])
    controls_parser.add_argument("--seed", type=int, default=42)
    controls_parser.add_argument("--no-hard-negatives", action="store_true")

    baseline_parser = subparsers.add_parser("train-baseline", help="Train migrated V12 k-mer baseline models.")
    baseline_parser.add_argument("--csv", type=Path, required=True)
    baseline_parser.add_argument("--out", type=Path, required=True)
    baseline_parser.add_argument("--model", choices=["logreg", "rf", "gb"], default="logreg")
    baseline_parser.add_argument("--k", type=int, default=4)
    baseline_parser.add_argument("--no-extra", action="store_true")

    pe_parser = subparsers.add_parser("train-pe-nac-plus", help="Train the migrated V10 PE-NAC++ detector.")
    _add_pe_training_args(pe_parser)

    ef_parser = subparsers.add_parser("train-ef-pe-nac-plus", help="Train the migrated V11 EF-PE-NAC++ detector.")
    _add_pe_training_args(ef_parser)

    deep_parser = subparsers.add_parser("train-deep-carrier", help="Train the migrated V12 DeepOpenCarrier model.")
    deep_parser.add_argument("--csv", type=Path, required=True)
    deep_parser.add_argument("--out", type=Path, required=True)
    deep_parser.add_argument("--encoder", choices=["cnn", "dilated_cnn", "transformer", "hybrid"], default="hybrid")
    deep_parser.add_argument("--epochs", type=int, default=10)
    deep_parser.add_argument("--batch-size", type=int, default=64)
    deep_parser.add_argument("--lr", type=float, default=1e-3)
    deep_parser.add_argument("--length", type=int, default=None)
    deep_parser.add_argument("--prototypes-per-class", type=int, default=4)
    deep_parser.add_argument("--hier-contrast-weight", type=float, default=0.15)
    deep_parser.add_argument("--prototype-weight", type=float, default=0.3)
    deep_parser.add_argument("--localization-weight", type=float, default=0.25)
    deep_parser.add_argument("--episodic-weight", type=float, default=0.2)
    deep_parser.add_argument("--episodic-prob", type=float, default=0.5)
    deep_parser.add_argument("--seed", type=int, default=42)
    deep_parser.add_argument("--device", default=None)

    openood_parser = subparsers.add_parser("evaluate-openood", help="Evaluate OpenOOD-style metrics from a predictions CSV.")
    openood_parser.add_argument("--pred", type=Path, required=True)
    openood_parser.add_argument("--out", type=Path, required=True)
    openood_parser.add_argument("--target", choices=["carrier", "unknown"], default="unknown")
    openood_parser.add_argument("--score-col", default=None)

    suite_parser = subparsers.add_parser("run-ood-suite", help="Run V12 OOD baseline suite using predictions and embeddings.")
    suite_parser.add_argument("--pred", type=Path, required=True)
    suite_parser.add_argument("--emb", type=Path, required=True)
    suite_parser.add_argument("--out", type=Path, required=True)
    suite_parser.add_argument("--target", choices=["carrier", "unknown"], default="unknown")
    suite_parser.add_argument("--reference-pred", type=Path, default=None)
    suite_parser.add_argument("--reference-emb", type=Path, default=None)

    fusion_parser = subparsers.add_parser("train-evidence-fusion", help="Train V11 evidence fusion on a predictions CSV.")
    fusion_parser.add_argument("--pred", type=Path, required=True)
    fusion_parser.add_argument("--out", type=Path, required=True)
    fusion_parser.add_argument("--model-out", type=Path, required=True)
    fusion_parser.add_argument("--target", choices=["carrier", "unknown"], default="unknown")
    fusion_parser.add_argument("--model-type", choices=["logreg", "gb"], default="logreg")
    fusion_parser.add_argument(
        "--calibration-pred",
        type=Path,
        default=None,
        help="Independent validation predictions used to fit the fusion model.",
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate-unknown",
        help="Fit V10 unknown-score calibration on validation predictions and apply it to evaluation predictions.",
    )
    calibrate_parser.add_argument("--calibration-pred", type=Path, required=True)
    calibrate_parser.add_argument("--pred", type=Path, required=True)
    calibrate_parser.add_argument("--out", type=Path, required=True)
    calibrate_parser.add_argument("--model-out", type=Path, required=True)

    analyze_parser = subparsers.add_parser("analyze-benchmark", help="Audit a V12 carrier benchmark and data leakage.")
    analyze_parser.add_argument("--csv", type=Path, required=True)
    analyze_parser.add_argument("--out", type=Path, required=True)

    card_parser = subparsers.add_parser("make-benchmark-card", help="Create a machine-readable V12 benchmark card.")
    card_parser.add_argument("--csv", type=Path, required=True)
    card_parser.add_argument("--out", type=Path, required=True)

    failure_parser = subparsers.add_parser("analyze-failures", help="Summarize V11/V12 false positives and negatives.")
    failure_parser.add_argument("--pred", type=Path, required=True)
    failure_parser.add_argument("--out", type=Path, required=True)
    failure_parser.add_argument("--score-col", default="special_prob")
    failure_parser.add_argument("--threshold", type=float, default=0.5)

    localization_parser = subparsers.add_parser(
        "evaluate-localization",
        help="Evaluate region, token, or V11 multi-scale carrier localization.",
    )
    localization_parser.add_argument("--pred", type=Path, required=True)
    localization_parser.add_argument("--out", type=Path, required=True)
    localization_parser.add_argument("--mode", choices=["region", "token", "multiscale"], default="multiscale")
    localization_parser.add_argument("--length", type=int, default=None)

    embedding_parser = subparsers.add_parser("plot-embeddings", help="Plot V12 PCA/t-SNE/UMAP embedding geometry.")
    embedding_parser.add_argument("--emb", type=Path, required=True)
    embedding_parser.add_argument("--pred", type=Path, required=True)
    embedding_parser.add_argument("--out", type=Path, required=True)
    embedding_parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    embedding_parser.add_argument("--color-by", default="open_target")
    embedding_parser.add_argument("--max-points", type=int, default=3000)
    embedding_parser.add_argument("--seed", type=int, default=42)

    leaderboard_parser = subparsers.add_parser("make-leaderboard", help="Summarize nested V12 metrics.json files.")
    leaderboard_parser.add_argument("--root", type=Path, required=True)
    leaderboard_parser.add_argument("--out-csv", type=Path, required=True)
    leaderboard_parser.add_argument("--out-md", type=Path, default=None)

    compare_parser = subparsers.add_parser("compare-predictions", help="Paired bootstrap comparison of two V12 runs.")
    compare_parser.add_argument("--pred-a", type=Path, required=True)
    compare_parser.add_argument("--pred-b", type=Path, required=True)
    compare_parser.add_argument("--out", type=Path, required=True)
    compare_parser.add_argument("--name-a", default="method_a")
    compare_parser.add_argument("--name-b", default="method_b")
    compare_parser.add_argument("--target", choices=["carrier", "unknown"], default="unknown")
    compare_parser.add_argument("--score-col-a", default=None)
    compare_parser.add_argument("--score-col-b", default=None)
    compare_parser.add_argument("--n-bootstrap", type=int, default=1000)
    compare_parser.add_argument("--seed", type=int, default=42)

    encoder_parser = subparsers.add_parser("run-encoder-comparison", help="Run the V12 deep encoder matrix.")
    encoder_parser.add_argument("--csv", type=Path, required=True)
    encoder_parser.add_argument("--out-root", type=Path, required=True)
    encoder_parser.add_argument("--encoders", nargs="+", default=["cnn", "dilated_cnn", "transformer", "hybrid"])
    encoder_parser.add_argument("--epochs", type=int, default=6)
    encoder_parser.add_argument("--batch-size", type=int, default=64)
    encoder_parser.add_argument("--length", type=int, default=None)
    encoder_parser.add_argument("--device", default=None)
    encoder_parser.add_argument("--seed", type=int, default=42)
    encoder_parser.add_argument("--dry-run", action="store_true")

    ablation_parser = subparsers.add_parser("run-ablations", help="Run PE-NAC++ or V12 deep component ablations.")
    ablation_parser.add_argument("--csv", type=Path, required=True)
    ablation_parser.add_argument("--out-root", type=Path, required=True)
    ablation_parser.add_argument("--model", choices=["pe", "deep"], default="deep")
    ablation_parser.add_argument("--encoder", choices=["cnn", "dilated_cnn", "transformer", "hybrid"], default="hybrid")
    ablation_parser.add_argument("--epochs", type=int, default=6)
    ablation_parser.add_argument("--batch-size", type=int, default=64)
    ablation_parser.add_argument("--device", default=None)
    ablation_parser.add_argument("--seed", type=int, default=42)
    ablation_parser.add_argument("--dry-run", action="store_true")

    holdout_parser = subparsers.add_parser("run-holdout-matrix", help="Run leave-one-codec open-set experiments.")
    holdout_parser.add_argument("--out-root", type=Path, required=True)
    holdout_parser.add_argument("--codecs", nargs="+", required=True)
    holdout_parser.add_argument("--calibration-codec", default="StegoCodec")
    holdout_parser.add_argument("--method", choices=["baseline", "pe", "ef", "deep"], default="deep")
    holdout_parser.add_argument("--encoder", choices=["cnn", "dilated_cnn", "transformer", "hybrid"], default="hybrid")
    holdout_parser.add_argument("--n-train", type=int, default=4000)
    holdout_parser.add_argument("--n-val", type=int, default=1000)
    holdout_parser.add_argument("--n-test", type=int, default=1500)
    holdout_parser.add_argument("--length", type=int, default=1000)
    holdout_parser.add_argument("--epochs", type=int, default=6)
    holdout_parser.add_argument("--batch-size", type=int, default=64)
    holdout_parser.add_argument("--background-fasta", type=Path, default=None)
    holdout_parser.add_argument("--device", default=None)
    holdout_parser.add_argument("--seed", type=int, default=42)
    holdout_parser.add_argument("--dry-run", action="store_true")

    v12_suite_parser = subparsers.add_parser("run-v12-suite", help="Run the repaired end-to-end V12 experiment suite.")
    v12_suite_parser.add_argument("--out-root", type=Path, required=True)
    v12_suite_parser.add_argument("--background-fasta", type=Path, default=None)
    v12_suite_parser.add_argument("--n-train", type=int, default=4000)
    v12_suite_parser.add_argument("--n-val", type=int, default=1000)
    v12_suite_parser.add_argument("--n-test", type=int, default=1500)
    v12_suite_parser.add_argument("--length", type=int, default=1000)
    v12_suite_parser.add_argument("--epochs", type=int, default=6)
    v12_suite_parser.add_argument("--batch-size", type=int, default=64)
    v12_suite_parser.add_argument("--encoders", nargs="+", default=["cnn", "dilated_cnn", "hybrid"])
    v12_suite_parser.add_argument("--device", default=None)
    v12_suite_parser.add_argument("--seed", type=int, default=42)
    v12_suite_parser.add_argument("--dry-run", action="store_true")

    extract_parser = subparsers.add_parser(
        "extract-candidates",
        help="Extract candidate sequences from FASTA using candidates.bed/candidates.csv.",
    )
    extract_parser.add_argument("--fasta", type=Path, required=True)
    extract_parser.add_argument("--candidates", type=Path, required=True)
    extract_parser.add_argument("--out", type=Path, required=True)
    extract_parser.add_argument("--out-fasta", type=Path, default=None)
    extract_parser.add_argument("--flank", type=int, default=0)
    extract_parser.add_argument("--target-length", type=int, default=None)
    extract_parser.add_argument("--min-acgt-fraction", type=float, default=0.0)
    extract_parser.add_argument("--max-candidates", type=int, default=None)
    extract_parser.add_argument("--split-label", default="real")

    score_parser = subparsers.add_parser(
        "score-candidates",
        help="Score extracted real candidate sequences with a trained V12 model checkpoint.",
    )
    score_parser.add_argument("--candidates", type=Path, required=True)
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--out", type=Path, required=True)
    score_parser.add_argument("--batch-size", type=int, default=64)
    score_parser.add_argument("--device", default=None)
    score_parser.add_argument("--include-sequence", action="store_true")
    score_parser.add_argument("--unknown-calibrator", type=Path, default=None)
    score_parser.add_argument("--evidence-fusion-model", type=Path, default=None)

    image_data_parser = subparsers.add_parser(
        "build-image-dataset",
        help="Build the leak-audited fixed grayscale image-in-genome localization dataset.",
    )
    image_data_parser.add_argument("--genome-manifest", type=Path, required=True)
    image_source = image_data_parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument("--image-root", type=Path)
    image_source.add_argument("--image-manifest", type=Path)
    image_data_parser.add_argument("--out", type=Path, required=True)
    image_data_parser.add_argument("--n-train", type=int, default=1000)
    image_data_parser.add_argument("--n-val", type=int, default=200)
    image_data_parser.add_argument("--n-test", type=int, default=300)
    image_data_parser.add_argument("--positive-fraction", type=float, default=0.5)
    image_data_parser.add_argument("--image-size", type=int, default=128)
    image_data_parser.add_argument(
        "--resize-short-side",
        type=int,
        default=None,
        help="Optionally resize the short image side before the centered square crop.",
    )
    image_data_parser.add_argument("--window-length", type=int, default=131072)
    image_data_parser.add_argument("--min-acgt-fraction", type=float, default=0.95)
    image_data_parser.add_argument(
        "--codecs",
        nargs="+",
        default=["direct"],
        help="Default positive codecs: direct encrypted constrained kmer cover lm.",
    )
    image_data_parser.add_argument("--train-codecs", nargs="+", default=None)
    image_data_parser.add_argument("--val-codecs", nargs="+", default=None)
    image_data_parser.add_argument("--test-codecs", nargs="+", default=None)
    image_data_parser.add_argument("--codec-key-file", type=Path, default=None)
    image_data_parser.add_argument("--kmer-order", type=int, default=3)
    image_data_parser.add_argument(
        "--ecc-symbols",
        type=int,
        default=0,
        help="Optional Reed-Solomon parity symbols per 255-byte codeword; zero disables ECC.",
    )
    image_data_parser.add_argument("--substitution-rate", type=float, default=0.0)
    image_data_parser.add_argument("--indel-rate", type=float, default=0.0)
    image_data_parser.add_argument(
        "--lm-model-file",
        type=Path,
        default=None,
        help="Serialized next-base model required when an L5 lm codec is selected.",
    )
    image_data_parser.add_argument("--lm-block-bytes", type=int, default=8)
    image_data_parser.add_argument("--lm-max-symbols-per-block", type=int, default=256)
    image_data_parser.add_argument("--lm-context-bases", type=int, default=4096)
    image_data_parser.add_argument("--lm-cdf-precision-bits", type=int, default=12)
    image_data_parser.add_argument("--lm-uniform-mix", type=float, default=0.20)
    image_data_parser.add_argument("--seed", type=int, default=42)

    l5_model_parser = subparsers.add_parser(
        "train-l5-context-model",
        help="Fit and freeze a deterministic FASTA k-mer next-base model for the L5 protocol MVP.",
    )
    l5_model_parser.add_argument("--fasta", type=Path, nargs="+", required=True)
    l5_model_parser.add_argument("--out", type=Path, required=True)
    l5_model_parser.add_argument("--order", type=int, default=6)
    l5_model_parser.add_argument("--pseudocount", type=float, default=0.5)
    l5_model_parser.add_argument("--minimum-context-count", type=int, default=4)
    l5_model_parser.add_argument("--max-bases", type=int, default=5_000_000)
    l5_model_parser.add_argument("--no-reverse-complement", action="store_true")

    l5_roundtrip_parser = subparsers.add_parser(
        "test-l5-roundtrip",
        help="Encode a file with lm_arithmetic_v1 and verify exact authorized recovery.",
    )
    l5_roundtrip_parser.add_argument("--input", type=Path, required=True)
    l5_roundtrip_parser.add_argument("--model", type=Path, required=True)
    l5_roundtrip_parser.add_argument("--codec-key-file", type=Path, required=True)
    l5_roundtrip_parser.add_argument("--dna-out", type=Path, required=True)
    l5_roundtrip_parser.add_argument("--recovered-out", type=Path, required=True)
    context_source = l5_roundtrip_parser.add_mutually_exclusive_group()
    context_source.add_argument("--context-sequence", default="")
    context_source.add_argument("--context-file", type=Path, default=None)
    l5_roundtrip_parser.add_argument("--sample-id", default="l5_roundtrip")
    l5_roundtrip_parser.add_argument("--ecc-symbols", type=int, default=0)
    l5_roundtrip_parser.add_argument("--block-bytes", type=int, default=8)
    l5_roundtrip_parser.add_argument("--max-symbols-per-block", type=int, default=256)
    l5_roundtrip_parser.add_argument("--context-bases", type=int, default=4096)
    l5_roundtrip_parser.add_argument("--cdf-precision-bits", type=int, default=12)
    l5_roundtrip_parser.add_argument("--uniform-mix", type=float, default=0.20)

    subparsers.add_parser("list-stego-codecs", help="List reversible image-to-DNA benchmark codecs.")
    stego_report_parser = subparsers.add_parser(
        "summarize-stego-dataset",
        help="Summarize capacity, cover change, GC/CpG, homopolymer, and k-mer traces by codec.",
    )
    stego_report_parser.add_argument("--dataset", type=Path, required=True)
    stego_report_parser.add_argument("--out", type=Path, required=True)

    stego_verify_parser = subparsers.add_parser(
        "verify-stego-roundtrip",
        help="Use oracle payload boundaries to verify exact codec and image recovery.",
    )
    stego_verify_parser.add_argument("--dataset", type=Path, required=True)
    stego_verify_parser.add_argument("--out", type=Path, required=True)
    stego_verify_parser.add_argument("--codec-key-file", type=Path, default=None)
    stego_verify_parser.add_argument(
        "--split", choices=["all", "train", "val", "test"], default="all"
    )
    stego_verify_parser.add_argument("--max-samples", type=int, default=None)
    stego_verify_parser.add_argument("--allow-failures", action="store_true")

    image_audit_parser = subparsers.add_parser(
        "audit-image-dataset",
        help="Audit donor, image, sequence, and genomic-window leakage in an image dataset.",
    )
    image_audit_parser.add_argument("--dataset", type=Path, required=True)
    image_audit_parser.add_argument("--out", type=Path, default=None)

    image_train_parser = subparsers.add_parser(
        "train-image-locator",
        help="Train a CNN or HyenaDNA image-presence and variable start/end locator.",
    )
    image_train_parser.add_argument("--dataset", type=Path, required=True)
    image_train_parser.add_argument("--out", type=Path, required=True)
    image_train_parser.add_argument("--epochs", type=int, default=10)
    image_train_parser.add_argument("--batch-size", type=int, default=4)
    image_train_parser.add_argument("--lr", type=float, default=2e-4)
    image_train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    image_train_parser.add_argument("--backbone", choices=["cnn", "hyenadna"], default="cnn")
    image_train_parser.add_argument("--d-model", type=int, default=128)
    image_train_parser.add_argument("--downsample-stride", type=int, default=256)
    image_train_parser.add_argument("--context", choices=["dilated_cnn", "transformer"], default="dilated_cnn")
    image_train_parser.add_argument("--context-layers", type=int, default=6)
    image_train_parser.add_argument("--dropout", type=float, default=0.1)
    image_train_parser.add_argument(
        "--hyena-model-name",
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
    )
    image_train_parser.add_argument("--hyena-revision", default=None)
    image_train_parser.add_argument("--hyena-local-files-only", action="store_true")
    image_train_parser.add_argument(
        "--hyena-fine-tune",
        choices=["frozen", "last_n", "full"],
        default="frozen",
    )
    image_train_parser.add_argument("--hyena-last-n-layers", type=int, default=2)
    image_train_parser.add_argument("--hyena-no-reverse-complement", action="store_true")
    image_train_parser.add_argument("--hyena-random-init", action="store_true")
    image_train_parser.add_argument("--gradient-checkpointing", action="store_true")
    image_train_parser.add_argument("--gradient-accumulation", type=int, default=1)
    image_train_parser.add_argument("--grad-clip", type=float, default=1.0)
    image_train_parser.add_argument("--presence-weight", type=float, default=1.0)
    image_train_parser.add_argument("--start-weight", type=float, default=1.0)
    image_train_parser.add_argument("--offset-weight", type=float, default=1.0)
    image_train_parser.add_argument("--segment-weight", type=float, default=0.5)
    image_train_parser.add_argument("--rc-consistency-weight", type=float, default=0.1)
    image_train_parser.add_argument("--workers", type=int, default=2)
    image_train_parser.add_argument("--device", default=None)
    image_train_parser.add_argument("--amp", action="store_true")
    image_train_parser.add_argument("--codec-key-file", type=Path, default=None)
    image_train_parser.add_argument("--seed", type=int, default=42)

    image_eval_parser = subparsers.add_parser(
        "evaluate-image-locator",
        help="Evaluate detection, variable-span localization, and codec-aware image recovery.",
    )
    image_eval_parser.add_argument("--dataset", type=Path, required=True)
    image_eval_parser.add_argument("--model", type=Path, required=True)
    image_eval_parser.add_argument("--out", type=Path, required=True)
    image_eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    image_eval_parser.add_argument("--batch-size", type=int, default=4)
    image_eval_parser.add_argument("--workers", type=int, default=2)
    image_eval_parser.add_argument("--device", default=None)
    image_eval_parser.add_argument("--save-images", action="store_true")
    image_eval_parser.add_argument("--max-saved-images", type=int, default=25)
    image_eval_parser.add_argument("--codec-key-file", type=Path, default=None)

    comparison_parser = subparsers.add_parser(
        "run-hyenadna-comparison",
        help="Compare CNN, random HyenaDNA, pretrained forward HyenaDNA, and pretrained RC HyenaDNA.",
    )
    comparison_parser.add_argument("--dataset", type=Path, required=True)
    comparison_parser.add_argument("--out-root", type=Path, required=True)
    comparison_parser.add_argument("--epochs", type=int, default=10)
    comparison_parser.add_argument("--batch-size", type=int, default=1)
    comparison_parser.add_argument("--lr", type=float, default=2e-4)
    comparison_parser.add_argument("--d-model", type=int, default=128)
    comparison_parser.add_argument("--downsample-stride", type=int, default=256)
    comparison_parser.add_argument(
        "--hyena-model-name",
        default="LongSafari/hyenadna-medium-160k-seqlen-hf",
    )
    comparison_parser.add_argument("--hyena-revision", default=None)
    comparison_parser.add_argument("--hyena-local-files-only", action="store_true")
    comparison_parser.add_argument("--workers", type=int, default=2)
    comparison_parser.add_argument("--device", default=None)
    comparison_parser.add_argument("--no-amp", action="store_true")
    comparison_parser.add_argument("--gradient-accumulation", type=int, default=8)
    comparison_parser.add_argument("--codec-key-file", type=Path, default=None)
    comparison_parser.add_argument("--seed", type=int, default=42)
    comparison_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "scan":
        _scan(args)
    elif args.command == "build-controls":
        _build_controls(args)
    elif args.command == "train-baseline":
        train_baseline_model(args.csv, args.out, model_name=args.model, k=args.k, use_extra=not args.no_extra)
    elif args.command == "train-pe-nac-plus":
        _train_pe_nac(args, variant="pe_nac_plus")
    elif args.command == "train-ef-pe-nac-plus":
        _train_pe_nac(args, variant="ef_pe_nac_plus")
    elif args.command == "train-deep-carrier":
        _train_deep(args)
    elif args.command == "evaluate-openood":
        evaluate_openood_metrics(args.pred, args.out, target=args.target, score_col=args.score_col)
    elif args.command == "run-ood-suite":
        run_ood_method_suite(
            args.pred,
            args.emb,
            args.out,
            target=args.target,
            reference_pred_path=args.reference_pred,
            reference_embeddings_path=args.reference_emb,
        )
    elif args.command == "train-evidence-fusion":
        train_evidence_fusion(
            args.pred,
            args.out,
            args.model_out,
            target=args.target,
            model_type=args.model_type,
            calibration_pred_path=args.calibration_pred,
        )
    elif args.command == "calibrate-unknown":
        report = calibrate_unknown_score(args.pred, args.calibration_pred, args.out, args.model_out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "analyze-benchmark":
        report = analyze_benchmark(args.csv, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "make-benchmark-card":
        card = make_benchmark_card(args.csv, args.out)
        print(json.dumps(card, ensure_ascii=False, indent=2))
    elif args.command == "analyze-failures":
        report = analyze_failures(args.pred, args.out, score_column=args.score_col, threshold=args.threshold)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "evaluate-localization":
        import pandas as pd

        frame = pd.read_csv(args.pred)
        if args.mode == "region":
            from .localization import localization_metrics_from_regions

            report = localization_metrics_from_regions(frame[frame.y == 1] if "y" in frame else frame)
        elif args.mode == "token":
            report = token_localization_metrics(frame, length=args.length)
        else:
            report = multiscale_localization_metrics(frame, length=args.length)
        save_json(report, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "plot-embeddings":
        report = plot_embeddings(
            args.emb,
            args.pred,
            args.out,
            method=args.method,
            color_by=args.color_by,
            max_points=args.max_points,
            seed=args.seed,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "make-leaderboard":
        frame = make_leaderboard(args.root, args.out_csv, args.out_md)
        print(frame.to_string(index=False))
    elif args.command == "compare-predictions":
        report = compare_predictions(
            args.pred_a,
            args.pred_b,
            args.out,
            name_a=args.name_a,
            name_b=args.name_b,
            target=args.target,
            score_column_a=args.score_col_a,
            score_column_b=args.score_col_b,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "run-encoder-comparison":
        run_deep_encoder_comparison(
            args.csv,
            args.out_root,
            encoders=args.encoders,
            epochs=args.epochs,
            batch_size=args.batch_size,
            length=args.length,
            device=args.device,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    elif args.command == "run-ablations":
        run_v12_ablations(
            args.csv,
            args.out_root,
            model=args.model,
            encoder=args.encoder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    elif args.command == "run-holdout-matrix":
        run_holdout_matrix(
            args.out_root,
            codecs=args.codecs,
            calibration_codec=args.calibration_codec,
            method=args.method,
            encoder=args.encoder,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            length=args.length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            background_fasta=args.background_fasta,
            dry_run=args.dry_run,
        )
    elif args.command == "run-v12-suite":
        run_v12_suite(
            args.out_root,
            background_fasta=args.background_fasta,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            length=args.length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            encoders=args.encoders,
            device=args.device,
            seed=args.seed,
            dry_run=args.dry_run,
        )
    elif args.command == "extract-candidates":
        rows = extract_candidate_sequences(
            args.fasta,
            args.candidates,
            args.out,
            out_fasta=args.out_fasta,
            flank=args.flank,
            target_length=args.target_length,
            min_acgt_fraction=args.min_acgt_fraction,
            max_candidates=args.max_candidates,
            split_label=args.split_label,
        )
        print(f"extracted_candidates={len(rows)} out={args.out}")
    elif args.command == "score-candidates":
        rows = score_candidates(
            args.candidates,
            args.model,
            args.out,
            batch_size=args.batch_size,
            device=args.device,
            include_sequence=args.include_sequence,
            unknown_calibrator=args.unknown_calibrator,
            evidence_fusion_model=args.evidence_fusion_model,
        )
        top_score = max((float(row.get("unknown_score", 0.0)) for row in rows), default=0.0)
        print(f"scored_candidates={len(rows)} top_unknown_score={top_score:.4f} out={args.out}")
    elif args.command == "build-image-dataset":
        manifest = build_image_genome_dataset(
            args.genome_manifest,
            args.out,
            image_root=args.image_root,
            image_manifest=args.image_manifest,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            positive_fraction=args.positive_fraction,
            image_size=args.image_size,
            resize_short_side=args.resize_short_side,
            window_length=args.window_length,
            min_acgt_fraction=args.min_acgt_fraction,
            codecs=args.codecs,
            train_codecs=args.train_codecs,
            val_codecs=args.val_codecs,
            test_codecs=args.test_codecs,
            codec_key=load_codec_key(args.codec_key_file),
            kmer_order=args.kmer_order,
            ecc_symbols=args.ecc_symbols,
            lm_model_path=args.lm_model_file,
            lm_block_bytes=args.lm_block_bytes,
            lm_max_symbols_per_block=args.lm_max_symbols_per_block,
            lm_context_bases=args.lm_context_bases,
            lm_cdf_precision_bits=args.lm_cdf_precision_bits,
            lm_uniform_mix=args.lm_uniform_mix,
            substitution_rate=args.substitution_rate,
            indel_rate=args.indel_rate,
            seed=args.seed,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif args.command == "train-l5-context-model":
        model = KmerProbabilityModel.from_fasta(
            args.fasta,
            order=args.order,
            pseudocount=args.pseudocount,
            minimum_context_count=args.minimum_context_count,
            include_reverse_complement=not args.no_reverse_complement,
            max_bases=args.max_bases,
        )
        model.save(args.out)
        print(json.dumps({**model.metadata(), "model_file": str(args.out.resolve())}, indent=2))
    elif args.command == "test-l5-roundtrip":
        context = _read_l5_context(args.context_sequence, args.context_file)
        payload = args.input.read_bytes()
        codec = make_stego_codec(
            "lm",
            key=load_codec_key(args.codec_key_file),
            ecc_symbols=args.ecc_symbols,
            lm_model_path=args.model,
            lm_block_bytes=args.block_bytes,
            lm_max_symbols_per_block=args.max_symbols_per_block,
            lm_context_bases=args.context_bases,
            lm_cdf_precision_bits=args.cdf_precision_bits,
            lm_uniform_mix=args.uniform_mix,
        )
        dna = codec.encode(payload, sample_id=args.sample_id, context=context)
        recovered = codec.decode(
            dna,
            expected_bytes=len(payload),
            sample_id=args.sample_id,
            context=context,
        )
        args.dna_out.parent.mkdir(parents=True, exist_ok=True)
        args.recovered_out.parent.mkdir(parents=True, exist_ok=True)
        args.dna_out.write_text(dna + "\n", encoding="ascii")
        args.recovered_out.write_bytes(recovered)
        report = {
            "exact": recovered == payload,
            "input_bytes": len(payload),
            "encoded_bases": len(dna),
            "effective_bits_per_base": len(payload) * 8 / len(dna) if dna else None,
            "dna_out": str(args.dna_out.resolve()),
            "recovered_out": str(args.recovered_out.resolve()),
            "runtime": codec.runtime_metadata(),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "list-stego-codecs":
        print(json.dumps(available_stego_codecs(), ensure_ascii=False, indent=2))
    elif args.command == "summarize-stego-dataset":
        report = summarize_stego_dataset(args.dataset, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "verify-stego-roundtrip":
        report = verify_stego_roundtrip_dataset(
            args.dataset,
            args.out,
            codec_key=load_codec_key(args.codec_key_file),
            split=args.split,
            max_samples=args.max_samples,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"] and not args.allow_failures:
            raise SystemExit(2)
    elif args.command == "audit-image-dataset":
        report = audit_image_genome_dataset(args.dataset)
        if args.out is not None:
            save_json(report, args.out)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["passed"]:
            raise SystemExit(2)
    elif args.command == "train-image-locator":
        report = train_image_locator(
            args.dataset,
            args.out,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            backbone=args.backbone,
            d_model=args.d_model,
            downsample_stride=args.downsample_stride,
            hyena_model_name=args.hyena_model_name,
            hyena_revision=args.hyena_revision,
            hyena_local_files_only=args.hyena_local_files_only,
            context=args.context,
            context_layers=args.context_layers,
            dropout=args.dropout,
            hyena_fine_tune=args.hyena_fine_tune,
            hyena_last_n_layers=args.hyena_last_n_layers,
            hyena_reverse_complement=not args.hyena_no_reverse_complement,
            hyena_pretrained=not args.hyena_random_init,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation=args.gradient_accumulation,
            grad_clip=args.grad_clip,
            presence_weight=args.presence_weight,
            start_weight=args.start_weight,
            offset_weight=args.offset_weight,
            segment_weight=args.segment_weight,
            rc_consistency_weight=args.rc_consistency_weight,
            workers=args.workers,
            device=args.device,
            amp=args.amp,
            codec_key_file=args.codec_key_file,
            seed=args.seed,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "evaluate-image-locator":
        report = evaluate_image_locator_model(
            args.dataset,
            args.model,
            args.out,
            split=args.split,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
            save_images=args.save_images,
            max_saved_images=args.max_saved_images,
            codec_key_file=args.codec_key_file,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "run-hyenadna-comparison":
        report = run_hyenadna_comparison(
            args.dataset,
            args.out_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            d_model=args.d_model,
            downsample_stride=args.downsample_stride,
            hyena_model_name=args.hyena_model_name,
            hyena_revision=args.hyena_revision,
            hyena_local_files_only=args.hyena_local_files_only,
            workers=args.workers,
            device=args.device,
            amp=not args.no_amp,
            gradient_accumulation=args.gradient_accumulation,
            codec_key_file=args.codec_key_file,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))


def _read_l5_context(sequence: str, path: Path | None) -> str:
    if path is None:
        value = str(sequence).upper()
    else:
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")]
        value = "".join(lines).upper()
    if any(base not in "ACGTN" for base in value):
        raise ValueError("L5 context may contain only A/C/G/T/N.")
    return value


def _add_pe_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--prototypes-per-class", type=int, default=4)
    parser.add_argument("--family-weight", type=float, default=0.35)
    parser.add_argument("--contrast-weight", type=float, default=0.1)
    parser.add_argument("--naturalness-weight", type=float, default=0.3)
    parser.add_argument("--prototype-weight", type=float, default=0.3)
    parser.add_argument("--localization-weight", type=float, default=0.25)
    parser.add_argument("--episodic-weight", type=float, default=0.2)
    parser.add_argument("--episodic-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)


def _scan(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    windows = iter_fasta_windows(
        args.fasta,
        window_size=args.window,
        step=args.step,
        min_acgt_fraction=args.min_acgt_fraction,
        source=args.source or str(args.fasta),
        species=args.species,
        max_windows=args.max_windows,
    )
    feature_rows = [compute_window_features(window, include_sequence=args.include_sequence) for window in windows]
    scored_rows, baseline = score_feature_rows(feature_rows)
    candidate_rows = [row for row in scored_rows if float(row["anomaly_score"]) >= args.threshold]
    if args.top_n and args.top_n > 0:
        candidate_rows = candidate_rows[: args.top_n]
    candidates = [row_to_hit(row) for row in candidate_rows]

    write_csv(args.out / "windows.features.csv", scored_rows)
    write_csv(args.out / "candidates.csv", candidate_rows)
    write_bed(args.out / "candidates.bed", candidates)
    write_summary_json(
        args.out / "summary.json",
        fasta=str(args.fasta),
        window_size=args.window,
        step=args.step,
        total_windows=len(scored_rows),
        candidates=candidates,
        threshold=args.threshold,
        baseline=baseline,
    )
    write_markdown_report(
        args.out / "mining_report.md",
        fasta=str(args.fasta),
        window_size=args.window,
        step=args.step,
        total_windows=len(scored_rows),
        candidates=candidates,
        threshold=args.threshold,
    )
    top_score = candidates[0].score if candidates else 0.0
    print(
        f"scored_windows={len(scored_rows)} candidates={len(candidates)} "
        f"threshold={args.threshold:.3f} top_score={top_score:.3f} out={args.out}"
    )


def _build_controls(args: argparse.Namespace) -> None:
    rows, manifest = build_carrier_benchmark(
        args.out,
        generator_kind=args.generator_kind,
        protocol=args.protocol,
        length=args.length,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        holdout_codecs=args.holdout_codecs,
        calibration_codecs=args.calibration_codecs,
        negative_rate=args.negative_rate,
        train_detectability=args.train_detectability,
        test_detectability=args.test_detectability,
        seed=args.seed,
        allow_hard_negatives=not args.no_hard_negatives,
        background_fasta=args.background_fasta,
    )
    known = sum(1 for row in rows if row["open_target"] == "known_special")
    unknown = sum(1 for row in rows if row["open_target"] == "unknown_anomaly")
    natural = sum(1 for row in rows if row["open_target"] == "natural")
    print(
        f"rows={len(rows)} natural={natural} known_special={known} unknown_anomaly={unknown} "
        f"generator={manifest['generator_kind']} out={args.out}"
    )


def _train_pe_nac(args: argparse.Namespace, *, variant: str) -> None:
    train_pe_nac_plus(
        args.csv,
        args.out,
        variant=variant,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        length=args.length,
        prototypes_per_class=args.prototypes_per_class,
        family_weight=args.family_weight,
        contrast_weight=args.contrast_weight,
        naturalness_weight=args.naturalness_weight,
        prototype_weight=args.prototype_weight,
        localization_weight=args.localization_weight,
        episodic_weight=args.episodic_weight,
        episodic_prob=args.episodic_prob,
        seed=args.seed,
        device=args.device,
    )


def _train_deep(args: argparse.Namespace) -> None:
    train_deep_carrier_model(
        args.csv,
        args.out,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        length=args.length,
        prototypes_per_class=args.prototypes_per_class,
        hier_contrast_weight=args.hier_contrast_weight,
        prototype_weight=args.prototype_weight,
        localization_weight=args.localization_weight,
        episodic_weight=args.episodic_weight,
        episodic_prob=args.episodic_prob,
        seed=args.seed,
        device=args.device,
    )
