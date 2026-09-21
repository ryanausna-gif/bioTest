from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _nested(mapping: dict, *path: str):
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def make_leaderboard(root: str | Path, out_csv: str | Path, out_md: str | Path | None = None):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Leaderboard generation requires pandas.") from exc

    root_path = Path(root)
    rows = []
    for metrics_path in sorted(root_path.rglob("metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "run": str(metrics_path.parent.relative_to(root_path)) or ".",
                "method": metrics.get("method", metrics_path.parent.name),
                "encoder": metrics.get("encoder", ""),
                "carrier_auroc": _nested(metrics, "carrier_detection", "auroc"),
                "carrier_aupr": _nested(metrics, "carrier_detection", "aupr"),
                "carrier_fpr95": _nested(metrics, "carrier_detection", "fpr95"),
                "unknown_auroc": _nested(metrics, "unknown_codec_detection", "auroc"),
                "unknown_aupr": _nested(metrics, "unknown_codec_detection", "aupr"),
                "unknown_fpr95": _nested(metrics, "unknown_codec_detection", "fpr95"),
                "hard_negative_fpr": metrics.get("hard_negative_fpr_at_0.5"),
                "localization_iou": _nested(metrics, "localization", "mean_iou"),
            }
        )
    if not rows:
        raise ValueError(f"No metrics.json files found under {root_path}")
    frame = pd.DataFrame(rows)
    output = Path(out_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    markdown_path = Path(out_md) if out_md else output.with_suffix(".md")
    try:
        markdown = frame.to_markdown(index=False)
    except ImportError:
        markdown = frame.to_csv(index=False)
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    return frame


def compare_predictions(
    pred_a: str | Path,
    pred_b: str | Path,
    out_path: str | Path,
    *,
    name_a: str = "method_a",
    name_b: str = "method_b",
    target: str = "unknown",
    score_column_a: str | None = None,
    score_column_b: str | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    try:
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("Paired prediction comparison requires numpy, pandas, and scikit-learn.") from exc
    from .model_metrics import openood_report, save_json

    left = pd.read_csv(pred_a)
    right = pd.read_csv(pred_b)
    join_key = None
    for candidate in ("sample_id", "candidate_id", "window_id"):
        if candidate in left.columns and candidate in right.columns:
            join_key = candidate
            break
    if join_key:
        right_columns = [join_key]
        right_columns += [column for column in right.columns if column != join_key]
        paired = left.merge(right[right_columns], on=join_key, suffixes=("_a", "_b"), validate="one_to_one")
        y_column = "y_a" if "y_a" in paired.columns else "y"
        unknown_column = "is_unknown_a" if "is_unknown_a" in paired.columns else "is_unknown"
    else:
        n = min(len(left), len(right))
        if n == 0:
            raise ValueError("Prediction files are empty.")
        left = left.iloc[:n].reset_index(drop=True)
        right = right.iloc[:n].reset_index(drop=True)
        paired = pd.DataFrame({"y": left["y"], "is_unknown": left.get("is_unknown", 0)})
        paired["score_a"] = left[score_column_a or ("unknown_score" if target == "unknown" else "special_prob")]
        paired["score_b"] = right[score_column_b or ("unknown_score" if target == "unknown" else "special_prob")]
        y_column = "y"
        unknown_column = "is_unknown"

    default_score = "unknown_score" if target == "unknown" else "special_prob"
    if join_key:
        raw_a = score_column_a or default_score
        raw_b = score_column_b or default_score
        col_a = f"{raw_a}_a" if f"{raw_a}_a" in paired.columns else raw_a
        col_b = f"{raw_b}_b" if f"{raw_b}_b" in paired.columns else raw_b
    else:
        col_a, col_b = "score_a", "score_b"
    if col_a not in paired or col_b not in paired:
        raise ValueError(f"Missing paired score columns: {col_a}, {col_b}")
    if target == "unknown":
        labels = ((paired[y_column].astype(int) == 1) & (paired[unknown_column].astype(int) == 1)).astype(int).to_numpy()
    else:
        labels = paired[y_column].astype(int).to_numpy()
    score_a = paired[col_a].astype(float).to_numpy()
    score_b = paired[col_b].astype(float).to_numpy()
    if len(np.unique(labels)) < 2:
        raise ValueError("Paired comparison requires both positive and negative target labels.")

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(max(1, n_bootstrap)):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_y = labels[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        differences.append(
            float(roc_auc_score(sampled_y, score_b[indices]) - roc_auc_score(sampled_y, score_a[indices]))
        )
    observed = float(roc_auc_score(labels, score_b) - roc_auc_score(labels, score_a))
    report = {
        "target": target,
        "n_paired": int(len(labels)),
        "join_key": join_key or "row_order",
        name_a: openood_report(labels, score_a),
        name_b: openood_report(labels, score_b),
        "auroc_difference_b_minus_a": observed,
        "bootstrap": {
            "n_requested": int(n_bootstrap),
            "n_valid": int(len(differences)),
            "ci95": None
            if not differences
            else [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))],
            "probability_b_better": None
            if not differences
            else float(np.mean(np.asarray(differences) > 0.0)),
            "seed": int(seed),
        },
    }
    save_json(report, out_path)
    return report


def run_commands(commands: list[list[object]], *, dry_run: bool = False) -> None:
    for command in commands:
        rendered = [str(value) for value in command]
        print("Running:", " ".join(rendered))
        if not dry_run:
            subprocess.check_call(rendered)


def _module_command(command: str, *args: object) -> list[object]:
    return [sys.executable, "-m", "genome_mining", command, *args]


def run_deep_encoder_comparison(
    csv_path: str | Path,
    out_root: str | Path,
    *,
    encoders: list[str],
    epochs: int = 6,
    batch_size: int = 64,
    length: int | None = None,
    device: str | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    commands = []
    for encoder in encoders:
        command = _module_command(
            "train-deep-carrier",
            "--csv",
            csv_path,
            "--out",
            root / encoder,
            "--encoder",
            encoder,
            "--epochs",
            epochs,
            "--batch-size",
            batch_size,
            "--seed",
            seed,
        )
        if length is not None:
            command += ["--length", length]
        if device:
            command += ["--device", device]
        commands.append(command)
    run_commands(commands, dry_run=dry_run)
    if not dry_run:
        make_leaderboard(root, root / "deep_encoder_leaderboard.csv")


def run_v12_ablations(
    csv_path: str | Path,
    out_root: str | Path,
    *,
    model: str = "deep",
    encoder: str = "hybrid",
    epochs: int = 6,
    batch_size: int = 64,
    device: str | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    if model == "deep":
        settings = {
            "full": [],
            "no_hierarchical_contrast": ["--hier-contrast-weight", "0"],
            "no_prototype": ["--prototype-weight", "0"],
            "no_localization": ["--localization-weight", "0"],
            "no_episodic": ["--episodic-weight", "0", "--episodic-prob", "0"],
            "single_prototype": ["--prototypes-per-class", "1"],
        }
        command_name = "train-deep-carrier"
        shared = ["--encoder", encoder]
    else:
        settings = {
            "full": [],
            "no_prototype": ["--prototype-weight", "0"],
            "no_contrast": ["--contrast-weight", "0"],
            "no_naturalness": ["--naturalness-weight", "0"],
            "no_localization": ["--localization-weight", "0"],
            "no_episodic": ["--episodic-weight", "0", "--episodic-prob", "0"],
            "single_prototype": ["--prototypes-per-class", "1"],
        }
        command_name = "train-pe-nac-plus"
        shared = []
    commands = []
    for name, extra in settings.items():
        command = _module_command(
            command_name,
            "--csv",
            csv_path,
            "--out",
            root / name,
            "--epochs",
            epochs,
            "--batch-size",
            batch_size,
            "--seed",
            seed,
            *shared,
            *extra,
        )
        if device:
            command += ["--device", device]
        commands.append(command)
    run_commands(commands, dry_run=dry_run)
    if not dry_run:
        make_leaderboard(root, root / "ablation_leaderboard.csv")


def run_holdout_matrix(
    out_root: str | Path,
    *,
    codecs: list[str],
    calibration_codec: str = "StegoCodec",
    method: str = "deep",
    encoder: str = "hybrid",
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1500,
    length: int = 1000,
    epochs: int = 6,
    batch_size: int = 64,
    device: str | None = None,
    seed: int = 42,
    background_fasta: str | Path | None = None,
    dry_run: bool = False,
) -> None:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    commands: list[list[object]] = []
    for codec in codecs:
        if codec == calibration_codec:
            raise ValueError("calibration_codec must differ from every final holdout codec.")
        experiment = root / f"holdout_{codec}"
        benchmark = experiment / "benchmark.csv"
        build = _module_command(
            "build-controls",
            "--out",
            benchmark,
            "--generator-kind",
            "realistic",
            "--length",
            length,
            "--n-train",
            n_train,
            "--n-val",
            n_val,
            "--n-test",
            n_test,
            "--holdout-codecs",
            codec,
            "--calibration-codecs",
            calibration_codec,
            "--seed",
            seed,
        )
        if background_fasta:
            build += ["--background-fasta", background_fasta]
        commands.append(build)
        if method == "baseline":
            train = _module_command(
                "train-baseline", "--csv", benchmark, "--out", experiment / "model", "--model", "logreg"
            )
        elif method == "pe":
            train = _module_command(
                "train-pe-nac-plus",
                "--csv",
                benchmark,
                "--out",
                experiment / "model",
                "--epochs",
                epochs,
                "--batch-size",
                batch_size,
                "--seed",
                seed,
            )
        elif method == "ef":
            train = _module_command(
                "train-ef-pe-nac-plus",
                "--csv",
                benchmark,
                "--out",
                experiment / "model",
                "--epochs",
                epochs,
                "--batch-size",
                batch_size,
                "--seed",
                seed,
            )
        else:
            train = _module_command(
                "train-deep-carrier",
                "--csv",
                benchmark,
                "--out",
                experiment / "model",
                "--encoder",
                encoder,
                "--epochs",
                epochs,
                "--batch-size",
                batch_size,
                "--seed",
                seed,
            )
        if device and method != "baseline":
            train += ["--device", device]
        commands.append(train)
    run_commands(commands, dry_run=dry_run)
    if not dry_run:
        make_leaderboard(root, root / "holdout_matrix_leaderboard.csv")


def run_v12_suite(
    out_root: str | Path,
    *,
    background_fasta: str | Path | None = None,
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1500,
    length: int = 1000,
    epochs: int = 6,
    batch_size: int = 64,
    encoders: list[str] | None = None,
    device: str | None = None,
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    benchmark = root / "v12_benchmark.csv"
    encoders = list(encoders or ["cnn", "dilated_cnn", "hybrid"])
    build = _module_command(
        "build-controls",
        "--out",
        benchmark,
        "--generator-kind",
        "realistic",
        "--length",
        length,
        "--n-train",
        n_train,
        "--n-val",
        n_val,
        "--n-test",
        n_test,
        "--holdout-codecs",
        "AdaptiveStegoCodec",
        "DNACryptoCodec",
        "RealisticSparseWatermarkCodec",
        "--calibration-codecs",
        "CodonStegoCodec",
        "--seed",
        seed,
    )
    if background_fasta:
        build += ["--background-fasta", background_fasta]
    commands = [build]
    commands.append(_module_command("analyze-benchmark", "--csv", benchmark, "--out", root / "benchmark_audit.json"))
    commands.append(
        _module_command("train-baseline", "--csv", benchmark, "--out", root / "logreg", "--model", "logreg")
    )
    pe_command = _module_command(
        "train-pe-nac-plus",
        "--csv",
        benchmark,
        "--out",
        root / "v10_pe_nac_plus",
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--seed",
        seed,
    )
    ef_command = _module_command(
        "train-ef-pe-nac-plus",
        "--csv",
        benchmark,
        "--out",
        root / "v11_ef_pe_nac_plus",
        "--epochs",
        epochs,
        "--batch-size",
        batch_size,
        "--seed",
        seed,
    )
    if device:
        pe_command += ["--device", device]
        ef_command += ["--device", device]
    commands.extend([pe_command, ef_command])
    run_commands(commands, dry_run=dry_run)

    v10 = root / "v10_pe_nac_plus"
    v11 = root / "v11_ef_pe_nac_plus"
    migrated_post = [
        _module_command(
            "calibrate-unknown",
            "--calibration-pred",
            v10 / "validation_predictions.csv",
            "--pred",
            v10 / "predictions.csv",
            "--out",
            v10 / "predictions_calibrated.csv",
            "--model-out",
            v10 / "unknown_calibrator.joblib",
        ),
        _module_command(
            "train-evidence-fusion",
            "--calibration-pred",
            v11 / "validation_predictions.csv",
            "--pred",
            v11 / "predictions.csv",
            "--out",
            v11 / "predictions_calibrated.csv",
            "--model-out",
            v11 / "evidence_fusion.joblib",
            "--target",
            "unknown",
        ),
        _module_command(
            "evaluate-localization",
            "--pred",
            v11 / "predictions.csv",
            "--out",
            v11 / "multiscale_localization.json",
            "--mode",
            "multiscale",
        ),
        _module_command(
            "analyze-failures",
            "--pred",
            v11 / "predictions.csv",
            "--out",
            v11 / "failure_analysis.json",
        ),
    ]
    run_commands(migrated_post, dry_run=dry_run)
    run_deep_encoder_comparison(
        benchmark,
        root / "deep_encoders",
        encoders=encoders,
        epochs=epochs,
        batch_size=batch_size,
        length=length,
        device=device,
        seed=seed,
        dry_run=dry_run,
    )
    selected = "hybrid" if "hybrid" in encoders else encoders[0]
    run_dir = root / "deep_encoders" / selected
    post = [
        _module_command(
            "calibrate-unknown",
            "--calibration-pred",
            run_dir / "validation_predictions.csv",
            "--pred",
            run_dir / "predictions.csv",
            "--out",
            run_dir / "predictions_calibrated.csv",
            "--model-out",
            run_dir / "unknown_calibrator.joblib",
        ),
        _module_command(
            "run-ood-suite",
            "--pred",
            run_dir / "predictions_calibrated.csv",
            "--emb",
            run_dir / "embeddings.npz",
            "--reference-pred",
            run_dir / "validation_predictions.csv",
            "--reference-emb",
            run_dir / "validation_embeddings.npz",
            "--out",
            run_dir / "ood_suite_unknown.json",
            "--target",
            "unknown",
        ),
        _module_command(
            "evaluate-localization",
            "--pred",
            run_dir / "predictions.csv",
            "--out",
            run_dir / "multiscale_localization.json",
            "--mode",
            "multiscale",
        ),
        _module_command(
            "analyze-failures",
            "--pred",
            run_dir / "predictions.csv",
            "--out",
            run_dir / "failure_analysis.json",
        ),
        _module_command(
            "plot-embeddings",
            "--pred",
            run_dir / "predictions.csv",
            "--emb",
            run_dir / "embeddings.npz",
            "--out",
            root / "figures" / f"{selected}_embedding_pca.png",
            "--method",
            "pca",
            "--color-by",
            "open_target",
        ),
    ]
    run_commands(post, dry_run=dry_run)
    if not dry_run:
        make_leaderboard(root, root / "v12_leaderboard.csv")
