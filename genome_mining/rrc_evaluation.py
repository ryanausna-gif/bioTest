"""Audited three-task evaluation with donor bootstrap and reusable detectors."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .rrc_data import audit_carriers, dna_signature

TASKS = (("natural", "ordinary"), ("ordinary", "rrc"), ("natural", "rrc"))


def score_metrics(y, scores, threshold):
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, scores = np.asarray(y), np.asarray(scores)
    predicted = scores > threshold
    return {"auroc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores)),
            "test_fpr": float(np.mean(predicted[y == 0])),
            "test_tpr": float(np.mean(predicted[y == 1]))}


def donor_intervals(y, scores, donors, threshold, repeats, seed):
    import numpy as np
    unique = sorted(set(donors))
    if repeats == 0 or len(unique) < 2:
        return None
    groups = {d: np.flatnonzero(np.asarray(donors) == d) for d in unique}
    rng, values = np.random.default_rng(seed), []
    for _ in range(repeats):
        indices = np.concatenate([groups[d] for d in rng.choice(unique, len(unique), replace=True)])
        values.append(score_metrics(np.asarray(y)[indices], np.asarray(scores)[indices], threshold))
    return {name: [float(x) for x in np.quantile([v[name] for v in values], [0.025, 0.975])]
            for name in values[0]}


def select_threshold(y, scores, fpr):
    import numpy as np
    negatives = np.asarray(scores)[np.asarray(y) == 0]
    return float(np.quantile(negatives, 1 - fpr, method="higher"))


def naturalness(rows):
    import math
    from .baseline_model import _max_run, _entropy
    result = {}
    for kind in ("natural", "ordinary", "rrc"):
        seqs = [r["sequence"] for r in rows if r["kind"] == kind]
        counts = Counter(s[i:i+4] for s in seqs for i in range(len(s)-3))
        result[kind] = {"samples": len(seqs),
                        "gc_mean": sum((s.count("G")+s.count("C"))/len(s) for s in seqs)/len(seqs),
                        "max_run_mean": sum(_max_run(s) for s in seqs)/len(seqs),
                        "base_entropy_mean": sum(_entropy(s) for s in seqs)/len(seqs), "counts": counts}
    for a, b in TASKS:
        ca, cb = result[a]["counts"], result[b]["counts"]
        na, nb, jsd = sum(ca.values()), sum(cb.values()), 0.0
        if not na or not nb:
            result[a + "_vs_" + b + "_4mer_jsd_bits"] = None
            continue
        for kmer in ca.keys() | cb.keys():
            p, q = ca[kmer]/na, cb[kmer]/nb
            middle = (p+q)/2
            jsd += (p * math.log2(p/middle) if p else 0) / 2
            jsd += (q * math.log2(q/middle) if q else 0) / 2
        result[a + "_vs_" + b + "_4mer_jsd_bits"] = jsd
    for kind in ("natural", "ordinary", "rrc"):
        result[kind].pop("counts")
    return result


def evaluate_experiment(args):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_curve
    from .baseline_model import featurize
    from .rrc_cli import read_csv, write_csv, write_json, new_output
    from .rrc_detectors import fit_cnn, predict_cnn, HyenaFeatures

    if not 0 < args.fpr < 1 or args.epochs < 1 or args.batch_size < 1 or args.bootstrap < 0:
        raise ValueError("Invalid evaluation parameters")
    dataset = Path(args.csv).parent
    run_path = dataset / "run.json"
    run = json.loads(run_path.read_text()) if run_path.exists() else None
    if run:
        if run.get("status") not in {None, "complete", "complete_with_failures"}:
            raise ValueError("Generation is incomplete; resume it before evaluation")
        if run.get("failed", run.get("attempted", 0)-run.get("successful", 0)) and not args.allow_failures:
            raise ValueError("Generation failures present; inspect failures.json or pass --allow-failures")
    rows = read_csv(args.csv)
    audit = audit_carriers(rows)
    out = new_output(args.out)
    write_json(out / "evaluation.json", {"status": "running", "detector": args.detector,
                                         "audit": audit, "dataset_run": run,
                                         "csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
                                         "seed": args.seed, "bootstrap": args.bootstrap})
    # Extract features once for all tasks; never include kind, donor, nonce or stop positions.
    encoder = None
    if args.detector == "logreg":
        all_features, feature_names = featurize([r["sequence"] for r in rows], k=4)
    elif args.detector == "hyena-probe":
        if not args.hf_model:
            raise ValueError("hyena-probe requires --hf-model")
        encoder = HyenaFeatures(args.hf_model, args.device, len(rows[0]["sequence"]), args.trust_remote_code)
        all_features = encoder.transform([r["sequence"] for r in rows])
        feature_names = [f"embedding_{i}" for i in range(all_features.shape[1])]
    report, predictions, roc_rows = {}, [], []
    for negative, positive in TASKS:
        task = negative + "_vs_" + positive
        task_dir = out / task
        task_dir.mkdir()
        sets = {}
        for split in ("train", "val", "test"):
            indices = [i for i, r in enumerate(rows) if r["split"] == split and r["kind"] in {negative, positive}]
            subset = [rows[i] for i in indices]
            y = np.asarray([int(r["kind"] == positive) for r in subset])
            if set(y) != {0, 1}:
                raise ValueError(f"Both classes required in {split} for {task}")
            x = [r["sequence"] for r in subset] if args.detector == "cnn" else all_features[indices]
            sets[split] = (x, y, subset)
        history = None
        if args.detector == "cnn":
            classifier, history = fit_cnn(sets["train"][:2], sets["val"][:2], epochs=args.epochs,
                batch_size=args.batch_size, device=args.device, seed=args.seed, out=task_dir)
            def predict(x):
                return predict_cnn(classifier, x, args.device, args.batch_size)
        else:
            classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=args.seed))
            classifier.fit(sets["train"][0], sets["train"][1])
            joblib.dump(classifier, task_dir / "classifier.joblib")
            def predict(x):
                return classifier.predict_proba(x)[:, 1]
        val, test = predict(sets["val"][0]), predict(sets["test"][0])
        threshold = select_threshold(sets["val"][1], val, args.fpr)
        y, test_rows = sets["test"][1], sets["test"][2]
        donors = [r["donor_id"] for r in test_rows]
        warning = []
        if sum(sets["val"][1] == 0) < 10 / args.fpr:
            warning.append("Too few validation negatives for a stable low-FPR estimate")
        if len(set(donors)) < 5:
            warning.append("Few test donors; donor-bootstrap interval is absent or unstable")
        if run and run.get("failed", 0):
            warning.append("Conditional on successful embedding; selection bias remains")
        report[task] = {**score_metrics(y, test, threshold), "threshold": threshold,
                        "target_val_fpr": args.fpr,
                        "actual_val_fpr": float(np.mean(val[sets["val"][1] == 0] > threshold)),
                        "test_samples": len(y), "test_donors": len(set(donors)),
                        "val_negatives": int(sum(sets["val"][1] == 0)),
                        "donor_bootstrap_95ci": donor_intervals(y, test, donors, threshold, args.bootstrap, args.seed),
                        "warnings": warning}
        metadata = {"format": "rrc_detector_v1", "detector": args.detector,
                    "negative": negative, "positive": positive, "threshold": threshold,
                    "sequence_length": len(rows[0]["sequence"]), "seed": args.seed,
                    "feature_names": feature_names if args.detector != "cnn" else None,
                    "encoder": encoder.adapter.metadata() if encoder else None,
                    "fit_donors": sorted({r["donor_id"] for r in rows if r["split"] in {"train", "val"}}),
                    "fit_sequence_signatures": sorted({dna_signature(r["sequence"]) for r in rows
                                                        if r["split"] in {"train", "val"}}),
                    "source_csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()}
        write_json(task_dir / "detector.json", metadata)
        if history is not None:
            write_json(task_dir / "training.json", history)
        for row, label, probability in zip(test_rows, y, test):
            predictions.append({"task": task, "sample_id": row["sample_id"], "pair_id": row["pair_id"],
                                "donor_id": row["donor_id"], "label": int(label), "score": float(probability),
                                "predicted_positive": int(probability > threshold)})
        fprs, tprs, _ = roc_curve(y, test)
        roc_rows.extend({"task": task, "fpr": float(f), "tpr": float(t)} for f, t in zip(fprs, tprs))
        print(json.dumps({"task": task, **report[task]}), flush=True)
    write_json(out / "report.json", report)
    write_csv(out / "predictions.csv", predictions)
    write_csv(out / "roc.csv", roc_rows)
    write_json(out / "diagnostics.json", {split: naturalness([r for r in rows if r["split"] == split]) for split in ("train", "val", "test")})
    lines = ["# RRC detector experiment", "", f"Detector: {args.detector}", "",
             "| Task | AUROC | Test FPR | Test TPR |", "|---|---:|---:|---:|"]
    for task, values in report.items():
        lines.append(f"| {task} | {values['auroc']:.4f} | {values['test_fpr']:.4f} | {values['test_tpr']:.4f} |")
    lines += ["", "Thresholds were selected on validation negatives only. Predictions use sequence-only features.",
              "Bootstrap samples whole test donors, retaining paired classes. It does not include training or calibration uncertainty.",
              "Chance-level performance is not a proof of distributional equality or steganographic security."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    evaluation = json.loads((out / "evaluation.json").read_text())
    evaluation["status"] = "complete"
    write_json(out / "evaluation.json", evaluation)


def score_saved(args):
    import joblib
    from .rrc_cli import read_csv, write_csv
    from .baseline_model import featurize
    from .rrc_detectors import cnn_model, predict_cnn, initialize_torch, HyenaFeatures
    root = Path(args.detector_dir)
    meta = json.loads((root / "detector.json").read_text())
    if meta["format"] != "rrc_detector_v1" or args.batch_size < 1:
        raise ValueError("Invalid detector format/batch size")
    if Path(args.out).exists():
        raise ValueError("Output already exists")
    rows = read_csv(args.csv)
    seqs = [r["sequence"] for r in rows]
    if not seqs or any(len(s) != meta["sequence_length"] or set(s)-set("ACGT") for s in seqs):
        raise ValueError("Sequence length/alphabet differs from saved detector")
    if meta["detector"] == "cnn":
        torch = initialize_torch(meta["seed"])
        classifier = cnn_model().to(args.device)
        classifier.load_state_dict(torch.load(root / "weights.pt", map_location=args.device, weights_only=True))
        scores = predict_cnn(classifier, seqs, args.device, args.batch_size)
    else:
        if meta["detector"] == "hyena-probe":
            if not args.hf_model:
                raise ValueError("Saved Hyena probe requires --hf-model")
            encoder = HyenaFeatures(args.hf_model, args.device, meta["sequence_length"], args.trust_remote_code)
            if encoder.adapter.metadata() != meta["encoder"]:
                raise ValueError("Frozen encoder fingerprint mismatch")
            features = encoder.transform(seqs)
        else:
            features, names = featurize(seqs, k=4)
            if names != meta["feature_names"]:
                raise ValueError("Feature definitions changed")
        scores = joblib.load(root / "classifier.joblib").predict_proba(features)[:, 1]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, [{"sample_id": r.get("sample_id", str(i)), "score": float(p),
                         "predicted_positive": int(p > meta["threshold"]),
                         "positive_class": meta["positive"]} for i, (r, p) in enumerate(zip(rows, scores))])


def evaluate_transfer(args):
    """Evaluate a locked detector on a target test split, without recalibration."""
    from types import SimpleNamespace
    from .rrc_cli import read_csv, write_csv, write_json, new_output
    meta = json.loads((Path(args.detector_dir) / "detector.json").read_text())
    if "fit_donors" not in meta or "fit_sequence_signatures" not in meta:
        raise ValueError("Legacy detector lacks transfer audit metadata; retrain with current version")
    if args.bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    run_path = Path(args.csv).parent / "run.json"
    if not run_path.exists():
        raise ValueError("Transfer evaluation requires the target dataset's run.json")
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete" or run.get("failed", 0):
        raise ValueError("Target generation must be complete without budget failures")
    rows = read_csv(args.csv)
    audit = audit_carriers(rows)
    target = [r for r in rows if r["split"] == "test" and r["kind"] in {meta["negative"], meta["positive"]}]
    if {r["kind"] for r in target} != {meta["negative"], meta["positive"]}:
        raise ValueError("Target test split needs both saved task classes")
    if set(meta["fit_donors"]) & {r["donor_id"] for r in target}:
        raise ValueError("Target test donors overlap source training/calibration donors")
    if set(meta["fit_sequence_signatures"]) & {dna_signature(r["sequence"]) for r in target}:
        raise ValueError("Target test sequences overlap source training/calibration sequences (including RC)")
    out = new_output(args.out)
    write_csv(out / "test_input.csv", target)
    score_saved(SimpleNamespace(detector_dir=args.detector_dir, csv=str(out/"test_input.csv"),
        out=str(out/"scores.csv"), device=args.device, batch_size=args.batch_size,
        hf_model=args.hf_model, trust_remote_code=args.trust_remote_code))
    scores = [float(r["score"]) for r in read_csv(out/"scores.csv")]
    labels = [int(r["kind"] == meta["positive"]) for r in target]
    donors = [r["donor_id"] for r in target]
    result = {"task": meta["negative"] + "_vs_" + meta["positive"],
              **score_metrics(labels, scores, meta["threshold"]),
              "threshold": meta["threshold"], "recalibrated": False, "audit": audit,
              "test_samples": len(target), "test_donors": len(set(donors)),
              "donor_bootstrap_95ci": donor_intervals(labels, scores, donors, meta["threshold"], args.bootstrap, args.seed),
              "source_csv_sha256": meta["source_csv_sha256"],
              "target_csv_sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
              "warning": "Same-length carriers only; near homology/pretraining overlap not audited. Few donors give unstable intervals."}
    write_json(out / "report.json", result)
    print(json.dumps(result, indent=2))
