from __future__ import annotations

import itertools
import json
import math
import zlib
from collections import Counter
from pathlib import Path


ALPHABET = "ACGT"


def _require_baseline_deps():
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError as exc:
        raise ImportError("Baseline training requires numpy, pandas, scikit-learn, and joblib.") from exc
    return (
        joblib,
        np,
        pd,
        GradientBoostingClassifier,
        RandomForestClassifier,
        LogisticRegression,
        Pipeline,
        LabelEncoder,
        StandardScaler,
    )


def kmer_vocab(k: int) -> list[str]:
    return ["".join(parts) for parts in itertools.product(ALPHABET, repeat=int(k))]


def _gc_content(sequence: str) -> float:
    seq = str(sequence).upper()
    acgt = [base for base in seq if base in ALPHABET]
    if not acgt:
        return 0.0
    return sum(1 for base in acgt if base in "GC") / len(acgt)


def _entropy(sequence: str) -> float:
    seq = [base for base in str(sequence).upper() if base in ALPHABET]
    if not seq:
        return 0.0
    counts = Counter(seq)
    out = 0.0
    for count in counts.values():
        p = count / len(seq)
        out -= p * math.log2(p)
    return out


def _compression_ratio(sequence: str) -> float:
    seq = str(sequence).upper().encode("ascii", errors="ignore")
    return len(zlib.compress(seq)) / max(1, len(seq))


def _max_run(sequence: str) -> int:
    best = current = 0
    last = ""
    for base in str(sequence).upper():
        if base == last:
            current += 1
        else:
            current = 1
            last = base
        best = max(best, current)
    return best


def featurize(sequences: list[str], k: int = 4, use_extra: bool = True):
    _joblib, np, *_ = _require_baseline_deps()
    vocab = kmer_vocab(k)
    index = {mer: idx for idx, mer in enumerate(vocab)}
    rows = []
    for sequence in sequences:
        seq = str(sequence).upper()
        kmer = np.zeros(len(vocab), dtype=np.float32)
        total = 0
        for idx in range(0, len(seq) - k + 1):
            mer = seq[idx : idx + k]
            if mer in index:
                kmer[index[mer]] += 1
                total += 1
        if total:
            kmer /= total
        parts = [kmer]
        if use_extra:
            parts.append(np.asarray([_gc_content(seq), _entropy(seq), _compression_ratio(seq), _max_run(seq), len(seq)], dtype=np.float32))
        rows.append(np.concatenate(parts))
    names = [f"{k}mer_{mer}" for mer in vocab]
    if use_extra:
        names += ["gc", "entropy", "compression", "max_run", "length"]
    return np.vstack(rows), names


def _make_model(name: str):
    (
        _joblib,
        _np,
        _pd,
        GradientBoostingClassifier,
        RandomForestClassifier,
        LogisticRegression,
        Pipeline,
        _LabelEncoder,
        StandardScaler,
    ) = _require_baseline_deps()
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=42)
    if name == "gb":
        return GradientBoostingClassifier(random_state=42)
    raise ValueError(f"Unknown baseline model: {name}")


def train_baseline_model(csv_path: str | Path, out_dir: str | Path, *, model_name: str = "logreg", k: int = 4, use_extra: bool = True) -> dict[str, object]:
    (
        joblib,
        _np,
        pd,
        _GradientBoostingClassifier,
        _RandomForestClassifier,
        _LogisticRegression,
        _Pipeline,
        LabelEncoder,
        _StandardScaler,
    ) = _require_baseline_deps()
    from .model_metrics import evaluate_open_set, save_json

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    x_train, feature_names = featurize(train.sequence.astype(str).tolist(), k=k, use_extra=use_extra)
    x_val, _ = featurize(val.sequence.astype(str).tolist(), k=k, use_extra=use_extra)
    x_test, _ = featurize(test.sequence.astype(str).tolist(), k=k, use_extra=use_extra)

    binary_model = _make_model(model_name)
    binary_model.fit(x_train, train.y.values)
    label_encoder = LabelEncoder()
    y_family = label_encoder.fit_transform(train.family.astype(str))
    family_model = _make_model(model_name)
    family_model.fit(x_train, y_family)

    special_prob = binary_model.predict_proba(x_test)[:, 1]
    family_prob = family_model.predict_proba(x_test)
    known_conf = family_prob.max(axis=1)
    pred_family = label_encoder.inverse_transform(family_prob.argmax(axis=1))
    metrics, pred = evaluate_open_set(test, special_prob, known_conf)
    pred["pred_family"] = pred_family

    val_special_prob = binary_model.predict_proba(x_val)[:, 1]
    val_family_prob = family_model.predict_proba(x_val)
    _val_metrics, val_pred = evaluate_open_set(val, val_special_prob, val_family_prob.max(axis=1))
    val_pred["pred_family"] = label_encoder.inverse_transform(val_family_prob.argmax(axis=1))
    pred.to_csv(output / "predictions.csv", index=False)
    val_pred.to_csv(output / "validation_predictions.csv", index=False)
    save_json(metrics, output / "metrics.json")
    joblib.dump(
        {
            "model_type": "v12_kmer_baseline",
            "binary_model": binary_model,
            "family_model": family_model,
            "label_encoder": label_encoder,
            "k": k,
            "use_extra": use_extra,
            "feature_names": feature_names,
        },
        output / "model.joblib",
    )
    print(json.dumps(metrics.get("carrier_detection", {}), indent=2))
    print("Saved:", output)
    return metrics
