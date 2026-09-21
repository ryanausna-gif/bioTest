from __future__ import annotations

from pathlib import Path


UNKNOWN_FEATURE_COLUMNS = [
    "special_prob",
    "known_conf",
    "naturalness_prob",
    "prototype_unknown_score",
    "mixture_prototype_unknown_score",
    "energy_unknown_score",
    "localization_max_score",
    "sparse_site_max_score",
]
UNKNOWN_DERIVED_FEATURES = ["carrier_x_unknown_conf", "carrier_x_non_natural"]


def _require_calibration_deps():
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "Unknown-score calibration requires numpy, scikit-learn, and joblib."
        ) from exc
    return joblib, np, LogisticRegression, Pipeline, StandardScaler


def _unknown_feature_values(df, name: str, np):
    zeros = np.zeros(len(df), dtype=float)
    if name in UNKNOWN_FEATURE_COLUMNS:
        return df[name].to_numpy(dtype=float) if name in df.columns else zeros
    special = df["special_prob"].to_numpy(dtype=float) if "special_prob" in df.columns else zeros
    if name == "carrier_x_unknown_conf":
        known = df["known_conf"].to_numpy(dtype=float) if "known_conf" in df.columns else zeros
        return special * (1.0 - known)
    if name == "carrier_x_non_natural":
        natural = (
            df["naturalness_prob"].to_numpy(dtype=float)
            if "naturalness_prob" in df.columns
            else zeros
        )
        return special * (1.0 - natural)
    if name == "zero":
        return zeros
    raise ValueError(f"Unknown calibration feature: {name}")


def build_unknown_features(df, feature_names: list[str] | None = None):
    _joblib, np, _LogisticRegression, _Pipeline, _StandardScaler = _require_calibration_deps()
    if feature_names is None:
        names = list(UNKNOWN_FEATURE_COLUMNS)
        if "special_prob" in df.columns and "known_conf" in df.columns:
            names.append("carrier_x_unknown_conf")
        if "special_prob" in df.columns and "naturalness_prob" in df.columns:
            names.append("carrier_x_non_natural")
    else:
        names = list(feature_names)
    if not names:
        names = ["zero"]
    values = [_unknown_feature_values(df, name, np) for name in names]
    return np.vstack(values).T, names


def fit_unknown_calibrator(df, out_path: str | Path | None = None):
    joblib, _np, LogisticRegression, Pipeline, StandardScaler = _require_calibration_deps()
    y = ((df["y"] == 1) & (df["is_unknown"] == 1)).astype(int).to_numpy()
    if len(set(y.tolist())) < 2:
        raise ValueError("Unknown-score calibration requires both known/in-distribution and unknown samples.")
    x, feature_names = build_unknown_features(df)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(x, y)
    artifact = {"model": model, "feature_names": feature_names, "target": "unknown"}
    if out_path is not None:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, output)
    return artifact


def apply_unknown_calibrator(df, artifact):
    if not isinstance(artifact, dict) or "model" not in artifact:
        artifact = {"model": artifact, "feature_names": None}
    x, _names = build_unknown_features(df, artifact.get("feature_names"))
    return artifact["model"].predict_proba(x)[:, 1]


def load_unknown_calibrator(path: str | Path):
    joblib, _np, _LogisticRegression, _Pipeline, _StandardScaler = _require_calibration_deps()
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "feature_names": None, "target": "unknown"}
