from __future__ import annotations

from pathlib import Path


BASE_FEATURES = [
    "special_prob",
    "known_conf",
    "naturalness_prob",
    "prototype_unknown_score",
    "mixture_prototype_unknown_score",
    "energy_unknown_score",
    "localization_max_score",
    "sparse_site_max_score",
]
DERIVED_FEATURES = [
    "carrier_x_unknown_conf",
    "carrier_x_non_natural",
    "energy_x_prototype",
]


def _require_fusion_deps():
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Evidence fusion requires numpy, scikit-learn, and joblib.") from exc
    return joblib, np, GradientBoostingClassifier, LogisticRegression, Pipeline, StandardScaler


def _feature_values(df, name: str, np):
    zeros = np.zeros(len(df), dtype=float)
    if name in BASE_FEATURES:
        return df[name].to_numpy(dtype=float) if name in df.columns else zeros
    special = df["special_prob"].to_numpy(dtype=float) if "special_prob" in df.columns else zeros
    known = df["known_conf"].to_numpy(dtype=float) if "known_conf" in df.columns else zeros
    natural = df["naturalness_prob"].to_numpy(dtype=float) if "naturalness_prob" in df.columns else zeros
    energy = df["energy_unknown_score"].to_numpy(dtype=float) if "energy_unknown_score" in df.columns else zeros
    prototype = (
        df["prototype_unknown_score"].to_numpy(dtype=float)
        if "prototype_unknown_score" in df.columns
        else zeros
    )
    if name == "carrier_x_unknown_conf":
        return special * (1.0 - known)
    if name == "carrier_x_non_natural":
        return special * (1.0 - natural)
    if name == "energy_x_prototype":
        return energy * prototype
    if name == "zero":
        return zeros
    raise ValueError(f"Unknown evidence feature: {name}")


def build_evidence_features(df, feature_names: list[str] | None = None):
    _joblib, np, _GradientBoostingClassifier, _LogisticRegression, _Pipeline, _StandardScaler = _require_fusion_deps()
    if feature_names is None:
        names = [name for name in BASE_FEATURES if name in df.columns]
        if "special_prob" in df.columns and "known_conf" in df.columns:
            names.append("carrier_x_unknown_conf")
        if "special_prob" in df.columns and "naturalness_prob" in df.columns:
            names.append("carrier_x_non_natural")
        if "energy_unknown_score" in df.columns and "prototype_unknown_score" in df.columns:
            names.append("energy_x_prototype")
        if not names:
            names = ["zero"]
    else:
        names = list(feature_names)
    values = [_feature_values(df, name, np) for name in names]
    return np.vstack(values).T, names


def fit_evidence_fusion(df, target: str = "unknown", model_type: str = "logreg"):
    _joblib, _np, GradientBoostingClassifier, LogisticRegression, Pipeline, StandardScaler = _require_fusion_deps()
    y = (
        ((df.y == 1) & (df.is_unknown == 1)).astype(int).to_numpy()
        if target == "unknown"
        else df.y.astype(int).to_numpy()
    )
    if len(set(y.tolist())) < 2:
        raise ValueError(f"Evidence fusion for target '{target}' requires both target classes.")
    x, names = build_evidence_features(df)
    model = (
        GradientBoostingClassifier(random_state=42)
        if model_type == "gb"
        else Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
    )
    model.fit(x, y)
    return model, names


def apply_evidence_fusion(df, model, feature_names: list[str] | None = None):
    x, _names = build_evidence_features(df, feature_names=feature_names)
    return model.predict_proba(x)[:, 1]


def save_model(model, names: list[str], path: str | Path, *, target: str = "unknown") -> None:
    joblib, _np, _GradientBoostingClassifier, _LogisticRegression, _Pipeline, _StandardScaler = _require_fusion_deps()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": names, "target": target}, output)


def load_model(path: str | Path):
    model, names, _target = load_artifact(path)
    return model, names


def load_artifact(path: str | Path):
    joblib, _np, _GradientBoostingClassifier, _LogisticRegression, _Pipeline, _StandardScaler = _require_fusion_deps()
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("feature_names", []), artifact.get("target", "unknown")
    return artifact, [], "unknown"
