from __future__ import annotations


def train_embedding_classifier(train_embeddings, train_labels):
    """Fit the lightweight V12 classifier head on external foundation-model embeddings."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Foundation embedding adapters require numpy and scikit-learn.") from exc

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model.fit(np.asarray(train_embeddings), np.asarray(train_labels).astype(int))
    return model
