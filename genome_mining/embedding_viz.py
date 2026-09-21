from __future__ import annotations

from pathlib import Path


def reduce_embeddings(embeddings, method: str = "pca", seed: int = 42):
    try:
        import numpy as np
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("Embedding reduction requires numpy and scikit-learn.") from exc

    emb = np.asarray(embeddings, dtype=float)
    if emb.ndim != 2 or len(emb) < 2:
        raise ValueError("Embedding visualization requires at least two 2D embedding rows.")
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(emb)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(1, min(30, len(emb) - 1, max(5, len(emb) // 20)))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=seed,
        ).fit_transform(emb)
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(emb)
        except ImportError:
            return PCA(n_components=2, random_state=seed).fit_transform(emb)
    raise ValueError(f"Unknown embedding reduction method: {method}")


def plot_embeddings(
    embeddings_path: str | Path,
    predictions_path: str | Path,
    out_path: str | Path,
    *,
    method: str = "pca",
    color_by: str = "open_target",
    max_points: int = 3000,
    seed: int = 42,
) -> dict[str, object]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Embedding plotting requires matplotlib, numpy, and pandas.") from exc

    frame = pd.read_csv(predictions_path)
    data = np.load(embeddings_path)
    key = "projection" if "projection" in data else "embedding"
    embeddings = data[key]
    n = min(len(frame), len(embeddings), max_points)
    if n < 2:
        raise ValueError("Embedding plotting requires at least two aligned prediction rows.")
    frame = frame.iloc[:n].reset_index(drop=True)
    xy = reduce_embeddings(embeddings[:n], method=method, seed=seed)
    if color_by in frame.columns:
        groups = frame[color_by].astype(str)
    elif "y" in frame.columns:
        groups = frame["y"].astype(str)
    else:
        groups = pd.Series(["all"] * n)

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, indices in groups.groupby(groups).groups.items():
        idx = list(indices)
        ax.scatter(xy[idx, 0], xy[idx, 1], s=8, alpha=0.65, label=str(name))
    ax.set_title(f"Embedding visualization ({method})")
    ax.set_xlabel("dimension 1")
    ax.set_ylabel("dimension 2")
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)

    coordinates = pd.DataFrame(
        {"x": xy[:, 0], "y": xy[:, 1], "group": groups.to_numpy(), "row_index": range(n)}
    )
    coordinates.to_csv(output.with_suffix(".csv"), index=False)
    return {
        "image": str(output),
        "coordinates": str(output.with_suffix(".csv")),
        "method": method,
        "color_by": color_by,
        "embedding_key": key,
        "n": n,
    }
