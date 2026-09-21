import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.neighbors import NearestNeighbors


def _sigmoid_standardized(x):
    x = np.asarray(x, dtype=float)
    z = (x - np.mean(x)) / (np.std(x) + 1e-8)
    return 1.0 / (1.0 + np.exp(-z))


def msp_score(logits):
    x = np.asarray(logits, dtype=float)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x); p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    return 1.0 - p.max(axis=1)


def energy_score(logits):
    x = np.asarray(logits, dtype=float)
    m = x.max(axis=1, keepdims=True)
    energy_known = m.squeeze(1) + np.log(np.exp(x - m).sum(axis=1))
    return _sigmoid_standardized(-energy_known)


def mahalanobis_score(train_emb, train_labels, test_emb, reg=1e-4):
    train_emb = np.asarray(train_emb, dtype=float)
    test_emb = np.asarray(test_emb, dtype=float)
    labels = np.asarray(train_labels)
    classes = sorted(set(labels.tolist()))
    cov = EmpiricalCovariance().fit(train_emb)
    precision = cov.precision_ + reg * np.eye(train_emb.shape[1])
    dists = []
    for c in classes:
        mu = train_emb[labels == c].mean(axis=0)
        diff = test_emb - mu
        d = np.einsum("bi,ij,bj->b", diff, precision, diff)
        dists.append(d)
    min_d = np.vstack(dists).min(axis=0)
    return _sigmoid_standardized(min_d)


def knn_ood_score(train_emb, test_emb, k=10):
    nn = NearestNeighbors(n_neighbors=min(k, len(train_emb))).fit(train_emb)
    d, _ = nn.kneighbors(test_emb)
    return _sigmoid_standardized(d.mean(axis=1))


def feature_norm_score(test_emb):
    return _sigmoid_standardized(-np.linalg.norm(test_emb, axis=1))
