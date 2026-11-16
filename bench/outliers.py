"""`outlier_scores` against the detectors people use, on three planted-outlier fixtures.

The score under test is `dist(x, its cluster centroid) / that cluster's RMS radius`, a per-cluster
z-score, so it is local by construction where a global distance-to-mean is not. Each fixture puts
5 % of `N = 4000` rows outside the clusters in a different way: uniform background noise (global
outliers), a ring just outside a tight cluster next to a diffuse one (local outliers), and two
sheared clusters (the case a spherical radius model is worst at). Every detector runs at its
documented defaults on the same rows; betula at `n_clusters=5`, `feature="full"`,
`max_leaves=0.05` (200 leaves), with the default `metric="radius"` and with
`metric="mahalanobis"`, which whitens by the cluster's pooled covariance instead.

| family | detectors |
|---|---|
| scikit-learn | `IsolationForest`, `LocalOutlierFactor` |
| PyOD | `CBLOF` (cluster-based), `ECOD`, `KNN` |

Detection quality only — ROC-AUC and average precision are exact for given scores, so the run is
fit for a loaded machine. Resilience to parameters and the internal-index filter stay in
`local/scratch/outlier_dami.py`.

    uv run --no-sync --with scikit-learn --with pandas --with pyod python bench/outliers.py

Writes `bench/results_outliers.csv`, one row per fixture, seed and detector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)


def blobs_with_uniform_noise(seed, n=4000, dim=8, k=5, contamination=0.05):
    """Compact Gaussian clusters plus a uniform background: the textbook global-outlier case."""
    rng = np.random.default_rng(seed)
    n_out = int(n * contamination)
    centres = rng.uniform(-10.0, 10.0, (k, dim))
    inliers = np.vstack([rng.normal(c, 0.6, (max((n - n_out) // k, 1), dim)) for c in centres])
    outliers = rng.uniform(-14.0, 14.0, (n_out, dim))
    x = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(len(inliers), dtype=int), np.ones(n_out, dtype=int)]
    return x, y


def clusters_of_unequal_density(seed, n=4000, dim=4, contamination=0.05):
    """One tight cluster and one diffuse one: a local outlier problem, where a global radius
    threshold has to fail and a per-cluster z-score need not."""
    rng = np.random.default_rng(seed)
    n_out = int(n * contamination)
    tight = rng.normal(np.zeros(dim), 0.25, ((n - n_out) // 2, dim))
    loose = rng.normal(np.full(dim, 8.0), 2.0, ((n - n_out) // 2, dim))
    # The outliers sit just outside the tight cluster, closer to the data mean than most of the
    # diffuse cluster's own members, so distance alone cannot separate them.
    directions = rng.normal(0.0, 1.0, (n_out, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    outliers = 1.6 * directions
    x = np.vstack([tight, loose, outliers])
    y = np.r_[np.zeros(len(tight) + len(loose), dtype=int), np.ones(n_out, dtype=int)]
    return x, y


def elongated_with_noise(seed, n=4000, dim=6, contamination=0.05):
    """Anisotropic (sheared) clusters: the case a spherical radius model is worst at."""
    rng = np.random.default_rng(seed)
    n_out = int(n * contamination)
    shear = rng.normal(0.0, 1.0, (dim, dim))
    per = (n - n_out) // 2
    a = rng.normal(0.0, 1.0, (per, dim)) * np.array([6.0] + [0.4] * (dim - 1))
    b = rng.normal(0.0, 1.0, (per, dim)) * np.array([6.0] + [0.4] * (dim - 1)) + 9.0
    inliers = np.vstack([a, b]) @ shear
    outliers = rng.uniform(-1.0, 1.0, (n_out, dim)) * 12.0 @ shear
    x = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(len(inliers), dtype=int), np.ones(n_out, dtype=int)]
    return x, y


FIXTURES = {
    "blobs+uniform": blobs_with_uniform_noise,
    "unequal-density": clusters_of_unequal_density,
    "elongated": elongated_with_noise,
}


def _betula(x, seed, metric):
    import betula_cluster

    est = betula_cluster.Betula(
        n_clusters=5, feature="full", threshold=0.0, max_leaves=0.05, seed=seed
    ).fit(x)
    scores = np.asarray(est.outlier_scores(x, metric))
    # An unreachable centroid scores +inf; rank it first rather than poison the AUC arithmetic.
    return np.where(np.isfinite(scores), scores, np.nanmax(scores[np.isfinite(scores)]) + 1.0)


def _iforest(x, seed):
    from sklearn.ensemble import IsolationForest

    return -IsolationForest(random_state=seed).fit(x).score_samples(x)


def _lof(x, seed):
    from sklearn.neighbors import LocalOutlierFactor

    return -LocalOutlierFactor().fit(x).negative_outlier_factor_


def _pyod(name, x, seed):
    from pyod.models.cblof import CBLOF
    from pyod.models.ecod import ECOD
    from pyod.models.knn import KNN

    if name == "cblof":
        return CBLOF(random_state=seed).fit(x).decision_scores_
    return {"ecod": ECOD, "knn": KNN}[name]().fit(x).decision_scores_


# name → scores(x, seed); higher is more outlying.
DETECTORS = {
    "betula": lambda x, s: _betula(x, s, "radius"),
    "betula-mahalanobis": lambda x, s: _betula(x, s, "mahalanobis"),
    "sklearn-iforest": _iforest,
    "sklearn-lof": _lof,
    "pyod-cblof": lambda x, s: _pyod("cblof", x, s),
    "pyod-ecod": lambda x, s: _pyod("ecod", x, s),
    "pyod-knn": lambda x, s: _pyod("knn", x, s),
}


def main() -> int:
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for fixture, make in FIXTURES.items():
        for seed in SEEDS:
            x, y = make(seed)
            for detector, scores in DETECTORS.items():
                s = scores(x, seed)
                rows.append(
                    {
                        "fixture": fixture,
                        "n": len(x),
                        "dim": x.shape[1],
                        "seed": seed,
                        "detector": detector,
                        "roc_auc": roc_auc_score(y, s),
                        "avg_precision": average_precision_score(y, s),
                    }
                )
                print(
                    f"{fixture:16s} seed={seed} {detector:16s} "
                    f"{rows[-1]['roc_auc']:.3f} / {rows[-1]['avg_precision']:.3f}",
                    flush=True,
                )
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).with_name("results_outliers.csv"), index=False)
    med = df.groupby(["fixture", "detector"], sort=False)[["roc_auc", "avg_precision"]].median()
    print("\nmedian of seeds, ROC-AUC / average precision")
    for (fixture, detector), r in med.iterrows():
        print(f"{fixture:16s} {detector:16s} {r.roc_auc:.3f} / {r.avg_precision:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
