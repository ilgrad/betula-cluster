"""Non-convex benchmark: betula spectral / leiden vs scikit-learn SpectralClustering.

Reproduces the "spectral clustering that scales" table in RESULTS.md. betula's spectral head runs on
the <= max_leaves CF microclusters, so it matches sklearn's quality on moons/circles at a fraction of
the cost; leiden (community detection) is included as an honest negative — it over-splits manifolds.

    python bench/spectral_nonconvex.py            # N = 30000
    python bench/spectral_nonconvex.py --n 100000
"""

from __future__ import annotations

import argparse
import time

import betula_cluster as bc
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import StandardScaler

BETULA_KW = dict(threshold=0.0, max_leaves=2000, seed=0)


def gen(name: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if name == "moons":
        x, y = make_moons(n_samples=n, noise=0.06, random_state=seed)
    else:
        x, y = make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=seed)
    return StandardScaler().fit_transform(x).astype(np.float64), y


def timed(fn, x):
    t = time.perf_counter()
    labels = fn(x)
    return labels, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000)
    args = ap.parse_args()

    methods = {
        "betula-spectral": lambda x: bc.fit_predict(x, 2, method="spectral", **BETULA_KW),
        "betula-leiden": lambda x: bc.fit_predict(x, 2, method="leiden", **BETULA_KW),
        "sklearn-spectral": lambda x: SpectralClustering(
            n_clusters=2, affinity="nearest_neighbors", n_neighbors=10, random_state=0
        ).fit_predict(x),
    }
    for ds in ("moons", "circles"):
        x, y = gen(ds, args.n)
        print(f"\n=== {ds}  N={args.n:,} ===")
        for name, fn in methods.items():
            labels, dt = timed(fn, x)
            print(f"  {name:18s} ARI={ari(labels, y):.3f}  time={dt:.3f}s  k={len(set(labels))}")


if __name__ == "__main__":
    main()
