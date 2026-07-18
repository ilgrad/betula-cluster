"""CF-weighted NMF vs point-level NMF for nonnegative data — the scale story.

Both pipelines reduce to `r` nonnegative components and then k-means. The scikit-learn path factorizes
the full `N×d` matrix (`O(N·d·r)` per iteration, holds an `N×r` code matrix); betula's
`projection="weighted-nmf"` factorizes the `M ≪ N` leaf **centroids** instead (König-Huygens makes this
equal the full-data NMF up to the within-leaf scatter constant), so NMF runs at BETULA scale and bounded
memory. At matching cluster quality the CF-weighted path pulls ahead in time as `N` grows.

Run: `.venv/bin/python bench/nmf_cf_weighted.py`
"""

import time

import betula_cluster as bc
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score as ari


def topics(n_per, d, k, seed):
    """`k` topics over `d` nonnegative features; each doc is a nonnegative mix dominated by one topic."""
    rng = np.random.default_rng(seed)
    h = np.abs(rng.normal(size=(k, d))) * (rng.random((k, d)) < 0.3)
    xs, ys = [], []
    for c in range(k):
        w = rng.random((n_per, k)) * 0.1
        w[:, c] += 1.0 + rng.random(n_per)
        xs.append(w @ h + 0.02 * rng.random((n_per, d)))
        ys += [c] * n_per
    return np.ascontiguousarray(np.vstack(xs)), np.array(ys)


def main():
    print("Nonnegative topic mixtures (d=60, k=4); NMF rank 8 → k-means.\n")
    print(f"{'N':>9} | {'sklearn NMF→km':>22} | {'cf-weighted-nmf':>22} | {'speedup':>8}")
    print("-" * 72)
    for n_per in (2_000, 10_000, 40_000):
        x, y = topics(n_per, 60, 4, 0)
        n = len(x)

        t = time.perf_counter()
        codes = NMF(8, init="nndsvda", tol=1e-3, max_iter=100, random_state=0).fit_transform(x)
        sk = bc.fit_predict(codes, 4, method="kmeans", feature="spherical", threshold=0.0, seed=0)
        t_sk = time.perf_counter() - t

        t = time.perf_counter()
        cf = bc.fit_predict(
            x,
            4,
            method="kmeans",
            feature="spherical",
            threshold=0.0,
            seed=0,
            max_leaves=4000,
            projection="weighted-nmf",
            projection_dim=8,
        )
        t_cf = time.perf_counter() - t

        print(
            f"{n:>9} | ARI {ari(y, sk):>5.3f}  {t_sk:>7.2f}s | "
            f"ARI {ari(y, cf):>5.3f}  {t_cf:>7.2f}s | {t_sk / max(t_cf, 1e-9):>6.1f}×"
        )
    print(
        "\nsklearn NMF factorizes all N points; the CF-weighted path factorizes the ≤4000 leaf centroids,"
    )
    print("so it stays fast + memory-bounded as N grows, at matching cluster quality.")


if __name__ == "__main__":
    main()
