"""CF-weighted NMF vs point-level NMF for nonnegative data — the scale story.

Both pipelines reduce to `r` nonnegative components and then k-means. The scikit-learn path factorizes
the full `N×d` matrix (`O(N·d·r)` per iteration, holds an `N×r` code matrix); betula's
`projection="weighted-nmf"` factorizes the `M ≪ N` leaf **centroids** instead (König-Huygens makes this
equal the full-data NMF up to the within-leaf scatter constant), so NMF runs at BETULA scale and bounded
memory.

What that buys first is **determinism**. The speed column crosses over: 0.2× / 0.7× / **1.3×** at
`N` = 8 k / 40 k / 160 k, so the CF-weighted path is slower below roughly 10⁵ and faster above it — the
factorization cost is bounded by the leaf count while scikit-learn's grows with `N`. Quote the crossover,
not a single ratio. The column that separates the two at every size is the seed spread: ARI 1.000 ±0.000
against scikit-learn's 0.812–0.991 ±0.37, because NMF is invariant to `(W D, D⁻¹H)` and betula returns a
canonical factorization where scikit-learn's does not. Reach for it for that, or for bounded memory
beyond what a dense NMF can hold.

Run: `uv run --no-sync --with scikit-learn python bench/nmf_cf_weighted.py`
"""

import statistics as st
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


SEEDS = range(8)


def main():
    print("Nonnegative topic mixtures (d=60, k=4); NMF rank 8 → k-means. Median over 8 seeds.\n")
    print(f"{'N':>9} | {'sklearn NMF→km':>28} | {'cf-weighted-nmf':>28} | {'speedup':>8}")
    print("-" * 88)
    for n_per in (2_000, 10_000, 40_000):
        x, y = topics(n_per, 60, 4, 0)
        n = len(x)

        sk, t_sk = [], time.perf_counter()
        for s in SEEDS:
            codes = NMF(8, init="nndsvda", tol=1e-3, max_iter=100, random_state=s).fit_transform(x)
            sk.append(
                ari(
                    y,
                    bc.fit_predict(
                        codes, 4, method="kmeans", feature="spherical", threshold=0.0, seed=s
                    ),
                )
            )
        t_sk = (time.perf_counter() - t_sk) / len(sk)

        cf, t_cf = [], time.perf_counter()
        for s in SEEDS:
            cf.append(
                ari(
                    y,
                    bc.fit_predict(
                        x,
                        4,
                        method="kmeans",
                        feature="spherical",
                        threshold=0.0,
                        seed=s,
                        max_leaves=4000,
                        projection="weighted-nmf",
                        projection_dim=8,
                    ),
                )
            )
        t_cf = (time.perf_counter() - t_cf) / len(cf)

        print(
            f"{n:>9} | ARI {st.median(sk):>5.3f} ±{max(sk) - min(sk):>5.3f}  {t_sk:>7.2f}s | "
            f"ARI {st.median(cf):>5.3f} ±{max(cf) - min(cf):>5.3f}  {t_cf:>7.2f}s | "
            f"{t_sk / max(t_cf, 1e-9):>6.1f}×"
        )
    print(
        "\nsklearn NMF factorizes all N points; the CF-weighted path factorizes the ≤4000 leaf centroids,"
    )
    print(
        "so its factorization cost is bounded by the leaf count, not N — which is why the speed column"
    )
    print("crosses over: a LOSS at 8k and 40k, a win at 160k. Quote the crossover, not one ratio.")
    print(
        "The ± column is the seed spread. NMF is invariant to (W D, D^-1 H), so an unpinned split lets"
    )
    print(
        "one component's arbitrary scale dominate the Euclidean geometry the head clusters; betula"
    )
    print(
        "returns a canonical factorization (unit-L2 parts, energy-ordered), sklearn's NMF does not."
    )


if __name__ == "__main__":
    main()
