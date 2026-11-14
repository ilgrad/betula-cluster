"""A stream of sentence embeddings against refits: does `partial_fit` lose anything, and what do the
rivals that cannot stream read when refitted from scratch at the same checkpoints?

The 20 newsgroups MiniLM embeddings (`bench/embed_news.py`) arrive in a random order, five orders.
At 20, 40, 60, 80 and 100 % of the stream every arm labels the rows seen so far:

- `betula-stream`: `vmf` (`kappa="shared"`, 20 clusters, 4 000 leaves), `partial_fit` on the new
  chunk, then `partial_fit()` for the global step; no refit.
- `betula-refit`: the same estimator fitted from scratch on every row seen.
- `sklearn-kmeans`, `umap+kmeans`, `umap+hdbscan` (BERTopic's cluster step: UMAP 5-D cosine,
  `min_cluster_size=10`) and `evoc`, each refitted from scratch, since none of them streams.

Quality only (ARI over the rows seen), seeds 0, 1, 2. The rivals are pulled per invocation:

    uv run --no-sync --with scikit-learn --with umap-learn --with hdbscan --with evoc \\
        --with pandas python bench/stream_vs_refit.py [order ...]

Writes `bench/results_stream_refit.csv`; orders given on the command line replace their rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embed_news import cached

K, LEAVES, SEEDS, ORDERS = 20, 4000, (0, 1, 2), (0, 1, 2, 3, 4)
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def run(order: int) -> list[dict]:
    import betula_cluster as bc
    import evoc
    import hdbscan
    import umap
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    x, y = cached()
    x = np.ascontiguousarray(x, dtype=np.float64)
    perm = np.random.default_rng(2000 + order).permutation(len(x))
    x, y = x[perm], y[perm]
    cuts = [int(f * len(x)) for f in FRACTIONS]
    rows = []

    def emit(seen, arm, seed, labels):
        labels = np.asarray(labels)
        rows.append(
            {
                "order": order,
                "seen": seen,
                "arm": arm,
                "seed": seed,
                "ari": adjusted_rand_score(y[:seen], labels),
                "n_clusters": len(set(labels[labels >= 0].tolist())),
                "noise_share": float(np.mean(labels < 0)),
            }
        )
        print(f"order {order} seen {seen:6d} {arm:15s} seed {seed}  ARI {rows[-1]['ari']:.3f}")

    def vmf(seed):
        return bc.Betula(
            method="vmf", kappa="shared", n_clusters=K, max_leaves=LEAVES, threshold=0.0, seed=seed
        )

    for seed in SEEDS:
        est, start = vmf(seed), 0
        for cp in cuts:
            est.partial_fit(x[start:cp])
            start = cp
            est.partial_fit()
            emit(cp, "betula-stream", seed, est.predict(x[:cp]))
            emit(cp, "betula-refit", seed, vmf(seed).fit_predict(x[:cp]))
    for seed in SEEDS:
        for cp in cuts:
            xs = x[:cp]
            emit(cp, "sklearn-kmeans", seed, KMeans(K, random_state=seed).fit_predict(xs))
            z = umap.UMAP(
                n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=seed
            ).fit_transform(xs)
            emit(cp, "umap+kmeans", seed, KMeans(K, random_state=seed).fit_predict(z))
            emit(cp, "umap+hdbscan", seed, hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(z))
            emit(cp, "evoc", seed, evoc.EVoC(random_state=seed).fit_predict(xs))
    return rows


def main(argv: list[str]) -> int:
    import pandas as pd

    orders = [int(a) for a in argv] or list(ORDERS)
    csv = HERE / "results_stream_refit.csv"
    kept = pd.read_csv(csv) if argv and csv.exists() else pd.DataFrame()
    if len(kept):
        kept = kept[~kept["order"].isin(orders)]
    for order in orders:
        kept = pd.concat([kept, pd.DataFrame(run(order))])
        kept.sort_values(["order", "arm", "seed", "seen"]).to_csv(csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
