"""A stream of sentence embeddings against refits: does `partial_fit` lose anything, what do the
rivals that cannot stream read when refitted from scratch at the same checkpoints, and what do the
rivals that do stream read on the same stream?

The 20 newsgroups MiniLM embeddings (`bench/embed_news.py`) arrive in a random order, five orders.
At 20, 40, 60, 80 and 100 % of the stream every arm labels the rows seen so far:

- `betula-stream`: `vmf` (`kappa="shared"`, 20 clusters, 4 000 leaves), `partial_fit` on the new
  chunk, then `partial_fit()` for the global step; no refit.
- `betula-refit`: the same estimator fitted from scratch on every row seen.
- `sklearn-kmeans`, `umap+kmeans`, `umap+hdbscan` (BERTopic's cluster step: UMAP 5-D cosine,
  `min_cluster_size=10`) and `evoc`, each refitted from scratch, since none of them streams.
- The rivals that learn in chunks (`STREAMERS`), each fed the new chunk the way its documentation
  streams, then asked for the labels of every row seen: scikit-learn's `MiniBatchKMeans` and
  `Birch`, BICO, BERTopic's online recipe, and River's `DBSTREAM` and `DenStream`, at their
  defaults and at the radius that suits this stream.

Quality only (ARI over the rows seen), seeds 0, 1, 2; an arm with no seed runs once per order. The
rivals are pulled per invocation (BICO is GPL-3.0, River BSD-3):

    uv run --no-sync --with scikit-learn --with umap-learn --with hdbscan --with evoc \\
        --with bico --with river --with pandas python bench/stream_vs_refit.py \\
        [--arms=a,b] [order ...]

Writes `bench/results_stream_refit.csv`. The orders given (all five by default) and the arms given
(all by default) replace their rows; every other row is kept. Each row records the libraries its arm
ran and their versions (`version`); rows written before the column existed leave it empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _provenance import versions
from embed_news import cached

K, LEAVES, SEEDS, ORDERS = 20, 4000, (0, 1, 2), (0, 1, 2, 3, 4)
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
REFITS = ("sklearn-kmeans", "umap+kmeans", "umap+hdbscan", "evoc")

# River's settings for this stream, each the best ARI on the first chunk of order 0 (3 580 rows):
# an oracle choice, generous to River. DBSTREAM's radius 0.5–1.5 at its default fading factor, and
# radius 0.7/0.9/1.1 × fading 0.01/0.001/0.0001, never found more than 6 clusters (best ARI 0.031).
# DenStream at radius 0.5 found 67 (ARI 0.125); at 0.7 it did not learn the 3 580 rows in 17
# minutes. Their defaults (DBSTREAM 1.0, DenStream 0.02) are sized for low-dimensional data; MiniLM
# rows sit 1.0–1.2 from their tenth neighbour.
DBSTREAM_TUNED = {"clustering_threshold": 0.9, "fading_factor": 0.0001}
DENSTREAM_TUNED = {"epsilon": 0.5}


class MiniBatchStream:
    """`MiniBatchKMeans.partial_fit`, one call per mini-batch of its own `batch_size` (1 024)."""

    seeded = True

    def __init__(self, seed: int):
        from sklearn.cluster import MiniBatchKMeans

        self.model = MiniBatchKMeans(K, random_state=seed)

    def learn(self, chunk: np.ndarray) -> None:
        step = self.model.batch_size
        for start in range(0, len(chunk), step):
            self.model.partial_fit(chunk[start : start + step])

    def labels(self, rows: np.ndarray) -> np.ndarray:
        return self.model.predict(rows)


class BertopicOnline:
    """BERTopic's online recipe (its *Online Topic Modeling* page) at the cluster step: an
    `IncrementalPCA` to 5 dimensions and a `MiniBatchKMeans`, each `partial_fit` per chunk of 1 000
    documents, the page's chunk size; the page's 50 clusters become the told count. `partial_fit`
    labels only the chunk it learns, so the rows seen are labelled as `transform` labels them."""

    seeded = True

    def __init__(self, seed: int):
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import IncrementalPCA

        self.pca = IncrementalPCA(n_components=5)
        self.model = MiniBatchKMeans(K, random_state=seed)

    def learn(self, chunk: np.ndarray) -> None:
        for start in range(0, len(chunk), 1000):
            part = chunk[start : start + 1000]
            self.pca.partial_fit(part)
            self.model.partial_fit(self.pca.transform(part))

    def labels(self, rows: np.ndarray) -> np.ndarray:
        return self.model.predict(self.pca.transform(rows))


class BirchStream:
    """`Birch.partial_fit` at its defaults (threshold 0.5, branching factor 50) and the told count:
    each call adds the chunk to the CF tree and reruns the global step."""

    seeded = False

    def __init__(self, seed: int):
        from sklearn.cluster import Birch

        self.model = Birch(n_clusters=K)

    def learn(self, chunk: np.ndarray) -> None:
        self.model.partial_fit(chunk)

    def labels(self, rows: np.ndarray) -> np.ndarray:
        return self.model.predict(rows)


class BicoStream:
    """BICO, its README's recipe for data larger than memory: `partial_fit` per chunk, then
    `partial_fit()` for the coreset (200 points per cluster) and its k-means (`fit_coreset=True`)."""

    seeded = True

    def __init__(self, seed: int):
        from bico import BICO

        self.model = BICO(n_clusters=K, random_state=seed, fit_coreset=True)

    def learn(self, chunk: np.ndarray) -> None:
        self.model.partial_fit(chunk)
        self.model.partial_fit()

    def labels(self, rows: np.ndarray) -> np.ndarray:
        return self.model.predict(rows)


class RiverStream:
    """River's `DBSTREAM` or `DenStream`: one `learn_one` per row, one `predict_one` per row seen."""

    seeded = False
    head, params = "", {}

    def __init__(self, seed: int):
        from river import cluster

        self.model = getattr(cluster, self.head)(**self.params)

    def learn(self, chunk: np.ndarray) -> None:
        for row in chunk:
            self.model.learn_one(dict(enumerate(row.tolist())))

    def labels(self, rows: np.ndarray) -> np.ndarray:
        return np.array([self.model.predict_one(dict(enumerate(r.tolist()))) for r in rows])


def _river(head: str, **params) -> type[RiverStream]:
    return type(head, (RiverStream,), {"head": head, "params": params})


STREAMERS = {
    "minibatch-stream": MiniBatchStream,
    "birch-stream": BirchStream,
    "bico-stream": BicoStream,
    "bertopic-online": BertopicOnline,
    "dbstream": _river("DBSTREAM"),
    "denstream": _river("DenStream"),
    "dbstream-tuned": _river("DBSTREAM", **DBSTREAM_TUNED),
    "denstream-tuned": _river("DenStream", **DENSTREAM_TUNED),
}
ARMS = ("betula-stream", "betula-refit", *REFITS, *STREAMERS)
# The libraries each arm runs, written into its rows' `version` column.
DISTS = {
    "betula-stream": ("betula-cluster",),
    "betula-refit": ("betula-cluster",),
    "sklearn-kmeans": ("scikit-learn",),
    "umap+kmeans": ("umap-learn", "pynndescent", "scikit-learn"),
    "umap+hdbscan": ("umap-learn", "pynndescent", "hdbscan"),
    "evoc": ("evoc",),
    "minibatch-stream": ("scikit-learn",),
    "birch-stream": ("scikit-learn",),
    "bico-stream": ("bico", "scikit-learn"),
    "bertopic-online": ("scikit-learn",),
    "dbstream": ("river",),
    "denstream": ("river",),
    "dbstream-tuned": ("river",),
    "denstream-tuned": ("river",),
}
assert set(DISTS) == set(ARMS), set(ARMS) ^ set(DISTS)


def refit(arm: str, xs: np.ndarray, seed: int) -> np.ndarray:
    """A rival that cannot stream, fitted from scratch on every row seen."""
    from sklearn.cluster import KMeans

    if arm == "sklearn-kmeans":
        return KMeans(K, random_state=seed).fit_predict(xs)
    if arm == "evoc":
        import evoc

        return evoc.EVoC(random_state=seed).fit_predict(xs)
    import hdbscan
    import umap

    z = umap.UMAP(
        n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=seed
    ).fit_transform(xs)
    if arm == "umap+kmeans":
        return KMeans(K, random_state=seed).fit_predict(z)
    return hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(z)


def run(order: int, arms: tuple[str, ...] = ARMS) -> list[dict]:
    import betula_cluster as bc
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
                "version": versions(*DISTS[arm]),
            }
        )
        print(
            f"order {order} seen {seen:6d} {arm:15s} seed {seed}  ARI {rows[-1]['ari']:.3f}",
            flush=True,
        )

    def vmf(seed):
        return bc.Betula(
            method="vmf", kappa="shared", n_clusters=K, max_leaves=LEAVES, threshold=0.0, seed=seed
        )

    if "betula-stream" in arms or "betula-refit" in arms:
        for seed in SEEDS:
            est, start = vmf(seed), 0
            for cp in cuts:
                est.partial_fit(x[start:cp])
                start = cp
                est.partial_fit()
                if "betula-stream" in arms:
                    emit(cp, "betula-stream", seed, est.predict(x[:cp]))
                if "betula-refit" in arms:
                    emit(cp, "betula-refit", seed, vmf(seed).fit_predict(x[:cp]))
    for seed in SEEDS:
        for cp in cuts:
            for arm in (a for a in REFITS if a in arms):
                emit(cp, arm, seed, refit(arm, x[:cp], seed))
    for arm in (a for a in STREAMERS if a in arms):
        streamer = STREAMERS[arm]
        for seed in SEEDS if streamer.seeded else SEEDS[:1]:
            model, start = streamer(seed), 0
            for cp in cuts:
                model.learn(x[start:cp])
                start = cp
                emit(cp, arm, seed, model.labels(x[:cp]))
    return rows


def main(argv: list[str]) -> int:
    import pandas as pd

    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    if only and set(only) - set(ARMS):
        raise SystemExit(f"unknown arms: {sorted(set(only) - set(ARMS))}")
    arms = tuple(only) if only else ARMS
    orders = [int(a) for a in argv if not a.startswith("--")] or list(ORDERS)
    csv = HERE / "results_stream_refit.csv"
    # Kept rows pass through as text: pandas 2 parses some floats one ulp off, and a re-run of
    # one order would then rewrite every other order's row.
    kept = (
        pd.read_csv(csv, dtype=str, keep_default_na=False)
        if argv and csv.exists()
        else pd.DataFrame()
    )
    if len(kept):
        kept = kept[~(kept["order"].astype(int).isin(orders) & kept["arm"].isin(arms))]

    def key(col):
        return col if col.name == "arm" else col.map(int)

    for order in orders:
        kept = pd.concat([kept, pd.DataFrame(run(order, arms))])
        kept.sort_values(["order", "arm", "seed", "seen"], key=key).to_csv(csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
