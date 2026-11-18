"""betula's heads against the methods people use on manifold data and on text embeddings.

`bench/comprehensive.py` measures betula against scikit-learn, and `bench/external_baselines.py`
against the fastest k-means and HDBSCAN on synthetic blobs. Neither holds the pipelines most users
reach for on images and embeddings: UMAP followed by a clusterer, EVoC, and Leiden on a point-level
k-nearest-neighbour graph. This harness puts every one of them on the same rows, with the same
seeds, under the same scoring, and writes the losses beside the wins.

None of the rivals is a project dependency, and none may become one. They are pulled per
invocation, and two of them are GPL (leidenalg, python-igraph); they run here as separate programs
and nothing in `python/betula_cluster` imports them:

    uv run --no-sync --with scikit-learn --with pandas --with umap-learn --with hdbscan \\
        --with evoc --with leidenalg --with python-igraph --with pacmap --with openTSNE \\
        --with phenograph --with deltric --with cplearn python bench/rivals.py [--quick] \\
        [--arms=a,b] [--datasets=a,b]

`--arms=` and `--datasets=` re-run only the named arms, the named datasets, or both, and keep every
other row of the CSV. The embedding boards read caches that `bench/embed_*.py` build once:
`news-minilm` (20 newsgroups under all-MiniLM-L6-v2, `bench/embed_news.py`), and the two
`evoc_bench_*` sets of the EVoC benchmark (`bench/embed_evoc.py`): `news-mpnet` (the same corpus
under all-mpnet-base-v2) and `cifar100-clip` (CIFAR-100 as CLIP image embeddings, 100 classes).
`load` also reads `arxiv-1m` (`bench/embed_arxiv.py`) for the timing campaign.

Rules (the plan's 12.4, kept here so the CSV can be read without it):

- Every arm runs in its own subprocess, so its peak RSS is its own and a rival that fails to import
  or exceeds `TIMEOUT` becomes a `DNF` row instead of taking the run down.
- Scores are over all rows, with noise counted as one more label. A method that leaves rows
  unclustered is also scored on its clustered rows alone, and its noise share is recorded: EVoC's
  own benchmark scores the clustered rows only, and the two numbers say different things.
- `told_k` separates the arms given the true number of clusters from those that choose it. A cell
  never compares one with the other.
- Rivals run at their documented defaults. The one tuned arm is UMAP's documented MNIST recipe,
  HDBSCAN(min_samples=10, min_cluster_size=500) on a 2-D layout with 30 neighbours, and its name
  says so.
- MNIST is raw pixels / 255. Standardising pixels lowers every method's score, point-level ones
  included, so it would measure the preprocessing rather than the clusterer.
- Wall time is recorded, and is only a result when `~/.claude/scripts/bench-ready.sh` passed
  before the run; the CSV records whether it did.
- Every row records the libraries its arm ran, with their versions (`version`, read in the arm's
  own process). Rows written before the column existed leave it empty.

Writes `bench/results_rivals.csv` (one row per arm, dataset and seed).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _provenance import jobs, versions
from _worker import Peak, cap_memory

SEEDS = (0, 1, 2)
TIMEOUT = 3600.0

# The leaf budget each betula arm gets, per dataset size: about five rows per leaf, the regime where
# §11 of the research plan measured the leaf partition's purity at 0.95. MNIST-70k also runs the two
# rows-graph arms at 8 000 and 20 000 leaves (`BUDGET_STUDY`), so the page can state what the budget
# buys and what it costs.
BUDGET = {20_000: 4_000, 70_000: 14_000, 17_901: 4_000, 18_119: 4_000, 59_999: 12_000}
BUDGET_STUDY = (8_000, 20_000)

# The unit-norm embedding sets. UMAP runs on them with the cosine metric, and the directional heads
# and BERTopic's pipeline run on them alone. `arxiv-1m` (`bench/embed_arxiv.py`) is the timing
# campaign's million-row set; no board of this harness reads it.
EMBEDDINGS = ("news-minilm", "news-mpnet", "cifar100-clip", "arxiv-1m")


def load(dataset: str, n: int, seed: int):
    """`(X float32, y)`: MNIST raw / 255, or one of the `EMBEDDINGS`; `n` below the set's size
    takes one seeded draw of its rows, the same for every seed (seeds vary the method)."""
    if dataset == "mnist":
        from _fixtures import fetch

        x, y, _ = fetch("mnist")
        x = x / 255.0
    elif dataset == "news-minilm":
        from embed_news import cached

        x, y = cached()
    elif dataset in ("news-mpnet", "cifar100-clip"):
        from embed_evoc import cached

        x, y = cached(dataset)
    elif dataset == "arxiv-1m":
        from embed_arxiv import cached

        x, y = cached()
    else:
        raise ValueError(dataset)
    if n < len(x):
        pick = np.random.default_rng(0).choice(len(x), n, replace=False)
        x, y = x[pick], y[pick]
    return np.ascontiguousarray(x, dtype=np.float32), np.asarray(y)


def _kmeans(z, k, seed):
    from sklearn.cluster import KMeans

    return KMeans(k, n_init=4, random_state=seed).fit_predict(z)


def umap_params(dataset):
    """UMAP's settings: BERTopic's on the embedding sets, the UMAP docs' MNIST recipe otherwise."""
    if dataset in EMBEDDINGS or dataset.startswith("arxiv"):
        return {"n_neighbors": 15, "min_dist": 0.0, "n_components": 5, "metric": "cosine"}
    return {"n_neighbors": 30, "min_dist": 0.0, "n_components": 2}


def _umap(x, seed, dataset, **kw):
    import umap

    return umap.UMAP(random_state=seed, **{**umap_params(dataset), **kw}).fit_transform(x)


def _leiden_knn(x, seed, n_neighbors=15):
    """Leiden at resolution 1 on the Jaccard-weighted symmetric kNN graph (the scanpy recipe)."""
    import igraph as ig
    import leidenalg as la
    from pynndescent import NNDescent
    from scipy import sparse

    idx = NNDescent(x, n_neighbors=n_neighbors + 1, random_state=seed).neighbor_graph[0][:, 1:]
    n, k = idx.shape
    member = sparse.csr_matrix(
        (np.ones(n * k, np.float32), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n)
    )
    und = (member + member.T).tocoo()
    keep = und.row < und.col
    i, j = und.row[keep], und.col[keep]
    inter = np.asarray(member[i].multiply(member[j]).sum(1)).ravel()
    g = ig.Graph(n=n, edges=np.column_stack([i, j]).tolist())
    g.es["weight"] = (inter / (2 * k - inter) + 1e-9).tolist()
    part = la.find_partition(
        g, la.RBConfigurationVertexPartition, weights="weight", seed=seed, n_iterations=-1
    )
    return np.asarray(part.membership)


def _cluprop(x, seed, n_neighbors=8):
    """The CluProp recipe (Zheng, Phibbs, Pham, arXiv 2605.00390 v2): Leiden modularity at
    resolution 1 on the union kNN graph, k = 8 under the cosine metric.

    The paper publishes no code and leaves three choices open; these are the ones that reproduce its
    MNIST-70k row (ARI 0.87 ± 0.01, 12 ± 1 clusters): eight neighbours besides the row itself, edge
    weight the cosine similarity `1 - d` (its text says "proportional to d", which reads 0.78), and
    igraph's `community_leiden` at its defaults.
    """
    import random

    import igraph as ig
    from pynndescent import NNDescent

    idx, dist = NNDescent(
        x, n_neighbors=n_neighbors + 1, metric="cosine", random_state=seed
    ).neighbor_graph
    n, k = idx.shape
    i, j, d = np.repeat(np.arange(n), k), idx.ravel(), dist.ravel()
    keep = i != j
    lo, hi, d = np.minimum(i, j)[keep], np.maximum(i, j)[keep], d[keep]
    _, first = np.unique(lo.astype(np.int64) * n + hi, return_index=True)
    g = ig.Graph(n=n, edges=np.column_stack([lo[first], hi[first]]).tolist())
    random.seed(seed)  # igraph draws from Python's `random`
    part = g.community_leiden(
        objective_function="modularity",
        weights=np.clip(1.0 - d[first], 1e-9, None).tolist(),
        resolution=1.0,
    )
    return np.asarray(part.membership)


def deltric(x, seed):
    from deltric import DelTriC

    # DelTriC takes no seed; its UMAP projection draws from numpy's global generator.
    np.random.seed(seed)
    return DelTriC().fit_predict(x)


def corespect(x, seed):
    import random

    from cplearn.corespect import CorespectModel
    from cplearn.corespect.config import CoreSpectConfig

    # CoreSPECT takes no seed either; its quickstart runs `fine_grained=True, propagate=True`.
    np.random.seed(seed)
    random.seed(seed)
    cfg = CoreSpectConfig().configure()
    return CorespectModel(x, **cfg.unpack()).run(fine_grained=True, propagate=True).labels_


def _betula(x, k, seed, n, **kw):
    import betula_cluster as bc

    # The estimator, not `fit_predict`: its defaults (feature, method options) are what users get.
    kw = {"max_leaves": BUDGET[n], **kw}
    est = bc.Betula(n_clusters=k, threshold=0.0, seed=seed, **kw)
    return est.fit_predict(x)


# name → (told_k, where, fit(x, k, seed, dataset, n) → labels). `-1` is noise. `where` is "all",
# "embeddings" (every set in `EMBEDDINGS`), one dataset, or one dataset at one size.
ARMS = {
    "betula-kmeans": (True, "all", lambda x, k, s, d, n: _betula(x, k, s, n, method="kmeans")),
    "betula-spectral-mass": (
        True,
        "mnist",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="spectral", leaf_graph="mass"),
    ),
    "betula-vmf-shared": (
        True,
        "embeddings",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="vmf", kappa="shared"),
    ),
    "betula-spherical-kmeans": (
        True,
        "embeddings",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="spherical-kmeans"),
    ),
    "betula-leiden": (False, "all", lambda x, k, s, d, n: _betula(x, k, s, n, method="leiden")),
    "betula-leiden-mass": (
        False,
        "all",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="leiden", leaf_graph="mass"),
    ),
    "betula-leiden-rows": (
        False,
        "all",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="leiden", leaf_graph="rows"),
    ),
    "betula-leiden-rows-auto": (
        True,
        "all",
        lambda x, k, s, d, n: _betula(
            x, k, s, n, method="leiden", leaf_graph="rows", resolution="auto"
        ),
    ),
    "betula-hdbscan": (False, "all", lambda x, k, s, d, n: _betula(x, k, s, n, method="hdbscan")),
    **{
        f"betula-leiden-rows{suffix} ({leaves} leaves)": (
            told,
            "mnist-70000",
            lambda x, k, s, d, n, leaves=leaves, kw=kw: _betula(
                x, k, s, n, max_leaves=leaves, **kw
            ),
        )
        for leaves in BUDGET_STUDY
        for suffix, told, kw in (
            ("", False, {"method": "leiden", "leaf_graph": "rows"}),
            ("-auto", True, {"method": "leiden", "leaf_graph": "rows", "resolution": "auto"}),
        )
    },
    "sklearn-kmeans": (True, "all", lambda x, k, s, d, n: _kmeans(x, k, s)),
    "umap+kmeans": (True, "all", lambda x, k, s, d, n: _kmeans(_umap(x, s, d), k, s)),
    "umap-0.6+kmeans": (
        True,
        "all",
        lambda x, k, s, d, n: _kmeans(_umap(x, s, d, compatibility_layout=False), k, s),
    ),
    "pacmap+kmeans": (True, "all", lambda x, k, s, d, n: _kmeans(_pacmap(x, s), k, s)),
    "opentsne+kmeans": (True, "all", lambda x, k, s, d, n: _kmeans(_tsne(x, s), k, s)),
    "umap+hdbscan (umap docs' mnist recipe)": (
        False,
        "mnist",
        lambda x, k, s, d, n: _hdbscan(_umap(x, s, d), min_samples=10, min_cluster_size=500),
    ),
    "bertopic default (umap+hdbscan mcs=10)": (
        False,
        "embeddings",
        lambda x, k, s, d, n: _hdbscan(_umap(x, s, d), min_cluster_size=10),
    ),
    "evoc": (False, "all", lambda x, k, s, d, n: _evoc(x, s)),
    "knn+leiden (scanpy recipe)": (False, "all", lambda x, k, s, d, n: _leiden_knn(x, s)),
    "cluprop recipe (union knn-8 + leiden)": (False, "all", lambda x, k, s, d, n: _cluprop(x, s)),
    "phenograph": (False, "all", lambda x, k, s, d, n: _phenograph(x, s)),
    "deltric": (False, "all", lambda x, k, s, d, n: deltric(x, s)),
    "cplearn-corespect": (False, "all", lambda x, k, s, d, n: corespect(x, s)),
}


# The libraries each rival arm runs, written into its rows' `version` column. A betula arm runs
# betula.
RIVAL_DISTS = {
    "sklearn-kmeans": ("scikit-learn",),
    "umap+kmeans": ("umap-learn", "pynndescent", "scikit-learn"),
    "umap-0.6+kmeans": ("umap-learn", "pynndescent", "scikit-learn"),
    "pacmap+kmeans": ("pacmap", "scikit-learn"),
    "opentsne+kmeans": ("openTSNE", "scikit-learn"),
    "umap+hdbscan (umap docs' mnist recipe)": ("umap-learn", "pynndescent", "hdbscan"),
    "bertopic default (umap+hdbscan mcs=10)": ("umap-learn", "pynndescent", "hdbscan"),
    "evoc": ("evoc",),
    "knn+leiden (scanpy recipe)": ("pynndescent", "leidenalg", "python-igraph"),
    "cluprop recipe (union knn-8 + leiden)": ("pynndescent", "python-igraph"),
    "phenograph": ("PhenoGraph", "leidenalg"),
    "deltric": ("deltric", "umap-learn"),
    "cplearn-corespect": ("cplearn",),
}
assert set(RIVAL_DISTS) == {a for a in ARMS if not a.startswith("betula")}


def _hdbscan(z, **kw):
    import hdbscan

    return hdbscan.HDBSCAN(**kw, **jobs("core_dist_n_jobs")).fit_predict(z)


def _evoc(x, seed):
    import evoc

    return evoc.EVoC(random_state=seed).fit_predict(x)


def _pacmap(x, seed):
    import pacmap

    return pacmap.PaCMAP(n_components=2, random_state=seed).fit_transform(x)


def _tsne(x, seed):
    from openTSNE import TSNE

    threads = {"n_jobs": os.cpu_count()} | jobs("n_jobs")
    return np.asarray(TSNE(random_state=seed, **threads).fit(x))


def _phenograph(x, seed):
    import phenograph

    labels, _, _ = phenograph.cluster(x, seed=seed, clustering_algo="leiden", **jobs("n_jobs"))
    return labels


def score(truth, labels) -> dict:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    labels = np.asarray(labels)
    noise = labels < 0
    rows = np.where(noise, labels.max() + 1, labels)
    out = {
        "ari": float(adjusted_rand_score(truth, rows)),
        "ami": float(adjusted_mutual_info_score(truth, rows)),
        "noise_share": float(noise.mean()),
        "n_clusters": len(np.unique(labels[~noise])),
    }
    if noise.any() and (~noise).sum() > 1:
        out["ari_clustered"] = float(adjusted_rand_score(truth[~noise], labels[~noise]))
    return out


def one(arm: str, dataset: str, n: int, seed: int) -> dict:
    cap_memory()
    x, y = load(dataset, n, seed)
    k = len(np.unique(y))
    peak = Peak()
    t0 = time.perf_counter()
    labels = ARMS[arm][2](x, k, seed, dataset, n)
    dt = time.perf_counter() - t0
    dists = ("betula-cluster",) if arm.startswith("betula") else RIVAL_DISTS[arm]
    return {"time_s": dt, "rss_mb": peak.mb(), **score(y, labels), "version": versions(*dists)}


def run_child(arm, dataset, n, seed) -> dict:
    cmd = [sys.executable, str(Path(__file__)), "--one", arm, dataset, str(n), str(seed)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout {TIMEOUT:.0f} s"}
    tail = out.stdout.strip().splitlines()
    if out.returncode != 0 or not tail:
        last = (out.stderr.strip().splitlines() or ["no output"])[-1]
        return {"error": f"rc={out.returncode}: {last[:160]}"}
    return json.loads(tail[-1])


def main(argv: list[str]) -> int:
    import pandas as pd

    quick = "--quick" in argv
    plan = [("mnist", 20_000), ("news-minilm", 17_901), ("news-mpnet", 18_119)] + (
        [] if quick else [("mnist", 70_000), ("cifar100-clip", 59_999)]
    )
    # `--arms=a,b` and `--datasets=a,b` re-run only those and replace their rows in the CSV.
    only, datasets = (
        next((a.split("=", 1)[1].split(",") for a in argv if a.startswith(flag)), None)
        for flag in ("--arms=", "--datasets=")
    )
    for given, known, what in ((only, ARMS, "arms"), (datasets, dict(plan), "datasets")):
        unknown = sorted(set(given or ()) - set(known))
        if unknown:
            raise SystemExit(f"unknown {what}: {unknown}")
    plan = [(d, n) for d, n in plan if not datasets or d in datasets]
    out = HERE / "results_rivals.csv"
    # Kept rows pass through as text: pandas 2 parses some floats one ulp off, and a re-run of
    # one arm would then rewrite every other arm's row.
    kept = (
        pd.read_csv(out, dtype=str, keep_default_na=False)
        if (only or datasets) and out.exists()
        else pd.DataFrame()
    )
    if len(kept):
        slices = pd.MultiIndex.from_arrays([kept["dataset"], kept["n"].astype(int)]).isin(plan)
        kept = kept[~(kept["arm"].isin(only or list(ARMS)) & slices)]
    ready = (
        subprocess.run(
            ["bash", str(Path.home() / ".claude/scripts/bench-ready.sh")], capture_output=True
        ).returncode
        == 0
    )
    seeds = SEEDS[:1] if quick else SEEDS
    rows = []
    for dataset, n in plan:
        for arm, (told_k, where, _) in ARMS.items():
            runs_here = where in ("all", dataset, f"{dataset}-{n}") or (
                where == "embeddings" and dataset in EMBEDDINGS
            )
            if not runs_here or (only and arm not in only):
                continue
            for seed in seeds:
                res = run_child(arm, dataset, n, seed)
                row = {"dataset": dataset, "n": n, "arm": arm, "told_k": told_k, "seed": seed}
                row |= {"dnf": res["error"]} if "error" in res else res
                row["timing_valid"] = ready
                rows.append(row)
                shown = res.get("error") or (
                    f"ARI {res['ari']:.3f}  AMI {res['ami']:.3f}  k={res['n_clusters']:<4}"
                    f" noise {res['noise_share']:.2f}  {res['time_s']:7.1f} s"
                    f"  {res['rss_mb']:7.0f} MB"
                )
                print(f"{dataset:12s} n={n:<6} {arm:42s} seed={seed}  {shown}", flush=True)
                pd.concat([kept, pd.DataFrame(rows)]).to_csv(out, index=False)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        a = sys.argv[2:]
        print(json.dumps(one(a[0], a[1], int(a[2]), int(a[3]))))
    else:
        sys.exit(main(sys.argv[1:]))
