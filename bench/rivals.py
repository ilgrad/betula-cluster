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
        --with phenograph python bench/rivals.py [--quick] [--arms=a,b]

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

from _provenance import versions
from _worker import Peak, cap_memory

SEEDS = (0, 1, 2)
TIMEOUT = 3600.0

# The leaf budget each betula arm gets, per dataset size: about five rows per leaf, the regime where
# §11 of the research plan measured the leaf partition's purity at 0.95. MNIST-70k also runs the two
# rows-graph arms at 8 000 and 20 000 leaves (`BUDGET_STUDY`), so the page can state what the budget
# buys and what it costs.
BUDGET = {20_000: 4_000, 70_000: 14_000, 17_901: 4_000}
BUDGET_STUDY = (8_000, 20_000)


def load(dataset: str, n: int, seed: int):
    """`(X float32, y)`: MNIST raw / 255, or the 20 newsgroups MiniLM embeddings."""
    if dataset == "mnist":
        from _fixtures import fetch

        x, y, _ = fetch("mnist")
        x = x / 255.0
        if n < len(x):
            pick = np.random.default_rng(0).choice(
                len(x), n, replace=False
            )  # one draw; seeds vary the method
            x, y = x[pick], y[pick]
    elif dataset == "news-minilm":
        from embed_news import cached

        x, y = cached()
    else:
        raise ValueError(dataset)
    return np.ascontiguousarray(x, dtype=np.float32), np.asarray(y)


def _kmeans(z, k, seed):
    from sklearn.cluster import KMeans

    return KMeans(k, n_init=4, random_state=seed).fit_predict(z)


def _umap(x, seed, dataset, **kw):
    import umap

    base = (
        {"n_neighbors": 15, "min_dist": 0.0, "n_components": 5, "metric": "cosine"}
        if dataset == "news-minilm"
        else {"n_neighbors": 30, "min_dist": 0.0, "n_components": 2}
    )
    return umap.UMAP(random_state=seed, **{**base, **kw}).fit_transform(x)


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


def _betula(x, k, seed, n, **kw):
    import betula_cluster as bc

    # The estimator, not `fit_predict`: its defaults (feature, method options) are what users get.
    kw = {"max_leaves": BUDGET[n], **kw}
    est = bc.Betula(n_clusters=k, threshold=0.0, seed=seed, **kw)
    return est.fit_predict(x)


# name → (told_k, datasets, fit(x, k, seed, dataset, n) → labels). `-1` is noise.
ARMS = {
    "betula-kmeans": (True, "both", lambda x, k, s, d, n: _betula(x, k, s, n, method="kmeans")),
    "betula-spectral-mass": (
        True,
        "mnist",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="spectral", leaf_graph="mass"),
    ),
    "betula-vmf-shared": (
        True,
        "news-minilm",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="vmf", kappa="shared"),
    ),
    "betula-spherical-kmeans": (
        True,
        "news-minilm",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="spherical-kmeans"),
    ),
    "betula-leiden": (False, "both", lambda x, k, s, d, n: _betula(x, k, s, n, method="leiden")),
    "betula-leiden-mass": (
        False,
        "both",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="leiden", leaf_graph="mass"),
    ),
    "betula-leiden-rows": (
        False,
        "both",
        lambda x, k, s, d, n: _betula(x, k, s, n, method="leiden", leaf_graph="rows"),
    ),
    "betula-leiden-rows-auto": (
        True,
        "both",
        lambda x, k, s, d, n: _betula(
            x, k, s, n, method="leiden", leaf_graph="rows", resolution="auto"
        ),
    ),
    "betula-hdbscan": (False, "both", lambda x, k, s, d, n: _betula(x, k, s, n, method="hdbscan")),
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
    "sklearn-kmeans": (True, "both", lambda x, k, s, d, n: _kmeans(x, k, s)),
    "umap+kmeans": (True, "both", lambda x, k, s, d, n: _kmeans(_umap(x, s, d), k, s)),
    "umap-0.6+kmeans": (
        True,
        "both",
        lambda x, k, s, d, n: _kmeans(_umap(x, s, d, compatibility_layout=False), k, s),
    ),
    "pacmap+kmeans": (True, "both", lambda x, k, s, d, n: _kmeans(_pacmap(x, s), k, s)),
    "opentsne+kmeans": (True, "both", lambda x, k, s, d, n: _kmeans(_tsne(x, s), k, s)),
    "umap+hdbscan (umap docs' mnist recipe)": (
        False,
        "mnist",
        lambda x, k, s, d, n: _hdbscan(_umap(x, s, d), min_samples=10, min_cluster_size=500),
    ),
    "bertopic default (umap+hdbscan mcs=10)": (
        False,
        "news-minilm",
        lambda x, k, s, d, n: _hdbscan(_umap(x, s, d), min_cluster_size=10),
    ),
    "evoc": (False, "both", lambda x, k, s, d, n: _evoc(x, s)),
    "knn+leiden (scanpy recipe)": (False, "both", lambda x, k, s, d, n: _leiden_knn(x, s)),
    "phenograph": (False, "both", lambda x, k, s, d, n: _phenograph(x, s)),
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
    "phenograph": ("PhenoGraph", "leidenalg"),
}
assert set(RIVAL_DISTS) == {a for a in ARMS if not a.startswith("betula")}


def _hdbscan(z, **kw):
    import hdbscan

    return hdbscan.HDBSCAN(**kw).fit_predict(z)


def _evoc(x, seed):
    import evoc

    return evoc.EVoC(random_state=seed).fit_predict(x)


def _pacmap(x, seed):
    import pacmap

    return pacmap.PaCMAP(n_components=2, random_state=seed).fit_transform(x)


def _tsne(x, seed):
    from openTSNE import TSNE

    return np.asarray(TSNE(random_state=seed, n_jobs=os.cpu_count()).fit(x))


def _phenograph(x, seed):
    import phenograph

    labels, _, _ = phenograph.cluster(x, seed=seed, clustering_algo="leiden")
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
    # `--arms=a,b` re-runs only those arms and replaces their rows in the existing CSV.
    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    out = HERE / "results_rivals.csv"
    # Kept rows pass through as text: pandas 2 parses some floats one ulp off, and a re-run of
    # one arm would then rewrite every other arm's row.
    kept = (
        pd.read_csv(out, dtype=str, keep_default_na=False)
        if only and out.exists()
        else pd.DataFrame()
    )
    if only:
        unknown = sorted(set(only) - set(ARMS))
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}")
        kept = kept[~kept["arm"].isin(only)] if len(kept) else kept
    ready = (
        subprocess.run(
            ["bash", str(Path.home() / ".claude/scripts/bench-ready.sh")], capture_output=True
        ).returncode
        == 0
    )
    plan = [("mnist", 20_000), ("news-minilm", 17_901)] + ([] if quick else [("mnist", 70_000)])
    seeds = SEEDS[:1] if quick else SEEDS
    rows = []
    for dataset, n in plan:
        for arm, (told_k, where, _) in ARMS.items():
            if where not in ("both", dataset, f"{dataset}-{n}") or (only and arm not in only):
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
