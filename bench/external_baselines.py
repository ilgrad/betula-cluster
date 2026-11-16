"""The strongest specialist per axis, not scikit-learn's general-purpose implementation.

`bench/comprehensive.py` measures betula against scikit-learn, which is the right *default*
comparison and the wrong *hardest* one: on raw k-means throughput the strongest published claim
belongs to FAISS, and on HDBSCAN at scale in low dimension it belongs to `fast_hdbscan`. Losing to
those is expected on some rows; the point of writing them down is that a loss becomes a numbered gap
instead of an absence.

Neither library is a project dependency and neither may become one — they are pulled per invocation:

    uv run --with faiss-cpu --with fast-hdbscan --with scikit-learn --with pandas \\
        python bench/external_baselines.py [--quality-only]

Writes `bench/results_external.csv`, which `bench/scoreboard.py` folds into the board as the
`vs-external` pairing. Every measurement runs in its own subprocess, the same isolation
`bench/_worker.py` uses, so peak RSS is that task's own and a library that cannot be imported takes
its row down rather than the run.

Each row is one (contest, size, row order, method) over a contest's seeds. `_worker.gen` stacks the
clusters one after another, so the rows arrive sorted by label; every case runs in that order and
on a seeded shuffle of it (permutation seed `1000 + seed`), because the tree is built in row order
and sorted input costs the `hdbscan` head up to 0.1 ARI. A seed is the method's own seed (FAISS
`seed`, scikit-learn `random_state`, betula `seed`); `fast_hdbscan` has none, so in label order its
seeds repeat one fit. The CSV row is the median-ARI run taken whole (an odd seed count has one), with
the range beside it. `--quality-only` leaves `time_s` empty: a run on a machine that is not fit to
time still yields the quality and memory cells.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# `setdefault`: a shell that exported OMP_NUM_THREADS (this machine exports 8) keeps its value, and
# the `threads` column records what each library actually ran with.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _worker import Peak, cap_memory, gen

# Each contest is one question, asked of every method in it, in both row orders.
CONTESTS: dict[str, dict[str, Any]] = {
    # The strongest "fastest k-means" claim in the ecosystem. FAISS fits Lloyd on a subsample with its
    # own SIMD kernels; betula clusters the leaf summary. Two FAISS configurations: its defaults, and
    # k-means++ seeding with three restarts — the cheapest setting that reaches betula's quality on
    # this fixture, still on the default subsample (see `fit`).
    "kmeans-speed": {
        "dataset": "highdim",
        "sizes": (200_000, 1_000_000),
        "seeds": tuple(range(11)),
        "methods": ("betula-kmeans", "faiss-kmeans", "faiss-kmeans-pp", "sklearn-kmeans"),
    },
    # `fast_hdbscan` is numba-compiled and restricted to low dimension and Euclidean, which is
    # exactly the regime where the CF summary has least to offer. Head against head: betula's
    # `hdbscan` at its defaults on 8 000 leaves, plain and with `canonical_order=True`.
    "hdbscan-scale": {
        "dataset": "blobs",
        "sizes": (100_000, 500_000),
        "seeds": tuple(range(5)),
        "methods": ("betula-hdbscan", "betula-hdbscan-canonical", "fast-hdbscan"),
    },
    # The same fixture and rival, against the heads that find the count without a density threshold.
    "countfree-scale": {
        "dataset": "blobs",
        "sizes": (100_000, 500_000),
        "seeds": tuple(range(5)),
        "methods": ("betula-gmm-full-bic", "betula-xmeans", "fast-hdbscan"),
    },
}

ORDERS = ("label", "shuffled")

TIMEOUT = 900.0


def min_points(n: int) -> int:
    """The smallest cluster the density contests ask for, in points, asked of every method."""
    return max(20, n // 400)


def fit(method: str, X, k: int, seed: int):
    """Return labels. Imports only what this method needs, so peak RSS is that method's own."""
    n = len(X)
    if method == "betula-kmeans":
        import betula_cluster as bc

        return bc.fit_predict(
            X,
            k,
            feature="spherical",
            method="kmeans",
            threshold=0.0,
            max_leaves=2000,
            n_shards=1,
            seed=seed,
        )
    if method.startswith("betula-"):
        import betula_cluster as bc

        kw: dict[str, Any]
        if method.startswith("betula-hdbscan"):
            kw = {"method": "hdbscan", "min_cluster_size": min_points(n), "max_leaves": 8000}
            kw["canonical_order"] = method.endswith("-canonical")
        elif method == "betula-gmm-full-bic":
            kw = {"method": "gmm-full", "n_clusters": 0}
        elif method == "betula-xmeans":
            kw = {"method": "xmeans", "n_clusters": 20}
        else:
            raise ValueError(method)
        return bc.Betula(threshold=0.0, seed=seed, **kw).fit_predict(X)
    if method.startswith("faiss-kmeans"):
        import faiss

        # `Kmeans` defaults to `max_points_per_centroid=256`, so at k=8 both arms train on 2 048 of
        # the n rows and assign the rest with one search. The default seeds each restart from a
        # random subset; `-pp` uses the k-means++ seeding FAISS has shipped since 1.14. A
        # restart stops once an iteration no longer lowers the objective (`early_stop_threshold`
        # 0), well before `niter` here: 12 of 25 iterations at seed 0 on 200 000 rows (1.16.0).
        x = np.ascontiguousarray(X, dtype=np.float32)
        km = faiss.Kmeans(
            x.shape[1], k, niter=25, nredo=3 if method.endswith("-pp") else 1, seed=seed
        )
        if method.endswith("-pp"):
            km.cp.init_method = faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS
        km.train(x)
        return km.index.search(x, 1)[1].ravel()
    if method == "fast-hdbscan":
        from fast_hdbscan import HDBSCAN

        return HDBSCAN(min_cluster_size=min_points(n), min_samples=10).fit_predict(
            np.ascontiguousarray(X, dtype=np.float64)
        )
    if method == "sklearn-kmeans":
        from sklearn.cluster import KMeans

        return KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    raise ValueError(method)


def provenance(method: str) -> tuple[str, str]:
    """(threads, version) the method ran with, read from the library after the fit."""
    if method.startswith("betula"):
        import betula_cluster as bc

        pool = os.environ.get("RAYON_NUM_THREADS") or str(os.cpu_count())
        build = os.environ.get("BETULA_GIT_REV", "")
        return f"rayon {pool}", f"betula-cluster {bc.__version__}{build and '+' + build}"
    if method.startswith("faiss"):
        import faiss

        return f"openmp {faiss.omp_get_max_threads()}", "faiss-cpu " + _dist("faiss-cpu")
    if method == "fast-hdbscan":
        import numba

        return f"numba {numba.get_num_threads()}", "fast-hdbscan " + _dist("fast-hdbscan")
    if method == "sklearn-kmeans":
        from threadpoolctl import threadpool_info

        omp = next(p["num_threads"] for p in threadpool_info() if p["internal_api"] == "openmp")
        return f"openmp {omp}", "scikit-learn " + _dist("scikit-learn")
    raise ValueError(method)


def _dist(name: str) -> str:
    return importlib.metadata.version(name)


def one(method: str, dataset: str, n: int, order: str, seed: int) -> dict:
    """One isolated measurement; prints a single JSON line for the parent to read."""
    cap_memory()
    peak = Peak()
    X, k = gen(dataset, n)
    # `gen` gives the last group the remainder, so an even split would misalign the labels by n % k.
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    truth = np.repeat(np.arange(k), sizes)
    if order == "shuffled":
        perm = np.random.default_rng(1000 + seed).permutation(n)
        X, truth = X[perm], truth[perm]
    t0 = time.perf_counter()
    labels = np.asarray(fit(method, X, k, seed))
    dt = time.perf_counter() - t0
    threads, version = provenance(method)

    from sklearn.metrics import adjusted_rand_score

    return {
        "time_s": dt,
        "rss_mb": peak.mb(),
        "n_clusters": len({int(v) for v in labels if v >= 0}),
        "noise": float(np.mean(labels < 0)),
        "ari": float(adjusted_rand_score(truth, labels)),
        "threads": threads,
        "version": version,
    }


def run_child(method: str, dataset: str, n: int, order: str, seed: int) -> dict:
    argv = [str(HERE / "external_baselines.py"), "--one", method, dataset, str(n), order, str(seed)]
    try:
        out = subprocess.run(
            [sys.executable, *argv], capture_output=True, text=True, timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    tail = out.stdout.strip().splitlines()
    if not tail:
        return {"error": "died", "rc": out.returncode, "stderr": out.stderr.strip()[-200:]}
    try:
        return json.loads(tail[-1])
    except json.JSONDecodeError:
        return {"error": "unparsable", "rc": out.returncode}


def median_run(runs: dict[int, dict]) -> dict:
    """The median-ARI run, whole (ties broken by seed), with the range over every seed beside it."""
    ranked = sorted(runs.items(), key=lambda sr: (sr[1]["ari"], sr[0]))
    seed, row = ranked[(len(ranked) - 1) // 2]
    aris = [r["ari"] for r in runs.values()]
    return {**row, "ari_min": min(aris), "ari_max": max(aris), "seeds": len(runs), "seed": seed}


def git_rev() -> str:
    try:
        rev = subprocess.run(
            ["git", "-C", str(HERE), "describe", "--always", "--dirty", "--abbrev=7"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return rev.replace("-", ".")


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quality-only", action="store_true", help="leave time_s empty (machine unfit to time)"
    )
    args = ap.parse_args()
    os.environ["BETULA_GIT_REV"] = git_rev()
    # numba compiles `fast_hdbscan` into its on-disk cache on the first call in a fresh environment:
    # 52 s and 651 MB against 1.4 s and 313 MB warm (100 000 rows). One unrecorded fit fills it.
    run_child("fast-hdbscan", "blobs", 20_000, "label", 0)

    cache: dict[tuple, dict] = {}  # a rival shared by two contests is fitted once per case
    rows = []
    for contest, spec in CONTESTS.items():
        for n in spec["sizes"]:
            for order in ORDERS:
                for method in spec["methods"]:
                    runs = {}
                    for seed in spec["seeds"]:
                        key = (method, spec["dataset"], n, order, seed)
                        if key not in cache:
                            cache[key] = run_child(*key)
                        if "error" not in cache[key]:
                            runs[seed] = cache[key]
                    if len(runs) < len(spec["seeds"]):
                        print(f"  {contest:15s} {method:24s} n={n:<8} {order:8s} skipped")
                        continue
                    res = median_run(runs)
                    if args.quality_only:
                        res["time_s"] = None
                    rows.append(
                        {
                            "dataset": spec["dataset"],
                            "n": n,
                            "contest": contest,
                            "order": order,
                            "method": method,
                            **res,
                        }
                    )
                    print(
                        f"  {contest:15s} {method:24s} n={n:<8} {order:8s} "
                        f"ARI {res['ari']:.4f} [{res['ari_min']:.4f}, {res['ari_max']:.4f}] "
                        f"k={res['n_clusters']} {res['rss_mb']:7.1f} MB",
                        flush=True,
                    )
    if not rows:
        print("no external rows produced — is faiss-cpu / fast-hdbscan importable?")
        return 1
    out = HERE / "results_external.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.name} ({len(rows)} rows) — fold it in with bench/scoreboard.py")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        method, dataset, n, order, seed = sys.argv[2:7]
        print(json.dumps(one(method, dataset, int(n), order, int(seed))))
    else:
        sys.exit(main())
