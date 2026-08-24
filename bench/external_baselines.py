"""The strongest specialist per axis, not scikit-learn's general-purpose implementation.

`bench/comprehensive.py` measures betula against scikit-learn, which is the right *default*
comparison and the wrong *hardest* one: on raw k-means throughput the strongest published claim
belongs to FAISS, and on HDBSCAN at scale in low dimension it belongs to `fast_hdbscan`. Losing to
those is expected on some rows; the point of writing them down is that a loss becomes a numbered gap
instead of an absence.

Neither library is a project dependency and neither may become one — they are pulled per invocation:

    uv run --with faiss-cpu --with fast-hdbscan --with scikit-learn --with pandas \\
        python bench/external_baselines.py

Writes `bench/results_external.csv`, which `bench/scoreboard.py` folds into the board as the
`vs-external` pairing. Every measurement runs in its own subprocess, the same isolation
`bench/_worker.py` uses, so peak RSS is that task's own and a library that cannot be imported takes
its row down rather than the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _worker import Peak, cap_memory, gen

# Each contest is one question, asked of every method in it. `n`/`d` are the workload; `k` the
# cluster count the generator planted.
CONTESTS = {
    # The strongest "fastest k-means" claim in the ecosystem. FAISS runs a fixed 25 iterations of
    # Lloyd over the raw points with its own SIMD kernels; betula clusters the leaf summary.
    "kmeans-speed": {
        "dataset": "highdim",
        "sizes": (200_000, 1_000_000),
        "methods": ("betula-kmeans", "faiss-kmeans", "sklearn-kmeans"),
    },
    # `fast_hdbscan` is numba-compiled and restricted to low dimension and Euclidean, which is
    # exactly the regime where the CF summary has least to offer.
    "hdbscan-scale": {
        "dataset": "blobs",
        "sizes": (100_000, 500_000),
        "methods": ("betula-hdbscan", "fast-hdbscan"),
    },
}

TIMEOUT = 900.0


def min_points(n: int) -> int:
    """The smallest cluster the density contests ask for, in points, asked of every method."""
    return max(20, n // 400)


def fit(method: str, X, k: int):
    """Return labels. Imports only what this method needs, so peak RSS is that method's own."""
    n = len(X)
    if method == "betula-kmeans":
        import betula_cluster as bc

        return bc.fit_predict(
            X, k, feature="spherical", method="kmeans", threshold=0.0, max_leaves=2000, n_jobs=1
        )
    if method == "betula-hdbscan":
        import betula_cluster as bc

        return bc.fit_predict(
            X,
            method="hdbscan",
            min_cluster_size=min_points(n),
            min_samples=10,
            threshold=0.0,
            max_leaves=2000,
            n_jobs=1,
        )
    if method == "faiss-kmeans":
        import faiss

        x = np.ascontiguousarray(X, dtype=np.float32)
        km = faiss.Kmeans(x.shape[1], k, niter=25, seed=0, verbose=False)
        km.train(x)
        return km.index.search(x, 1)[1].ravel()
    if method == "fast-hdbscan":
        from fast_hdbscan import HDBSCAN

        return HDBSCAN(min_cluster_size=min_points(n), min_samples=10).fit_predict(
            np.ascontiguousarray(X, dtype=np.float64)
        )
    if method == "sklearn-kmeans":
        from sklearn.cluster import KMeans

        return KMeans(k, n_init=10, random_state=0).fit_predict(X)
    raise ValueError(method)


def one(method: str, dataset: str, n: int) -> dict:
    """One isolated measurement; prints a single JSON line for the parent to read."""
    cap_memory()
    peak = Peak()
    X, k = gen(dataset, n)
    # `gen` gives the last group the remainder, so an even split would misalign the labels by n % k.
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    truth = np.repeat(np.arange(k), sizes)
    t0 = time.perf_counter()
    labels = np.asarray(fit(method, X, k))
    dt = time.perf_counter() - t0

    from sklearn.metrics import adjusted_rand_score

    return {
        "time_s": dt,
        "rss_mb": peak.mb(),
        "n_clusters": len({int(v) for v in labels if v >= 0}),
        "ari": float(adjusted_rand_score(truth, labels)),
    }


def run_child(method: str, dataset: str, n: int) -> dict:
    try:
        out = subprocess.run(
            [sys.executable, str(HERE / "external_baselines.py"), "--one", method, dataset, str(n)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
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


def main() -> int:
    import pandas as pd

    rows = []
    for contest, spec in CONTESTS.items():
        for n in spec["sizes"]:
            for method in spec["methods"]:
                res = run_child(method, spec["dataset"], n)
                if "error" in res:
                    print(f"  {contest:14s} {method:16s} n={n:<8} skipped ({res['error']})")
                    continue
                rows.append(
                    {
                        "dataset": spec["dataset"],
                        "n": n,
                        "contest": contest,
                        "method": method,
                        **res,
                    }
                )
                print(
                    f"  {contest:14s} {method:16s} n={n:<8} "
                    f"{res['time_s']:7.2f}s  {res['rss_mb']:8.1f} MB  ARI {res['ari']:.4f}"
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
        print(json.dumps(one(sys.argv[2], sys.argv[3], int(sys.argv[4]))))
    else:
        sys.exit(main())
