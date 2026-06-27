"""Standalone, single-task benchmark worker — invoked as a fresh OS process by `comprehensive.py`.

Running each measurement in its own `subprocess` (not a `multiprocessing` child) guarantees a clean
address space, so `ru_maxrss` is *this task's* peak — not the parent's. Generates its own data
(pure NumPy, no sklearn dependency for data) and imports only the library the chosen method needs, so
the measured peak RSS reflects what that method actually costs.

Usage (prints one JSON line):
    python _worker.py fit        <method> <dataset> <n>
    python _worker.py stream     <n> <d> <chunk>      # betula chunked partial_fit (bounded memory)
    python _worker.py oneshot_km <n> <d>              # sklearn KMeans, must hold the whole array
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import contextlib
import json
import resource
import sys
import threading
import time

import numpy as np

MEM_CAP = 14 * 1024**3


def cap_memory():
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, MEM_CAP))


class Peak:
    """Sample this process's *current* RSS from /proc/self/statm and track the max.

    Unlike `ru_maxrss`, this reflects the post-`exec` program only — `ru_maxrss` retains the
    fork-time high-water mark of the (possibly large) parent that launched us, which would make every
    isolated worker report the parent's RSS.
    """

    def __init__(self):
        self.max = 0
        self._pg = resource.getpagesize()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _sample(self):
        try:
            with open("/proc/self/statm") as f:
                self.max = max(self.max, int(f.read().split()[1]) * self._pg)
        except OSError:
            pass

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(0.004)

    def mb(self) -> float:
        self._sample()
        self._stop.set()
        return self.max / 1e6


def gen(dataset: str, n: int, seed: int = 0):
    """Pure-NumPy standardized blobs (no sklearn, so a betula-only worker stays lean)."""
    rng = np.random.default_rng(seed)
    if dataset == "blobs":
        d, k, spread = 2, 6, 1.0
    elif dataset == "highdim":
        d, k, spread = 20, 8, 1.0
    else:
        raise ValueError(dataset)
    centers = rng.uniform(-10, 10, size=(k, d))
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    X = np.vstack([centers[i] + spread * rng.standard_normal((s, d)) for i, s in enumerate(sizes)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X.astype(np.float64), k


def load_real_worker(dataset: str):
    """Load a full real dataset (standardized) inside the worker, for the real-scale headline."""
    from sklearn.preprocessing import StandardScaler

    if dataset == "covtype":
        from sklearn.datasets import fetch_covtype

        d = fetch_covtype()
        x, y, k = d.data, d.target.astype(int) - 1, 7
    elif dataset == "mnist":
        from sklearn.datasets import fetch_openml

        d = fetch_openml("mnist_784", version=1, as_frame=False)
        x, y, k = d.data, d.target.astype(int), 10
    else:
        raise ValueError(dataset)
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float64)).astype(np.float64)
    return x, np.asarray(y), k


def fit_method(method: str, X, k: int, n: int):
    bkw = dict(threshold=0.0, max_leaves=2000, seed=0, n_jobs=1)
    mcs = max(20, n // 400)
    if method.startswith("betula"):
        import betula_cluster as bc

        kind = method.split("-", 1)[1]
        if kind == "kmeans":
            return bc.fit_predict(X, k, feature="spherical", method="kmeans", **bkw)
        if kind == "gmm":
            return bc.fit_predict(X, k, feature="diagonal", method="gmm", **bkw)
        if kind == "gmm-full":
            return bc.fit_predict(X, k, feature="full", method="gmm-full", **bkw)
        if kind == "ward":
            return bc.fit_predict(X, k, feature="diagonal", method="ward", **bkw)
        if kind == "hdbscan":
            return bc.fit_predict(X, method="hdbscan", min_cluster_size=mcs, min_samples=10, **bkw)
    if method == "sklearn-kmeans":
        from sklearn.cluster import KMeans

        return KMeans(k, n_init=10, random_state=0).fit_predict(X)
    if method == "sklearn-minibatch":
        from sklearn.cluster import MiniBatchKMeans

        return MiniBatchKMeans(k, n_init=3, random_state=0).fit_predict(X)
    if method == "sklearn-birch":
        from sklearn.cluster import Birch

        return Birch(n_clusters=k).fit_predict(X)
    if method == "sklearn-gmm":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(k, random_state=0).fit(X).predict(X)
    if method == "sklearn-ward":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(n_clusters=k).fit_predict(X)
    if method == "sklearn-hdbscan":
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(min_cluster_size=mcs, min_samples=10).fit_predict(X)
    raise ValueError(method)


def main() -> dict:
    cap_memory()
    peak = Peak()
    kind = sys.argv[1]
    if kind == "fit":
        method, dataset, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
        X, k = gen(dataset, n)
        t0 = time.perf_counter()
        labels = np.asarray(fit_method(method, X, k, n))
        dt = time.perf_counter() - t0
        return {
            "time_s": dt,
            "rss_mb": peak.mb(),
            "n_clusters": len({int(v) for v in labels if v >= 0}),
        }
    if kind == "stream":
        n, d, chunk = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        import betula_cluster as bc

        est = bc.Betula(
            n_clusters=8,
            feature="diagonal",
            method="kmeans",
            threshold=0.0,
            max_leaves=2000,
            seed=0,
        )
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        done = 0
        while done < n:  # never materialize the full array
            c = min(chunk, n - done)
            est.partial_fit(rng.standard_normal((c, d)))
            done += c
        est.partial_fit()
        return {"time_s": time.perf_counter() - t0, "rss_mb": peak.mb()}
    if kind == "oneshot_km":
        n, d = int(sys.argv[2]), int(sys.argv[3])
        from sklearn.cluster import KMeans

        X = np.random.default_rng(0).standard_normal((n, d))  # in-core: the whole array at once
        t0 = time.perf_counter()
        KMeans(8, n_init=1, random_state=0).fit(X)
        return {"time_s": time.perf_counter() - t0, "rss_mb": peak.mb()}
    if kind == "real_fit":
        method, dataset = sys.argv[2], sys.argv[3]
        X, y, k = load_real_worker(dataset)
        t0 = time.perf_counter()
        labels = np.asarray(fit_method(method, X, k, len(X)))
        dt = time.perf_counter() - t0
        from sklearn.metrics import adjusted_rand_score

        return {
            "time_s": dt,
            "rss_mb": peak.mb(),
            "n_clusters": len({int(v) for v in labels if v >= 0}),
            "ari": round(float(adjusted_rand_score(y, labels)), 3),
        }
    raise ValueError(kind)


if __name__ == "__main__":
    try:
        print(json.dumps(main()))
    except MemoryError:
        print(json.dumps({"error": "OOM"}))
    except Exception as e:
        print(json.dumps({"error": type(e).__name__, "msg": str(e)[:120]}))
