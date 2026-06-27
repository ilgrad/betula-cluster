"""Comprehensive, honest benchmark: betula-cluster vs scikit-learn.

Two parts, both written to be *fair* and *reproducible*:

* **Quality** — every method on every dataset (standardized) at a fixed `N`, scored with external
  metrics (ARI, AMI, V-measure, vs ground truth) and internal metrics (silhouette on a sample,
  Davies-Bouldin, Calinski-Harabasz), plus wall-clock fit time.
* **Scaling** — fit time and **peak process RSS** vs `N`, each run isolated in a spawned subprocess
  (its own `ru_maxrss`, a `RLIMIT_AS` cap, and a timeout) so a method that explodes in memory fails
  gracefully instead of taking down the host. This surfaces betula's memory-bounded CF-tree.

Outputs CSVs + seaborn plots into `bench/` and `bench/plots/`. Honest by construction: every cell is
guarded, failures are recorded (not hidden), and both wins and losses are reported.

Run: `.venv/bin/python bench/comprehensive.py [--quick]`
"""

from __future__ import annotations

import os

# Pin BLAS threads before importing numpy so timings are comparable and RLIMIT_AS isn't tripped.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
PLOTS.mkdir(exist_ok=True)
TIMEOUT = 180.0  # per scaling fit


# ── datasets (standardized) ───────────────────────────────────────────────────────────────────────
def gen_dataset(name: str, n: int, seed: int = 0):
    """Return (X float64 standardized, y or None, k)."""
    from sklearn.datasets import make_blobs, make_circles, make_moons
    from sklearn.preprocessing import StandardScaler

    if name == "blobs":
        X, y = make_blobs(n_samples=n, centers=6, cluster_std=1.0, random_state=seed)
        k = 6
    elif name == "aniso":
        X, y = make_blobs(n_samples=n, centers=3, cluster_std=0.8, random_state=seed)
        X = X @ np.array([[0.6, -0.6], [-0.4, 0.8]])  # skew into anisotropic clusters
        k = 3
    elif name == "varied":
        X, y = make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], centers=3, random_state=seed)
        k = 3
    elif name == "moons":
        X, y = make_moons(n_samples=n, noise=0.06, random_state=seed)
        k = 2
    elif name == "circles":
        X, y = make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=seed)
        k = 2
    elif name == "highdim":
        X, y = make_blobs(n_samples=n, n_features=20, centers=8, cluster_std=1.0, random_state=seed)
        k = 8
    else:
        raise ValueError(name)
    return StandardScaler().fit_transform(X).astype(np.float64), y, k


# ── methods ─────────────────────────────────────────────────────────────────────────────────────
BETULA_KW = dict(threshold=0.0, max_leaves=2000, seed=0, n_jobs=1)


def methods(k: int, n: int) -> dict:
    """name -> (fn(X) -> labels, scales_to_large)."""
    import betula_cluster as bc
    from sklearn.cluster import (
        HDBSCAN,
        AgglomerativeClustering,
        Birch,
        KMeans,
        MiniBatchKMeans,
    )
    from sklearn.mixture import GaussianMixture

    mcs = max(20, n // 400)
    return {
        "betula-kmeans": (
            lambda X: bc.fit_predict(X, k, feature="spherical", method="kmeans", **BETULA_KW),
            True,
        ),
        "betula-gmm": (
            lambda X: bc.fit_predict(X, k, feature="diagonal", method="gmm", **BETULA_KW),
            True,
        ),
        "betula-gmm-full": (
            lambda X: bc.fit_predict(X, k, feature="full", method="gmm-full", **BETULA_KW),
            True,
        ),
        "betula-ward": (
            lambda X: bc.fit_predict(X, k, feature="diagonal", method="ward", **BETULA_KW),
            True,
        ),
        "betula-hdbscan": (
            lambda X: bc.fit_predict(
                X, method="hdbscan", min_cluster_size=mcs, min_samples=10, **BETULA_KW
            ),
            True,
        ),
        "sklearn-kmeans": (lambda X: KMeans(k, n_init=10, random_state=0).fit_predict(X), True),
        "sklearn-minibatch": (
            lambda X: MiniBatchKMeans(k, n_init=3, random_state=0).fit_predict(X),
            True,
        ),
        "sklearn-birch": (lambda X: Birch(n_clusters=k).fit_predict(X), True),
        "sklearn-gmm": (lambda X: GaussianMixture(k, random_state=0).fit(X).predict(X), True),
        "sklearn-ward": (lambda X: AgglomerativeClustering(n_clusters=k).fit_predict(X), False),
        "sklearn-hdbscan": (
            lambda X: HDBSCAN(min_cluster_size=mcs, min_samples=10).fit_predict(X),
            False,
        ),
    }


# ── quality ───────────────────────────────────────────────────────────────────────────────────────
def score(X, y, labels):
    from sklearn import metrics

    out = {}
    uniq = set(int(v) for v in labels)
    out["n_clusters"] = len([u for u in uniq if u >= 0])
    out["noise_frac"] = float(np.mean(np.asarray(labels) < 0))
    if y is not None:
        out["ARI"] = metrics.adjusted_rand_score(y, labels)
        out["AMI"] = metrics.adjusted_mutual_info_score(y, labels)
        out["V"] = metrics.v_measure_score(y, labels)
    mask = np.asarray(labels) >= 0
    try:
        if len(uniq - {-1}) >= 2 and mask.sum() > len(uniq):
            out["silhouette"] = metrics.silhouette_score(
                X[mask],
                np.asarray(labels)[mask],
                sample_size=min(5000, int(mask.sum())),
                random_state=0,
            )
            out["davies_bouldin"] = metrics.davies_bouldin_score(X[mask], np.asarray(labels)[mask])
            out["calinski_harabasz"] = metrics.calinski_harabasz_score(
                X[mask], np.asarray(labels)[mask]
            )
    except Exception:
        pass
    return out


def run_quality(n: int, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        X, y, k = gen_dataset(ds, n)
        for name, (fn, _) in methods(k, n).items():
            rec = {"dataset": ds, "method": name}
            try:
                t0 = time.perf_counter()
                labels = np.asarray(fn(X))
                rec["time_s"] = time.perf_counter() - t0
                rec.update(score(X, y, labels))
            except Exception as e:
                rec["error"] = type(e).__name__
            rows.append(rec)
            print(
                f"  quality {ds:8s} {name:18s} ARI={rec.get('ARI', float('nan')):.3f} t={rec.get('time_s', float('nan')):.2f}s"
            )
    return pd.DataFrame(rows)


# ── isolated runner: each task is a fresh `subprocess` (clean address space → honest peak RSS) ──────
WORKER = str(HERE / "_worker.py")

# (method name, scales-to-large?) — the O(N^2) methods are capped to small N.
SCALING_METHODS = [
    ("betula-kmeans", True),
    ("betula-gmm", True),
    ("betula-gmm-full", True),
    ("betula-ward", True),
    ("betula-hdbscan", True),
    ("sklearn-kmeans", True),
    ("sklearn-minibatch", True),
    ("sklearn-birch", True),
    ("sklearn-gmm", True),
    ("sklearn-ward", False),
    ("sklearn-hdbscan", False),
]


def _run_worker(argv: list[str], timeout: float) -> dict:
    import subprocess

    try:
        out = subprocess.run(
            [sys.executable, WORKER, *argv], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    try:
        import json

        return json.loads(line)
    except Exception:
        return {"error": "died", "rc": out.returncode}


def run_scaling(sizes: list[int], dataset: str = "blobs") -> pd.DataFrame:
    rows = []
    for name, big in SCALING_METHODS:
        for n in sizes:
            if not big and n > 30_000:
                rows.append({"method": name, "n": n, "error": "skipped (O(N^2))"})
                continue
            res = _run_worker(["fit", name, dataset, str(n)], TIMEOUT)
            rows.append({"method": name, "n": n, **res})
            print(f"  scaling {name:18s} n={n:>9,} {res}")
    return pd.DataFrame(rows)


def run_memory(sizes: list[int], d: int = 20, chunk: int = 50_000) -> pd.DataFrame:
    """Peak RSS: betula streaming (chunked `partial_fit`) vs sklearn KMeans one-shot, at large N."""
    rows = []
    for label, argv in [
        ("betula (streaming)", lambda n: ["stream", str(n), str(d), str(chunk)]),
        ("sklearn-kmeans (one-shot)", lambda n: ["oneshot_km", str(n), str(d)]),
    ]:
        for n in sizes:
            res = _run_worker(argv(n), TIMEOUT * 3)
            rows.append({"method": label, "n": n, "dense_array_gb": n * d * 8 / 1e9, **res})
            print(f"  memory {label:26s} n={n:>10,} {res}")
    return pd.DataFrame(rows)


# ── plots ─────────────────────────────────────────────────────────────────────────────────────────
def make_plots(q: pd.DataFrame, s: pd.DataFrame, m: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk", palette="deep")

    if "ARI" in q:
        piv = q.pivot_table(index="method", columns="dataset", values="ARI")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            piv,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "ARI"},
        )
        ax.set_title("Clustering quality (ARI vs ground truth)")
        fig.tight_layout()
        fig.savefig(PLOTS / "quality_ari.png", dpi=110)
        plt.close(fig)

    sc = s[s["time_s"].notna()] if "time_s" in s else s.iloc[0:0]
    if len(sc):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(data=sc, x="n", y="time_s", hue="method", marker="o", ax=ax)
        ax.set(
            xscale="log",
            yscale="log",
            title="Fit time vs N (one-shot)",
            xlabel="N (points)",
            ylabel="fit time (s)",
        )
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()
        fig.savefig(PLOTS / "scaling_time.png", dpi=110)
        plt.close(fig)

    mm = m[m["rss_mb"].notna()] if "rss_mb" in m else m.iloc[0:0]
    if len(mm):
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.lineplot(data=mm, x="n", y="rss_mb", hue="method", marker="o", ax=ax)
        # reference: the dense float64 array any in-core method must hold
        ax.plot(m["n"], m["dense_array_gb"] * 1024, "k--", alpha=0.5, label="dense array (N×d×8)")
        ax.set(
            xscale="log",
            yscale="log",
            title="Peak memory vs N — streaming stays bounded",
            xlabel="N (points)",
            ylabel="peak RSS (MB)",
        )
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS / "memory_streaming.png", dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller N for a fast smoke run")
    args = ap.parse_args()

    qn = 8_000 if args.quick else 30_000
    sizes = [5_000, 20_000] if args.quick else [10_000, 50_000, 200_000, 1_000_000]
    mem_sizes = (
        [200_000, 1_000_000]
        if args.quick
        else [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
    )
    datasets = ["blobs", "aniso", "varied", "moons", "circles", "highdim"]

    print(f"[quality] N={qn:,}")
    q = run_quality(qn, datasets)
    q.to_csv(HERE / "results_quality.csv", index=False)
    print(f"[scaling] sizes={sizes}")
    s = run_scaling(sizes)
    s.to_csv(HERE / "results_scaling.csv", index=False)
    print(f"[memory] streaming vs one-shot, sizes={mem_sizes}")
    m = run_memory(mem_sizes)
    m.to_csv(HERE / "results_memory.csv", index=False)
    make_plots(q, s, m)
    print("wrote results_{quality,scaling,memory}.csv, plots/*.png")
    return q, s, m


if __name__ == "__main__":
    main()
