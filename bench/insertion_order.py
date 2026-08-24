"""Insertion-order sensitivity: the defining property of the BIRCH family, measured.

A CF-tree is built by streaming points in, and every routing decision is made against the tree as
it stands at that moment. Reorder the input and you get a different tree — that is not a bug, it is
the price of a single pass. Every BIRCH-class library inherits it and none of them publish the size
of the effect, so "betula is order-sensitive" has been an assertion here rather than a number.

The study isolates it with a control. Two groups of `P` runs per cell:

    vary="order"  P permutations of the row order, estimator seed held at 0
    vary="seed"   identity order, estimator seeds 0..P-1

and reports the same statistics for both. Without the seed arm the order spread is unreadable: the
k-means and GMM heads are non-convex and would show a spread from restarts alone. `ward` is the
built-in check on the harness itself — it is deterministic given a leaf set, so its seed arm must
report exactly zero spread, and any other number means the two arms are not isolating what they
claim.

The cell that motivated this (task 27, 2026-08-23) is `digits` at a leaf budget above `n`: there the
four routing distances produce an *identical* leaf set, yet the `ward` head returned 0.6224-0.6525
ARI across them. With no compression and no head randomness, a 0.030 spread can only be leaf
*ordering* — the tie-break order in which equal-distance leaves are visited. So the budgets below
deliberately straddle that regime: one at no compression at all, two where compression is real.

    uv run --no-sync --with scikit-learn --with pandas python bench/insertion_order.py

Writes `bench/results_order.csv` (one row per cell) and prints a table. Small datasets on purpose —
this measures a property of the tree, not throughput, and P x 2 x heads full fits per cell is the
cost of a spread rather than a point estimate.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Single-threaded on purpose, and this one is load-bearing rather than tidy: a parallel reduction
# reorders its floating-point sums, so with rayon free to pick a schedule the run-to-run spread
# would carry a thread-timing term that neither arm of this study is trying to measure.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

P = 8
HEADS = ("kmeans", "gmm", "ward")

# `budget` is `max_leaves`. The first entry of each list is at or above `n` on purpose: it is the
# zero-compression control, and it is also the setting several published rows were measured at.
DATASETS = {
    "digits": {"n": None, "k": 10, "budgets": (4000, 360, 90)},
    "covtype-20k": {"n": 20_000, "k": 7, "budgets": (20_000, 2000, 300)},
    "mnist-10k": {"n": 10_000, "k": 10, "budgets": (10_000, 1000, 200)},
}


def load(name: str):
    """Standardized `(X, y, k)`. Subsampled datasets take a fixed permutation, never a head slice —
    covtype and MNIST both ship class-ordered, and a head slice would hand back a truncated label
    set that quietly makes every ARI in the table a different question."""
    from sklearn.preprocessing import StandardScaler

    spec = DATASETS[name]
    if name == "digits":
        from sklearn.datasets import load_digits

        d = load_digits()
        x, y = np.asarray(d.data, dtype=np.float64), np.asarray(d.target, dtype=int)
    else:
        from _worker import load_real_worker

        base = "covtype" if name.startswith("covtype") else "mnist"
        x, y, _ = load_real_worker(base)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(x))[: spec["n"]]
        return np.ascontiguousarray(x[idx]), y[idx], spec["k"]
    x = StandardScaler().fit_transform(x)
    return np.ascontiguousarray(x), y, spec["k"]


def one_fit(x, k, head, budget, seed, perm):
    """One full fit under a given row order; labels come back in the *original* row order."""
    import betula_cluster as bc

    est = bc.Betula(
        n_clusters=k,
        method=head,
        feature="spherical",
        threshold=0.0,
        max_leaves=budget,
        seed=seed,
    )
    t0 = time.perf_counter()
    permuted = est.fit_predict(x[perm])
    dt = time.perf_counter() - t0
    labels = np.empty_like(permuted)
    labels[perm] = permuted
    weights = np.asarray(est.microcluster_weights_, dtype=np.float64)
    return {
        "labels": labels,
        "leaves": int(est.n_leaves_),
        "max_leaf_weight": float(weights.max()) if weights.size else 0.0,
        "time_s": dt,
    }


def summarize(runs, truth, ari) -> dict:
    """Spread across a group of runs: against the truth, and against each other."""
    aris = np.array([ari(truth, r["labels"]) for r in runs])
    pair = [
        ari(runs[i]["labels"], runs[j]["labels"])
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    leaves = np.array([r["leaves"] for r in runs])
    return {
        "ari_min": float(aris.min()),
        "ari_median": float(np.median(aris)),
        "ari_max": float(aris.max()),
        "ari_spread": float(aris.max() - aris.min()),
        "pairwise_ari_mean": float(np.mean(pair)),
        "pairwise_ari_min": float(np.min(pair)),
        "leaves_min": int(leaves.min()),
        "leaves_max": int(leaves.max()),
        "max_leaf_weight_max": float(max(r["max_leaf_weight"] for r in runs)),
        "time_s_median": float(np.median([r["time_s"] for r in runs])),
    }


def main() -> int:
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    rows = []
    for name, spec in DATASETS.items():
        x, y, k = load(name)
        n = len(x)
        rng = np.random.default_rng(0)
        perms = [np.arange(n)] + [rng.permutation(n) for _ in range(P - 1)]
        identity = np.arange(n)
        print(f"\n{name}  n={n}  d={x.shape[1]}  k={k}")
        for budget in spec["budgets"]:
            for head in HEADS:
                for vary in ("order", "seed"):
                    if vary == "order":
                        runs = [one_fit(x, k, head, budget, 0, p) for p in perms]
                    else:
                        runs = [one_fit(x, k, head, budget, s, identity) for s in range(P)]
                    stat = summarize(runs, y, ari)
                    rows.append(
                        {
                            "dataset": name,
                            "n": n,
                            "max_leaves": budget,
                            "head": head,
                            "vary": vary,
                            "p": P,
                            **stat,
                        }
                    )
                    print(
                        f"  budget {budget:<6} {head:<7} vary={vary:<5} "
                        f"ARI {stat['ari_median']:.4f} "
                        f"[{stat['ari_min']:.4f}, {stat['ari_max']:.4f}] "
                        f"spread {stat['ari_spread']:.4f}  "
                        f"pairwise {stat['pairwise_ari_mean']:.4f}  "
                        f"leaves {stat['leaves_min']}-{stat['leaves_max']}"
                    )

    out = HERE / "results_order.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.name} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
