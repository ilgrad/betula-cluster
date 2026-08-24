"""Quality against the leaf budget, crossed with the routing distance — "how many leaves do I need".

The published quality tables fix `max_leaves` and vary the head. That answers the wrong question for
a summarization library: the user's knob is the budget, and the tables say nothing about what a
smaller one costs. Worse, the `digits` row runs at `max_leaves=4000` against `n=1797` — **zero
compression**. Every leaf is a point, so that row is raw-point clustering behind a betula wrapper
and cannot support any claim about summarization at all. The first budget in each sweep below
reproduces that setting on purpose, so the curve shows what it is worth.

The third axis is the routing distance. Task 27 (2026-08-23) measured the routing lever and found
its effect is a function of *compression*, not of the head: at 1797 singleton leaves on digits all
four distances produce an identical leaf set, while on covtype-20k at 5.2x compression `ward`
routing moved the maximum leaf weight from 295 to 88. So a `max_leaves x distance` sweep is what
settles when the distance is worth setting, and a one-budget comparison never could.

Two derived columns feed task 60 (Zador-form fit -> effective intrinsic dimension):

    compression   n / realised leaves
    mean_sq_radius  sum_i w_i r_i^2 / n, the summary's mean squared quantization error

Zador's theorem says the latter falls like `m^(-2/d_eff)` for a good quantizer of an intrinsically
`d_eff`-dimensional source, so the slope of `log mean_sq_radius` against `log m` is an estimate of
`-2/d_eff` that needs no labels at all.

    uv run --no-sync --with scikit-learn --with pandas python bench/leaf_budget.py

Writes `bench/results_budget.csv`. ARI is the median of three seeds per cell, as the project's
measurement discipline requires; the seed spread is carried alongside so a cell whose median is
meaningless says so.
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

from insertion_order import load  # same standardization, same subsample, one definition

SEEDS = (0, 1, 2)
DISTANCES = ("euclidean", "manhattan", "ward", "average")
HEADS = ("kmeans", "gmm", "ward")

# The `ward` head is O(m^2) in the leaf count, so it is not asked for budgets it would spend the
# afternoon on. Nothing is silently dropped: the skipped cells are printed and counted.
WARD_HEAD_MAX_LEAVES = 2000

DATASETS = {
    # First entry is at or above `n`: the zero-compression control, and the setting the published
    # digits row was measured at.
    "digits": (4000, 1797, 900, 450, 225, 112, 90, 45),
    "covtype-20k": (20_000, 8000, 4000, 2000, 1000, 500, 250, 100),
    "mnist-10k": (10_000, 4000, 2000, 1000, 500, 250, 100),
}


def one_fit(x, y, k, head, budget, distance, seed, ari):
    import betula_cluster as bc

    est = bc.Betula(
        n_clusters=k,
        method=head,
        feature="spherical",
        threshold=0.0,
        max_leaves=budget,
        distance=distance,
        seed=seed,
    )
    t0 = time.perf_counter()
    labels = est.fit_predict(x)
    dt = time.perf_counter() - t0
    w = np.asarray(est.microcluster_weights_, dtype=np.float64)
    r = np.asarray(est.microcluster_radii_, dtype=np.float64)
    return {
        "ari": float(ari(y, labels)),
        "leaves": int(est.n_leaves_),
        "max_leaf_weight": float(w.max()) if w.size else 0.0,
        # sum_i w_i r_i^2 / n -- the summary's mean squared quantization error. `r` is the leaf RMS
        # radius, so w*r^2 is that leaf's scatter S_i exactly, not an approximation of it.
        "mean_sq_radius": float((w * r * r).sum() / max(w.sum(), 1.0)),
        "time_s": dt,
    }


def main() -> int:
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    rows, skipped = [], 0
    for name, budgets in DATASETS.items():
        x, y, k = load(name)
        n = len(x)
        print(f"\n{name}  n={n}  d={x.shape[1]}  k={k}")
        for budget in budgets:
            for distance in DISTANCES:
                for head in HEADS:
                    if head == "ward" and budget > WARD_HEAD_MAX_LEAVES:
                        skipped += 1
                        continue
                    got = [one_fit(x, y, k, head, budget, distance, s, ari) for s in SEEDS]
                    aris = np.array([g["ari"] for g in got])
                    mid = got[int(np.argsort(aris)[len(aris) // 2])]
                    rows.append(
                        {
                            "dataset": name,
                            "n": n,
                            "max_leaves": budget,
                            "distance": distance,
                            "head": head,
                            "seeds": len(SEEDS),
                            "ari": float(np.median(aris)),
                            "ari_min": float(aris.min()),
                            "ari_max": float(aris.max()),
                            "leaves": mid["leaves"],
                            "compression": n / max(mid["leaves"], 1),
                            "max_leaf_weight": mid["max_leaf_weight"],
                            "mean_sq_radius": mid["mean_sq_radius"],
                            "time_s": float(np.median([g["time_s"] for g in got])),
                        }
                    )
                    r = rows[-1]
                    print(
                        f"  budget {budget:<6} {distance:<9} {head:<7} "
                        f"ARI {r['ari']:.4f} [{r['ari_min']:.4f}, {r['ari_max']:.4f}]  "
                        f"leaves {r['leaves']:<6} x{r['compression']:.1f}  "
                        f"maxw {r['max_leaf_weight']:.0f}  "
                        f"msr {r['mean_sq_radius']:.4g}  {r['time_s']:.2f}s"
                    )

    out = HERE / "results_budget.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        f"\nwrote {out.name} ({len(rows)} rows); "
        f"{skipped} ward-head cells skipped above max_leaves={WARD_HEAD_MAX_LEAVES}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
