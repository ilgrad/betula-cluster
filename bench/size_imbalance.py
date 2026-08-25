"""Size imbalance: what a *geometric* leaf budget costs when the mass is not spread geometrically.

scikit-learn's Birch issue #22854 reports that Birch returns far fewer clusters than asked for when
one cluster dominates the mass. Both scikit-learn's Birch and this library allocate leaves the same
way -- a single global absorption radius, raised until the leaf count fits -- so the issue is a
property of the CF-tree family rather than of one implementation, and it is worth a measured row
instead of an anecdote.

Two fixtures, one mass profile: 80% of the points in a tight core, 20% spread across five diffuse
minorities ten times wider.

    structured   the core holds two true clusters 2.0 apart, well   (k = 7)
                 inside the minorities' own spread -- so collapsing
                 the core destroys information no head can recover
    flat         the control: same masses, no internal structure,   (k = 6)
                 where collapsing the core costs nothing

`flat` is what makes this a finding rather than a fixture. If the collapse were a scoring artefact it
would show there too; it does not.

Columns beyond ARI say *where* the budget went: `leaves` realised against the budget (fill), `thr`
the absorption radius the rebuild settled on, `maxw` the heaviest leaf and `top1` its share of the
total mass. A `top1` near 0.8 with a 90%+ fill is the whole finding -- the budget is spent, and spent
on the 20% of the mass that happens to be spread out.

    uv run --no-sync --with scikit-learn --with pandas python bench/size_imbalance.py

Writes `bench/results_imbalance.csv`. Every cell is the median of three seeds with the spread
carried alongside, as the project's measurement discipline requires.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Single-threaded on purpose: a parallel reduction reorders its floating-point sums, and the
# seed-to-seed spread this study reports would then carry a thread-timing term as well.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent

N = 100_000
D = 10
BUDGETS = (250, 1000, 4000)
SEEDS = (0, 1, 2)
HEADS = ("kmeans", "ward", "gmm")

# The `ward` head is O(m^2) in the leaf count; above this it is skipped and the skip is counted.
WARD_HEAD_MAX_LEAVES = 2000


def fixture(kind: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """80% of the mass in a tight core, 20% across five minorities ten times wider.

    The separation inside the `structured` core (2.0 before standardization) is deliberately much
    smaller than the minorities' spread (1.0 each, centres up to 20 apart): a global absorption
    radius large enough to bound the leaf count is then already larger than the structure that has
    to survive.
    """
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-10.0, 10.0, size=(6, D))
    blocks, labels = [], []
    n_major, n_minor = int(0.80 * N), int(0.04 * N)
    if kind == "structured":
        half = n_major // 2
        offset = np.zeros(D)
        offset[0] = 1.0
        for i, m in enumerate((half, n_major - half)):
            blocks.append(
                centres[0] + (offset if i else -offset) + 0.1 * rng.standard_normal((m, D))
            )
            labels.append(np.full(m, i))
        nxt = 2
    elif kind == "flat":
        blocks.append(centres[0] + 0.1 * rng.standard_normal((n_major, D)))
        labels.append(np.zeros(n_major))
        nxt = 1
    else:
        raise ValueError(kind)
    for j in range(1, 6):
        blocks.append(centres[j] + 1.0 * rng.standard_normal((n_minor, D)))
        labels.append(np.full(n_minor, nxt + j - 1))
    x = np.vstack(blocks)
    y = np.concatenate(labels).astype(int)
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    return np.ascontiguousarray(x), y, int(y.max()) + 1


def betula_cell(x, y, k, head, budget, seed, ari):
    import betula_cluster as bc

    est = bc.Betula(
        n_clusters=k,
        feature="spherical",
        method=head,
        threshold=0.0,
        max_leaves=budget,
        seed=seed,
    )
    t0 = time.perf_counter()
    labels = np.asarray(est.fit_predict(x))
    dt = time.perf_counter() - t0
    w = np.asarray(est.microcluster_weights_, dtype=np.float64)
    return {
        "ari": float(ari(y, labels)),
        "leaves": int(w.size),
        "threshold": float(est.threshold_),
        "max_leaf_weight": float(w.max()),
        "top1_mass": float(w.max() / w.sum()),
        "time_s": dt,
    }


def sklearn_birch_cell(x, y, k, seed, ari):
    """scikit-learn's own Birch, at its default threshold -- the implementation #22854 is filed
    against. It takes no leaf budget, so `leaves` is what its threshold happened to produce and the
    budget column is left at zero."""
    from sklearn.cluster import Birch

    brc = Birch(n_clusters=k, threshold=0.5)
    t0 = time.perf_counter()
    labels = brc.fit_predict(x)
    dt = time.perf_counter() - t0
    centres = np.asarray(brc.subcluster_centers_, dtype=np.float64)
    # Birch exposes no leaf weights, so the heaviest leaf is recovered by assigning the points.
    d2 = (
        (x * x).sum(1)[:, None] - 2.0 * x @ centres.T + (centres * centres).sum(1)[None, :]
    ).argmin(1)
    counts = np.bincount(d2, minlength=centres.shape[0]).astype(np.float64)
    return {
        "ari": float(ari(y, labels)),
        "leaves": int(centres.shape[0]),
        "threshold": 0.5,
        "max_leaf_weight": float(counts.max()),
        "top1_mass": float(counts.max() / counts.sum()),
        "time_s": dt,
    }


def sklearn_birch_matched_cell(x, y, k, budget, ari, steps=10):
    """The same Birch, with its threshold bisected until its subcluster count matches `budget`.

    Its default threshold is not a like-for-like setting -- it produced ~7 000 subclusters above,
    against budgets of 250 to 4 000 -- and a rival compared only at its defaults is a rival
    misreported. Birch is deterministic, so this runs once per budget rather than per seed; the
    realised count is carried in `leaves` so a bisection that did not converge is visible.
    """
    from sklearn.cluster import Birch

    lo, hi = 0.01, 20.0
    best = None
    t0 = time.perf_counter()
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        brc = Birch(n_clusters=k, threshold=mid).fit(x)
        got = int(np.asarray(brc.subcluster_centers_).shape[0])
        if best is None or abs(got - budget) < abs(best[1] - budget):
            best = (brc, got)
        if got > budget:  # too many subclusters -> a wider radius merges more
            lo = mid
        else:
            hi = mid
    dt = time.perf_counter() - t0
    brc, got = best
    centres = np.asarray(brc.subcluster_centers_, dtype=np.float64)
    assign = (
        (x * x).sum(1)[:, None] - 2.0 * x @ centres.T + (centres * centres).sum(1)[None, :]
    ).argmin(1)
    counts = np.bincount(assign, minlength=centres.shape[0]).astype(np.float64)
    return {
        "ari": float(ari(y, brc.predict(x))),
        "leaves": got,
        "threshold": float(brc.threshold),
        "max_leaf_weight": float(counts.max()),
        "top1_mass": float(counts.max() / counts.sum()),
        "time_s": dt,
    }


def sklearn_kmeans_cell(x, y, k, seed, ari):
    """The no-budget control: k-means on the raw points has nothing to misallocate."""
    from sklearn.cluster import KMeans

    t0 = time.perf_counter()
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(x)
    dt = time.perf_counter() - t0
    return {
        "ari": float(ari(y, labels)),
        "leaves": 0,
        "threshold": float("nan"),
        "max_leaf_weight": float("nan"),
        "top1_mass": float("nan"),
        "time_s": dt,
    }


def summarize(name, n, k, method, budget, got):
    aris = np.array([g["ari"] for g in got])
    mid = got[int(np.argsort(aris)[len(aris) // 2])]
    return {
        "dataset": name,
        "n": n,
        "k": k,
        "method": method,
        "max_leaves": budget,
        "seeds": len(got),
        "ari": float(np.median(aris)),
        "ari_min": float(aris.min()),
        "ari_max": float(aris.max()),
        "leaves": mid["leaves"],
        "fill": (mid["leaves"] / budget) if budget else float("nan"),
        "threshold": mid["threshold"],
        "max_leaf_weight": mid["max_leaf_weight"],
        "top1_mass": mid["top1_mass"],
        "time_s": float(np.median([g["time_s"] for g in got])),
    }


def main() -> int:
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    rows, skipped = [], 0
    for name in ("structured", "flat"):
        x, y, k = fixture(name)
        n = len(x)
        print(f"\n== {name}  n={n} d={D} k={k}", flush=True)
        for method, cell in (
            ("sklearn-kmeans", sklearn_kmeans_cell),
            ("sklearn-birch", sklearn_birch_cell),
        ):
            got = [cell(x, y, k, s, ari) for s in SEEDS]
            rows.append(summarize(name, n, k, method, 0, got))
            r = rows[-1]
            print(
                f"  {method:<16} ARI {r['ari']:.4f} [{r['ari_min']:.4f}, {r['ari_max']:.4f}]  "
                f"leaves {r['leaves']:<6} top1 {r['top1_mass']:.3f}  {r['time_s']:.1f}s",
                flush=True,
            )
        for budget in BUDGETS:
            got = [sklearn_birch_matched_cell(x, y, k, budget, ari)]
            rows.append(summarize(name, n, k, "sklearn-birch-matched", budget, got))
            r = rows[-1]
            print(
                f"  budget {budget:<5} sklearn-birch-matched "
                f"ARI {r['ari']:.4f}  leaves {r['leaves']:<6} thr {r['threshold']:.3f}  "
                f"top1 {r['top1_mass']:.3f}  {r['time_s']:.1f}s",
                flush=True,
            )
            for head in HEADS:
                if head == "ward" and budget > WARD_HEAD_MAX_LEAVES:
                    skipped += 1
                    continue
                got = [betula_cell(x, y, k, head, budget, s, ari) for s in SEEDS]
                rows.append(summarize(name, n, k, f"betula-{head}", budget, got))
                r = rows[-1]
                print(
                    f"  budget {budget:<5} betula-{head:<7} "
                    f"ARI {r['ari']:.4f} [{r['ari_min']:.4f}, {r['ari_max']:.4f}]  "
                    f"leaves {r['leaves']:<6} fill {r['fill']:.2f}  thr {r['threshold']:.3f}  "
                    f"maxw {r['max_leaf_weight']:.0f}  top1 {r['top1_mass']:.3f}  "
                    f"{r['time_s']:.1f}s",
                    flush=True,
                )

    out = HERE / "results_imbalance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(
        f"\nwrote {out.name} ({len(rows)} rows); "
        f"{skipped} ward-head cells skipped above max_leaves={WARD_HEAD_MAX_LEAVES}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
