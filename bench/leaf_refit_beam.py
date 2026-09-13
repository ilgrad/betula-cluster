"""Does making `leaf_refit` consistent with `route_beam` buy anything, or only stop a mismatch?

Since 1.1 the leaf refit routes at the estimator's `route_beam` instead of always greedily. That
change is justified by consistency alone — a Lloyd step is only a Lloyd step against the rule it is
read back with, and the old pass fitted the leaves to a partition none of the estimator's own
queries reproduce. Consistency is not quality, though, and the two are separate claims. This file
measures the second one so the record does not have to imply it.

Four arms per cell, `route_beam` × `leaf_refit` over {1, 8} × {0, 1}. Two of them are also a check
on the scoping rule rather than on the change: at `leaf_refit=0` the width must not reach the build
at all, so `kmeans` and `gmm` — which label from their own `k` centres and never consult the tree —
must read *identical* at both widths, and only `ward` may move. If a b=8/r=0 cell differs from
b=1/r=0 under `kmeans`, the width has leaked somewhere it was never meant to go and the rest of the
table is uninterpretable.

The free `fit_predict` cannot express the question: it takes no `route_beam`, by the same rule that
keeps the fractional `max_leaves` on the estimator. So this harness uses `Betula` where
`bench/leaf_refit.py` uses the free function — the two tables are otherwise the same fixtures, the
same budgets and the same seeds, and the b=1 column here should reproduce the arrival column there.

Setup as in `bench/leaf_refit.py`: **raw features**, no scaling, everything else at its default;
`digits` is all 1797 rows at `max_leaves=90`, `mnist-10k` the first 10 000 rows of OpenML
`mnist_784` at 200. ARI against the class labels, median of seeds 0/1/2, identity row order.

    uv run --no-sync --with scikit-learn --with pandas python bench/leaf_refit_beam.py

`--with pandas` is not optional: `fetch_openml` needs it to parse the dense ARFF, and without it
the `mnist-10k` half of the table cannot be built at all.

The pre-1.1 arm — a greedy refit queried at a wide route — is not a parameter combination any
build can express, so it is measured by reverting the one line that carries the width
(`refit_tree` passing `1` instead of `self.route_beam`) and rebuilding. The other three arms cannot
move under that revert, which makes them the control: if they differ between the two builds,
something other than the change under test did.

Writes `bench/results_refit_beam.csv` (one row per cell) and prints the table.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from betula_cluster import Betula
from sklearn.metrics import adjusted_rand_score as ari

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _fixtures

SEEDS = (0, 1, 2)
HEADS = ("kmeans", "ward", "gmm")
ARMS = ((1, 0), (1, 1), (8, 0), (8, 1))


def quality(x, truth, k, ml, head, beam, refit):
    """Median ARI over [`SEEDS`] and the total fit time, since the wide pass costs a linear factor
    in the width on its route and a quality table that hides the price is half a result."""
    scores, t0 = [], time.perf_counter()
    for s in SEEDS:
        est = Betula(
            n_clusters=k,
            method=head,
            max_leaves=ml,
            seed=s,
            leaf_refit=refit,
            route_beam=beam,
        )
        scores.append(ari(truth, est.fit_predict(x)))
    return float(np.median(scores)), time.perf_counter() - t0


def datasets():
    """Raw, unscaled features. `np.ascontiguousarray` because `fetch_openml` hands back a
    column-major array and the row order the engine sees should be the one this file claims."""
    dx, dy, _ = _fixtures.fetch("digits")
    mx, my, _ = _fixtures.fetch("mnist")
    return {
        "digits": (np.ascontiguousarray(dx), dy, 10, 90),
        "mnist-10k": (np.ascontiguousarray(mx[:10000]), my[:10000], 10, 200),
    }


def main() -> None:
    rows = []
    head_fmt = " ".join(f"{f'b={b} r={r}':>9}" for b, r in ARMS)
    print(f"{'cell':<28} {head_fmt} {'d_refit@1':>10} {'d_refit@8':>10} {'x time':>7}")
    for name, (x, truth, k, ml) in datasets().items():
        for head in HEADS:
            cell = {"dataset": name, "max_leaves": ml, "head": head}
            for beam, refit in ARMS:
                score, secs = quality(x, truth, k, ml, head, beam, refit)
                cell[f"ari_b{beam}_r{refit}"] = score
                cell[f"secs_b{beam}_r{refit}"] = secs
            cell["delta_refit_b1"] = cell["ari_b1_r1"] - cell["ari_b1_r0"]
            cell["delta_refit_b8"] = cell["ari_b8_r1"] - cell["ari_b8_r0"]
            cell["refit_cost_ratio"] = cell["secs_b8_r1"] / cell["secs_b1_r1"]
            rows.append(cell)
            scores = " ".join(f"{cell[f'ari_b{b}_r{r}']:9.3f}" for b, r in ARMS)
            print(
                f"{name + ' ml=' + str(ml) + ' ' + head:<28} {scores} "
                f"{cell['delta_refit_b1']:+10.3f} {cell['delta_refit_b8']:+10.3f} "
                f"{cell['refit_cost_ratio']:7.2f}",
                flush=True,
            )

    out = HERE / "results_refit_beam.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
