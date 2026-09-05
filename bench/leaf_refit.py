"""What `leaf_refit` is worth, and what it is worth *on top of* `canonical_order`.

A leaf CF is an absorption **history**: the rows that happened to arrive while that entry was the
nearest one. The tree those rows finished in routes a different partition, so the head reads a
summary of the wrong sets. `leaf_refit=k` runs `k` passes of: route every row through the finished
tree, rebuild each leaf CF from exactly the rows it wins, drop the entries that win none.

`canonical_order` attacks the same defect from the other end — it fixes *which* prototypes exist,
by sorting the rows before the first insert — so the two could plausibly overlap. The hypothesis
this file was written to test is that they do: a net built by sweeping the space should already sit
close to its own routed partition, leaving `leaf_refit` less to repair. Four arms per cell measure
it directly rather than arguing about it.

**This file exists in `bench/` and not in a scratch directory on purpose.** The table it feeds
(`docs/USAGE.md`) previously had no reproducible source: its rows turned out to have come from at
least two different preprocessings, and the discrepancy between them is what exposed the
column-major ingest bug. A published number whose script is not in the repository is a number
nobody can check.

Setup, stated because the last version of this table did not state one: **raw features** — no
scaling, no normalisation — with every parameter other than the ones under test left at its default.
`digits` is all 1797 rows; `mnist-10k` is the first 10 000 rows of OpenML `mnist_784`. ARI against
the class labels, median of seeds 0/1/2, identity row order.

    uv run --no-sync --with scikit-learn python bench/leaf_refit.py

Writes `bench/results_refit.csv` (one row per cell) and prints the table.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from betula_cluster import fit_predict
from sklearn.datasets import fetch_openml, load_digits
from sklearn.metrics import adjusted_rand_score as ari

HERE = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
HEADS = ("kmeans", "ward", "gmm")


def quality(x, truth, k, ml, head, refit, canonical):
    """Median ARI over [`SEEDS`] — a single run is not a result on a non-convex head."""
    return float(
        np.median(
            [
                ari(
                    truth,
                    fit_predict(
                        x,
                        k,
                        method=head,
                        max_leaves=ml,
                        seed=s,
                        leaf_refit=refit,
                        canonical_order=canonical,
                    ),
                )
                for s in SEEDS
            ]
        )
    )


def datasets():
    """Raw, unscaled features. `np.ascontiguousarray` is not decoration: `fetch_openml` hands back a
    column-major array, and reading one as rows was a live bug until 2026-09-05 — the row order the
    engine sees should be the one this file claims it sees, not one numpy happens to supply."""
    d = load_digits()
    m = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    return {
        "digits": (np.ascontiguousarray(d.data, dtype=np.float64), d.target, 10, 90),
        "mnist-10k": (
            np.ascontiguousarray(m.data[:10000], dtype=np.float64),
            m.target[:10000].astype(int),
            10,
            200,
        ),
    }


def main() -> None:
    rows = []
    print(
        f"{'cell':<28} {'arr r=0':>8} {'arr r=1':>8} {'can r=0':>8} {'can r=1':>8} "
        f"{'d_arr':>7} {'d_can':>7}"
    )
    for name, (x, truth, k, ml) in datasets().items():
        for head in HEADS:
            cell = {"dataset": name, "max_leaves": ml, "head": head}
            for arm, canonical in (("arrival", False), ("canonical", True)):
                for refit in (0, 1):
                    cell[f"{arm}_refit{refit}"] = quality(x, truth, k, ml, head, refit, canonical)
            cell["delta_arrival"] = cell["arrival_refit1"] - cell["arrival_refit0"]
            cell["delta_canonical"] = cell["canonical_refit1"] - cell["canonical_refit0"]
            rows.append(cell)
            print(
                f"{name + ' ml=' + str(ml) + ' ' + head:<28} "
                f"{cell['arrival_refit0']:8.3f} {cell['arrival_refit1']:8.3f} "
                f"{cell['canonical_refit0']:8.3f} {cell['canonical_refit1']:8.3f} "
                f"{cell['delta_arrival']:+7.3f} {cell['delta_canonical']:+7.3f}",
                flush=True,
            )

    out = HERE / "results_refit.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
