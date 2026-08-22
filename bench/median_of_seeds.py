"""Reduce per-seed benchmark runs to the published tables by taking the element-wise median.

Clustering quality is seed-dependent and EM is non-convex, so a single run is not a result. The
quality suites of ``comprehensive.py`` are run once per seed with ``--seed S --tag _sS``; this script
collapses those runs into the canonical ``results_*.csv`` that the docs quote.

The median, not the mean: one restart that lands in a bad local optimum should not drag the published
number, and with an odd number of seeds the median is an *observed* run rather than a synthetic point
between runs.

Run: ``python bench/median_of_seeds.py --seeds 0 1 2``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# table stem -> the columns that identify a row (everything else is a measurement)
TABLES = {
    "results_quality": ["dataset", "method"],
    "results_real": ["dataset", "n", "method"],
    "results_real_hires": ["dataset", "n", "max_leaves", "method"],
    "results_real_normalize": ["dataset", "n", "d", "method"],
}


def reduce_table(stem: str, keys: list[str], seeds: list[int]) -> pd.DataFrame | None:
    parts = []
    for s in seeds:
        f = HERE / f"{stem}_s{s}.csv"
        if not f.exists():
            print(f"  {stem}: missing {f.name} — table skipped")
            return None
        parts.append(pd.read_csv(f).assign(_seed=s))
    df = pd.concat(parts, ignore_index=True)

    # A cell that errored carries no measurement in any seed; keep the message rather than emit a
    # row of NaNs that reads like a silent zero.
    err = None
    if "error" in df.columns:
        err = df.groupby(keys, dropna=False)["error"].first()
        df = df.drop(columns=["error"])

    num = [c for c in df.columns if c not in keys and c != "_seed"]
    out = df.groupby(keys, dropna=False)[num].median(numeric_only=True)
    if err is not None:
        out = out.join(err)
    out = out.reset_index()

    # Preserve the row order of the first seed's file: the tables are read by humans top to bottom.
    order = parts[0][keys].drop_duplicates()
    out = order.merge(out, on=keys, how="left")
    out.to_csv(HERE / f"{stem}.csv", index=False)

    # The median alone hides how much of it is luck. Measured here: digits `betula-kmeans` ranges
    # over 0.443-0.571 across three seeds, and the single-seed table this replaces happened to
    # publish the top of that range as the result. The spread ships next to the medians so a reader
    # can see which rows are seed-stable and which are not.
    metric = next((c for c in ("ARI", "ARI_norm_on") if c in df.columns), None)
    if metric is not None:
        g = df.groupby(keys, dropna=False)[metric]
        spread = pd.DataFrame({"min": g.min(), "median": g.median(), "max": g.max()})
        spread["range"] = spread["max"] - spread["min"]
        spread = order.merge(spread.reset_index(), on=keys, how="left")
        spread.to_csv(HERE / f"{stem}_spread.csv", index=False)

    print(f"  {stem}: {len(out)} rows from {len(seeds)} seeds -> {stem}.csv")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print(f"[median] seeds={args.seeds}")
    written = [
        stem for stem, keys in TABLES.items() if reduce_table(stem, keys, args.seeds) is not None
    ]

    # Provenance sidecar rather than a comment line in the CSVs: a `#` header would break every naive
    # `pd.read_csv` consumer, and the seed list has to survive next to the numbers it produced.
    (HERE / "results_seeds.json").write_text(
        json.dumps({"seeds": args.seeds, "tables": written}, indent=2) + "\n"
    )
    print(f"[median] wrote results_seeds.json ({len(written)} tables)")


if __name__ == "__main__":
    main()
