"""Reduce per-seed benchmark runs to the published tables: each row is the median run.

Clustering quality is seed-dependent and EM is non-convex, so a single run is not a result. The
quality suites of ``comprehensive.py`` are run once per seed with ``--seed S --tag _sS``; this script
collapses those runs into the canonical ``results_*.csv`` that the docs quote.

The median, not the mean: one restart that lands in a bad local optimum should not drag the published
number, and with an odd number of seeds the median is an *observed* run rather than a synthetic point
between runs. The run is chosen by the table's ARI column and the whole row is taken from it, so the
cluster count, noise share and time beside a published ARI are that run's. A median per column would
print one seed's ARI beside another seed's cluster count.

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
    files = {s: HERE / f"{stem}_s{s}.csv" for s in seeds}
    missing = [f.name for f in files.values() if not f.exists()]
    if len(missing) == len(files):
        print(f"  {stem}: not run for any seed — table skipped")
        return None
    if missing:
        # A median over the seeds that happen to exist is a different statistic under the same
        # name; the old code dropped the whole table here without failing.
        raise SystemExit(f"{stem}: missing {', '.join(missing)}; rerun those seeds")
    parts = [pd.read_csv(f).assign(_seed=s) for s, f in files.items()]
    df = pd.concat(parts, ignore_index=True)
    metric = next(c for c in ("ARI", "ARI_norm_on") if c in df.columns)

    # The median run of each row: rank the runs by the metric (ties by seed) and take the middle
    # one. A cell that errored has no metric in any seed and keeps its first seed's row, error
    # message included, rather than a row of NaNs that reads like a silent zero.
    ranked = df.sort_values([metric, "_seed"], na_position="last", kind="stable")
    by_key = ranked.groupby(keys, dropna=False, sort=False)
    middle = ((by_key[metric].transform("count") - 1) // 2).clip(lower=0)
    out = ranked[by_key.cumcount() == middle].drop(columns="_seed")

    # Preserve the row order of the first seed's file: the tables are read by humans top to bottom.
    order = parts[0][keys].drop_duplicates()
    out = order.merge(out, on=keys, how="left")
    out.to_csv(HERE / f"{stem}.csv", index=False)

    # The median alone hides how much of it is luck. Measured here: digits `betula-kmeans` ranges
    # over 0.443-0.571 across three seeds, and the single-seed table this replaces happened to
    # publish the top of that range as the result. The spread ships next to the medians so a reader
    # can see which rows are seed-stable and which are not.
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
    if len(args.seeds) % 2 == 0:
        raise SystemExit("an even number of seeds has no middle run; pass an odd number")

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
