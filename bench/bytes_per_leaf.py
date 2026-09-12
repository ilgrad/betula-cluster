"""Measure the marginal resident bytes per CF-tree leaf, against `_bytes_per_leaf`.

`python/betula_cluster/__init__.py:_bytes_per_leaf` is the whole of what `memory_budget_mb` means:
the budget is divided by it to get `max_leaves`. Its constants come from this harness, so this is
where they are re-derived when the tree layout or a cluster feature changes. Until 1.1 the formula
counted only the declared arrays and had never been checked against a process; it under-predicted
every cell from 1.15x to 19.5x.

Method: one point per subprocess, several `max_leaves` per (feature, dim), and the **slope** of RSS
against the *realised* leaf count. An absolute RSS carries the interpreter, the input array, numpy's
arenas and rayon's stacks; all of those are constant across a sweep at fixed (feature, dim, n), so
they cancel in the slope and only the tree is left. That is also exactly the quantity the model
claims, since it is used as a divisor of a budget.

`RAYON_NUM_THREADS=1` throughout: a thread pool's per-thread allocator arenas are constant across
the sweep but large, and one thread keeps the residual noise well under the effect.

    RAYON_NUM_THREADS=1 uv run --no-sync python bench/bytes_per_leaf.py sweep
    RAYON_NUM_THREADS=1 uv run --no-sync python bench/bytes_per_leaf.py point --feature full ...
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys

PAGE = os.sysconf("SC_PAGE_SIZE")

# (feature, dim) -> (n_rows, [max_leaves ...]). `full` stores a packed d×(d+1)/2 scatter per leaf,
# so its grid is scaled down to keep the biggest cell near 1 GB rather than 10.
GRID: dict[tuple[str, int], tuple[int, list[int]]] = {
    **{
        (f, d): (n, [500, 1000, 2000, 4000])
        for f in ("spherical", "diagonal", "fd")
        for d, n in ((2, 200_000), (20, 200_000), (54, 200_000), (784, 50_000))
    },
    ("full", 2): (200_000, [500, 1000, 2000, 4000]),
    ("full", 20): (200_000, [500, 1000, 2000, 4000]),
    ("full", 54): (200_000, [250, 500, 1000, 2000]),
    ("full", 784): (50_000, [50, 100, 200, 400]),
}


def rss_bytes() -> int:
    """Resident set size now, from `/proc/self/statm` (field 2 is resident pages)."""
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * PAGE


def point(feature: str, dim: int, n: int, max_leaves: int) -> dict[str, object]:
    import numpy as np
    from betula_cluster import Betula

    rng = np.random.default_rng(0)
    # Uniform rows, not blobs: the tree must actually reach `max_leaves` for the slope to be the
    # slope of a full tree rather than of whatever structure the data happens to have.
    x = rng.random((n, dim))
    gc.collect()
    before = rss_bytes()

    # `method="kmeans"` for every cell, not the default: the default `gmm` *refuses*
    # `feature="spherical"` (a scalar scatter distorts its log-determinant), which silently dropped a
    # quarter of the grid on the first run. A centroid head accepts all four features, so one head
    # across the whole sweep also keeps the head's own allocations out of the comparison.
    est = Betula(n_clusters=2, method="kmeans", feature=feature, max_leaves=max_leaves, seed=0).fit(
        x
    )
    gc.collect()
    after = rss_bytes()

    return {
        "feature": feature,
        "dim": dim,
        "n": n,
        "max_leaves": max_leaves,
        "n_leaves": int(est.n_leaves_),
        "rss_before": before,
        "rss_after": after,
        "delta": after - before,
    }


def run_sweep(only: str | None) -> None:
    from betula_cluster import _bytes_per_leaf  # type: ignore[attr-defined]

    env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
    rows = []
    for (feature, dim), (n, leaf_grid) in GRID.items():
        if only and only != feature:
            continue
        points = []
        for max_leaves in leaf_grid:
            cmd = [
                sys.executable, __file__, "point",
                "--feature", feature, "--dim", str(dim),
                "--n", str(n), "--max-leaves", str(max_leaves),
            ]  # fmt: skip
            out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
            if out.returncode != 0:
                print(f"FAILED {feature} d={dim} leaves={max_leaves}: {out.stderr[-400:]}")
                break
            rec = json.loads(out.stdout.strip().splitlines()[-1])
            points.append(rec)
            print(
                f"  {feature:<10} d={dim:<4} leaves={rec['n_leaves']:<6} "
                f"rss={rec['rss_after'] / 2**20:8.1f} MiB"
            )
        if len(points) < 2:
            continue
        # Ordinary least squares of RSS on the realised leaf count. The intercept is the fixed cost
        # and is reported only as a sanity check -- it should be near the input array's size.
        xs = [float(p["n_leaves"]) for p in points]
        ys = [float(p["rss_after"]) for p in points]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((v - mx) ** 2 for v in xs)
        slope = sum((v - mx) * (w - my) for v, w in zip(xs, ys, strict=True)) / sxx if sxx else 0.0
        ss_tot = sum((w - my) ** 2 for w in ys)
        ss_res = sum((w - (my + slope * (v - mx))) ** 2 for v, w in zip(xs, ys, strict=True))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        modelled = _bytes_per_leaf(feature, dim)
        rows.append({
            "feature": feature, "dim": dim, "measured": slope, "modelled": modelled,
            "ratio": slope / modelled if modelled else float("nan"), "r2": r2,
            "intercept_mib": (my - slope * mx) / 2**20, "points": points,
        })  # fmt: skip

    print(
        f"\n{'feature':<10} {'dim':>5} {'measured B/leaf':>16} {'modelled':>10} "
        f"{'measured/modelled':>18} {'r2':>7} {'fixed MiB':>10}"
    )
    for r in rows:
        print(
            f"{r['feature']:<10} {r['dim']:>5} {r['measured']:>16,.0f} {r['modelled']:>10,} "
            f"{r['ratio']:>18.2f} {r['r2']:>7.4f} {r['intercept_mib']:>10.1f}"
        )
    with open("bench/results_bytes_per_leaf.json", "w") as fh:
        json.dump(rows, fh, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("point")
    p.add_argument("--feature", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--max-leaves", type=int, required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--only", default=None, help="restrict to one feature")
    args = ap.parse_args()

    if args.cmd == "point":
        print(json.dumps(point(args.feature, args.dim, args.n, args.max_leaves)))
    else:
        run_sweep(args.only)


if __name__ == "__main__":
    main()
