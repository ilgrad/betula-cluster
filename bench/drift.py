"""Drift detection on the streaming heads: does the alarm land in the change window, and how often
does it fire when nothing changed?

`DenStream` and `DbStream` both fade old data at `2^(-lambda*dt)`. That is a *schedule*: it forgets
at a fixed rate whether or not anything changed, and a wrong lambda is silent in both directions —
too small and the model never forgets, too large and it forgets structure that was never stale. The
ADWIN detector (Bifet & Gavalda, SDM 2007) answers the other question, "did the stream change", from
the data, at a stated false-positive ceiling of delta = 0.002. It reports; it does not act.

The statistic it watches is the distance from each incoming point to the nearest micro-cluster, in
units of the micro-cluster radius. Two arms, both needed:

    drift        stationary prefix, then one of four changes -> the alarm must land in the window
    stationary   nothing ever changes                        -> alarms here are the false positives

The false-positive arm is measured *after a warm-up*, and that is not a thumb on the scale: while
the model is still opening its first micro-clusters the mean routing distance genuinely falls, so an
alarm there is the statistic changing, not the detector being wrong. `warmup` is reported per row so
the discount is visible.

The four changes are not interchangeable. `jump` (the whole cloud translates) is the easy case and
the one every drift paper shows; `spread` (same centres, variance up) and `split` (one cluster
becomes two) move the routing distance far less; `gradual` ramps the centres over 1000 points rather
than switching, which is the case an abrupt-change detector is least suited to and the one worth
publishing a negative for if it fails.

    uv run --no-sync --with pandas python bench/drift.py

Writes `bench/results_drift.csv` (one row per head x scenario x seed) and prints a table.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent

SEEDS = (0, 1, 2)
DIM = 2
WARMUP = 2000  # points before the false-positive count starts, and before any change
AFTER = 2000  # points streamed after the change
CHECK_CLOCK = 32  # ADWIN tests on a clock, so an alarm can be this late and no later
K_TRUE = 3
SEP = 8.0  # centre separation of the stationary mixture, in sigma

# Radius and decay per head. Chosen so the stationary model is settled — a lambda fast enough to
# prune micro-clusters as fast as they form has no baseline to depart from, and the detector
# correctly reports nothing (pinned in the Rust suite, not re-litigated here).
HEADS = {
    "denstream": {"eps": 1.0, "decay": 0.001, "beta": 0.5, "mu": 4.0},
    "dbstream": {"r": 1.0, "decay": 0.001, "alpha": 0.5, "min_weight": 2.0},
}


def centres(rng: np.random.Generator) -> np.ndarray:
    """`K_TRUE` well-separated centres on a circle of radius `SEP`."""
    phase = rng.uniform(0.0, 2.0 * np.pi)
    ang = phase + np.arange(K_TRUE) * 2.0 * np.pi / K_TRUE
    return SEP * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def mixture(rng: np.random.Generator, mu: np.ndarray, n: int, scale: float = 1.0) -> np.ndarray:
    """`n` points from an equal-weight isotropic Gaussian mixture at centres `mu`."""
    which = rng.integers(len(mu), size=n)
    return mu[which] + scale * rng.normal(size=(n, DIM))


def scenario(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """`(before, after)` — the stationary prefix and what follows it."""
    rng = np.random.default_rng(seed)
    mu = centres(rng)
    before = mixture(rng, mu, WARMUP)
    if name == "stationary":
        after = mixture(rng, mu, AFTER)
    elif name == "jump":
        after = mixture(rng, mu + 50.0, AFTER)
    elif name == "spread":
        after = mixture(rng, mu, AFTER, scale=4.0)
    elif name == "split":
        # Each cluster becomes two, `SEP` apart along x — the same total mass, new centres.
        offset = np.array([SEP / 2, 0.0])
        split = np.vstack([mu + offset, mu - offset])
        after = mixture(rng, split, AFTER)
    elif name == "gradual":
        # Centres ramp to +50 over the first 1000 points, then hold.
        ramp = np.clip(np.arange(AFTER) / 1000.0, 0.0, 1.0)[:, None]
        after = mixture(rng, mu, AFTER) + 50.0 * ramp
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return before, after


SCENARIOS = ("stationary", "jump", "spread", "split", "gradual")


def run(head: str, name: str, seed: int) -> dict[str, object]:
    """Stream the prefix, read the detector, then stream the rest one check clock at a time.

    Chunked rather than in one call because `last_alarm` is the *most recent* report, and on a
    change that keeps changing (`gradual`, `spread`) the detector correctly keeps firing — reading
    it once at the end would report the last of those as the latency. One clock per chunk is the
    detector's own resolution, so nothing is lost by not going finer.
    """
    import betula_cluster as bc

    est = (bc.DenStream if head == "denstream" else bc.DbStream)(**HEADS[head])
    before, after = scenario(name, seed)
    est.partial_fit(before)
    warm = est.drift_
    seen, raised, latency = 0, 0, None
    peak = float(warm["distance"])
    for chunk in np.array_split(after, len(after) // CHECK_CLOCK):
        est.partial_fit(chunk)
        seen += len(chunk)
        now = est.drift_
        fresh = int(now["alarms"]) - int(warm["alarms"]) - raised
        if fresh and latency is None:
            latency = seen
        raised += fresh
        peak = max(peak, float(now["distance"]))
    end = est.drift_
    return {
        "head": head,
        "scenario": name,
        "seed": seed,
        "warmup_alarms": int(warm["alarms"]),
        "alarms": raised,
        "latency": latency,
        "distance_before": float(warm["distance"]),
        "distance_peak": peak,
        "distance_after": float(end["distance"]),
        "window_before": int(warm["window"]),
        "window_after": int(end["window"]),
    }


def main() -> int:
    import pandas as pd

    rows = [run(head, name, seed) for head in HEADS for name in SCENARIOS for seed in SEEDS]
    df = pd.DataFrame(rows)
    out = HERE / "results_drift.csv"
    df.to_csv(out, index=False)

    print(
        f"{len(SEEDS)} seeds, {WARMUP} warm-up + {AFTER} points, delta = 0.002, "
        f"check clock {CHECK_CLOCK}\n"
    )
    print(
        f"{'head':<10} {'scenario':<11} {'fired':>7} {'alarms':>7} {'first alarm':>13} "
        f"{'distance: before  peak  after':>32}"
    )
    for (head, name), g in df.groupby(["head", "scenario"], sort=False):
        lat = g["latency"].dropna()
        lat_s = f"{lat.median():.0f} pts" if len(lat) else "never"
        print(
            f"{head:<10} {name:<11} {len(lat)}/{len(g):<5} {g['alarms'].sum():>7} {lat_s:>13} "
            f"{g['distance_before'].median():>16.2f} {g['distance_peak'].median():>6.2f} "
            f"{g['distance_after'].median():>6.2f}"
        )

    fp = df[df.scenario == "stationary"]
    rate = fp["alarms"].sum() / (len(fp) * AFTER)
    print(
        f"\nfalse positives after warm-up: {fp['alarms'].sum()} in {len(fp) * AFTER} points "
        f"({rate:.5f}); delta allows {0.002:.5f}"
    )
    print(f"wrote {out.name} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
