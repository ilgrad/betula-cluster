"""Past-window queries on `WindowStream`: is the clustering of a window the stream has moved past
any good, and how much mass from outside the window does the answer carry?

The fixture is `stream_reference.py`'s: 4000 2-D points, stationary for 2000 and then changed
(`stationary`, a `jump` of +50, or a `split` of each cluster into two). Timestamps are the row
index, so a frame of width 10 holds ten points; 400 frames pass through an index of `capacity`
frames, so every window below is answered from merged frames. After the whole stream is in, three
windows are asked for:

    before   [0, 1999]     the settled prefix, half a stream back
    after    [2000, 3999]  everything since the change
    recent   [3500, 3999]  the last 500 points

For each, `cluster_window(t0, t1, k)` with `k` the number of components among the window's own
rows; each of those rows takes the nearest returned centre, and the ARI is against the component it
was drawn from. The oracle is k-means on exactly the window's rows. `excess` is the weight
`window_moments` returns over the window's true row count, minus one: the mass of frames that reach
into the window from outside it.

The frame-merge rule changed in 1.5.0 (the oldest frame used to absorb every overflow). The same
script measures 1.4.0 from PyPI, which is not a dependency and is pulled per invocation:

    uv run --no-sync --with scikit-learn python bench/window_queries.py
    uv run --no-project --with betula-cluster==1.4.0 --with scikit-learn \\
        python bench/window_queries.py --tag 1.4.0

Writes `bench/results_window.csv` (appending the rows of a tagged run).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from stream_reference import CHUNK, SCENARIOS, SEEDS, labelled_stream

FRAME_WIDTH = 10.0
CAPACITIES = (16, 64)
WINDOWS = {"before": (0, 1999), "after": (2000, 3999), "recent": (3500, 3999)}


def window_rows(ws, x: np.ndarray, y: np.ndarray, t0: int, t1: int, seed: int) -> dict[str, float]:
    import betula_cluster as bc
    from sklearn.metrics import adjusted_rand_score as ari

    rows, truth = x[t0 : t1 + 1], y[t0 : t1 + 1]
    k = len(np.unique(truth))
    got = ws.cluster_window(float(t0), float(t1), k)
    if got is None:
        stream_ari = float("nan")
    else:
        centers = got[0]
        nearest = ((rows[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        stream_ari = ari(truth, nearest)
    weight = ws.window_moments(float(t0), float(t1))["weight"]
    oracle = bc.Betula(n_clusters=k, method="kmeans", seed=seed).fit_predict(rows)
    return {
        "k": k,
        "ari": stream_ari,
        "oracle_ari": ari(truth, oracle),
        "excess": weight / len(rows) - 1.0,
    }


def main() -> None:
    import betula_cluster as bc

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="current")
    args = parser.parse_args()
    out = HERE / "results_window.csv"
    fields = [
        "version",
        "scenario",
        "seed",
        "capacity",
        "window",
        "k",
        "ari",
        "oracle_ari",
        "excess",
    ]
    rows = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            x, y = labelled_stream(scenario, seed)
            t = np.arange(len(x), dtype=np.float64)
            for capacity in CAPACITIES:
                ws = bc.WindowStream(frame_width=FRAME_WIDTH, capacity=capacity, seed=seed)
                for lo in range(0, len(x), CHUNK):
                    ws.partial_fit(x[lo : lo + CHUNK], t[lo : lo + CHUNK])
                ws.close_frame()
                for name, (t0, t1) in WINDOWS.items():
                    rows.append(
                        {
                            "version": args.tag,
                            "scenario": scenario,
                            "seed": seed,
                            "capacity": capacity,
                            "window": name,
                            **window_rows(ws, x, y, t0, t1, seed),
                        }
                    )
    keep = []
    if args.tag != "current" and out.exists():
        with out.open() as f:
            keep = [r for r in csv.DictReader(f) if r["version"] != args.tag]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep + rows)
    print(f"betula-cluster {bc.__version__}  (tag {args.tag})")
    print(
        f"{'scenario':<11}{'capacity':>9}{'window':>8}{'ARI med':>9}{'oracle':>8}{'excess med':>12}"
    )
    for scenario in SCENARIOS:
        for capacity in CAPACITIES:
            for name in WINDOWS:
                cell = [
                    r
                    for r in rows
                    if r["scenario"] == scenario
                    and r["capacity"] == capacity
                    and r["window"] == name
                ]
                print(
                    f"{scenario:<11}{capacity:>9}{name:>8}"
                    f"{np.median([r['ari'] for r in cell]):>9.3f}"
                    f"{np.median([r['oracle_ari'] for r in cell]):>8.3f}"
                    f"{np.median([r['excess'] for r in cell]):>12.3f}"
                )


if __name__ == "__main__":
    main()
