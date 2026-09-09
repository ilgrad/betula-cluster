"""The streaming heads against River, the reference implementation of the same three algorithms.

`bench/drift.py` measures whether the drift alarm lands in the change window; nothing measured
whether the *clustering* underneath it is any good, because there was no reference to measure it
against. River (`river-ml/river`) ships DenStream, DBSTREAM and CluStream from the same papers, so
the comparison is implementation-against-implementation at matched parameters, not a comparison of
algorithms.

River is **not** a project dependency and may not become one — it is pulled per invocation:

    uv run --no-sync --with river --with pandas python bench/stream_reference.py

Two numbers per chunk of a 4000-point stream: the ARI of the labels the model gives that chunk's own
points, and the wall time to learn it. The stream is stationary for 2000 points and then changes, so
the ARI curve shows both the settled quality and the recovery.

**What this cannot measure, and the row that is a loss by construction.** CluStream's contribution is
the pyramidal time frame: snapshots that let a caller ask, after the fact, for the clustering of an
arbitrary past window. Nothing in this library does that — a `partial_fit` tree carries one summary
at the current time and cannot be rewound — so the retrospective-query row is not a number we lose,
it is a capability we do not have. It is written down here rather than left out of the table.

Writes `bench/results_stream.csv`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from drift import AFTER, DIM, K_TRUE, SEP, WARMUP, centres

SEEDS = (0, 1, 2)
CHUNK = 250
SCENARIOS = ("stationary", "jump", "split")

# One radius and one decay for every head that takes them, so a difference in the table is the
# implementation and not the tuning. `drift.py`'s settled values.
EPS, DECAY = 1.0, 0.001


def labelled_stream(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The `drift.py` fixture, carrying the component each point was drawn from."""
    rng = np.random.default_rng(seed)
    mu = centres(rng)

    def draw(centres_now: np.ndarray, n: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        which = rng.integers(len(centres_now), size=n)
        return centres_now[which] + scale * rng.normal(size=(n, DIM)), which

    x0, y0 = draw(mu, WARMUP)
    if name == "stationary":
        x1, y1 = draw(mu, AFTER)
    elif name == "jump":
        x1, y1 = draw(mu + 50.0, AFTER)
    elif name == "split":
        offset = np.array([SEP / 2, 0.0])
        # Each component becomes two, so the truth after the change has 2*K_TRUE labels.
        x1, y1 = draw(np.vstack([mu + offset, mu - offset]), AFTER)
    else:
        raise ValueError(f"unknown scenario {name!r}")
    return np.vstack([x0, x1]), np.concatenate([y0, y1 + K_TRUE])


def run_betula(
    head: str, x: np.ndarray, eps: float = EPS
) -> tuple[list[float], list[float], list[int]]:
    import betula_cluster as bc

    if head == "denstream":
        model = bc.DenStream(eps=eps, decay=DECAY, beta=0.5, mu=4.0)
    elif head == "dbstream":
        model = bc.DbStream(r=eps, decay=DECAY, alpha=0.5, min_weight=2.0)
    else:
        raise ValueError(head)
    preds, times, counts = [], [], []
    for chunk in np.array_split(x, len(x) // CHUNK):
        t0 = time.perf_counter()
        model.partial_fit(chunk)
        times.append(time.perf_counter() - t0)
        labels = np.asarray(model.predict(chunk))
        preds.append(labels)
        counts.append(int(model.n_clusters_))
    return preds, times, counts


def run_river(
    head: str, x: np.ndarray, eps: float = EPS
) -> tuple[list[float], list[float], list[int]]:
    from river import cluster

    if head == "denstream":
        model = cluster.DenStream(
            epsilon=eps, decaying_factor=DECAY, beta=0.5, mu=4.0, n_samples_init=CHUNK
        )
    elif head == "dbstream":
        model = cluster.DBSTREAM(
            clustering_threshold=eps,
            fading_factor=DECAY,
            intersection_factor=0.5,
            minimum_weight=2.0,
        )
    elif head == "clustream":
        # No matched radius: CluStream is bounded by a micro-cluster *count*, not by a radius, and
        # asks for the macro count up front. K_TRUE is the friendliest setting it can be given.
        model = cluster.CluStream(n_macro_clusters=K_TRUE, seed=0)
    else:
        raise ValueError(head)
    preds, times, counts = [], [], []
    for chunk in np.array_split(x, len(x) // CHUNK):
        rows = [{i: float(v) for i, v in enumerate(row)} for row in chunk]
        t0 = time.perf_counter()
        for row in rows:
            model.learn_one(row)
        times.append(time.perf_counter() - t0)
        labels = np.array([model.predict_one(row) for row in rows])
        preds.append(labels)
        counts.append(len(np.unique(labels)))
    return preds, times, counts


METHODS = {
    "betula DenStream": lambda x: run_betula("denstream", x),
    "river DenStream": lambda x: run_river("denstream", x),
    "betula DbStream": lambda x: run_betula("dbstream", x),
    "river DBSTREAM": lambda x: run_river("dbstream", x),
    "river CluStream": lambda x: run_river("clustream", x),
}


#: Radii the sweep tries, so "matched parameters" can be told apart from "a better implementation".
RADII = (0.5, 1.0, 2.0, 4.0, 8.0)


def sweep() -> dict[str, float]:
    """Each density head at its own best radius, not only at the shared one.

    A radius means what its own paper's implementation says it means, and the two libraries need
    not agree. If River fragments the stationary fixture into 16 clusters at `eps=1.0`, the useful
    question is what it does at the radius that suits it — otherwise the main table is measuring a
    units mismatch and calling it quality.
    """
    from sklearn.metrics import adjusted_rand_score as ari

    x, y = labelled_stream("stationary", 0)
    settled = slice(WARMUP, None)
    runners = {
        "betula DenStream": lambda r: run_betula("denstream", x, eps=r),
        "river DenStream": lambda r: run_river("denstream", x, eps=r),
        "betula DbStream": lambda r: run_betula("dbstream", x, eps=r),
        "river DBSTREAM": lambda r: run_river("dbstream", x, eps=r),
    }
    print(f"\n== radius sweep, stationary, seed 0 (ARI / clusters over the last {AFTER} points)")
    print("  " + f"{'method':<20}" + "".join(f"{r:>16}" for r in RADII))
    best: dict[str, float] = {}
    for name, fn in runners.items():
        cells, scores = [], []
        for r in RADII:
            preds, _times, counts = fn(r)
            labels = np.concatenate(preds)[settled]
            scores.append(ari(y[settled], labels))
            cells.append(f"{scores[-1]:.3f} / {int(np.median(counts))}")
        best[name] = RADII[int(np.argmax(scores))]
        print(f"  {name:<20}" + "".join(f"{c:>16}" for c in cells))
    print("  best radius: " + ", ".join(f"{n}={r}" for n, r in best.items()))
    return best


def tuned(best: dict[str, float]) -> None:
    """What the change costs each head *at its own best radius*, so the recovery rows cannot be
    dismissed as an artefact of the shared one."""
    from sklearn.metrics import adjusted_rand_score as ari

    runners = {
        "betula DenStream": lambda x, r: run_betula("denstream", x, eps=r),
        "river DenStream": lambda x, r: run_river("denstream", x, eps=r),
        "betula DbStream": lambda x, r: run_betula("dbstream", x, eps=r),
        "river DBSTREAM": lambda x, r: run_river("dbstream", x, eps=r),
        "river CluStream": lambda x, _r: run_river("clustream", x),
    }
    print("\n== at each head's own best radius, median of seeds " + str(list(SEEDS)))
    print(f"  {'method':<20}{'radius':>8}" + "".join(f"{s:>22}" for s in SCENARIOS))
    for name, fn in runners.items():
        radius = best.get(name, float("nan"))
        cells = []
        for scenario in SCENARIOS:
            before, after = [], []
            for seed in SEEDS:
                x, y = labelled_stream(scenario, seed)
                preds, _times, _counts = fn(x, radius)
                labels = np.concatenate(preds)
                before.append(ari(y[:WARMUP], labels[:WARMUP]))
                after.append(ari(y[WARMUP:], labels[WARMUP:]))
            cells.append(f"{np.median(before):.3f} -> {np.median(after):.3f}")
        print(f"  {name:<20}{radius:>8.1f}" + "".join(f"{c:>22}" for c in cells))


def main() -> None:
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score as ari

    rows = []
    settled = WARMUP // CHUNK  # the last chunk before the change
    for scenario in SCENARIOS:
        for seed in SEEDS:
            x, y = labelled_stream(scenario, seed)
            chunks = np.array_split(np.arange(len(x)), len(x) // CHUNK)
            for name, fn in METHODS.items():
                preds, times, counts = fn(x)
                for i, (idx, lab, t, k) in enumerate(
                    zip(chunks, preds, times, counts, strict=True)
                ):
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "method": name,
                            "chunk": i,
                            "phase": "before" if i < settled else "after",
                            "ari": ari(y[idx], lab),
                            "learn_s": t,
                            "us_per_point": 1e6 * t / len(idx),
                            "clusters": k,
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results_stream.csv", index=False)

    for scenario in SCENARIOS:
        block = df[df.scenario == scenario]
        print(f"\n== {scenario}  (chunks of {CHUNK}, {WARMUP} points before the change)")
        print(
            f"  {'method':<20}{'ARI before':>12}{'ARI after':>11}{'clusters':>10}{'us/point':>10}"
        )
        for name in METHODS:
            m = block[block.method == name]
            before, after = m[m.phase == "before"], m[m.phase == "after"]
            print(
                f"  {name:<20}{before.ari.median():>12.4f}{after.ari.median():>11.4f}"
                f"{m.clusters.median():>10.0f}{m.us_per_point.median():>10.1f}"
            )


if __name__ == "__main__":
    main()
    tuned(sweep())
