"""CF-tree *build* benchmark: betula-cluster (Phase-1 only) vs betulars (the reference builder).

Both libraries bulk-build a CF-tree from a 2D float64 array entirely in Rust, with matched
parameters (capacity 32, maxleaves 1000, threshold 0, VII / "spherical"). We time only the build:
betulars in its constructor, betula-cluster via ``partial_fit`` (Phase-1 insertion, no clustering).

Run: ``VIRTUAL_ENV=.venv uv run --no-project --with numpy --with scikit-learn \
        python bench/build_vs_betulars.py``
(``betulars`` and the editable ``betula_cluster`` must already be in the venv.)
"""

from __future__ import annotations

import gc
import time

import betulars
import numpy as np
from betula_cluster import Betula
from sklearn.datasets import make_blobs

SIZES = (50_000, 100_000, 200_000, 500_000, 1_000_000)
DIM = 10
K = 10
REPS = 9
CAPACITY = 32
MAXLEAVES = 1000


def _best(fn, reps: int) -> float:
    """Minimum wall-clock over ``reps`` runs (least contaminated by GC / scheduler jitter)."""
    best = float("inf")
    gc.disable()
    try:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
    finally:
        gc.enable()
    return best


def _betulars_build(x: np.ndarray) -> int:
    model = betulars.Betula(x, capacity=CAPACITY, maxleaves=MAXLEAVES, threshold=0.0, feature="vii")
    return model.num_clusters


def _betula_build(x: np.ndarray) -> int:
    est = Betula(
        feature="spherical",
        branching=CAPACITY,
        leaf_cap=CAPACITY,
        max_leaves=MAXLEAVES,
        threshold=0.0,
    )
    est.partial_fit(x)
    return est.n_leaves_


def main() -> None:
    print(
        f"build vs betulars  (d={DIM}, k={K}, capacity={CAPACITY}, maxleaves={MAXLEAVES}, "
        f"threshold=0, best of {REPS})\n"
    )
    header = f"{'N':>10} | {'betulars':>9} | {'betula':>9} | {'ratio':>6} | leaves (blrs/btl)"
    print(header)
    print("-" * len(header))
    for n in SIZES:
        x, _ = make_blobs(n_samples=n, n_features=DIM, centers=K, random_state=0)
        x = np.ascontiguousarray(x, dtype=np.float64)
        leaves_ref = _betulars_build(x)
        leaves_btl = _betula_build(x)
        t_ref = _best(lambda x=x: _betulars_build(x), REPS)
        t_btl = _best(lambda x=x: _betula_build(x), REPS)
        ratio = t_btl / t_ref
        flag = "  <-- betula wins" if ratio < 1.0 else ""
        print(
            f"{n:>10} | {t_ref:>9.4f} | {t_btl:>9.4f} | {ratio:>5.2f}x | "
            f"{leaves_ref:>5}/{leaves_btl:<5}{flag}"
        )


if __name__ == "__main__":
    main()
