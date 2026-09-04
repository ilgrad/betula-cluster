"""Training workload for a profile-guided build of the extension.

PGO optimises for the branches a profile actually saw, so what this script covers is what the wheel
gets tuned for. It covers the insert path at both ends of the `d` range and every Phase-3 head,
because those are different code: insertion is branch- and pointer-bound, the heads are `O(m²)`
scans over leaf features, and a profile of only one of them would pessimise the other.

Measured on this repo (AMD Zen 3, 3 alternating repetitions, median, `RAYON_NUM_THREADS=1`):
insertion at `1 000 000 × 20` runs **0.948 s → 0.841 s (1.13×)** with the profile, and at
`200 000 × 784` **2.643 s → 2.747 s** — a wash to slightly slower, the two ranges overlapping. The
win is in the low-dimensional regime where the time is branches and pointer chasing rather than
kernel arithmetic, which is also the regime the streaming use case lives in.

    RUSTFLAGS="-C llvm-args=--inline-threshold=1000 -Cprofile-generate=/tmp/pgo-data" \
        maturin develop --release
    python bench/pgo_train.py
    llvm-profdata merge -o /tmp/pgo.profdata /tmp/pgo-data
    RUSTFLAGS="-C llvm-args=--inline-threshold=1000 -Cprofile-use=/tmp/pgo.profdata" \
        maturin develop --release

`llvm-profdata` is the one from `rustup component add llvm-tools-preview`, under
`~/.rustup/toolchains/<toolchain>/lib/rustlib/<host>/bin/` — a system `llvm-profdata` of a different
major version will refuse the raw profiles.
"""

from __future__ import annotations

import betula_cluster as bc
import numpy as np

# (n, d, max_leaves) — small enough that the whole run is a couple of minutes, wide enough in `d`
# that both the SIMD kernel and the scalar tail are represented.
SHAPES = [(400_000, 20, 4_000), (200_000, 128, 4_000), (60_000, 784, 2_000)]
HEADS = ["kmeans", "gmm", "ward", "hdbscan"]


def blobs(n: int, d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(0.0, 3.0, size=(k, d))
    out = np.empty((n, d), dtype=np.float64)
    for lo in range(0, n, 20_000):
        hi = min(n, lo + 20_000)
        out[lo:hi] = centres[rng.integers(0, k, size=hi - lo)]
        out[lo:hi] += rng.normal(0.0, 1.0, size=(hi - lo, d))
    return out


def main() -> None:
    for n, d, max_leaves in SHAPES:
        x = blobs(n, d, 10, seed=0)
        for method in HEADS:
            labels = bc.fit_predict(
                x, 10, method=method, threshold=0.0, max_leaves=max_leaves, seed=0, n_jobs=1
            )
            print(f"{method:<8} n={n:<8} d={d:<4} clusters={len(set(labels.tolist()))}")
        # The streaming path takes different branches from the one-shot one: chunked absorption,
        # repeated rebuilds, and a finalize that reads the tree rather than the rows.
        est = bc.Betula(n_clusters=10, threshold=0.0, max_leaves=max_leaves, seed=0)
        for lo in range(0, n, 50_000):
            est.partial_fit(x[lo : lo + 50_000])
        est.partial_fit()
        est.predict(x[:10_000])
        print(f"stream   n={n:<8} d={d:<4} leaves={est.n_leaves_} rebuilds={est.n_rebuilds_}")


if __name__ == "__main__":
    main()
