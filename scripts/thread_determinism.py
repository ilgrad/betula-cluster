"""The thread count must not enter the answer — over every head, not just the one that broke.

`src/clustering/nmf.rs` once summed `WᵀX` with a rayon work-stealing `fold`/`reduce`. Floating-point
addition does not associate, so the result was a function of how the threads interleaved: twenty
runs at ``RAYON_NUM_THREADS=8`` produced twenty different ``components_``, and none matched the
single-threaded run. It was invisible at one thread, and `tests/test_python.py` now pins that one
head with a subprocess test.

The promise, though, is library-wide: any future parallel float reduction reintroduces the defect
silently, in whichever head it lands in. This script is the library-wide form — every head, fitted
in a child process at several pool sizes, compared by digest. It lives outside the pytest suite on
purpose: it costs one process per (head, pool size), which is minutes, and the suite is run far
more often than a parallel reduction is added.

    uv run --no-sync python scripts/thread_determinism.py            # every head
    uv run --no-sync python scripts/thread_determinism.py -m gmm     # one configuration

Exit status is 1 if any head's digest depends on the pool size.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

# Pool sizes to compare. The 8 appears twice deliberately: a work-stealing reduction is not merely
# different from the single-threaded answer, it is different from *itself* run again, and the repeat
# is what catches the second kind without needing the first to disagree.
POOLS = ("1", "8", "8")

# Every float-valued fitted attribute, hashed when the head sets it. Listed rather than discovered
# so a new attribute is a deliberate addition to what this promises, not a silent widening.
FITTED = (
    "microcluster_centers_",
    "microcluster_radii_",
    "microcluster_weights_",
    "microcluster_proba_",
    "cluster_centers_",
    "cluster_radii_",
    "cluster_sizes_",
    "components_",
    "threshold_",
    "reconstruction_err_",
)

# Every head `_kind_of` accepts, with what its fixture needs. `normalize` puts the rows on the unit
# sphere for the directional heads; `k` is None for the heads that discover their own count.
HEADS: dict[str, dict[str, object]] = {
    "kmeans": {},
    "xmeans": {"k": None},
    "gmm": {},
    "gmm-full": {},
    "gmm-toeplitz": {},
    "gmm-toeplitz-full": {},
    "gmm-toeplitz-gs": {},
    "mppca": {},
    "mfa": {},
    "ward": {},
    "average": {},
    "weighted": {},
    "centroid": {},
    "median": {},
    "spectral": {},
    "leiden": {"k": None},
    "leiden-cpm": {"k": None},
    "kmedoids": {},
    "fuzzy-cmeans": {},
    "spherical-kmeans": {"normalize": True},
    "vmf": {"normalize": True},
    "watson": {"normalize": True},
    "hyperbolic": {},
    "dc-center": {},
    "dc-median": {},
    "hdbscan": {"k": None},
    "scale-space": {"k": None},
}

# The projections are where the one historical defect actually lived, and no `method` reaches them:
# `WᵀX` runs only when `projection` is set. A head alone would have let `nmf.rs` through, so each
# projection gets its own configuration. NMF needs a non-negative matrix, hence `fixture`.
CONFIGS: dict[str, dict[str, object]] = {
    **{name: {**spec, "method": name} for name, spec in HEADS.items()},
    "kmeans+weighted-nmf": {
        "method": "kmeans",
        "projection": "weighted-nmf",
        "projection_dim": 8,
        "fixture": "nonneg",
    },
    "kmeans+weighted-nmf-kl": {
        "method": "kmeans",
        "projection": "weighted-nmf-kl",
        "projection_dim": 8,
        "fixture": "nonneg",
    },
    "kmeans+svd": {"method": "kmeans", "projection": "svd", "projection_dim": 8},
}


def digest(config: str) -> str:
    """Fit `config` on its fixture and hash everything the fit is allowed to depend on."""
    import betula_cluster
    import numpy as np

    spec = dict(CONFIGS[config])
    fixture, discovers_k = spec.pop("fixture", None), "k" in spec
    spec.pop("k", None)
    spec.pop("normalize", None)

    # Overlapping clusters of unequal mass and spread, not five separated balls. On separable data
    # every head returns the same partition -- nine of these configurations produced one identical
    # digest -- and then the answer is a property of the fixture rather than of the head, so a
    # reduction inside the head's own arithmetic has nothing to move. Here `average` chains,
    # `scale-space` finds two modes where `ward` finds five, and `kmedoids` disagrees with both.
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=2.2, size=(5, 12))
    masses, spreads = (1400, 1000, 700, 500, 400), (0.6, 1.0, 1.4, 0.8, 1.8)
    x = np.vstack(
        [
            c + rng.normal(scale=s, size=(m, 12))
            for c, m, s in zip(centres, masses, spreads, strict=True)
        ]
    )
    if CONFIGS[config].get("normalize"):
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    elif fixture == "nonneg":
        x = np.abs(x)

    kw: dict[str, object] = {**spec, "max_leaves": 300, "seed": 0}
    if not discovers_k:
        kw["n_clusters"] = 5
    est = betula_cluster.Betula(**kw)  # type: ignore[arg-type]
    labels = est.fit_predict(x)

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    # Labels are a coarse view: a reduction can move a centre by an ulp and still land every point
    # in the same cluster. Hash the fitted parameters the head exposes as well, at full precision.
    for name in FITTED:
        try:
            # A head that does not produce the attribute raises rather than returning None, so the
            # "does this head have one?" question is asked the only way the API answers it.
            arr = getattr(est, name)
        except (AttributeError, ValueError):
            continue
        h.update(name.encode())
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def check(configs: list[str]) -> int:
    failures = []
    for method in configs:
        seen: dict[str, str] = {}
        for run, pool in enumerate(POOLS):
            done = subprocess.run(
                [sys.executable, __file__, "--digest", method],
                env=dict(os.environ, RAYON_NUM_THREADS=pool),
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if done.returncode != 0:
                print(f"{method:<24} ERROR at RAYON_NUM_THREADS={pool}: {done.stderr[-300:]}")
                failures.append(method)
                break
            seen[f"run{run}-threads{pool}"] = done.stdout.strip()
        else:
            unique = set(seen.values())
            status = "ok" if len(unique) == 1 else "THREAD-DEPENDENT"
            print(f"{method:<24} {status:<17} {next(iter(unique))[:16]}")
            if len(unique) != 1:
                print(f"  {json.dumps(seen, indent=2)}")
                failures.append(method)

    if failures:
        print(f"\n{len(failures)} head(s) whose answer depends on the pool size: {failures}")
        return 1
    print(f"\n{len(configs)} configurations, all pool-size independent.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--digest", metavar="CONFIG", help="child mode: print one digest and exit")
    ap.add_argument("-m", "--method", action="append", help="restrict to these configurations")
    args = ap.parse_args()

    if args.digest:
        print(digest(args.digest))
        return 0
    return check(args.method or list(CONFIGS))


if __name__ == "__main__":
    sys.exit(main())
