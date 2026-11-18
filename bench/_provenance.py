"""Which library versions produced a benchmark row, and the thread count its arm was given.

Kept apart from `_worker.py`, whose import sets thread-count variables: the harnesses that fit in
their own process import this one without that side effect.
"""

from __future__ import annotations

import importlib.metadata
import os

# The thread count given to every arm that takes one as an argument (openTSNE, PhenoGraph, hdbscan,
# OPTICS, DBSCAN, k-prototypes, IsolationForest, LOF). `None` leaves each library at its own
# default, which is what the quality runs use; `bench/timings.py` sets it in the process it times,
# so that every arm runs on the same count. Arms that read the count from the environment (OpenMP,
# rayon, numba) take it from there instead.
JOBS: int | None = None


def jobs(argument: str) -> dict[str, int]:
    """`{argument: JOBS}` for a library's thread-count keyword, or nothing while `JOBS` is unset."""
    return {} if JOBS is None else {argument: JOBS}


def versions(*dists: str) -> str:
    """`name version` of each library an arm runs, `; `-joined, read in the process that ran it.

    The rival harnesses write it into every row's `version` column: a rival release can change its
    labels (River 0.27.0 changed DenStream's, fast-hdbscan 0.3.4 its noise), and a row without a
    version cannot be told stale. Between two releases betula's version names no build, so
    `BETULA_GIT_REV`, when set, is appended to it.
    """
    rev = os.environ.get("BETULA_GIT_REV", "")
    out = []
    for dist in dists:
        v = importlib.metadata.version(dist)
        out.append(f"{dist} {v}+{rev}" if dist == "betula-cluster" and rev else f"{dist} {v}")
    return "; ".join(out)
