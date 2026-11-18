"""The real-data fixtures, defined once.

A "20 000-row draw from covtype" is not one thing, and the difference is not small: the same `ward`
head reads **0.0861** under one rule and **0.1416** under the other. Both rules are in the published
record and both are defensible, so this module keeps both — under names that say which is which:

`draw_then_scale` — the **sub** rule
    Take `n` rows, *then* fit the scaler on those rows. This is what a practitioner holding `n` rows
    has: there is no population to standardize against. The draw moves with the seed, so a
    median-of-seeds table averages over three subsamples as well as three initializations.

`scale_then_draw` — the **pop** rule
    Fit the scaler on all 581 012 rows, *then* take the permutation. The subsample inherits the
    population's scale, so a 20 000-row cell and the full-covtype cell measure the same geometry;
    the draw is fixed at `rng(0)`, so a study sweeping a knob holds the data still.

Under either rule the covtype rows are *drawn*, never sliced off the front: that file is ordered in
spatial blocks, and class 4 is 0.47 % of the data against 10.8 % of the first 20 000 rows, so a head
slice is a different question wearing the same name. OpenML MNIST is not like that — its first
10 000 rows carry all ten classes within a few per cent of their population shares — which is why
`bench/leaf_refit.py` is allowed to take one.

Every harness in `bench/` loads through here, and that is the whole point of the module. The two
rules were originally spelled out inline in four separate files; one-off scripts then wrote three
more variants of their own, and two tables that cannot be compared ended up in adjacent sections of
`bench/RESULTS.md` under one name. One definition per rule is what stops the next one appearing.

`sklearn` is imported inside the functions on purpose: `bench/_worker.py` measures peak RSS and
imports only what the method under test needs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

REAL_CAP = 20_000  # subsample big real sets so the O(N^2) baselines stay feasible and comparable


def fetch(dataset: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Raw `(X float64, y, k)` for a real dataset — no scaler, no subsample."""
    if dataset == "digits":
        from sklearn.datasets import load_digits

        x, y, k = (*load_digits(return_X_y=True), 10)
    elif dataset == "mnist":
        x, y, k = (*_mnist(), 10)
    elif dataset == "covtype":
        from sklearn.datasets import fetch_covtype

        d = fetch_covtype()
        x, y, k = d.data, d.target.astype(int) - 1, 7
    else:
        raise ValueError(f"unknown real dataset: {dataset}")
    return np.asarray(x, dtype=np.float64), np.asarray(y), k


def _mnist() -> tuple[np.ndarray, np.ndarray]:
    """OpenML's `mnist_784`, parsed once and kept as two `.npy` files beside the other caches.

    `fetch_openml` parses the ARFF text again on every call, in pure Python (70 000 × 785 values),
    and every bench process that loads MNIST paid for it: 114 of the 744 children the timing
    campaign was prepared with. The pixels are integers 0–255 and are kept as `uint8`; the float64
    array returned is the parse's bit for bit, which is checked when the cache is written.
    """
    from sklearn.datasets import get_data_home

    base = Path(get_data_home()) / "betula_bench"
    xp, yp = base / "mnist_784-X.npy", base / "mnist_784-y.npy"
    if xp.exists() and yp.exists():
        return np.load(xp).astype(np.float64), np.load(yp)
    from sklearn.datasets import fetch_openml

    # `parser="liac-arff"`: the default ("auto" → pandas) raises on scikit-learn 1.9.1 with
    # pandas 3.1, and the harness used to read that as "offline" and drop the dataset.
    d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    x, y = np.asarray(d.data, dtype=np.float64), np.asarray(d.target.astype(int))
    pixels = x.astype(np.uint8)
    if not np.array_equal(pixels, x):
        raise ValueError("mnist_784 holds a pixel that is not an integer 0-255: not cached")
    base.mkdir(parents=True, exist_ok=True)
    for path, array in ((xp, pixels), (yp, y)):
        # Written aside and renamed, so a process that starts mid-write never reads half a file.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp.open("wb") as f:
            np.save(f, array)
        os.replace(tmp, path)
    return x, y


def draw_then_scale(x, y, n: int | None = REAL_CAP, seed: int = 0):
    """The **sub** rule: draw `n` rows with `rng(seed).choice`, then standardize those rows."""
    if n is not None and len(x) > n:
        idx = np.random.default_rng(seed).choice(len(x), n, replace=False)
        x, y = x[idx], y[idx]
    return _standardize(x), y


def scale_then_draw(x, y, n: int | None = None, seed: int = 0):
    """The **pop** rule: standardize the population, then take `rng(seed).permutation(...)[:n]`."""
    x = _standardize(x)
    if n is not None and len(x) > n:
        idx = np.random.default_rng(seed).permutation(len(x))[:n]
        x, y = np.ascontiguousarray(x[idx]), y[idx]
    return x, y


def load_sub(dataset: str, n: int | None = REAL_CAP, seed: int = 0):
    """`(X, y, k)` under the **sub** rule."""
    x, y, k = fetch(dataset)
    return (*draw_then_scale(x, y, n, seed), k)


def load_pop(dataset: str, n: int | None = None, seed: int = 0):
    """`(X, y, k)` under the **pop** rule. `n=None` is the whole population, standardized."""
    x, y, k = fetch(dataset)
    return (*scale_then_draw(x, y, n, seed), k)


def _standardize(x):
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(x).astype(np.float64)
