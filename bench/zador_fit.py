"""Effective intrinsic dimension from the leaf budget, with no labels — Zador's form fitted.

`bench/leaf_budget.py` records, per cell, the realised number of leaves `m` and

    mean_sq_radius = sum_i w_i r_i^2 / n,

the summary's mean squared quantization error. Zador's theorem gives the asymptotics of an optimal
`m`-point quantizer of a source with intrinsic dimension `d`:

    D(m) ~ C * m^(-2/d),

so `log D` against `log m` is a straight line of slope `-2/d`, and `d_eff = -2 / slope` is an
estimate of the source's intrinsic dimension that never sees a label. A CF tree is not an optimal
quantizer, which biases the constant `C`; the slope is the part the theorem pins down, so that is
what is reported.

Two things the fit has to refuse rather than fudge:

  * A cell at zero compression (`m == n`) has `mean_sq_radius == 0` and no logarithm. Those rows are
    the raw-point rows the budget sweep includes on purpose, and they are dropped here.
  * A dataset whose realised leaf count saturates below the budget contributes repeated `m`, which
    inflates the fit's confidence without adding information. Cells are collapsed to unique `m`.

The fit is cross-checked against TWO-NN (Facco et al., *Estimating the intrinsic dimension of
datasets by a minimal neighborhood information*, Sci. Rep. 7, 2017), which estimates the same
quantity from nearest-neighbour ratios and knows nothing about quantizers. Zador's slope is an
asymptotic statement about an *optimal* quantizer; a CF tree is a greedy one, and `m^(1/d)` at these
budgets is barely above 1, so the two are expected to disagree and the size of the disagreement is
the finding, not a nuisance.

    uv run --no-sync --with pandas --with numpy --with scikit-learn python bench/zador_fit.py

Prints one row per (dataset, distance): the fitted slope with its standard error, the R^2, the
implied `d_eff` with the error propagated onto it, the TWO-NN estimate on the raw data, and the
ambient dimension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AMBIENT = {"digits": 64, "covtype-20k": 54, "mnist-10k": 784, "blobs": 2}


def fit(m: np.ndarray, d: np.ndarray) -> tuple[float, float, float]:
    """Least-squares slope of log d on log m, its standard error, and R^2.

    Six or seven budgets over a little more than a decade is not many, so the slope ships with the
    uncertainty that implies rather than to three decimals of false precision.
    """
    x, y = np.log(m), np.log(d)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    dof = len(x) - 2
    sxx = float(((x - x.mean()) ** 2).sum())
    se = float(np.sqrt((resid**2).sum() / dof / sxx)) if dof > 0 and sxx > 0 else float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), se, r2


def two_nn(x: np.ndarray, seed: int = 0) -> float:
    """TWO-NN intrinsic dimension (Facco et al. 2017).

    With `mu = r2/r1` the ratio of the two nearest-neighbour distances, `mu` is Pareto(1, d) on a
    `d`-dimensional manifold of locally constant density, so `-log(1 - F(mu))` is `d log mu` and the
    slope through the origin is the estimate. The top decile is discarded, as the paper prescribes,
    because that tail is where the constant-density assumption fails.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    if len(x) > 5000:
        x = x[rng.choice(len(x), 5000, replace=False)]
    d, _ = NearestNeighbors(n_neighbors=3).fit(x).kneighbors(x)
    r1, r2 = d[:, 1], d[:, 2]
    keep = r1 > 0
    mu = np.sort(r2[keep] / r1[keep])
    mu = mu[: int(0.9 * len(mu))]
    f = np.arange(1, len(mu) + 1) / (len(mu) + 1)
    xs, ys = np.log(mu), -np.log1p(-f)
    return float((xs @ ys) / (xs @ xs))


def raw_data(dataset: str) -> np.ndarray | None:
    """The points the budget sweep summarised, for the TWO-NN control."""
    from sklearn.datasets import fetch_covtype, fetch_openml, load_digits

    rng = np.random.default_rng(0)
    if dataset == "digits":
        return load_digits().data.astype(np.float64)
    if dataset == "covtype-20k":
        y = fetch_covtype()
        return y.data[rng.choice(len(y.data), 20_000, replace=False)].astype(np.float64)
    if dataset == "mnist-10k":
        y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        return y.data[:10_000].astype(np.float64)
    return None


def main() -> None:
    df = pd.read_csv("bench/results_budget.csv")
    # The quantization error is a property of the tree, not of the head that reads it, so every head
    # in a cell reports the same number; keep one row per (dataset, distance, budget).
    df = df.drop_duplicates(subset=["dataset", "distance", "max_leaves"])
    df = df[df["mean_sq_radius"] > 0.0]

    control: dict[str, float] = {}
    for dataset in sorted(df["dataset"].unique()):
        x = raw_data(dataset)
        if x is not None:
            control[dataset] = two_nn(x)

    print(
        f"{'dataset':12} {'distance':10} {'cells':>5} {'m range':>13} "
        f"{'slope':>7} {'se':>6} {'R^2':>6} {'d_eff':>6} {'+-':>6} {'two-nn':>7} {'ambient':>7}"
    )
    for (dataset, distance), g in df.groupby(["dataset", "distance"]):
        g = g.drop_duplicates(subset=["leaves"]).sort_values("leaves")
        if len(g) < 3:
            print(f"{dataset:12} {distance:10} {len(g):5}  too few points to fit")
            continue
        m = g["leaves"].to_numpy(dtype=float)
        d = g["mean_sq_radius"].to_numpy(dtype=float)
        slope, se, r2 = fit(m, d)
        d_eff = -2.0 / slope if slope < 0 else float("nan")
        # d_eff = -2/slope, so its uncertainty is |d/dslope| * se = 2 se / slope^2.
        d_se = 2.0 * se / slope**2 if slope < 0 else float("nan")
        span = f"{int(m.min())}-{int(m.max())}"
        print(
            f"{dataset:12} {distance:10} {len(g):5} {span:>13} "
            f"{slope:7.3f} {se:6.3f} {r2:6.3f} {d_eff:6.2f} {d_se:6.2f} "
            f"{control.get(dataset, float('nan')):7.2f} {AMBIENT.get(dataset, 0):7}"
        )


if __name__ == "__main__":
    main()
