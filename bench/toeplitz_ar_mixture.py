"""Synthetic covariance-structure benchmarks for the `gmm-toeplitz` heads.

Two scenarios where the clustering signal lives ENTIRELY in the autocovariance (every window is rescaled
to unit marginal variance, so a diagonal / centroid model is blind and a dense full covariance is
singular at `N_k < d`):

1. **AR-mixture** — components are AR(1) / AR(2) / white noise. The banded `gmm-toeplitz` (AR) head is
   the matched model; the general `gmm-toeplitz-full` (dense positive-definite Toeplitz) head is a
   superset and tracks it (occasionally edging it).
2. **Long-lag echo** — components differ only by an echo at lag `K ∈ {16, 28, 40}`, all beyond the AR
   order cap (`w_max = 10`). AR(w) is structurally unable to represent a lag-`K > w` autocovariance
   spike, so only the general `gmm-toeplitz-full` head recovers the components — the case that motivates
   the non-AR rung of the Toeplitz ladder (`docs/adr/001-gmm-toeplitz.md`).

Both tables are the **median of seeds 0, 1, 2** — the seed drives the data draw, betula's initialization
and scikit-learn's `random_state` together, so a cell's range covers both sources of luck. The per-cell
min/median/max ships in `results_toeplitz_spread.csv` next to the medians in `results_toeplitz.csv`.

Run: `uv run --no-sync --with scikit-learn python bench/toeplitz_ar_mixture.py`
"""

import csv
from pathlib import Path

import betula_cluster as bc
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent

SPECS = ([0.8], [1.1, -0.4], [])  # AR(1) a=0.8 · AR(2) [1.1,-0.4] · white-noise control
ECHO_LAGS = (16, 28, 40)  # single-echo MA lags, all > w_max=10 (unreachable by AR(w))
SEEDS = (0, 1, 2)


def ar_windows(n, d, a, rng):
    """`n` length-`d` windows from a zero-mean AR(len(a)) process, unit marginal variance."""
    a = np.asarray(a, float)
    w = len(a)
    out = np.empty((n, d))
    for k in range(n):
        buf = np.zeros(d + 256)
        e = rng.normal(size=d + 256)
        for t in range(w, d + 256):
            buf[t] = (a * buf[t - w : t][::-1]).sum() + e[t] if w else e[t]
        win = buf[256:]
        out[k] = (win - win.mean()) / win.std()
    return out


def echo_windows(n, d, lag, rng):
    """`n` length-`d` windows of a single-echo MA process `x_t = e_t + 0.7·e_{t−lag}`, unit variance.

    Its autocovariance is nonzero only at lags 0 and `lag`; for `lag > w_max` no AR(w) can represent it.
    """
    out = np.empty((n, d))
    for k in range(n):
        e = rng.normal(size=d + lag)
        win = e[lag:] + 0.7 * e[:d]
        out[k] = (win - win.mean()) / win.std()
    return out


def make_mixture(gen, params, d, per, seed):
    rng = np.random.default_rng(seed)
    xs = [gen(per, d, p, rng) for p in params]
    y = np.concatenate([np.full(per, c) for c in range(len(params))])
    return np.ascontiguousarray(np.vstack(xs), dtype=np.float64), y


def fit(method, feature, x, seed, k=3):
    return np.asarray(
        bc.fit_predict(x, k, method=method, feature=feature, threshold=0.0, seed=seed)
    )


def sweep(gen, params, d, per, methods):
    """`{method: [ARI per seed]}` for one window length, over `SEEDS`.

    The seed drives the draw and both libraries' initialization, so the spread it produces is the
    honest one: a reader cannot tell a lucky draw from a lucky restart, and neither can we.
    """
    out = {name: [] for name in methods}
    for seed in SEEDS:
        x, y = make_mixture(gen, params, d, per, seed=seed)
        for name, (method, feature) in methods.items():
            if method == "sk-diag":
                lab = GaussianMixture(
                    3, covariance_type="diag", n_init=8, random_state=seed
                ).fit_predict(x)
            elif method == "sk-full":
                lab = GaussianMixture(
                    3, covariance_type="full", reg_covar=1e-3, n_init=8, random_state=seed
                ).fit_predict(x)
            else:
                lab = fit(method, feature, x, seed)
            out[name].append(ari(y, lab))
    return out


def emit(rows, spread_rows, scenario, d, per, sweeps):
    for name, vals in sweeps.items():
        lo, mid, hi = min(vals), float(np.median(vals)), max(vals)
        rows.append(
            {"scenario": scenario, "d": d, "N_k_over_d": per / d, "method": name, "ARI": mid}
        )
        spread_rows.append(
            {
                "scenario": scenario,
                "d": d,
                "method": name,
                "min": lo,
                "median": mid,
                "max": hi,
                "range": hi - lo,
            }
        )
    return [float(np.median(v)) for v in sweeps.values()]


def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


AR_METHODS = {
    "toe-AR": ("gmm-toeplitz", "spherical"),
    "toe-full": ("gmm-toeplitz-full", "spherical"),
    "toe-gs": ("gmm-toeplitz-gs", "spherical"),
    "b-diag": ("gmm", "diagonal"),
    "b-full": ("gmm-full", "full"),
    "sk-diag": ("sk-diag", None),
    "sk-full": ("sk-full", None),
}

ECHO_METHODS = {
    "toeplitz-AR": ("gmm-toeplitz", "spherical"),
    "toeplitz-full": ("gmm-toeplitz-full", "spherical"),
    "toeplitz-gs": ("gmm-toeplitz-gs", "spherical"),
    "betula-diag": ("gmm", "diagonal"),
}


def main():
    per = 30
    rows, spread_rows = [], []
    print(f"AR-mixture: 3 components (AR(1) 0.8 · AR(2) [1.1,-0.4] · white), {per} windows each")
    print(f"median of seeds {list(SEEDS)}; ranges in results_toeplitz_spread.csv\n")
    hdr = f"{'d':>5} {'N_k/d':>6} | {'toe-AR':>8} {'toe-full':>8} {'toe-gs':>8} {'b-diag':>8} {'b-full':>8} {'sk-diag':>8} {'sk-full':>8}"
    print(hdr)
    print("-" * len(hdr))
    for d in (32, 64, 128, 256):
        row = emit(rows, spread_rows, "ar", d, per, sweep(ar_windows, SPECS, d, per, AR_METHODS))
        print(
            f"{d:>5} {per / d:>6.2f} | {row[0]:>8.3f} {row[1]:>8.3f} {row[2]:>8.3f} {row[3]:>8.3f} {row[4]:>8.3f} {row[5]:>8.3f} {row[6]:>8.3f}"
        )
    print(
        "\nBoth Toeplitz heads recover the components (improving with d); the general 'full' head tracks"
    )
    print(
        "the matched AR head, while diagonal is blind and dense full covariance is singular here.\n"
    )

    print(
        f"Long-lag echo: 3 components, echo lag K ∈ {ECHO_LAGS} (all > w_max=10), {per} windows each"
    )
    print(f"median of seeds {list(SEEDS)}\n")
    hdr2 = f"{'d':>5} {'N_k/d':>6} | {'toeplitz-AR':>11} {'toeplitz-full':>13} {'toeplitz-gs':>11} {'betula-diag':>11}"
    print(hdr2)
    print("-" * len(hdr2))
    for d in (64, 96, 128, 192):
        row = emit(
            rows,
            spread_rows,
            "echo",
            d,
            per,
            sweep(echo_windows, ECHO_LAGS, d, per, ECHO_METHODS),
        )
        print(
            f"{d:>5} {per / d:>6.2f} | {row[0]:>11.3f} {row[1]:>13.3f} {row[2]:>11.3f} {row[3]:>11.3f}"
        )
    print(
        "\nAR(w≤10) is blind to a lag-K>10 spike (≈ chance). The general 'full' (dense covariance) head"
    )
    print(
        "captures all lags; 'gs' (GS-MLE precision) captures lags within its order cap (≤16) — the two"
    )
    print("non-AR rungs of the Toeplitz ladder.")

    write_csv(HERE / "results_toeplitz.csv", rows, ["scenario", "d", "N_k_over_d", "method", "ARI"])
    write_csv(
        HERE / "results_toeplitz_spread.csv",
        spread_rows,
        ["scenario", "d", "method", "min", "median", "max", "range"],
    )
    worst = max(spread_rows, key=lambda r: r["range"])
    print(
        f"\nWidest cell: {worst['scenario']} d={worst['d']} {worst['method']} "
        f"{worst['min']:.3f}–{worst['max']:.3f} (range {worst['range']:.3f}). "
        "Wrote results_toeplitz.csv + results_toeplitz_spread.csv."
    )


if __name__ == "__main__":
    main()
