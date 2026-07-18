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

Run: `.venv/bin/python bench/toeplitz_ar_mixture.py`
"""

import betula_cluster as bc
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

SPECS = ([0.8], [1.1, -0.4], [])  # AR(1) a=0.8 · AR(2) [1.1,-0.4] · white-noise control
ECHO_LAGS = (16, 28, 40)  # single-echo MA lags, all > w_max=10 (unreachable by AR(w))


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


def fit(method, feature, x, k=3):
    return np.asarray(bc.fit_predict(x, k, method=method, feature=feature, threshold=0.0, seed=1))


def main():
    per = 30
    print(f"AR-mixture: 3 components (AR(1) 0.8 · AR(2) [1.1,-0.4] · white), {per} windows each\n")
    hdr = f"{'d':>5} {'N_k/d':>6} | {'toe-AR':>8} {'toe-full':>8} {'toe-gs':>8} {'b-diag':>8} {'b-full':>8} {'sk-diag':>8} {'sk-full':>8}"
    print(hdr)
    print("-" * len(hdr))
    for d in (32, 64, 128, 256):
        x, y = make_mixture(ar_windows, SPECS, d, per, seed=1)
        toe = fit("gmm-toeplitz", "spherical", x)
        tof = fit("gmm-toeplitz-full", "spherical", x)
        tgs = fit("gmm-toeplitz-gs", "spherical", x)
        bdi = fit("gmm", "diagonal", x)
        bfu = fit("gmm-full", "full", x)
        skd = GaussianMixture(3, covariance_type="diag", n_init=8, random_state=0).fit_predict(x)
        skf = GaussianMixture(
            3, covariance_type="full", reg_covar=1e-3, n_init=8, random_state=0
        ).fit_predict(x)
        row = [ari(y, v) for v in (toe, tof, tgs, bdi, bfu, skd, skf)]
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
        f"Long-lag echo: 3 components, echo lag K ∈ {ECHO_LAGS} (all > w_max=10), {per} windows each\n"
    )
    hdr2 = f"{'d':>5} {'N_k/d':>6} | {'toeplitz-AR':>11} {'toeplitz-full':>13} {'toeplitz-gs':>11} {'betula-diag':>11}"
    print(hdr2)
    print("-" * len(hdr2))
    for d in (64, 96, 128, 192):
        x, y = make_mixture(echo_windows, ECHO_LAGS, d, per, seed=1)
        toe = fit("gmm-toeplitz", "spherical", x)
        tof = fit("gmm-toeplitz-full", "spherical", x)
        tgs = fit("gmm-toeplitz-gs", "spherical", x)
        bdi = fit("gmm", "diagonal", x)
        row = [ari(y, v) for v in (toe, tof, tgs, bdi)]
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


if __name__ == "__main__":
    main()
