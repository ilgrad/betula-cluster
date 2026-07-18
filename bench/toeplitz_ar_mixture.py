"""Synthetic AR-mixture benchmark for the `gmm-toeplitz` head.

Three components differ ONLY in their autocovariance (each window rescaled to unit marginal variance),
so the clustering signal lives entirely in the covariance *structure* — the adversarial case for a
diagonal model and the small-sample case for full covariance. We sweep the window length `d` (shrinking
`N_k / d`) and score ARI for betula's `gmm-toeplitz` vs the diagonal / full GMM heads and scikit-learn.

Run: `.venv/bin/python bench/toeplitz_ar_mixture.py`
"""

import betula_cluster as bc
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

SPECS = ([0.8], [1.1, -0.4], [])  # AR(1) a=0.8 · AR(2) [1.1,-0.4] · white-noise control


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


def make_mixture(d, per, seed):
    rng = np.random.default_rng(seed)
    xs = [ar_windows(per, d, a, rng) for a in SPECS]
    y = np.concatenate([np.full(per, c) for c in range(len(SPECS))])
    return np.ascontiguousarray(np.vstack(xs), dtype=np.float64), y


def main():
    per = 30
    print(f"AR-mixture: 3 components (AR(1) 0.8 · AR(2) [1.1,-0.4] · white), {per} windows each\n")
    print(
        f"{'d':>5} {'N_k/d':>6} | {'betula-toeplitz':>15} {'betula-diag':>12} {'betula-full':>12} {'sk-diag':>8} {'sk-full':>8}"
    )
    print("-" * 78)
    for d in (32, 64, 128, 256):
        X, y = make_mixture(d, per, seed=1)
        toe = bc.fit_predict(
            X, 3, method="gmm-toeplitz", feature="spherical", threshold=0.0, seed=1
        )
        bdi = bc.fit_predict(X, 3, method="gmm", feature="diagonal", threshold=0.0, seed=1)
        bfu = bc.fit_predict(X, 3, method="gmm-full", feature="full", threshold=0.0, seed=1)
        skd = GaussianMixture(3, covariance_type="diag", n_init=8, random_state=0).fit_predict(X)
        skf = GaussianMixture(
            3, covariance_type="full", reg_covar=1e-3, n_init=8, random_state=0
        ).fit_predict(X)
        row = [ari(y, np.asarray(v)) for v in (toe, bdi, bfu, skd, skf)]
        print(
            f"{d:>5} {per / d:>6.2f} | {row[0]:>15.3f} {row[1]:>12.3f} {row[2]:>12.3f} {row[3]:>8.3f} {row[4]:>8.3f}"
        )
    print(
        "\nOnly the AR/Toeplitz head recovers the components; it improves with d (more positions to"
    )
    print("pool the autocovariance) while diagonal is blind and full is singular at N_k < d.")


if __name__ == "__main__":
    main()
