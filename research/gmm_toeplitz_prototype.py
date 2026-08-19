"""Prototype: Toeplitz / AR-structured GMM covariance for clustering ordered signals.

Validates the `gmm-toeplitz` head at proposal time (see `docs/adr/001-gmm-toeplitz.md`).
The ladder that ADR describes is now complete: banded AR (`gmm-toeplitz`) shipped in
0.2.0, the general dense-Toeplitz `gmm-toeplitz-full` in 0.3.0, and the full-order
Gohberg-Semencul MLE `gmm-toeplitz-gs` in 0.5.0. Claim: when each row is an ordered, wide-sense
stationary signal and the number of samples per component `N_k` is small relative to
the window length `d`, a Toeplitz / AR(w)-structured covariance clusters better than
both **diagonal** (blind to neighbour correlation) and **full** covariance
(ill-conditioned at `N_k < d`).

Data — a mixture of AR processes that differ ONLY in their autocovariance (each is
rescaled to unit marginal variance), so the clustering signal lives entirely in the
covariance *structure*: the adversarial case for diagonal and the small-sample case
for full covariance.

Covariance model — the AR(w) precision is estimated by Levinson-Durbin on the pooled
biased sample autocovariance and represented as the banded whitening filter `A`
(1 on the diagonal, `-a_j` on the j-th sub-diagonal): `Gamma = A^T A / sigma^2`, which
is positive-definite by construction (`sigma^2 > 0`) and is the classic banded /
Gohberg-Semencul-adjacent inverse covariance of an AR(w) process. Cost O(d*w) per
component; the E-step needs only the whitening residual energy `||A x||^2`.

Run: `.venv/bin/python research/gmm_toeplitz_prototype.py`
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

# --------------------------------------------------------------------------- data


def ar_windows(
    n: int, d: int, a: list[float], rng: np.random.Generator, sigma: float = 1.0
) -> np.ndarray:
    """`n` length-`d` windows from a zero-mean AR(len(a)) process (burn-in discarded)."""
    a = np.asarray(a, float)
    w = len(a)
    burn = 256
    out = np.empty((n, d))
    for k in range(n):
        buf = np.zeros(d + burn)
        e = rng.normal(0.0, sigma, d + burn)
        for t in range(w, d + burn):
            buf[t] = (a * buf[t - w:t][::-1]).sum() + e[t] if w else e[t]
        out[k] = buf[burn:]
    return out


def make_mixture(d: int, per: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """3 AR components distinguishable only by autocovariance, each unit marginal variance."""
    rng = np.random.default_rng(seed)
    specs = [[0.8], [1.1, -0.4], []]  # AR(1) a=0.8 · AR(2) [1.1,-0.4] · white-noise control
    blocks, labels = [], []
    for c, a in enumerate(specs):
        Xc = ar_windows(per, d, a, rng)
        Xc = (Xc - Xc.mean()) / Xc.std()  # kill the marginal-variance cue
        blocks.append(Xc)
        labels.append(np.full(per, c))
    return np.vstack(blocks), np.concatenate(labels)


# ------------------------------------------------------------------- AR machinery


def levinson(r: np.ndarray, w: int) -> tuple[np.ndarray, float]:
    """Levinson-Durbin: autocovariance r[0..w] -> AR coeffs a[1..w] and innovation var."""
    a = np.zeros(w)
    e = max(r[0], 1e-12)
    for m in range(1, w + 1):
        acc = r[m] - (a[: m - 1] * r[1:m][::-1]).sum()
        k = acc / e
        k = float(np.clip(k, -0.999, 0.999))  # keep the whitening filter stable
        new = a.copy()
        new[m - 1] = k
        new[: m - 1] = a[: m - 1] - k * a[: m - 1][::-1]
        a = new
        e *= 1.0 - k * k
        e = max(e, 1e-12)
    return a, e


def autocov(y: np.ndarray, weights: np.ndarray, w: int) -> np.ndarray:
    """Pooled biased weighted autocovariance r[0..w] over rows `y` (already centred)."""
    d = y.shape[1]
    wsum = weights.sum()
    r = np.empty(w + 1)
    for tau in range(w + 1):
        prod = (y[:, : d - tau] * y[:, tau:]).sum(axis=1)  # per-row lag-tau scatter
        r[tau] = (weights * prod).sum() / (wsum * d)  # biased (divide by d) => PD
    return r


def ar_loglik(X: np.ndarray, mu: float, a: np.ndarray, e: float) -> np.ndarray:
    """Per-row conditional AR log-density using the whitening residual energy."""
    w = len(a)
    d = X.shape[1]
    y = X - mu
    resid = y[:, w:].copy()
    for j in range(1, w + 1):
        resid -= a[j - 1] * y[:, w - j : d - j]
    energy = (resid * resid).sum(axis=1)
    return -0.5 * ((d - w) * np.log(2 * np.pi * e) + energy / e)


def bic_order(y: np.ndarray, weights: np.ndarray, w_max: int) -> int:
    """Pick the AR order by BIC on the pooled component (small, bounded grid)."""
    d = y.shape[1]
    n_eff = weights.sum() * d
    best_w, best_bic = 1, np.inf
    r = autocov(y, weights, w_max)
    for w in range(1, w_max + 1):
        a, e = levinson(r[: w + 1], w)
        ll = (weights * ar_loglik(y + 0.0, 0.0, a, e)).sum()
        bic = -2 * ll + w * np.log(n_eff)
        if bic < best_bic:
            best_bic, best_w = bic, w
    return best_w


# --------------------------------------------------------------- Toeplitz-GMM EM


def gmm_toeplitz(
    X: np.ndarray, k: int, w_max: int = 6, n_init: int = 8, max_iter: int = 100, seed: int = 0
):
    """Covariance-only GMM whose components are AR(w) (Toeplitz precision). Best of n_init."""
    rng = np.random.default_rng(seed)
    n = len(X)
    best = None
    for _ in range(n_init):
        resp = rng.dirichlet(np.ones(k), size=n)
        prev = -np.inf
        params = None
        for _ in range(max_iter):
            # M-step
            params = []
            for c in range(k):
                wt = resp[:, c] + 1e-9
                mu = (wt * X.mean(axis=1)).sum() / wt.sum()
                y = X - mu
                wc = bic_order(y, wt, w_max)
                r = autocov(y, wt, wc)
                a, e = levinson(r, wc)
                params.append((np.log(wt.sum() / n), mu, a, e))
            # E-step
            logr = np.empty((n, k))
            for c, (lpi, mu, a, e) in enumerate(params):
                logr[:, c] = lpi + ar_loglik(X, mu, a, e)
            mx = logr.max(axis=1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(logr - mx).sum(axis=1))
            resp = np.exp(logr - lse[:, None])
            ll = lse.sum()
            if ll - prev < 1e-4 * abs(prev):
                break
            prev = ll
        if best is None or prev > best[0]:
            best = (prev, resp.argmax(axis=1), params)
    return best[1]


# ------------------------------------------------------------------------- driver


def main() -> None:
    per, k = 30, 3
    print(f"AR-mixture: 3 components (AR(1) 0.8 · AR(2) [1.1,-0.4] · white), {per} windows each\n")
    print(f"{'d (window)':>10} {'N_k/d':>7} | {'gmm-diag':>9} {'gmm-full':>9} {'gmm-toeplitz':>13}")
    print("-" * 58)
    for d in (32, 64, 128, 256):
        X, y = make_mixture(d, per, seed=1)
        diag = GaussianMixture(k, covariance_type="diag", n_init=8, random_state=0).fit_predict(X)
        gm = GaussianMixture(k, covariance_type="full", reg_covar=1e-3, n_init=8, random_state=0)
        full = gm.fit_predict(X)
        toe = gmm_toeplitz(X, k, w_max=6, seed=1)
        ad, af, at = ari(y, diag), ari(y, full), ari(y, toe)
        print(f"{d:>10} {per / d:>7.2f} | {ad:>9.3f} {af:>9.3f} {at:>13.3f}")
    print("\nExpectation: diag ~0 (equal marginals), full degrades as N_k/d shrinks,")
    print("toeplitz stays high (few params, captures the autocovariance distinction).")


if __name__ == "__main__":
    main()
