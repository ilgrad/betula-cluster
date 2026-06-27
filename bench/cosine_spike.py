"""Spike: does native cosine geometry beat `normalize + Euclidean` for embedding clustering?

We already support clustering L2-normalized data (a one-line preprocessing step). The open question
is whether a *native* cosine/spherical CF would add quality over that. This script measures, on
embedding-like data where cosine and Euclidean genuinely disagree (cluster signal is in the
*direction*, magnitude is noise — varying-norm vectors), four pipelines:

  1. betula on RAW vectors          (Euclidean, norm dominates → expected to fail)
  2. betula on L2-NORMALIZED vectors (what we already support)
  3. sklearn KMeans on normalized    (Euclidean-on-sphere reference)
  4. true spherical k-means (numpy)  (cosine assignment + mean-direction centroid — the CEILING)

If (2) ≈ (4): `normalize + Euclidean` already captures cosine → native cosine is not worth it for
quality (ship a `normalize=True` convenience instead). If (2) << (4): native cosine matters.

Run: uv run --with numpy --with scikit-learn --with <wheel> python bench/cosine_spike.py
"""

from __future__ import annotations

import time

import betula_cluster
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

D = 128
K = 8
PER = 1000
SEED = 0


def normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def make_data(sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Direction-clustered, varying-norm vectors: clusters share a direction; magnitude is noise."""
    rng = np.random.default_rng(seed)
    centers = normalize(rng.standard_normal((K, D)))
    xs, ys = [], []
    for c in range(K):
        dirs = normalize(centers[c] + sigma * rng.standard_normal((PER, D)))
        scales = rng.lognormal(mean=0.0, sigma=1.5, size=(PER, 1))  # wide magnitude spread
        xs.append(scales * dirs)
        ys += [c] * PER
    x = np.vstack(xs).astype(np.float64)
    return x, np.array(ys)


def _sph_once(
    xn: np.ndarray, k: int, rng: np.random.Generator, iters: int
) -> tuple[np.ndarray, float]:
    c = normalize(xn[rng.choice(len(xn), k, replace=False)].copy())
    lab = (xn @ c.T).argmax(1)
    for _ in range(iters):
        newc = np.zeros_like(c)
        for j in range(k):
            m = xn[lab == j]
            if len(m):
                newc[j] = m.sum(0)
        n = np.linalg.norm(newc, axis=1, keepdims=True)
        newc = np.where(
            n > 1e-12, newc / np.maximum(n, 1e-12), c
        )  # empty cluster keeps old centroid
        if np.allclose(newc, c):
            break
        c = newc
        lab = (xn @ c.T).argmax(1)
    obj = float((xn @ c.T).max(1).sum())  # total cosine similarity to assigned centroid
    return lab, obj


def spherical_kmeans(
    xn: np.ndarray, k: int, seed: int, n_init: int = 10, iters: int = 100
) -> np.ndarray:
    """True spherical k-means (cosine assignment + mean-direction centroid), best of `n_init`."""
    rng = np.random.default_rng(seed)
    best_lab, best_obj = None, -np.inf
    for _ in range(n_init):
        lab, obj = _sph_once(xn, k, rng, iters)
        if obj > best_obj:
            best_lab, best_obj = lab, obj
    return best_lab


def betula(x: np.ndarray, method: str) -> np.ndarray:
    return np.asarray(
        betula_cluster.fit_predict(x, K, feature="diagonal", method=method, max_leaves=400, seed=1)
    )


def run(sigma: float) -> dict[str, float]:
    x, y = make_data(sigma, SEED)
    xn = normalize(x)
    out: dict[str, float] = {}
    t = time.perf_counter()
    out["betula-km-raw"] = ari(y, betula(x, "kmeans"))
    out["betula-km-norm"] = ari(y, betula(xn, "kmeans"))
    out["t"] = time.perf_counter() - t
    out["betula-gmm-norm"] = ari(y, betula(xn, "gmm"))
    out["sklearn-km-norm"] = ari(y, KMeans(K, n_init=10, random_state=1).fit_predict(xn))
    out["spherical-ceiling"] = ari(y, spherical_kmeans(xn, K, SEED))
    return out


def main() -> None:
    print(f"cosine spike — d={D} k={K} n={K * PER}, varying-norm direction clusters\n")
    cols = [
        "betula-km-raw",
        "betula-km-norm",
        "betula-gmm-norm",
        "sklearn-km-norm",
        "spherical-ceiling",
    ]
    print(f"{'sigma':>6} | " + " | ".join(f"{c:>17}" for c in cols) + " | betula s")
    print("-" * 116)
    rows = []
    for sigma in (0.05, 0.1, 0.2, 0.4):
        r = run(sigma)
        rows.append(r)
        print(f"{sigma:>6.2f} | " + " | ".join(f"{r[c]:>17.3f}" for c in cols) + f" | {r['t']:.2f}")

    # Isolate geometry from head/compression:
    #   normalize lift  = betula-km-norm − betula-km-raw   (does normalizing fix Euclidean?)
    #   two-phase gap   = sklearn-km-norm − betula-km-norm (does betula's compression lose signal?)
    #   ceiling gap     = spherical-ceiling − betula-km-norm (room a native cosine CF could recover)
    lift = np.mean([r["betula-km-norm"] - r["betula-km-raw"] for r in rows])
    twophase = np.mean([r["sklearn-km-norm"] - r["betula-km-norm"] for r in rows])
    cosine_head = np.mean([r["spherical-ceiling"] - r["sklearn-km-norm"] for r in rows])
    print("\nVERDICT")
    print(f"  normalize lift      (betula-km-norm − betula-km-raw):     {lift:+.3f}")
    print(f"  two-phase gap       (sklearn-km-norm − betula-km-norm):   {twophase:+.3f}")
    print(f"  cosine-head value   (spherical-ceiling − sklearn-km-norm): {cosine_head:+.3f}")
    if abs(cosine_head) < 0.03 and lift > 0.1:
        print(
            "  → normalize+Euclidean == spherical (a native cosine HEAD adds nothing). The remaining"
        )
        print("    betula gap is the CF-tree's RESOLUTION on the sphere — recovered by raising")
        print("    max_leaves (see the max_leaves probe), NOT a geometry problem.")
        print("    DECISION: ship `normalize=True`; do NOT build a native cosine CF.")
    else:
        print("  → cosine head shows measurable value beyond normalized-Euclidean; reconsider.")


if __name__ == "__main__":
    main()
