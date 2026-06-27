"""Which E-step is best for GMM clustering on BETULA cluster features?

Each "point" fed to global clustering is a weighted CF_i = (n_i, mu_i, Sigma_i) summarising
n_i original points.  We compare three responsibility formulas (identical weighted M-step,
identical init -> isolates the E-step):

  A) plug-in mean : log r ∝ log π_k + log N(mu_i | mu_k, Sigma_k)            (ignores Σ_i)
  B) convolution  : log r ∝ log π_k + log N(mu_i | mu_k, Sigma_k + Sigma_i)  (paper, exp. density)
  C) expected-log : log r  = log π_k + log N(mu_i | mu_k, Sigma_k) − ½ tr(Σ_k⁻¹ Σ_i)

Shared M-step folds within-CF spread into the component covariance:
  Σ_k = Σ_i w_ik (Σ_i + (μ_i−μ_k)(μ_i−μ_k)ᵀ) / N_k,   w_ik = n_i r_ik.

Metric: Adjusted Rand Index of the ORIGINAL points (each point inherits its CF's label)
vs ground truth.  Baselines: full-data GMM-EM (gold) and hard k-means on the CFs.
"""
from __future__ import annotations

import numpy as np

SEED = 0xBE7012A


# ---------- metrics ----------
def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ca, cb = np.unique(a), np.unique(b)
    ia = {c: i for i, c in enumerate(ca)}
    ib = {c: i for i, c in enumerate(cb)}
    cont = np.zeros((len(ca), len(cb)), dtype=np.int64)
    for x, y in zip(a, b):
        cont[ia[x], ib[y]] += 1
    comb2 = lambda x: x * (x - 1) // 2
    sc = sum(comb2(int(v)) for v in cont.sum(1))
    sk = sum(comb2(int(v)) for v in cont.sum(0))
    s = sum(comb2(int(v)) for v in cont.ravel())
    tot = comb2(len(a))
    exp = sc * sk / tot
    mx = 0.5 * (sc + sk)
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


# ---------- gaussian helpers ----------
def logpdf(x, mu, cov):  # x: (m,d), single cov
    d = x.shape[1]
    _, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    delta = x - mu
    quad = np.einsum("ia,ab,ib->i", delta, inv, delta)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def logpdf_percov(x, mu, covs):  # x: (m,d), covs: (m,d,d)
    d = x.shape[1]
    _, logdet = np.linalg.slogdet(covs)
    inv = np.linalg.inv(covs)
    delta = x - mu
    quad = np.einsum("ia,iab,ib->i", delta, inv, delta)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def kmeanspp(x, k, rng, weights=None):
    n = len(x)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    idx = rng.choice(n, p=w / w.sum())
    centers = [x[idx]]
    d2 = ((x - x[idx]) ** 2).sum(1)
    for _ in range(1, k):
        p = w * d2
        idx = rng.choice(n, p=p / p.sum())
        centers.append(x[idx])
        d2 = np.minimum(d2, ((x - x[idx]) ** 2).sum(1))
    return np.array(centers)


# ---------- full-data GMM (gold) ----------
def gmm_raw(x, k, rng, iters=80, ridge=1e-6):
    n, d = x.shape
    mu = kmeanspp(x, k, rng)
    cov = np.stack([np.cov(x.T, bias=True) + ridge * np.eye(d)] * k)
    pi = np.full(k, 1 / k)
    r = None
    for _ in range(iters):
        logr = np.stack([np.log(pi[c]) + logpdf(x, mu[c], cov[c]) for c in range(k)], 1)
        logr -= logsumexp(logr, axis=1, keepdims=True)
        r = np.exp(logr)
        nk = r.sum(0) + 1e-12
        pi = nk / n
        mu = (r.T @ x) / nk[:, None]
        for c in range(k):
            dl = x - mu[c]
            cov[c] = np.einsum("i,ia,ib->ab", r[:, c], dl, dl) / nk[c] + ridge * np.eye(d)
    return r.argmax(1)


# ---------- build CFs (micro-clusters via k-means) ----------
def kmeans(x, m, rng, iters=40):
    mu = x[rng.choice(len(x), m, replace=False)]
    a = np.zeros(len(x), int)
    for _ in range(iters):
        d2 = ((x[:, None, :] - mu[None, :, :]) ** 2).sum(2)
        a = d2.argmin(1)
        newmu = np.array([x[a == j].mean(0) if np.any(a == j) else mu[j] for j in range(m)])
        if np.allclose(newmu, mu):
            break
        mu = newmu
    return a


def build_cfs(x, assign, m, d):
    ns, mus, sigs, ids = [], [], [], []
    for j in range(m):
        pts = x[assign == j]
        if len(pts) == 0:
            continue
        ns.append(len(pts))
        mus.append(pts.mean(0))
        sigs.append(np.atleast_2d(np.cov(pts.T, bias=True)) if len(pts) > 1 else np.zeros((d, d)))
        ids.append(j)
    remap = {j: i for i, j in enumerate(ids)}
    return (np.array(ns, float), np.array(mus), np.array(sigs),
            np.array([remap[j] for j in assign]))


# ---------- GMM on CFs, shared init, three E-step variants ----------
def gmm_cf(n_i, mu_i, sig_i, k, init, variant, iters=100, ridge=1e-6):
    m, d = mu_i.shape
    mu, cov, pi = (a.copy() for a in init)
    r = None
    for _ in range(iters):
        logr = np.zeros((m, k))
        for c in range(k):
            if variant == "A":
                logr[:, c] = np.log(pi[c]) + logpdf(mu_i, mu[c], cov[c])
            elif variant == "B":
                logr[:, c] = np.log(pi[c]) + logpdf_percov(mu_i, mu[c], cov[c][None] + sig_i)
            else:  # C
                invk = np.linalg.inv(cov[c])
                tr = np.einsum("ab,iab->i", invk, sig_i)
                logr[:, c] = np.log(pi[c]) + logpdf(mu_i, mu[c], cov[c]) - 0.5 * tr
        logr -= logsumexp(logr, axis=1, keepdims=True)
        r = np.exp(logr)
        w = n_i[:, None] * r
        nk = w.sum(0) + 1e-12
        pi = nk / nk.sum()
        mu = (w.T @ mu_i) / nk[:, None]
        for c in range(k):
            dl = mu_i - mu[c]
            between = np.einsum("i,ia,ib->ab", w[:, c], dl, dl)
            within = np.einsum("i,iab->ab", w[:, c], sig_i)
            cov[c] = (within + between) / nk[c] + ridge * np.eye(d)
    return r.argmax(1)


def kmeans_cf(n_i, mu_i, k, init_mu, iters=100):
    mu = init_mu.copy()
    a = None
    for _ in range(iters):
        d2 = ((mu_i[:, None, :] - mu[None, :, :]) ** 2).sum(2)
        a = d2.argmin(1)
        newmu = np.array([
            (n_i[a == c, None] * mu_i[a == c]).sum(0) / n_i[a == c].sum()
            if np.any(a == c) else mu[c] for c in range(k)
        ])
        if np.allclose(newmu, mu):
            break
        mu = newmu
    return a


# ---------- data ----------
def gen(n, k, sep, rng, imbalance=False):
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centers = sep * np.column_stack([np.cos(ang), np.sin(ang)])
    if imbalance:
        wts = np.array([0.5, 0.25, 0.15, 0.10])[:k]
        wts = wts / wts.sum()
    else:
        wts = np.full(k, 1 / k)
    sizes = rng.multinomial(n, wts)
    xs, ys = [], []
    for c in range(k):
        a = rng.uniform(0, np.pi)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        scale = np.diag(rng.uniform(0.5, 1.5, 2))
        L = rot @ scale
        pts = centers[c] + rng.standard_normal((sizes[c], 2)) @ L.T
        xs.append(pts)
        ys.append(np.full(sizes[c], c))
    return np.vstack(xs), np.concatenate(ys)


def main():
    rng = np.random.default_rng(SEED)
    K, N, D = 4, 20000, 2
    print(f"GMM clustering on CF summaries — ARI vs ground truth (K={K}, N={N}, d={D})")
    print(f"{'scenario':<26}{'gold(raw)':>10}{'kmeans-CF':>10}"
          f"{'A:plugin':>10}{'B:conv':>9}{'C:explog':>10}")
    print("-" * 75)
    for imb in (False, True):
        for sep in (2.5, 4.0, 6.0):
            for m in (40, 150):
                rng = np.random.default_rng(SEED + int(sep * 10) + m + (1000 if imb else 0))
                x, y = gen(N, K, sep, rng, imbalance=imb)
                gold = ari(y, gmm_raw(x, K, rng))
                assign = kmeans(x, m, rng)
                n_i, mu_i, sig_i, pt2cf = build_cfs(x, assign, m, D)
                # shared init for all CF-GMM variants
                mu0 = kmeanspp(mu_i, K, rng, weights=n_i)
                cov0 = np.stack([np.cov(mu_i.T, aweights=n_i, bias=True) + 1e-6 * np.eye(D)] * K)
                pi0 = np.full(K, 1 / K)
                init = (mu0, cov0, pi0)
                res = {}
                for v in ("A", "B", "C"):
                    cf_lab = gmm_cf(n_i, mu_i, sig_i, K, init, v)
                    res[v] = ari(y, cf_lab[pt2cf])
                km = ari(y, kmeans_cf(n_i, mu_i, K, mu0)[pt2cf])
                tag = f"{'imb' if imb else 'bal'} sep={sep} m={m}"
                print(f"{tag:<26}{gold:>10.3f}{km:>10.3f}"
                      f"{res['A']:>10.3f}{res['B']:>9.3f}{res['C']:>10.3f}")
    print("\nHigher ARI = better. A/B/C share identical init + M-step; only the E-step differs.")
    print("Looking for: which CF E-step gets closest to gold(raw) and beats k-means, esp. coarse m.")


if __name__ == "__main__":
    main()
