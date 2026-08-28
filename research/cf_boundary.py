"""What a cluster feature can and cannot answer, checked rather than asserted.

Backs `research/RESULTS-cf-boundary.md`. A cluster feature is a sum-decomposition in the Deep Sets
sense (Zaheer et al. 2017), `f(X) = phi(sum_u psi(x_u))` with `psi(x) = (1, x, x x^T)`, so it
carries the permutation-invariant polynomials of degree <= 2 in the points and nothing else. Three
legs, one per section below, and each is a number rather than a paragraph:

  (a) exactness   objectives claimed exact from the summary are recomputed both ways; the residual
                  has to sit at machine precision, not at "close enough"
  (b) limits      an exhaustive integer search for two point sets whose features agree *bitwise* and
                  whose pairwise geometry does not, then what that costs single linkage and DBSCAN
  (c) cost        the identity WCSS_points = WCSS_summary + sum of leaf scatters, and -- for an
                  index that is *not* exact -- how fast the gap closes as the leaf budget grows

Legs (a) and (b) are reimplemented here from the definitions rather than called from the crate: the
point is to have a second, independently written arithmetic to disagree with, which is the same
discipline `research/gmm_cf_estep.py` follows.

    uv run --no-sync python research/cf_boundary.py
    uv run --no-sync --with scikit-learn python research/cf_boundary.py   # adds the digits row
"""

from __future__ import annotations

import itertools

import numpy as np

SEED = 0xB0DA12


# ────────────────────────────── the summary itself ──────────────────────────────


def summarize(x: np.ndarray, leaf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(weight, mean, ssd)` per leaf — the three fields every shipped feature model carries."""
    m = int(leaf.max()) + 1
    w = np.bincount(leaf, minlength=m).astype(float)
    mu = np.stack([x[leaf == lf].mean(0) for lf in range(m)])
    ssd = np.array([((x[leaf == lf] - mu[lf]) ** 2).sum() for lf in range(m)])
    return w, mu, ssd


def leader(x: np.ndarray, radius: float) -> np.ndarray:
    """Leader clustering: a BIRCH leaf is exactly this, a ball of bounded radius, first fit wins."""
    centers: list[np.ndarray] = []
    leaf = np.empty(len(x), dtype=np.int64)
    for i, p in enumerate(x):
        for j, c in enumerate(centers):
            if np.linalg.norm(p - c) <= radius:
                leaf[i] = j
                break
        else:
            leaf[i] = len(centers)
            centers.append(p)
    return leaf


def lloyd(mu: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted Lloyd from a k-means++ draw; enough to fix *a* partition, not to be a good one."""
    c = [mu[rng.integers(len(mu))]]
    for _ in range(k - 1):
        d2 = np.min([((mu - q) ** 2).sum(1) for q in c], axis=0)
        p = w * d2
        pick = rng.choice(len(mu), p=p / p.sum()) if p.sum() > 0 else rng.integers(len(mu))
        c.append(mu[pick])
    c = np.stack(c)
    for _ in range(100):
        lab = np.argmin(((mu[:, None, :] - c[None, :, :]) ** 2).sum(-1), axis=1)
        nxt = np.stack(
            [
                (w[lab == j, None] * mu[lab == j]).sum(0) / w[lab == j].sum()
                if (lab == j).any()
                else c[j]
                for j in range(k)
            ]
        )
        if np.allclose(nxt, c):
            break
        c = nxt
    return lab


# ────────────────────────── (a) the objectives that are exact ──────────────────────────


def indices_from_points(x: np.ndarray, lab: np.ndarray, k: int) -> dict[str, float]:
    """WCSS, Ward's first merge cost, Calinski-Harabasz and Davies-Bouldin, from the raw points."""
    n = len(x)
    mu = np.stack([x[lab == j].mean(0) for j in range(k)])
    nk = np.array([(lab == j).sum() for j in range(k)], dtype=float)
    wcss = sum(((x[lab == j] - mu[j]) ** 2).sum() for j in range(k))
    between = (nk * ((mu - x.mean(0)) ** 2).sum(1)).sum()
    radius = np.array([np.sqrt(((x[lab == j] - mu[j]) ** 2).sum() / nk[j]) for j in range(k)])
    return {
        "wcss": wcss,
        "ward": min(
            nk[a] * nk[b] / (nk[a] + nk[b]) * ((mu[a] - mu[b]) ** 2).sum()
            for a, b in itertools.combinations(range(k), 2)
        ),
        "calinski_harabasz": (between / (k - 1)) / (wcss / (n - k)),
        "davies_bouldin": davies_bouldin(mu, radius),
    }


def indices_from_summary(
    w: np.ndarray, mu: np.ndarray, ssd: np.ndarray, lab: np.ndarray, k: int
) -> dict[str, float]:
    """The same four, from `(weight, mean, ssd)` per leaf and a whole-leaf assignment.

    Every one reduces to `sum_{x in cl} ||x - c||^2 = sum_{leaf} (S_l + w_l ||mu_l - c||^2)`, the
    Koenig-Huygens split, for a `c` that is itself an affine function of the moments.
    """
    n = w.sum()
    ck = np.stack([(w[lab == j, None] * mu[lab == j]).sum(0) / w[lab == j].sum() for j in range(k)])
    nk = np.array([w[lab == j].sum() for j in range(k)])
    wcss = sum(
        ssd[lab == j].sum() + (w[lab == j] * ((mu[lab == j] - ck[j]) ** 2).sum(1)).sum()
        for j in range(k)
    )
    grand = (w[:, None] * mu).sum(0) / n
    between = (nk * ((ck - grand) ** 2).sum(1)).sum()
    radius = np.array(
        [
            np.sqrt(
                (ssd[lab == j].sum() + (w[lab == j] * ((mu[lab == j] - ck[j]) ** 2).sum(1)).sum())
                / nk[j]
            )
            for j in range(k)
        ]
    )
    return {
        "wcss": wcss,
        "ward": min(
            nk[a] * nk[b] / (nk[a] + nk[b]) * ((ck[a] - ck[b]) ** 2).sum()
            for a, b in itertools.combinations(range(k), 2)
        ),
        "calinski_harabasz": (between / (k - 1)) / (wcss / (n - k)),
        "davies_bouldin": davies_bouldin(ck, radius),
    }


def davies_bouldin(centers: np.ndarray, radius: np.ndarray) -> float:
    """RMS-radius Davies-Bouldin; the radius is the only place the summary enters."""
    k = len(centers)
    return float(
        np.mean(
            [
                max(
                    (radius[a] + radius[b]) / np.linalg.norm(centers[a] - centers[b])
                    for b in range(k)
                    if b != a
                )
                for a in range(k)
            ]
        )
    )


def leg_a(rng: np.random.Generator) -> None:
    print("(a) objectives claimed exact from the summary, recomputed both ways\n")
    x = rng.normal(0, 1, (400, 3))
    leaf = rng.integers(0, 20, 400)
    cluster_of_leaf = rng.integers(0, 4, 20)
    lab = cluster_of_leaf[leaf]
    w, mu, ssd = summarize(x, leaf)
    a = indices_from_points(x, lab, 4)
    b = indices_from_summary(w, mu, ssd, cluster_of_leaf, 4)
    print(f"    {'index':<20} {'from points':>16} {'from summary':>16} {'relative':>10}")
    for key in a:
        rel = abs(a[key] - b[key]) / max(abs(a[key]), 1e-300)
        print(f"    {key:<20} {a[key]:>16.10f} {b[key]:>16.10f} {rel:>10.1e}")


# ─────────────────────── (b) the pair no feature can separate ───────────────────────


def integer_twins(n: int, lo: int, hi: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Integer multisets of size `n` sharing `(sum, sum of squares)` but not their gap multiset.

    Integers on purpose: the two features then agree *exactly* in binary floating point, so the
    impossibility does not rest on a tolerance anywhere.
    """
    by_moment: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for s in itertools.combinations_with_replacement(range(lo, hi + 1), n):
        by_moment.setdefault((sum(s), sum(v * v for v in s)), []).append(s)
    out = []
    for group in by_moment.values():
        for p, q in itertools.combinations(group, 2):
            if pairwise(p) != pairwise(q):
                out.append((p, q))
    return out


def pairwise(s: tuple[int, ...]) -> list[int]:
    return sorted(abs(a - b) for a, b in itertools.combinations(s, 2))


def gaps(pts: np.ndarray) -> list[int]:
    """The distinct gaps along `x` — what a linkage or a density head actually reads."""
    return sorted({int(v) for v in pairwise(tuple(pts[:, 0].astype(int)))})


def single_linkage_heights(pts: np.ndarray) -> list[float]:
    """Kruskal on the complete graph: the merge heights *are* the sorted accepted edge weights."""
    n = len(pts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = sorted(
        (float(np.linalg.norm(pts[i] - pts[j])), i, j)
        for i, j in itertools.combinations(range(n), 2)
    )
    heights = []
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
            heights.append(d)
    return heights


def dbscan(pts: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    """Textbook DBSCAN, written out so the comparison depends on no library's tie-breaking."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    core = (d <= eps).sum(1) >= min_pts
    labels = np.full(len(pts), -1)
    cluster = 0
    for i in range(len(pts)):
        if not core[i] or labels[i] != -1:
            continue
        stack, labels[i] = [i], cluster
        while stack:
            j = stack.pop()
            for m in np.flatnonzero(d[j] <= eps):
                if labels[m] == -1:
                    labels[m] = cluster
                    if core[m]:
                        stack.append(m)
        cluster += 1
    return labels


def leg_b() -> None:
    print("\n(b) two sets one feature cannot tell apart\n")
    twins = integer_twins(4, -3, 3)
    print(f"    exhaustive search, n = 4, coordinates in [-3, 3]: {len(twins)} twin pairs")
    for n in (5, 6):
        print(f"    {'':<43}n = {n}: {len(integer_twins(n, -2, 2))} pairs in [-2, 2]")

    a = np.array([[-3.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    b = np.array([[-3.0, 0.0], [0.0, 0.0], [0.0, 0.0], [3.0, 0.0]])
    for name, pts in (("A", a), ("B", b)):
        dev = pts - pts.mean(0)
        print(
            f"\n    {name}: w = {len(pts)}  mean = {pts.mean(0)}  scatter = {(dev.T @ dev).ravel()}"
        )
    dev_a, dev_b = a - a.mean(0), b - b.mean(0)
    same = (
        len(a) == len(b)
        and np.array_equal(a.mean(0), b.mean(0))
        and np.array_equal(dev_a.T @ dev_a, dev_b.T @ dev_b)
    )
    print(f"\n    features identical bitwise: {same}")
    print(
        f"    single-linkage heights   A {single_linkage_heights(a)}  B {single_linkage_heights(b)}"
    )
    print(f"    pairwise distances       A {gaps(a)}  B {gaps(b)}")
    la, lb = dbscan(a, 2.0, 2), dbscan(b, 2.0, 2)
    print(
        f"    DBSCAN eps=2 minPts=2    A {la} ({la.max() + 1} clusters)  "
        f"B {lb} ({lb.max() + 1} clusters, {(lb < 0).sum()} noise)"
    )

    # Two more shapes, so a test that wants a second fixture does not have to trust one pair.
    print("\n    two further exact pairs, with the moments they share:")
    for p, q in (
        ((-2, -1, 1, 1, 1), (-2, 0, 0, 0, 2)),
        ((-2, 0, 0, 0, 1, 1), (-1, -1, -1, 1, 1, 1)),
    ):
        assert (sum(p), sum(v * v for v in p)) == (sum(q), sum(v * v for v in q))
        assert pairwise(p) != pairwise(q)
        print(
            f"      n = {len(p)}  {p} vs {q}   sum {sum(p)}, sum of squares "
            f"{sum(v * v for v in p)};  gaps {pairwise(p)} vs {pairwise(q)}"
        )


# ──────────────────── (c) what the summary costs, and how fast it closes ────────────────────


def medoid_silhouette(
    w: np.ndarray, mu: np.ndarray, ssd: np.ndarray, lab: np.ndarray, k: int
) -> float:
    """The crate's `validity::medoid_silhouette`, re-derived: squared metric, leaf spread included.

    With one point per leaf (`w = 1`, `ssd = 0`) this is the exact medoid silhouette of the points,
    which is what makes the two sides of the gap below the *same* index rather than two indices.
    """
    ck = np.stack([(w[lab == j, None] * mu[lab == j]).sum(0) / w[lab == j].sum() for j in range(k)])
    medoid = [
        int(np.flatnonzero(lab == j)[np.argmin(((mu[lab == j] - ck[j]) ** 2).sum(1))])
        for j in range(k)
    ]
    spread = ssd / w
    to = ((mu[:, None, :] - mu[None, medoid, :]) ** 2).sum(-1) + spread[:, None]
    own = to[np.arange(len(mu)), lab]
    other = np.min(np.where(np.arange(k)[None, :] == lab[:, None], np.inf, to), axis=1)
    s = np.where(other > 0.0, 1.0 - own / np.maximum(other, 1e-300), 0.0)
    return float((w * s).sum() / w.sum())


def leg_c(rng: np.random.Generator) -> None:
    print("\n(c) the identity, and the rate for an index that is not exact\n")

    x = rng.normal(0, 1, (60, 3))
    leaf = rng.integers(0, 6, 60)
    cluster_of_leaf = np.array([0, 0, 0, 1, 1, 1])
    w, mu, ssd = summarize(x, leaf)
    exact = indices_from_points(x, cluster_of_leaf[leaf], 2)["wcss"]
    # Zero scatter on purpose: this is the summary *without* the leaf-SSD term, so that what is
    # left over on the right-hand side is exactly the quantity the identity says it should be.
    summary = indices_from_summary(w, mu, np.zeros_like(ssd), cluster_of_leaf, 2)["wcss"]
    print(
        f"    WCSS_points {exact:.10f} = WCSS_summary {summary:.10f} + leaf scatter"
        f" {ssd.sum():.10f}   (residual {exact - summary - ssd.sum():.1e})"
    )

    print(
        f"\n    {'dataset':<10} {'radius':>7} {'leaves':>7} {'rms leaf':>9} "
        f"{'sil(points)':>12} {'sil(summary)':>13} {'gap':>9} {'gap/rms':>8}"
    )
    for name, data, k, ladder in datasets(rng):
        base = medoid_silhouette(
            np.ones(len(data)),
            data,
            np.zeros(len(data)),
            lloyd(data, np.ones(len(data)), k, rng),
            k,
        )
        for radius in ladder:
            leaf = leader(data, radius)
            w, mu, ssd = summarize(data, leaf)
            lab_leaf = lloyd(mu, w, k, rng)
            rms = float(np.sqrt(ssd.sum() / w.sum()))
            got = medoid_silhouette(w, mu, ssd, lab_leaf, k)
            # The reference is the same index on the same partition read back onto single points.
            ref = medoid_silhouette(
                np.ones(len(data)), data, np.zeros(len(data)), lab_leaf[leaf], k
            )
            print(
                f"    {name:<10} {radius:>7.2f} {len(w):>7} {rms:>9.4f} {ref:>12.4f} "
                f"{got:>13.4f} {abs(got - ref):>9.4f} {abs(got - ref) / max(rms, 1e-12):>8.3f}"
            )
        print(f"    {name:<10} {'points':>7} {len(data):>7} {0.0:>9.4f} {base:>12.4f}")


def datasets(rng: np.random.Generator):
    mu = rng.normal(0, 6, (4, 2))
    blobs = np.concatenate([m + rng.normal(0, 1, (250, 2)) for m in mu])
    out = [("blobs2d", blobs, 4, (2.0, 1.5, 1.0, 0.7, 0.4))]
    try:
        from sklearn.datasets import load_digits
    except ImportError:
        print("    (digits row skipped -- run with --with scikit-learn)")
        return out
    d = load_digits().data
    out.append(("digits64", d / 16.0, 10, (2.2, 1.9, 1.6, 1.3, 1.0)))
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    leg_a(rng)
    leg_b()
    leg_c(rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
