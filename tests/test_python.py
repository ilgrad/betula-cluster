"""End-to-end tests of the Python bindings (run with `pytest`).

Covers every public surface: the one-shot `fit_predict` (all heads, auto-k, f32, χ² absorption)
and the streaming `Betula` estimator, plus the error contract.
"""

import collections
import itertools
import math
import warnings
from math import comb

import betula_cluster
import numpy as np
import pytest


def ari(a, b):
    a = list(map(int, a))
    b = list(map(int, b))
    cont = collections.Counter(zip(a, b, strict=True))
    ra = collections.Counter(a)
    rb = collections.Counter(b)
    s = sum(comb(v, 2) for v in cont.values())
    sa = sum(comb(v, 2) for v in ra.values())
    sb = sum(comb(v, 2) for v in rb.values())
    tot = comb(len(a), 2)
    exp = sa * sb / tot
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else (s - exp) / (mx - exp)


def n_labels(labels):
    return len({int(v) for v in labels if v >= 0})


def test_version_is_exposed():
    v = betula_cluster.__version__
    assert isinstance(v, str) and v  # non-empty string
    assert v[0].isdigit() and "." in v  # looks like a real version (installed metadata)


@pytest.fixture(scope="module")
def blobs():
    """Four well-separated 2-D Gaussian blobs; returns (X float64, y)."""
    rng = np.random.default_rng(0)
    centers = [[0, 0], [9, 0], [0, 9], [9, 9]]
    xs, ys = [], []
    for c, ctr in enumerate(centers):
        xs.append(rng.normal(ctr, 0.6, (600, 2)))
        ys += [c] * 600
    return np.vstack(xs).astype(np.float64), np.array(ys)


@pytest.fixture(scope="module")
def moons():
    """Two interleaving half-moons (non-convex); returns (X, y)."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, np.pi, 700)
    x = np.vstack([np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]])
    x = x + rng.normal(0, 0.06, x.shape)
    y = np.array([0] * 700 + [1] * 700)
    return x.astype(np.float64), y


# ── one-shot fit_predict ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "feature,method",
    [
        ("spherical", "kmeans"),
        ("diagonal", "gmm"),
        ("full", "gmm-full"),
        ("fd", "gmm-full"),
        ("fd", "mppca"),
        ("diagonal", "ward"),
        ("spherical", "spectral"),
    ],
)
def test_fit_predict_recovers_blobs(blobs, feature, method):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x, 4, feature=feature, method=method, threshold=0.05, max_leaves=300, seed=1
    )
    assert ari(labels, y) > 0.95


@pytest.mark.parametrize(
    "feature,method",
    [("diagonal", "gmm"), ("full", "gmm-full"), ("fd", "mppca"), ("diagonal", "ward")],
)
def test_auto_k_selects_true_count_when_n_clusters_zero(blobs, feature, method):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x, n_clusters=0, feature=feature, method=method, threshold=0.05, max_leaves=300, seed=1
    )
    assert n_labels(labels) == 4
    assert ari(labels, y) > 0.95


# ── directional heads (spherical-kmeans / vMF) ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sphere_blobs():
    """Three vMF-like clusters on the unit hypersphere; returns (X unit-norm float64, y)."""
    rng = np.random.default_rng(2)
    d, k, per = 16, 3, 500
    centers = rng.normal(size=(k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xs, ys = [], []
    for c, ctr in enumerate(centers):
        pts = ctr + 0.25 * rng.normal(size=(per, d))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        xs.append(pts)
        ys += [c] * per
    return np.vstack(xs).astype(np.float64), np.array(ys)


@pytest.mark.parametrize("method", ["spherical-kmeans", "vmf"])
def test_directional_recovers_sphere_blobs(sphere_blobs, method):
    x, y = sphere_blobs
    labels = betula_cluster.fit_predict(x, 3, method=method, threshold=0.05, max_leaves=300, seed=1)
    assert ari(labels, y) > 0.9


def test_vmf_auto_k_selects_true_count(sphere_blobs):
    x, y = sphere_blobs
    labels = betula_cluster.fit_predict(
        x, n_clusters=0, method="vmf", threshold=0.05, max_leaves=300, seed=1
    )
    assert n_labels(labels) == 3
    assert ari(labels, y) > 0.9


def test_vmf_predict_proba_is_true_posterior(sphere_blobs):
    x, y = sphere_blobs
    est = betula_cluster.Betula(
        method="vmf", n_clusters=3, threshold=0.05, max_leaves=300, seed=1
    ).fit(x)
    proba = est.predict_proba(x)
    assert proba.shape == (len(x), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert ari(proba.argmax(axis=1), y) > 0.9


@pytest.fixture(scope="module")
def ar_windows():
    """Three AR components distinguishable ONLY by autocovariance (unit marginal variance)."""
    rng = np.random.default_rng(3)
    d, per = 64, 18
    specs = ([0.8], [1.1, -0.4], [])  # AR(1) · AR(2) · white-noise control
    xs, ys = [], []
    for c, a in enumerate(specs):
        a = np.asarray(a, float)
        w = len(a)
        for _ in range(per):
            buf = np.zeros(d + 200)
            e = rng.normal(size=d + 200)
            for t in range(w, d + 200):
                buf[t] = (a * buf[t - w : t][::-1]).sum() + e[t] if w else e[t]
            win = buf[200:]
            win = (win - win.mean()) / win.std()
            xs.append(win)
            ys.append(c)
    return np.vstack(xs).astype(np.float64), np.array(ys)


def test_gmm_toeplitz_separates_ar_mixture(ar_windows):
    x, y = ar_windows
    kw = dict(feature="spherical", threshold=0.0, seed=1)
    toe = betula_cluster.fit_predict(x, 3, method="gmm-toeplitz", **kw)
    diag = betula_cluster.fit_predict(x, 3, method="gmm", **kw)
    a_toe, a_diag = ari(toe, y), ari(diag, y)
    assert a_toe > 0.6, f"toeplitz ARI = {a_toe}"
    assert a_toe > a_diag + 0.3, f"toeplitz {a_toe} should beat diagonal {a_diag}"


def test_gmm_toeplitz_auto_k(ar_windows):
    x, y = ar_windows
    labels = betula_cluster.fit_predict(
        x, n_clusters=0, method="gmm-toeplitz", feature="spherical", threshold=0.0, seed=1
    )
    assert n_labels(labels) == 3
    assert ari(labels, y) > 0.6


def test_gmm_toeplitz_predict_proba(ar_windows):
    x, y = ar_windows
    est = betula_cluster.Betula(
        method="gmm-toeplitz", n_clusters=3, feature="spherical", threshold=0.0, seed=1
    ).fit(x)
    proba = est.predict_proba(x)
    assert proba.shape == (len(x), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert ari(proba.argmax(axis=1), y) > 0.6


def test_gmm_toeplitz_full_clusters_ar_mixture(ar_windows):
    x, y = ar_windows
    kw = dict(feature="spherical", threshold=0.0, seed=1)
    full = betula_cluster.fit_predict(x, 3, method="gmm-toeplitz-full", **kw)
    diag = betula_cluster.fit_predict(x, 3, method="gmm", **kw)
    a_full, a_diag = ari(full, y), ari(diag, y)
    assert a_full > 0.5, f"toeplitz-full ARI = {a_full}"
    assert a_full > a_diag + 0.2, f"toeplitz-full {a_full} should beat diagonal {a_diag}"


def test_gmm_toeplitz_full_auto_k_and_proba(ar_windows):
    x, _ = ar_windows
    est = betula_cluster.Betula(
        method="gmm-toeplitz-full", n_clusters=0, feature="spherical", threshold=0.0, seed=1
    ).fit(x)
    proba = est.predict_proba(x)
    assert proba.shape[0] == len(x)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_gmm_toeplitz_gs_clusters_ar_mixture(ar_windows):
    x, y = ar_windows
    kw = dict(feature="spherical", threshold=0.0, seed=1)
    gs = betula_cluster.fit_predict(x, 3, method="gmm-toeplitz-gs", **kw)
    diag = betula_cluster.fit_predict(x, 3, method="gmm", feature="diagonal", threshold=0.0, seed=1)
    assert ari(gs, y) > 0.5, f"gs ARI {ari(gs, y)} (diag {ari(diag, y)})"
    assert ari(gs, y) > ari(diag, y) + 0.2


def test_gmm_toeplitz_gs_predict_proba(ar_windows):
    x, _ = ar_windows
    est = betula_cluster.Betula(
        method="gmm-toeplitz-gs", n_clusters=3, feature="spherical", threshold=0.0, seed=1
    ).fit(x)
    proba = est.predict_proba(x)
    assert proba.shape == (len(x), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.fixture
def nmf_topics():
    """Nonnegative documents: each a nonnegative mix of one of 3 latent parts (NMF's setting)."""
    rng = np.random.default_rng(4)
    h = np.abs(rng.normal(size=(3, 30))) * (rng.random((3, 30)) < 0.4)
    xs, ys = [], []
    for c in range(3):
        for _ in range(200):
            w = np.zeros(3)
            w[c] = 1.0 + rng.random()
            xs.append(w @ h + 0.03 * rng.random(30))
            ys.append(c)
    return np.ascontiguousarray(xs), np.array(ys)


def test_projection_weighted_nmf_module(nmf_topics):
    x, y = nmf_topics
    labels = betula_cluster.fit_predict(
        x,
        3,
        method="kmeans",
        feature="spherical",
        threshold=0.0,
        seed=0,
        projection="weighted-nmf",
        projection_dim=6,
    )
    assert n_labels(labels) == 3
    assert ari(labels, y) > 0.9


def test_projection_weighted_nmf_kl(nmf_topics):
    x, y = nmf_topics
    labels = betula_cluster.fit_predict(
        x,
        3,
        method="kmeans",
        feature="spherical",
        threshold=0.0,
        seed=0,
        projection="weighted-nmf-kl",
        projection_dim=6,
    )
    assert n_labels(labels) == 3
    assert ari(labels, y) > 0.8
    # the KL variant is honoured through the streaming estimator too, and rejects signed input
    est = betula_cluster.Betula(
        method="kmeans",
        n_clusters=3,
        feature="spherical",
        threshold=0.0,
        seed=0,
        projection="weighted-nmf-kl",
        projection_dim=6,
    )
    assert ari(est.fit_predict(x), y) > 0.8
    xneg = x.copy()
    xneg[0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        est.fit_predict(xneg)


def test_projection_weighted_nmf_estimator(nmf_topics):
    x, y = nmf_topics
    kw = dict(
        n_clusters=3,
        method="kmeans",
        feature="spherical",
        threshold=0.0,
        seed=0,
        projection="weighted-nmf",
        projection_dim=6,
    )
    assert ari(betula_cluster.Betula(**kw).fit_predict(x), y) > 0.9
    est = betula_cluster.Betula(**kw).fit(x)  # fit / predict round-trip also honours the projection
    assert ari(est.predict(x), y) > 0.9


def test_projection_get_params_roundtrip():
    est = betula_cluster.Betula(projection="weighted-nmf", projection_dim=6)
    assert est.get_params()["projection"] == "weighted-nmf"
    assert est.get_params()["projection_dim"] == 6
    assert est.get_params()["projection_max_iter"] == 100
    est.set_params(projection="none")
    assert est.get_params()["projection"] == "none"


def test_projection_rejects_negative(nmf_topics):
    x, _ = nmf_topics
    xneg = x.copy()
    xneg[0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        betula_cluster.fit_predict(xneg, 3, method="kmeans", projection="weighted-nmf")
    with pytest.raises(ValueError, match="nonnegative"):
        betula_cluster.Betula(projection="weighted-nmf", method="kmeans").fit(xneg)
    with pytest.raises(ValueError, match="nonnegative"):
        betula_cluster.Betula(projection="weighted-nmf", method="kmeans").fit_predict(xneg)


def test_projection_accepts_sparse(nmf_topics):
    sp = pytest.importorskip("scipy.sparse")
    x, y = nmf_topics
    csr = sp.csr_matrix(x)
    est = betula_cluster.Betula(
        n_clusters=3,
        method="kmeans",
        feature="spherical",
        threshold=0.0,
        seed=0,
        projection="weighted-nmf",
        projection_dim=6,
    )
    assert ari(est.fit_predict(csr), y) > 0.9
    assert est.components_.shape == (6, x.shape[1])


def test_projection_rejects_sparse_with_negative_values():
    sp = pytest.importorskip("scipy.sparse")
    csr = sp.random(60, 20, density=0.3, format="csr", random_state=0)
    csr.data = -np.abs(csr.data)
    with pytest.raises(ValueError, match="nonnegative"):
        betula_cluster.Betula(projection="weighted-nmf", method="kmeans").fit_predict(csr)


@pytest.mark.parametrize("method", ["kmeans", "spherical-kmeans"])
def test_kmeans_labels_are_the_nearest_centre(method):
    """k-means assigns a point to its nearest centre; `predict` must return that partition.

    Reading the label off the leaf the point routes to instead is an approximate
    nearest-microcluster search, and the tree descent is greedy: in high dimension it lands on the
    wrong leaf often enough to disagree with the model on a fifth of the points.
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4000, 64)) + rng.integers(0, 6, 4000)[:, None] * 3.0
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    est = betula_cluster.Betula(n_clusters=6, method=method, max_leaves=400, normalize=True, seed=1)
    labels = est.fit_predict(x)
    centers = np.asarray(est.cluster_centers_, dtype=np.float64)
    live = np.flatnonzero((centers != 0).any(axis=1))
    if method == "spherical-kmeans":
        centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    d = (x**2).sum(1)[:, None] - 2 * x @ centers[live].T + (centers[live] ** 2).sum(1)[None, :]
    assert np.array_equal(labels, live[d.argmin(1)])


def test_predict_reaches_a_cluster_whose_radius_is_zero():
    """A cluster holding one coincident group has radius 0; `predict` must still return it.

    The Voronoi rule filters clusters on their *weight*, and the stats helper yields radii before
    weights for clusters (the opposite of its leaf order). Reading them in leaf order drops every
    zero-radius cluster from the rule, which makes it unreachable through `predict` even though it
    owns the point sitting exactly on its centre.
    """
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(size=(300, 4)), np.full((3, 4), 40.0)])
    est = betula_cluster.Betula(n_clusters=2, method="kmeans", max_leaves=64, seed=0).fit(x)
    centers = np.asarray(est.cluster_centers_, dtype=np.float64)
    outlier = int(np.argmin(((centers - 40.0) ** 2).sum(1)))
    assert np.asarray(est.cluster_radii_)[outlier] == 0.0, "the outlier cluster is not degenerate"
    assert est.predict(np.full((1, 4), 40.0))[0] == outlier


def test_projection_components_are_canonical(nmf_topics):
    x, _ = nmf_topics
    est = betula_cluster.Betula(
        n_clusters=3,
        method="kmeans",
        feature="spherical",
        threshold=0.0,
        seed=0,
        projection="weighted-nmf",
        projection_dim=6,
    ).fit(x)
    h = est.components_
    assert h.shape == (6, x.shape[1])
    assert (h >= 0).all()
    # NMF is invariant to (W D, D^-1 H); the codes are consumed as Euclidean features, so the split
    # is pinned down: unit-L2 component rows, ordered by descending energy.
    assert np.allclose(np.linalg.norm(h, axis=1), 1.0)
    assert 0.0 <= est.reconstruction_err_ <= 1.0


def test_projection_max_iter_is_independent_of_head_budget(nmf_topics):
    x, y = nmf_topics
    kw = dict(
        n_clusters=3,
        method="kmeans",
        feature="spherical",
        threshold=0.0,
        seed=0,
        projection="weighted-nmf",
        projection_dim=6,
    )
    # A bigger factorization budget fits the leaf centroids at least as well, whatever the head does
    coarse = betula_cluster.Betula(**kw, max_iter=100, projection_max_iter=3).fit(x)
    fine = betula_cluster.Betula(**kw, max_iter=100, projection_max_iter=100).fit(x)
    assert fine.reconstruction_err_ <= coarse.reconstruction_err_
    assert ari(fine.fit_predict(x), y) > 0.9


def test_projection_accessors_require_a_projection(nmf_topics):
    x, _ = nmf_topics
    est = betula_cluster.Betula(
        n_clusters=3, method="kmeans", feature="spherical", threshold=0.0, seed=0
    ).fit(x)
    with pytest.raises(ValueError, match="components_ is only available"):
        _ = est.components_
    with pytest.raises(ValueError, match="reconstruction_err_ is only available"):
        _ = est.reconstruction_err_


def test_projection_invalid_args():
    x = np.abs(np.random.default_rng(0).normal(size=(60, 8)))
    with pytest.raises(ValueError, match="projection must be"):
        betula_cluster.fit_predict(x, 3, method="kmeans", projection="bogus")
    with pytest.raises(ValueError, match="projection_max_iter must be"):
        betula_cluster.fit_predict(
            x, 3, method="kmeans", projection="weighted-nmf", projection_max_iter=0
        )
    with pytest.raises(ValueError, match="projection_dim must be"):
        betula_cluster.fit_predict(
            x, 3, method="kmeans", projection="weighted-nmf", projection_dim=0
        )


def test_directional_methods_force_normalization(sphere_blobs):
    x, y = sphere_blobs
    est = betula_cluster.Betula(method="vmf", n_clusters=3, threshold=0.05, max_leaves=300, seed=1)
    # get_params stays verbatim (normalize left False) so sklearn clone / set_params round-trip…
    assert est.get_params()["normalize"] is False
    # …yet non-unit input still clusters correctly: the engine normalizes for directional heads.
    labels = est.fit(x * 9.0).predict(x * 9.0)
    assert ari(labels, y) > 0.9


def test_directional_auto_threshold_pilots_on_normalized_data():
    rng = np.random.default_rng(5)
    d, k, per = 8, 3, 1800  # 5400 rows > the auto-threshold pilot cap (4000) → the pilot fires
    centers = rng.normal(size=(k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xs, ys = [], []
    for c, ctr in enumerate(centers):
        pts = ctr + 0.25 * rng.normal(size=(per, d))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        xs.append(pts)
        ys += [c] * per
    x, y = np.vstack(xs), np.array(ys)
    est = betula_cluster.Betula(
        method="vmf", n_clusters=3, threshold="auto", max_leaves=100, seed=1
    )
    labels = est.fit_predict(x)
    assert ari(labels, y) > 0.85


def test_scale_space_recovers_blobs_without_k(blobs):
    # scale-space picks the number of density modes by persistence — no n_clusters needed.
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x, method="scale-space", threshold=0.05, max_leaves=300, seed=1
    )
    assert n_labels(labels) == 4
    assert ari(labels, y) > 0.9


def test_spectral_separates_moons_where_kmeans_fails(moons):
    # The non-convex arms need a fine microcluster resolution for the affinity graph to follow the
    # manifold, so pair spectral with a small threshold (many leaves).
    x, y = moons
    kw = dict(n_clusters=2, threshold=0.004, max_leaves=600, seed=0)
    spectral = betula_cluster.fit_predict(x, method="spectral", **kw)
    kmeans = betula_cluster.fit_predict(x, method="kmeans", **kw)
    assert ari(spectral, y) > 0.9  # spectral recovers the two moons
    assert ari(kmeans, y) < 0.6  # a centroid head cuts straight across them


@pytest.mark.parametrize("method,resolution", [("leiden", 1.0), ("leiden-cpm", 0.03)])
def test_leiden_detects_communities_without_k(blobs, method, resolution):
    x, y = blobs
    # Leiden discovers the count from the microcluster graph — n_clusters is ignored. Pair it with a
    # moderate threshold; a fine graph over-splits (modularity's resolution limit). CPM's γ is on a
    # smaller, density scale.
    labels = betula_cluster.fit_predict(
        x, 99, method=method, threshold=0.3, max_leaves=400, resolution=resolution, seed=1
    )
    assert n_labels(labels) == 4  # four communities found despite n_clusters=99
    assert ari(labels, y) > 0.95


def test_leiden_resolution_controls_granularity(blobs):
    x, _ = blobs
    kw = dict(n_clusters=99, method="leiden", threshold=0.3, max_leaves=400, seed=1)
    coarse = n_labels(betula_cluster.fit_predict(x, resolution=1.0, **kw))
    fine = n_labels(betula_cluster.fit_predict(x, resolution=4.0, **kw))
    assert fine > coarse  # higher γ ⇒ more, smaller communities


def test_leiden_covariance_aware_recovers_blobs(blobs):
    # covariance_weight adds a log-Euclidean shape term to the Leiden affinity (feature="full");
    # on well-separated blobs it must recover the communities and not degrade them.
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=99,
        feature="full",
        method="leiden",
        covariance_weight=0.3,
        threshold=0.3,
        max_leaves=400,
        seed=1,
    )
    labels = est.fit_predict(x)
    assert est.get_params()["covariance_weight"] == 0.3  # verbatim (sklearn clone/set_params)
    assert n_labels(labels) >= 4  # recovers the structure (the shape term may split a little finer)
    assert ari(labels, y) > 0.85


def test_leiden_tangent_aware_recovers_blobs(blobs):
    # tangent_weight adds a Grassmann subspace term (GeoBETULA); on well-separated blobs it must
    # still recover the communities. get_params stays verbatim for sklearn clone / set_params.
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=99,
        feature="full",
        method="leiden",
        tangent_weight=0.5,
        tangent_rank=1,
        threshold=0.3,
        max_leaves=400,
        seed=1,
    )
    labels = est.fit_predict(x)
    assert est.get_params()["tangent_weight"] == 0.5
    assert est.get_params()["tangent_rank"] == 1
    assert n_labels(labels) >= 4
    assert ari(labels, y) > 0.85


def test_float32_reproduces_float64_on_normal_range(blobs):
    x, y = blobs
    kw = dict(feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1)
    l64 = betula_cluster.fit_predict(x, 4, **kw)
    l32 = betula_cluster.fit_predict(x.astype(np.float32), 4, **kw)
    assert np.asarray(l32).dtype == np.int64
    assert ari(l64, y) > 0.95 and ari(l32, y) > 0.95
    assert ari(l32, l64) > 0.99  # f32 path agrees with f64 on moderate-range data


@pytest.mark.parametrize(
    "absorb",
    ["euclidean", "manhattan", "average", "diameter", "ward", "radius", "chi2", "subspace"],
)
def test_absorption_modes_recover_blobs(blobs, absorb):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x,
        4,
        feature="diagonal",
        method="gmm",
        absorb=absorb,
        chi2_scale=0.5,
        max_leaves=300,
        seed=1,  # within-cluster var ≈ 0.36
    )
    assert ari(labels, y) > 0.95


def test_chi2_absorption_composes_with_float32(blobs):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x.astype(np.float32),
        4,
        feature="diagonal",
        method="gmm",
        absorb="chi2",
        chi2_p=0.9,
        chi2_scale=0.5,
        max_leaves=300,
        seed=1,
    )
    assert ari(labels, y) > 0.95


def test_chi2_without_scale_raises(blobs):
    x, _ = blobs
    with pytest.raises(ValueError):
        betula_cluster.fit_predict(x, 4, method="gmm", absorb="chi2")  # chi2_scale defaults to 0


def test_subspace_without_scale_raises(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match=r"subspace"):
        betula_cluster.fit_predict(x, 4, method="gmm", absorb="subspace")


def _concentric_subspaces(n, d, k, rank, seed):
    """k clusters sharing one centre, each isotropic inside its own random rank-`rank` subspace.

    Every centroid coincides, so a gate that reads only the distance to a leaf's mean has no signal
    to work with and only the orientation of the leaf's own basis can separate them.
    """
    rng = np.random.default_rng(seed)
    blocks, truth = [], []
    for i in range(k):
        basis = np.linalg.qr(rng.standard_normal((d, rank)))[0].T
        blocks.append(
            rng.standard_normal((n // k, rank)) @ basis + 0.05 * rng.standard_normal((n // k, d))
        )
        truth.append(np.full(n // k, i))
    x = np.vstack(blocks)
    return (x - x.mean(0)) / (x.std(0) + 1e-12), np.concatenate(truth)


def _leaf_purity(truth, leaves):
    """Weighted fraction of points sharing their leaf's majority true class."""
    by_leaf = collections.defaultdict(collections.Counter)
    for t, leaf in zip(map(int, truth), map(int, leaves), strict=True):
        by_leaf[leaf][t] += 1
    return sum(c.most_common(1)[0][1] for c in by_leaf.values()) / len(truth)


def test_subspace_gate_beats_chi2_where_only_orientation_separates():
    x, truth = _concentric_subspaces(3000, 40, 3, 3, seed=0)
    got = {}
    for absorb in ("chi2", "subspace"):
        est = betula_cluster.Betula(
            n_clusters=3,
            feature="fd",
            method="kmeans",
            absorb=absorb,
            chi2_scale=0.01,
            max_leaves=300,
            seed=0,
        )
        est.fit(x)
        got[absorb] = (_leaf_purity(truth, est.assign_microclusters(x)), est.n_leaves_)
    (pure_chi2, n_chi2), (pure_sub, n_sub) = got["chi2"], got["subspace"]
    # A gate that merely splits more finely buys purity for free, so the counts have to stay close.
    assert n_sub <= 1.3 * n_chi2, f"subspace bought purity with leaves: {n_sub} vs {n_chi2}"
    assert pure_sub > pure_chi2, f"subspace {pure_sub:.4f} did not beat chi2 {pure_chi2:.4f}"


def test_mppca_separates_subspaces_a_diagonal_covariance_cannot():
    """The head's reason to exist, on the fixture that isolates it.

    Every cluster shares one centre and every per-dimension variance, so neither the centroid nor a
    diagonal covariance carries any signal — only the orientation of the subspace does. `gmm` has to
    fail here and `mppca` has to succeed, at the same leaves and the same seed.
    """
    x, truth = _concentric_subspaces(6000, 40, 3, 3, seed=0)
    got = {}
    for method, kw in (("gmm", {}), ("mppca", {"rank": 3})):
        est = betula_cluster.Betula(
            n_clusters=3, feature="fd", method=method, max_leaves=300, seed=0, **kw
        )
        got[method] = ari(est.fit_predict(x), truth)
    assert got["gmm"] < 0.3, f"diagonal ARI {got['gmm']:.4f}: fixture is not discriminating"
    assert got["mppca"] > 0.9, f"mppca ARI {got['mppca']:.4f}"


def test_mppca_rank_is_clamped_below_the_dimension(blobs):
    """`rank >= dim` would leave no isotropic residual for σ² to explain. The head clamps rather
    than erroring, so a caller who asks for more subspace than the data has still gets a fit."""
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x, 4, feature="fd", method="mppca", rank=64, threshold=0.05, max_leaves=300, seed=1
    )
    assert ari(labels, y) > 0.95


def test_mppca_rank_zero_is_a_spherical_mixture(blobs):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x, 4, feature="fd", method="mppca", rank=0, threshold=0.05, max_leaves=300, seed=1
    )
    assert ari(labels, y) > 0.95


@pytest.mark.parametrize("distance", ["euclidean", "manhattan", "ward", "average"])
def test_routing_distance_modes(blobs, distance):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x,
        4,
        feature="diagonal",
        method="gmm",
        distance=distance,
        threshold=0.05,
        max_leaves=300,
        seed=1,
    )
    assert ari(labels, y) > 0.9


def test_invalid_distance_raises(blobs):
    x, _ = blobs
    with pytest.raises(ValueError):
        betula_cluster.fit_predict(x, distance="bogus")


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_parallel_build_recovers_blobs(blobs, n_jobs):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x,
        4,
        feature="diagonal",
        method="gmm",
        threshold=0.05,
        max_leaves=300,
        seed=1,
        n_jobs=n_jobs,
    )
    assert ari(labels, y) > 0.95  # parallel shard+merge gives a valid summary, clusters recover


def test_streaming_decay_runs(blobs):
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=4,
        feature="diagonal",
        method="gmm",
        decay=0.9,
        threshold=0.05,
        max_leaves=300,
        seed=1,
    )
    for chunk in np.array_split(x, 4):
        est.partial_fit(chunk)
    est.partial_fit()
    assert ari(est.predict(x), y) > 0.9  # decay weights recent chunks; static blobs still recover


def test_hdbscan_separates_moons(moons):
    x, y = moons
    labels = betula_cluster.fit_predict(
        x,
        method="hdbscan",
        threshold=0.01,
        max_leaves=3000,
        min_samples=5,
        min_cluster_size=5,
    )
    assert n_labels(labels) >= 2
    assert ari(labels, y) > 0.9


def test_hdbscan_graph_degree_recovers_the_exact_partition(moons):
    """`graph_degree > 0` swaps the complete graph for a bounded-degree proximity graph. The
    approximation is in which edges the MST may choose from, not in the clustering criterion, so on
    a fixture with a clean density gap it has to land on the same partition."""
    x, y = moons
    kwargs = dict(method="hdbscan", threshold=0.01, max_leaves=3000, min_samples=5)
    exact = betula_cluster.fit_predict(x, min_cluster_size=5, **kwargs)
    graph = betula_cluster.fit_predict(x, min_cluster_size=5, graph_degree=16, seed=1, **kwargs)
    assert n_labels(graph) == n_labels(exact)
    assert ari(graph, y) > 0.9


def test_hdbscan_graph_degree_is_a_floor_not_a_ceiling(blobs):
    """The core distance is read off the graph, so a degree below what `min_samples` needs would
    truncate the walk and underestimate it. The head raises the request to `min_samples`/leaf-mass,
    which is observable: two different requests below the floor land on the same graph and the same
    labels, where an honoured request would not. A degree above the floor is honoured and
    differs."""
    x, y = blobs
    kwargs = dict(
        method="hdbscan",
        threshold=0.05,
        max_leaves=300,
        min_samples=25,
        min_cluster_size=25,
        seed=1,
    )
    one = betula_cluster.fit_predict(x, graph_degree=1, **kwargs)
    two = betula_cluster.fit_predict(x, graph_degree=2, **kwargs)
    wide = betula_cluster.fit_predict(x, graph_degree=64, **kwargs)
    assert np.array_equal(one, two)
    assert not np.array_equal(one, wide)
    assert ari(one, y) > 0.8  # the cheapest legal graph is still a clustering, not a degenerate one
    assert ari(wide, y) > 0.99  # and a wide one recovers what the exact complete graph finds


def test_hdbscan_graph_degree_roundtrips_through_get_params():
    est = betula_cluster.Betula(method="hdbscan", graph_degree=24)
    assert est.get_params()["graph_degree"] == 24
    assert betula_cluster.Betula(**est.get_params()).get_params()["graph_degree"] == 24
    assert betula_cluster.Betula().get_params()["graph_degree"] == 0  # default = exact


# ── streaming Betula estimator ───────────────────────────────────────────────────────────────


def test_streaming_partial_fit_matches_oneshot(blobs):
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )
    idx = np.random.default_rng(2).permutation(len(x))
    for chunk in np.array_split(idx, 5):
        est.partial_fit(x[chunk])
    est.partial_fit()  # finalize global clustering (sklearn-Birch style)
    labels = est.predict(x)
    assert ari(labels, y) > 0.95
    assert est.n_clusters_ == 4
    assert est.n_leaves_ > 0


def test_estimator_fit_predict_and_auto_k(blobs):
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=0, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )
    labels = est.fit_predict(x)
    assert ari(labels, y) > 0.95
    assert est.n_clusters_ == 4


def test_estimator_predict_on_new_points(blobs):
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )
    est.fit(x)
    held = x[::3]
    assert ari(est.predict(held), y[::3]) > 0.95


def test_streaming_with_chi2_absorption(blobs):
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=4,
        feature="diagonal",
        method="gmm",
        absorb="chi2",
        chi2_scale=0.5,
        max_leaves=300,
        seed=1,
    )
    for chunk in np.array_split(x, 5):
        est.partial_fit(chunk)
    est.partial_fit()
    assert ari(est.predict(x), y) > 0.95


def test_estimator_chi2_without_scale_raises(blobs):
    x, _ = blobs
    # sklearn convention: __init__ records params verbatim; validation fires when the engine builds.
    with pytest.raises(ValueError):
        betula_cluster.Betula(absorb="chi2").fit(x)  # chi2_scale defaults to 0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_streaming_dtype(blobs, dtype):
    x, y = blobs
    xd = x.astype(dtype)
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )
    for chunk in np.array_split(xd, 4):
        est.partial_fit(chunk)
    est.partial_fit()
    assert ari(est.predict(xd), y) > 0.95
    assert est.n_clusters_ == 4


def test_streaming_float32_matches_float64(blobs):
    x, _ = blobs
    out = {}
    for dtype in (np.float64, np.float32):
        est = betula_cluster.Betula(
            n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
        )
        out[dtype] = np.asarray(est.fit_predict(x.astype(dtype)))
    assert ari(out[np.float32], out[np.float64]) > 0.95  # f32 tree agrees with f64 on this data


def test_save_load_roundtrip(blobs, tmp_path):
    x, _ = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )
    est.fit(x)
    before = np.asarray(est.predict(x))
    path = str(tmp_path / "model.bin")
    est.save(path)
    loaded = betula_cluster.Betula.load(path)
    assert np.array_equal(before, np.asarray(loaded.predict(x)))
    assert loaded.n_clusters_ == est.n_clusters_


def test_a_snapshot_written_by_0_6_0_still_loads():
    """A committed snapshot from the released 0.6.0 wheel, loaded by the current build.

    `test_save_load_roundtrip` writes and reads with the same build, so it cannot see a schema
    break -- it would pass just as happily if `SCHEMA_VERSION` had been bumped and every older file
    rejected. `CFTree` gained a field after 0.6.0 (`merged_since_rebalance`, `#[serde(default)]`),
    and the claim that older snapshots survive it is only worth what a foreign-version file proves.

    Regenerate with `tests/data/gen_snapshot.py`, whose docstring carries the invocation.
    """
    import json
    from pathlib import Path

    data = Path(__file__).parent / "data"
    want = json.loads((data / "v2_0.6.0.json").read_text())
    blob = (data / "v2_0.6.0.betula").read_bytes()
    assert b"merged_since_rebalance" not in blob, (
        "the fixture already carries the field whose absence it exists to exercise, so it was "
        "written by a build at or after this one -- regenerate it from a released wheel, and run "
        "the generator outside the repository: `uv run --no-project` inside it still resolves the "
        "project's own .venv and silently snapshots the local build"
    )

    centres = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)]
    x = np.array(
        [[cx + i * 0.1, cy + j * 0.1] for cx, cy in centres for i in range(10) for j in range(10)],
        dtype=np.float64,
    )

    est = betula_cluster.Betula.load(str(data / "v2_0.6.0.betula"))

    assert est.n_clusters_ == want["n_clusters_"]
    assert est.n_leaves_ == want["n_leaves_"]
    assert np.asarray(est.cluster_sizes_).tolist() == want["cluster_sizes_"]
    assert np.allclose(np.asarray(est.cluster_centers_), np.asarray(want["cluster_centers_"]))
    assert np.array_equal(np.asarray(est.predict(x)), np.asarray(want["labels"]))


def test_pickle_roundtrip(blobs):
    import pickle

    x, _ = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="full", method="gmm-full", threshold=0.05, max_leaves=300, seed=1
    )
    est.fit(x)
    restored = pickle.loads(pickle.dumps(est))
    assert np.array_equal(np.asarray(est.predict(x)), np.asarray(restored.predict(x)))


# ── normalize (cosine geometry via L2-normalized rows) ───────────────────────────────────────────


@pytest.fixture(scope="module")
def direction_blobs():
    """Varying-norm vectors whose cluster signal is the *direction* (magnitude is noise)."""
    rng = np.random.default_rng(0)
    d, k, per = 32, 4, 300
    centers = rng.standard_normal((k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    xs, ys = [], []
    for c in range(k):
        dirs = centers[c] + 0.05 * rng.standard_normal((per, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        xs.append(rng.lognormal(0.0, 1.0, (per, 1)) * dirs)  # wide magnitude spread
        ys += [c] * per
    return np.vstack(xs).astype(np.float64), np.array(ys)


def test_normalize_recovers_direction_clusters(direction_blobs):
    x, y = direction_blobs
    kw = dict(feature="diagonal", method="kmeans", max_leaves=2000, seed=1)
    raw = betula_cluster.fit_predict(x, 4, normalize=False, **kw)
    nrm = betula_cluster.fit_predict(x, 4, normalize=True, **kw)
    assert ari(raw, y) < 0.5  # raw Euclidean is dominated by magnitude → fails
    assert ari(nrm, y) > 0.85  # normalizing onto the unit sphere recovers the direction clusters


def test_normalize_param_roundtrips():
    est = betula_cluster.Betula(n_clusters=4, normalize=True)
    assert est.get_params()["normalize"] is True
    assert betula_cluster.Betula(**est.get_params()).get_params()["normalize"] is True


def test_normalize_survives_save_load(direction_blobs, tmp_path):
    x, _ = direction_blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="kmeans", normalize=True, max_leaves=2000, seed=1
    )
    est.fit(x)
    before = np.asarray(est.predict(x))
    path = str(tmp_path / "model.bin")
    est.save(path)
    loaded = betula_cluster.Betula.load(path)
    assert loaded.get_params()["normalize"] is True  # persisted via the engine, recovered on load
    assert np.array_equal(before, np.asarray(loaded.predict(x)))  # same space ⇒ same labels


# ── inspectability (dataset structure) ──────────────────────────────────────────────────────────


def _fitted(blobs):
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )
    est.fit(x)
    return est, x, y


def test_microcluster_stats_shapes_and_mass(blobs):
    est, x, _ = _fitted(blobs)
    nlv = est.n_leaves_
    assert est.microcluster_centers_.shape == (nlv, x.shape[1])
    assert est.microcluster_weights_.shape == (nlv,)
    assert est.microcluster_radii_.shape == (nlv,)
    assert np.all(est.microcluster_radii_ >= 0)
    assert abs(est.microcluster_weights_.sum() - len(x)) < 1e-6  # mass conserved


def test_cluster_centers_recover_blob_centers(blobs):
    est, _, _ = _fitted(blobs)
    centers = est.cluster_centers_
    assert centers.shape == (4, 2)
    truth = np.array([[0, 0], [9, 0], [0, 9], [9, 9]], dtype=float)
    for t in truth:  # each true center has a recovered centroid nearby (order-independent)
        assert np.min(np.linalg.norm(centers - t, axis=1)) < 1.0


def test_outlier_scores_flag_injected_point(blobs):
    est, x, _ = _fitted(blobs)
    xo = np.vstack([x, [[100.0, 100.0]]])
    scores = est.outlier_scores(xo)
    assert scores.shape == (len(xo),)
    assert scores[-1] > np.percentile(scores[:-1], 99)


def _sheared_ribbon(seed=0, n=4000):
    """One cluster elongated along a direction that is *not* a coordinate axis."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, 1.0, (n, 2)) * np.array([6.0, 0.25])
    angle = np.deg2rad(37.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return pts @ rot.T, rot


def test_outlier_scores_mahalanobis_equals_radius_on_an_isotropic_cluster():
    # The 2^d hypercube corners have per-dimension variance exactly 1 and no cross-covariance, so
    # the pooled covariance is exactly (R^2/d)*I and the two metrics are the same number. This pins
    # the calibration: the whitened score is a refinement of the scalar one, not a second scale.
    cube = np.array(list(itertools.product([-1.0, 1.0], repeat=5)))
    est = betula_cluster.Betula(n_clusters=1, feature="full", threshold=0.9, seed=0).fit(cube)
    scalar = np.asarray(est.outlier_scores(cube, "radius"))
    whitened = np.asarray(est.outlier_scores(cube, "mahalanobis"))
    assert np.allclose(scalar, whitened, rtol=1e-5)  # the variance ridge is relative, at 1e-6


def test_outlier_scores_mahalanobis_separates_the_axes_the_radius_conflates():
    # Two probes the same Euclidean distance from the centroid: one along the cluster's long axis
    # (an ordinary member), one along its short axis (far outside it). A scalar RMS radius is the
    # trace of the covariance, so it cannot tell them apart; whitening by the covariance must.
    rows, rot = _sheared_ribbon()
    probes = np.array([[7.0, 0.0], [0.0, 7.0]]) @ rot.T
    est = betula_cluster.Betula(n_clusters=1, feature="full", threshold=0.5, seed=0).fit(rows)
    scalar = np.asarray(est.outlier_scores(probes, "radius"))
    whitened = np.asarray(est.outlier_scores(probes, "mahalanobis"))
    # The fitted centroid is not exactly the origin, so the two probes are equidistant only to
    # within the sampling error of the centre — 0.15% here, against a 24x separation when whitened.
    assert scalar[1] == pytest.approx(scalar[0], rel=1e-2)
    assert whitened[1] > 10.0 * whitened[0]


def test_outlier_scores_rejects_an_unknown_metric(blobs):
    est, x, _ = _fitted(blobs)
    with pytest.raises(ValueError, match="metric must be 'radius' or 'mahalanobis'"):
        est.outlier_scores(x, "euclidean")


def test_find_outliers_passes_the_metric_through():
    rows, rot = _sheared_ribbon()
    off_axis = np.array([[0.0, 7.0]]) @ rot.T
    xo = np.vstack([rows, off_axis])
    est = betula_cluster.Betula(n_clusters=1, feature="full", threshold=0.5, seed=0).fit(rows)
    assert est.find_outliers(xo, top_k=5, metric="mahalanobis")[0] == len(rows)


def test_summary_reports_structure(blobs):
    est, _, _ = _fitted(blobs)
    s = est.summary()
    assert s["n_samples"] == 2400
    assert s["n_clusters"] == 4
    assert s["n_microclusters"] == est.n_leaves_
    assert s["mean_microcluster_radius"] >= 0


def test_validity_scores_the_true_grouping_above_a_shuffled_one(blobs):
    x, _ = blobs
    kwargs = dict(feature="diagonal", method="kmeans", threshold=0.05, max_leaves=300, seed=1)
    good = betula_cluster.Betula(n_clusters=4, **kwargs).fit(x).validity()
    # Four clusters where there are four blobs, against sixteen splinters of the same data.
    split = betula_cluster.Betula(n_clusters=16, **kwargs).fit(x).validity()
    assert good["calinski_harabasz"] > split["calinski_harabasz"]
    assert good["davies_bouldin"] < split["davies_bouldin"]
    assert good["medoid_silhouette"] > split["medoid_silhouette"]
    assert good["medoid_silhouette"] <= 1.0


def test_validity_agrees_with_sklearn_on_the_indices_that_are_exact():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    x = np.vstack([rng.normal(c, 0.5, (300, 2)) for c in ([0, 0], [7, 0], [0, 7])])
    # threshold=0 with a leaf budget above N gives one leaf per point, so the leaf summary is the
    # data and the exact index must agree with sklearn's point-level one to floating-point noise.
    est = betula_cluster.Betula(
        n_clusters=3, feature="spherical", threshold=0.0, max_leaves=4000, seed=0
    )
    labels = est.fit_predict(x)
    got = est.validity()["calinski_harabasz"]
    want = sk.calinski_harabasz_score(x, labels)
    assert abs(got - want) < 1e-6 * want


def test_validity_requires_a_finalized_clustering(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=3, max_leaves=300)
    est.partial_fit(x)  # a tree, but no head has run over it yet
    with pytest.raises(ValueError, match="finalize first"):
        est.validity()


def test_summary_mmd_falls_as_the_leaf_budget_stops_throwing_data_away(blobs):
    x, _ = blobs
    kwargs = dict(feature="spherical", method="kmeans", n_clusters=4, threshold=0.0, seed=1)

    def mmd(max_leaves):
        est = betula_cluster.Betula(max_leaves=max_leaves, **kwargs)
        est.partial_fit(x)  # a tree is enough; the number is a property of the summary
        return est.summary_mmd(x, bandwidth=1.5)

    coarse, fine = mmd(16), mmd(1000)
    assert 0.0 <= fine < coarse, f"{fine} against {coarse}"


def test_summary_mmd_vanishes_when_the_summary_kept_every_point(blobs):
    x, _ = blobs
    # threshold=0 with a budget above N is one leaf per point: the surrogate *is* the sample.
    est = betula_cluster.Betula(
        n_clusters=4, feature="spherical", threshold=0.0, max_leaves=4000, seed=0
    )
    est.partial_fit(x)
    assert est.summary_mmd(x, bandwidth=2.0) < 1e-9


def test_summary_mmd_defaults_to_the_median_heuristic_and_needs_no_labels(blobs):
    est, x, _ = _fitted(blobs)
    auto = est.summary_mmd(x)
    assert np.isfinite(auto)
    assert auto != est.summary_mmd(x, bandwidth=0.05)


def test_summary_mmd_rejects_a_sample_of_the_wrong_width(blobs):
    est, x, _ = _fitted(blobs)
    with pytest.raises(ValueError, match="columns but the summary"):
        est.summary_mmd(np.hstack([x, x]))


def test_summary_mmd_requires_a_tree(blobs):
    est = betula_cluster.Betula(n_clusters=3, max_leaves=300)
    with pytest.raises(AttributeError, match="not fitted yet"):
        est.summary_mmd(blobs[0])


def test_find_outliers_returns_injected(blobs):
    est, x, _ = _fitted(blobs)
    xo = np.vstack([x, [[100.0, 100.0]]])
    out = est.find_outliers(xo, top_k=5)
    assert len(out) == 5
    assert len(xo) - 1 in set(out.tolist())
    # scores must come back in descending order (the injected outlier is the most extreme → first)
    scores = np.asarray(est.outlier_scores(xo))
    assert list(scores[out]) == sorted(scores[out], reverse=True)
    assert out[0] == len(xo) - 1
    assert est.find_outliers(xo, top_k=0).size == 0  # empty top-k → empty result


def test_sample_representatives(blobs):
    est, x, _ = _fitted(blobs)
    reps = est.sample_representatives(x, k=3)
    assert set(reps) == {0, 1, 2, 3}
    assert all(len(idx) == 3 for idx in reps.values())


def test_find_near_duplicates(blobs):
    x, _ = blobs
    dup = np.repeat([[50.0, 50.0]], 6, axis=0)  # 6 identical points, isolated from the blobs
    xd = np.vstack([x, dup]).astype(np.float64)
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=400, seed=1
    )
    est.fit(xd)
    groups = est.find_near_duplicates(xd, radius=0.1)
    dup_idx = set(range(len(x), len(xd)))
    assert any(dup_idx.issubset(set(g.tolist())) for g in groups)


def test_near_duplicate_pairs(blobs):
    from itertools import combinations

    x, _ = blobs
    dup = np.repeat([[50.0, 50.0]], 4, axis=0)  # 4 identical points, isolated → one microcluster
    xd = np.vstack([x, dup]).astype(np.float64)
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=400, seed=1
    ).fit(xd)

    pairs = est.near_duplicate_pairs(xd, threshold=0.999)
    assert pairs.shape[1] == 3
    found = {(int(i), int(j)) for _, i, j in pairs}
    planted = set(combinations(range(len(x), len(xd)), 2))  # all 6 pairs among the 4 duplicates
    assert planted.issubset(found)
    assert pairs[:, 0].max() <= 1.0 + 1e-9  # cosine is bounded
    assert pairs[:, 0].min() >= 0.999  # everything returned clears the threshold
    assert (pairs[:, 1] < pairs[:, 2]).all()  # canonical i < j
    # ordered by similarity descending
    assert list(pairs[:, 0]) == sorted(pairs[:, 0], reverse=True)
    # an unreachable threshold yields an empty (0, 3) result
    assert est.near_duplicate_pairs(xd, threshold=1.01).shape == (0, 3)


def test_inspection_before_fit_raises():
    est = betula_cluster.Betula()
    with pytest.raises(AttributeError):
        _ = est.microcluster_centers_


def test_cluster_centers_before_finalize_raises(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, threshold=0.05, max_leaves=300)
    est.partial_fit(x)  # streamed but not finalized
    with pytest.raises(ValueError):
        _ = est.cluster_centers_


def test_all_inspection_accessors(blobs):
    est, x, _ = _fitted(blobs)
    assert est.microcluster_weights_.shape == est.microcluster_radii_.shape
    assert est.cluster_radii_.shape[0] == est.cluster_centers_.shape[0]
    assert est.cluster_sizes_.shape[0] == est.cluster_centers_.shape[0]
    assert est.n_rebuilds_ >= 0
    assert est.threshold_ >= 0.0
    assert est.assign_microclusters(x).shape == (len(x),)
    assert "Betula(" in repr(est)  # exercises __repr__


def test_unfitted_accessors_raise():
    est = betula_cluster.Betula()
    for attr in ("n_clusters_", "n_leaves_", "n_rebuilds_", "threshold_"):
        with pytest.raises(AttributeError):
            getattr(est, attr)
    with pytest.raises(ValueError):
        est.save("/tmp/betula_never_written.bin")  # raises before writing


# ── mass-balanced leaf budget (`balance`) ────────────────────────────────────────────────────────


def _dense_core_with_a_diffuse_halo(n=6000, seed=0):
    """80% of the mass in a core an order of magnitude tighter than the rest.

    This is the shape a single global absorption radius cannot serve: the radius that bounds the
    leaf count is already wider than the core, so the core lands in one leaf while the halo keeps
    splitting.
    """
    rng = np.random.default_rng(seed)
    core = rng.normal(0.0, 0.05, (int(0.8 * n), 4))
    halo = rng.normal(0.0, 5.0, (n - int(0.8 * n), 4))
    return np.vstack([core, halo]).astype(np.float64)


def _heaviest_leaf_share(balance):
    est = betula_cluster.Betula(
        n_clusters=4,
        feature="spherical",
        method="kmeans",
        threshold=0.0,
        max_leaves=200,
        seed=0,
        balance=balance,
    )
    est.fit(_dense_core_with_a_diffuse_halo())
    w = np.asarray(est.microcluster_weights_, dtype=np.float64)
    return float(w.max() / w.sum())


def test_balance_bounds_the_share_of_mass_one_leaf_may_hold():
    plain = _heaviest_leaf_share(None)
    capped = _heaviest_leaf_share(4.0)
    assert plain > 0.5, "the fixture must collapse without the cap, or it tests nothing"
    # 200 leaves ⇒ an ideal share of 1/200; `balance=4` allows four times that.
    assert capped <= 4.0 / 200.0
    assert capped < plain


def test_balance_leaves_the_budget_a_hard_bound():
    """`max_leaves` outranks the cap: an unreachable balance must not push the tree over budget."""
    est = betula_cluster.Betula(
        n_clusters=4, method="kmeans", threshold=0.0, max_leaves=200, seed=0, balance=0.001
    )
    est.fit(_dense_core_with_a_diffuse_halo())
    assert est.n_leaves_ <= 200


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_balance_nonpositive_raises(blobs, bad):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, balance=bad)
    with pytest.raises(ValueError):
        est.fit(x)


def test_balance_param_roundtrips():
    est = betula_cluster.Betula(n_clusters=4, balance=4.0)
    assert est.get_params()["balance"] == 4.0
    assert betula_cluster.Betula(**est.get_params()).get_params()["balance"] == 4.0
    assert (
        betula_cluster.Betula().get_params()["balance"] is None
    )  # default is the geometric budget


def test_balance_reaches_the_one_shot_path():
    x = _dense_core_with_a_diffuse_halo()
    kwargs = dict(n_clusters=4, feature="spherical", method="kmeans", max_leaves=200, seed=0)
    plain = betula_cluster.fit_predict(x, **kwargs)
    capped = betula_cluster.fit_predict(x, balance=4.0, **kwargs)
    assert not np.array_equal(plain, capped)


# ── robust CF (Huber / winsorized insertion) ─────────────────────────────────────────────────────


def test_huber_k_caps_absorbed_outlier_pull(blobs):
    # Winsorization's guarantee: an extreme point folded into a mature microcluster has its pull on
    # the centroid capped at the cluster scale. A huge threshold forces one microcluster, so the
    # metric is exactly that centroid: unclipped the outlier drags it, with `huber_k` it is clamped
    # to ~k·σ and the centroid barely moves.
    del blobs
    rng = np.random.default_rng(3)
    tight = rng.normal([0.0, 0.0], 0.3, (200, 2))
    data = np.vstack([tight, [[20.0, 0.0]]]).astype(np.float64)  # one far point on the +x axis

    def centroid_x(huber_k):
        est = betula_cluster.Betula(
            n_clusters=1,
            feature="diagonal",
            method="kmeans",
            threshold=1e6,
            seed=1,
            huber_k=huber_k,
        )
        est.fit(data)
        w = est.microcluster_weights_
        return abs(float(est.microcluster_centers_[int(np.argmax(w))][0]))

    plain = centroid_x(None)
    robust = centroid_x(2.0)
    assert robust < plain  # clipped outlier pulls the centroid far less
    assert robust < 0.05 < plain  # robust ≈ k·σ/n; plain ≈ 20/n


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_huber_k_nonpositive_raises(blobs, bad):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, huber_k=bad)
    with pytest.raises(ValueError):
        est.fit(x)


def test_huber_k_param_roundtrips():
    est = betula_cluster.Betula(n_clusters=4, huber_k=2.5)
    assert est.get_params()["huber_k"] == 2.5
    assert betula_cluster.Betula(**est.get_params()).get_params()["huber_k"] == 2.5
    assert betula_cluster.Betula().get_params()["huber_k"] is None  # default disables it


def test_huber_k_survives_save_load(blobs, tmp_path):
    x, _ = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="kmeans", threshold=0.05, seed=1, huber_k=2.0
    )
    est.fit(x)
    path = str(tmp_path / "robust.bin")
    est.save(path)
    loaded = betula_cluster.Betula.load(path)
    assert loaded.get_params()["huber_k"] == 2.0  # persisted via the engine, recovered on load


# ── constrained clustering (must-link / cannot-link, COP-KMeans) ──────────────────────────────────


def _cop(**kw):
    params = dict(n_clusters=4, feature="diagonal", method="kmeans", threshold=0.0, seed=1)
    params.update(kw)
    return betula_cluster.Betula(**params)


def test_must_link_puts_points_in_same_cluster(blobs):
    # Rows 0 and 600 sit in different blobs; a must-link forces them into one cluster regardless.
    x, _ = blobs
    est = _cop().fit(x, must_link=[(0, 600)])
    labels = np.asarray(est.predict(x))
    assert labels[0] == labels[600]


def test_cannot_link_splits_points(blobs):
    # Rows 0 and 1 are both in blob 0 (same cluster unconstrained); a cannot-link forces them apart.
    x, _ = blobs
    plain = np.asarray(_cop().fit(x).predict(x))
    assert plain[0] == plain[1]
    est = _cop().fit(x, cannot_link=[(0, 1)])
    labels = np.asarray(est.predict(x))
    assert labels[0] != labels[1]


def test_unconstrained_path_unaffected_by_none(blobs):
    # Passing must_link=None / cannot_link=None must reproduce the plain fit exactly.
    x, y = blobs
    a = np.asarray(_cop().fit(x).predict(x))
    b = np.asarray(_cop().fit(x, must_link=None, cannot_link=None).predict(x))
    assert np.array_equal(a, b)
    assert ari(a, y) > 0.95


def test_fit_predict_honours_constraints(blobs):
    x, _ = blobs
    labels = np.asarray(_cop().fit_predict(x, must_link=[(0, 600)]))
    assert labels.shape == (len(x),)
    assert labels[0] == labels[600]


def test_constraints_accept_ndarray_pairs(blobs):
    x, _ = blobs
    ml = np.array([[0, 600]], dtype=np.int64)
    est = _cop().fit(x, must_link=ml)
    labels = np.asarray(est.predict(x))
    assert labels[0] == labels[600]


def test_constraints_require_kmeans(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match="kmeans"):
        _cop(method="gmm").fit(x, must_link=[(0, 1)])


def test_same_microcluster_cannot_link_raises():
    # Two identical rows collapse into one microcluster (threshold 0 absorbs only exact duplicates),
    # so a cannot-link between them is infeasible at the microcluster granularity.
    x = np.array([[0.0, 0.0], [0.0, 0.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="same microcluster"):
        _cop(n_clusters=2).fit(x, cannot_link=[(0, 1)])


def test_infeasible_constraints_raise():
    # Three mutually cannot-linked points need three clusters; n_clusters=2 cannot satisfy them.
    x = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="infeasible"):
        _cop(n_clusters=2).fit(x, cannot_link=[(0, 1), (0, 2), (1, 2)])


def test_constraint_shape_validation(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match=r"\(m, 2\)"):
        _cop().fit(x, must_link=[(0, 1, 2)])


def test_constraint_row_index_out_of_range(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match="out of range"):
        _cop().fit(x, must_link=[(0, 10**9)])


def test_sparse_with_constraints_raises(blobs):
    sp = pytest.importorskip("scipy.sparse")
    x, _ = blobs
    with pytest.raises(ValueError, match="dense"):
        _cop().fit(sp.csr_matrix(x), must_link=[(0, 1)])


# ── mixed numeric + categorical clustering (k-prototypes) ─────────────────────────────────────────


@pytest.fixture(scope="module")
def mixed():
    """Two clusters: numeric blobs (cols 0,1) each with a distinct dominant category (col 2)."""
    rng = np.random.default_rng(0)
    a = np.c_[rng.normal([0, 0], 0.4, (150, 2)), np.zeros(150)]
    b = np.c_[rng.normal([8, 8], 0.4, (150, 2)), np.ones(150)]
    x = np.vstack([a, b]).astype(np.float64)
    y = np.array([0] * 150 + [1] * 150)
    return x, y


def test_kprototypes_recovers_mixed_blobs(mixed):
    x, y = mixed
    kp = betula_cluster.KPrototypes(n_clusters=2, categorical=[2], seed=1)
    labels = np.asarray(kp.fit_predict(x))
    assert ari(labels, y) > 0.95
    assert kp.n_clusters_ == 2
    assert kp.cluster_centroids_.shape == (2, 2)  # two numeric dims
    assert kp.cluster_modes_.shape == (2, 1)  # one categorical dim


def test_kprototypes_categorical_breaks_numeric_tie():
    # Numerically coincident points; only the categorical attribute separates the two groups.
    n = 100
    x = np.c_[np.zeros((2 * n, 1)), np.array([0] * n + [1] * n, dtype=float)]
    y = np.array([0] * n + [1] * n)
    kp = betula_cluster.KPrototypes(n_clusters=2, categorical=[1], gamma=1.0, seed=2)
    # The numeric part is a single point, so the summary is exactly two micro-clusters -- one per
    # requested cluster. That trips the leaf-budget floor, correctly: the partition here is entirely
    # the summary's and the head has no freedom left. It happens to be the right answer anyway.
    with pytest.warns(UserWarning, match=r"leaves per cluster"):
        labels = np.asarray(kp.fit_predict(x))
    assert ari(labels, y) > 0.99


def test_kprototypes_predict_on_new_points(mixed):
    x, y = mixed
    kp = betula_cluster.KPrototypes(n_clusters=2, categorical=[2], seed=1).fit(x)
    held = x[::3]
    assert ari(np.asarray(kp.predict(held)), y[::3]) > 0.95


def test_kprototypes_get_params_roundtrip():
    kp = betula_cluster.KPrototypes(n_clusters=3, categorical=[0, 2], gamma=0.7)
    params = kp.get_params()
    assert params["categorical"] == [0, 2]
    assert params["gamma"] == 0.7
    clone = betula_cluster.KPrototypes(**params)
    assert clone.get_params()["categorical"] == [0, 2]
    assert clone.set_params(n_clusters=5).get_params()["n_clusters"] == 5


def test_kprototypes_requires_categorical(mixed):
    x, _ = mixed
    with pytest.raises(ValueError, match="categorical column"):
        betula_cluster.KPrototypes(n_clusters=2, categorical=[]).fit(x)


def test_kprototypes_requires_numeric():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="numeric column"):
        betula_cluster.KPrototypes(n_clusters=2, categorical=[0, 1]).fit(x)


def test_kprototypes_cat_index_out_of_range(mixed):
    x, _ = mixed
    with pytest.raises(ValueError, match="out of range"):
        betula_cluster.KPrototypes(n_clusters=2, categorical=[5]).fit(x)


@pytest.mark.parametrize("bad", [-1.0, 0.5])
def test_kprototypes_bad_codes_raise(bad):
    x = np.array([[0.0, 0.0], [1.0, bad], [2.0, 1.0]], dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative integer"):
        betula_cluster.KPrototypes(n_clusters=2, categorical=[1]).fit(x)


def test_kprototypes_predict_dim_mismatch(mixed):
    x, _ = mixed
    kp = betula_cluster.KPrototypes(n_clusters=2, categorical=[2], seed=1).fit(x)
    with pytest.raises(ValueError, match="dimension mismatch"):
        kp.predict(x[:, :2])


def test_kprototypes_unfitted_raises():
    kp = betula_cluster.KPrototypes(n_clusters=2, categorical=[2])
    with pytest.raises(AttributeError):
        _ = kp.n_clusters_
    with pytest.raises(AttributeError):
        kp.predict(np.zeros((2, 3)))


def test_kprototypes_gamma_override_and_repr(mixed):
    x, y = mixed
    kp = betula_cluster.KPrototypes(n_clusters=2, categorical=[2], gamma=5.0, seed=1)
    labels = np.asarray(kp.fit_predict(x))
    assert ari(labels, y) > 0.95
    assert "KPrototypes" in repr(kp)


def test_kprototypes_set_params_invalid():
    kp = betula_cluster.KPrototypes()
    with pytest.raises(ValueError, match="Invalid parameter"):
        kp.set_params(bogus=1)


# ── error contract ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [{"method": "bogus"}, {"feature": "bogus"}, {"absorb": "bogus"}],
)
def test_invalid_option_raises(blobs, kwargs):
    x, _ = blobs
    with pytest.raises(ValueError):
        betula_cluster.fit_predict(x, **kwargs)


def test_empty_array_raises():
    with pytest.raises(ValueError):
        betula_cluster.fit_predict(np.empty((0, 3)))


def test_integer_array_raises(blobs):
    x, _ = blobs
    with pytest.raises((ValueError, TypeError)):
        betula_cluster.fit_predict(x.astype(np.int64))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_input_raises(blobs, bad):
    x, _ = blobs
    xb = x.copy()
    xb[0, 0] = bad
    with pytest.raises(ValueError):
        betula_cluster.fit_predict(xb, 4)


def test_nonfinite_streaming_raises(blobs):
    x, _ = blobs
    xb = x.copy()
    xb[5, 1] = np.nan
    with pytest.raises(ValueError):
        betula_cluster.Betula(n_clusters=4).partial_fit(xb)


# ── scikit-learn parameter protocol ──────────────────────────────────────────────────────────


def test_get_params_returns_constructor_args():
    est = betula_cluster.Betula(n_clusters=4, feature="diagonal", method="gmm", threshold=0.05)
    p = est.get_params()
    assert p["n_clusters"] == 4
    assert p["feature"] == "diagonal"
    assert p["method"] == "gmm"
    assert p["threshold"] == 0.05
    # round-trips through the constructor (what sklearn.clone relies on)
    assert betula_cluster.Betula(**p).get_params() == p


def test_set_params_updates_and_refits(blobs):
    x, y = blobs
    est = betula_cluster.Betula(n_clusters=2, threshold=0.05, max_leaves=300, seed=1)
    assert est.set_params(n_clusters=4, feature="diagonal", method="gmm") is est
    assert est.get_params()["n_clusters"] == 4
    assert ari(est.fit_predict(x), y) > 0.95


def test_set_params_invalid_key_raises():
    with pytest.raises(ValueError):
        betula_cluster.Betula().set_params(bogus=1)


def test_set_params_invalid_value_raises(blobs):
    x, _ = blobs
    est = betula_cluster.Betula().set_params(method="nope")  # recorded verbatim, not yet validated
    with pytest.raises(ValueError):
        est.fit(x)  # invalid value rejected when the engine builds


def test_sklearn_pipeline_smoke(blobs):
    pipeline = pytest.importorskip("sklearn.pipeline")
    pre = pytest.importorskip("sklearn.preprocessing")
    x, y = blobs
    pipe = pipeline.Pipeline(
        [
            ("scale", pre.StandardScaler()),
            (
                "cluster",
                betula_cluster.Betula(
                    n_clusters=4, feature="diagonal", method="gmm", max_leaves=300, seed=1
                ),
            ),
        ]
    )
    labels = pipe.fit_predict(x)
    assert ari(labels, y) > 0.9
    pipe.set_params(cluster__n_clusters=4)  # nested param access via the estimator's get/set_params


def test_sklearn_clone_roundtrip(blobs):
    base = pytest.importorskip("sklearn.base")
    x, y = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )
    cloned = base.clone(est)
    assert cloned.get_params() == est.get_params()
    assert ari(cloned.fit_predict(x), y) > 0.95


def test_predict_before_fit_raises(blobs):
    x, _ = blobs
    with pytest.raises(ValueError):
        betula_cluster.Betula().predict(x)


def test_predict_dim_mismatch_raises(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, threshold=0.05, max_leaves=300)
    est.fit(x)
    with pytest.raises(ValueError):
        est.predict(x[:, :1])


# ── Mapper topology ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def dumbbell():
    """Two dense 2-D blobs joined by a thin bridge — a clear topological bottleneck."""
    rng = np.random.default_rng(3)
    a = rng.normal([0.0, 0.0], 0.3, (600, 2))
    b = rng.normal([10.0, 0.0], 0.3, (600, 2))
    neck = np.c_[np.linspace(1.5, 8.5, 40), np.zeros(40)] + rng.normal(0, 0.05, (40, 2))
    return np.vstack([a, neck, b]).astype(np.float64)


def _mapped(data):
    est = betula_cluster.Betula(
        feature="spherical", method="hdbscan", threshold=0.0, max_leaves=300
    ).fit(data)
    return est, est.mapper(lens="coordinate", coordinate=0, resolution=8, gain=0.4, link_scale=3.0)


def test_mapper_coordinate_lens_finds_bridge(dumbbell):
    _est, g = _mapped(dumbbell)
    assert g.n_nodes >= 3
    assert g.n_edges >= 2
    assert g.node_centroids.shape == (g.n_nodes, 2)
    assert len(g.bridges) >= 1  # the neck between the blobs is a bridge
    assert np.all(g.bridges < g.n_edges)  # bridges index valid edges
    # CF-aware edge overlap: Bhattacharyya coefficient per edge, in [0, 1]; the bridge across the
    # sparse neck has lower distributional overlap than the densest within-blob edge.
    assert g.edge_overlap.shape == (g.n_edges,)
    assert np.all((g.edge_overlap >= 0.0) & (g.edge_overlap <= 1.0))
    assert g.edge_overlap[g.bridges].min() < g.edge_overlap.max()


def test_mapper_persistence_diagram(dumbbell):
    _est, g = _mapped(dumbbell)
    # both filtrations: one (birth, death) per node, finite classes on/above the diagonal
    for filt in ("overlap", "lens"):
        d = g.persistence(filt)
        assert d.shape == (g.n_nodes, 2)
        fin = d[np.isfinite(d[:, 1])]
        assert np.all(fin[:, 1] >= fin[:, 0] - 1e-9)
    # essentials (= connected components) carry inf; finite_only drops them.
    overlap = g.persistence("overlap")
    assert np.isinf(overlap[:, 1]).sum() >= 1
    finite = g.persistence("overlap", finite_only=True)
    assert finite.shape[0] >= 1
    assert np.all(np.isfinite(finite[:, 1]))
    # the dominant finite bar ranks the neck: its death is the sparsest bridge's Bhattacharyya gap.
    assert abs(finite[:, 1].max() - (1.0 - g.edge_overlap[g.bridges].min())) < 1e-9
    with pytest.raises(ValueError):
        g.persistence("nonsense")


@pytest.mark.parametrize("lens", ["density", "radius", "l2norm", "coordinate", "eccentricity"])
def test_mapper_lenses_run_and_conserve_mass(blobs, lens):
    est, x, _ = _fitted(blobs)
    g = est.mapper(lens=lens, resolution=6, gain=0.3)
    assert g.n_nodes == len(g.node_members) == g.node_mass.shape[0] == g.node_bin.shape[0]
    assert g.node_centroids.shape == (g.n_nodes, x.shape[1])
    assert g.edges.shape[1] == 3
    w = est.microcluster_weights_
    for members, mass in zip(g.node_members, g.node_mass, strict=True):
        assert np.all(members < est.n_leaves_)  # members are valid microcluster indices
        assert abs(w[members].sum() - mass) < 1e-6  # node mass == sum of its microclusters


def test_mapper_branch_points_have_high_degree(dumbbell):
    _est, g = _mapped(dumbbell)
    deg = np.zeros(g.n_nodes, dtype=int)
    for a, b, _w in g.edges:
        deg[a] += 1
        deg[b] += 1
    assert all(deg[i] >= 3 for i in g.branch_points)


def test_mapper_to_networkx_round_trips(dumbbell):
    nx = pytest.importorskip("networkx")
    _est, g = _mapped(dumbbell)
    graph = g.to_networkx()
    assert graph.number_of_nodes() == g.n_nodes
    assert graph.number_of_edges() == g.n_edges
    n_bridge_edges = sum(1 for _a, _b, d in graph.edges(data=True) if d["bridge"])
    assert n_bridge_edges == len(g.bridges)
    assert all(0.0 <= d["overlap"] <= 1.0 for _a, _b, d in graph.edges(data=True))
    assert isinstance(nx.Graph(), type(graph))


def test_mapper_before_fit_raises():
    with pytest.raises(AttributeError):
        betula_cluster.Betula().mapper()


def test_mapper_invalid_lens_raises(blobs):
    est, _, _ = _fitted(blobs)
    with pytest.raises(ValueError):
        est.mapper(lens="nonsense")


def test_mapper_coordinate_out_of_range_raises(blobs):
    est, _, _ = _fitted(blobs)
    with pytest.raises(ValueError):
        est.mapper(lens="coordinate", coordinate=99)


def test_mapper_stability_persistence_curve():
    rng = np.random.default_rng(0)
    t = rng.uniform(0, 2 * np.pi, 5000)
    x = (np.c_[3.0 * np.cos(t), 3.0 * np.sin(t)] + 0.18 * rng.standard_normal((5000, 2))).astype(
        np.float64
    )
    est = betula_cluster.Betula(
        feature="spherical", method="hdbscan", threshold=0.0, max_leaves=220
    ).fit(x)
    keys = {
        "resolution",
        "n_nodes",
        "n_edges",
        "n_branch_points",
        "n_bridges",
        "n_components",
        "n_loops",
    }

    kw = dict(lens="coordinate", coordinate=0, gain=0.4, link_scale=2.5, min_node_mass=20)
    rows = est.mapper_stability(resolutions=[8, 12, 16], **kw)
    assert len(rows) == 3
    assert all(set(r) == keys for r in rows)
    assert all(r["n_components"] >= 1 and r["n_loops"] >= 0 for r in rows)
    # a ring carries a persistent loop (β₁ == 1) — its closing edge exercises the cycle branch
    assert max(r["n_loops"] for r in rows) >= 1

    # default resolution sweep runs and returns one row per resolution
    assert len(est.mapper_stability(**kw)) == len(range(4, 30, 2))


def test_mapper_stability_before_fit_raises():
    with pytest.raises(AttributeError):
        betula_cluster.Betula().mapper_stability()


def test_mapper_stability_rejects_resolution_kwarg(blobs):
    est, _, _ = _fitted(blobs)
    with pytest.raises(ValueError):
        est.mapper_stability(resolution=5)  # `resolution` is swept; must use `resolutions=`


# ── coreset / soft assignment / diagnostics / representatives ─────────────────────────────────


def test_predict_proba_gmm_is_posterior(blobs):
    est, x, _ = _fitted(blobs)  # method="gmm"
    p = est.predict_proba(x)
    assert p.shape[0] == len(x)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(p.argmax(axis=1), np.asarray(est.predict(x)))
    pr = est.microcluster_proba_
    assert pr.shape[0] == est.n_leaves_
    assert np.allclose(pr.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_kmeans_heuristic_and_confidence(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="kmeans", threshold=0.05, max_leaves=300, seed=1
    ).fit(x)
    p = est.predict_proba(x)
    assert p.shape == (len(x), est.cluster_centers_.shape[0])
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
    c = est.assignment_confidence(x)
    assert c.shape == (len(x),)
    assert np.all((c >= 0.0) & (c <= 1.0))


def test_export_coreset_conserves_mass(blobs):
    est, x, _ = _fitted(blobs)
    cs = est.export_coreset()
    assert cs.centers.shape == (est.n_leaves_, x.shape[1])
    assert cs.weights.shape == (est.n_leaves_,)
    assert cs.radii.shape == (est.n_leaves_,)
    assert abs(cs.n_points - len(x)) < 1e-6


def test_diagnostics_reports_compression_and_clusters(blobs):
    est, x, _ = _fitted(blobs)
    d = est.diagnostics()
    assert d["n_samples"] == len(x)
    assert d["compression_ratio"] > 1.0
    assert d["n_clusters"] == est.n_clusters_
    assert d["cluster_mass_max"] >= d["cluster_mass_min"]
    assert d["microcluster_radius_p99"] >= d["microcluster_radius_p50"] >= 0.0


def test_diagnostics_before_finalize_omits_cluster_block(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, threshold=0.05, max_leaves=300).partial_fit(x)
    d = est.diagnostics()
    assert "n_microclusters" in d
    assert "n_clusters" not in d


@pytest.mark.parametrize("method", ["medoid", "boundary", "outlier", "diverse"])
def test_representatives_are_cluster_members(blobs, method):
    est, x, _ = _fitted(blobs)
    reps = est.representatives(x, 0, method=method, k=5)
    assert 0 < len(reps) <= 5
    assert np.all(np.asarray(est.predict(x))[reps] == 0)


def test_representatives_empty_and_bad_method(blobs):
    est, x, _ = _fitted(blobs)
    assert est.representatives(x, 9999).size == 0  # out-of-range cluster → no members
    with pytest.raises(ValueError):
        est.representatives(x, 0, method="nope")


def test_cluster_profile_geometry(blobs):
    est, _, _ = _fitted(blobs)
    prof = est.cluster_profile(0)
    assert prof["cluster_id"] == 0
    assert prof["size"] > 0
    assert len(prof["center"]) == 2
    assert len(prof["nearest_clusters"]) >= 1
    assert all(np.isfinite(nc["distance"]) for nc in prof["nearest_clusters"])


# ── memory budget / drift / active learning ───────────────────────────────────────────────────


def test_memory_budget_controls_resolution(blobs):
    x, _ = blobs
    kw = dict(feature="spherical", method="kmeans", n_clusters=4, threshold=0.0)
    small = betula_cluster.Betula(memory_budget_mb=0.05, **kw).fit(x)
    big = betula_cluster.Betula(memory_budget_mb=50.0, **kw).fit(x)
    assert small.effective_max_leaves_ < big.effective_max_leaves_
    assert small.n_leaves_ <= small.effective_max_leaves_


def test_memory_budget_none_uses_configured_max_leaves(blobs):
    est, _, _ = _fitted(blobs)  # no budget → effective == configured max_leaves
    assert est.effective_max_leaves_ == est.max_leaves


def test_memory_budget_helper_clamps_and_scales():
    grow = betula_cluster._budget_max_leaves
    assert grow(1e-6, 10, "spherical", 32) == 33  # floor at branching + 1
    assert grow(10.0, 10, "spherical", 32) > grow(1.0, 10, "spherical", 32)


def test_memory_budget_clone_roundtrip():
    base = pytest.importorskip("sklearn.base")
    est = betula_cluster.Betula(memory_budget_mb=128.0)
    assert base.clone(est).get_params()["memory_budget_mb"] == 128.0


def test_snapshot_and_compare(blobs):
    est, x, _ = _fitted(blobs)
    s1 = est.snapshot()
    assert s1["n_clusters"] == est.n_clusters_
    same = betula_cluster.Betula.compare_snapshots(s1, s1)  # identical → ~zero drift
    assert same["n_clusters_before"] == same["n_clusters_after"]
    assert same["max_centroid_shift_radii"] == pytest.approx(0.0, abs=1e-9)
    assert len(same["matches"]) == s1["n_clusters"]
    assert all(np.isfinite(m["mass_ratio"]) for m in same["matches"])
    shifted = betula_cluster.Betula(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    ).fit(x + 5.0)
    drift = betula_cluster.Betula.compare_snapshots(s1, shifted.snapshot())
    assert drift["max_centroid_shift_radii"] > 0.0


@pytest.mark.parametrize("strategy", ["uncertain", "outlier"])
def test_active_learning_batch(blobs, strategy):
    est, x, _ = _fitted(blobs)
    idx = est.active_learning_batch(x, n=50, strategy=strategy)
    assert 0 < len(idx) <= 50
    assert np.all((idx >= 0) & (idx < len(x)))


def test_active_learning_bad_strategy(blobs):
    est, x, _ = _fitted(blobs)
    with pytest.raises(ValueError):
        est.active_learning_batch(x, strategy="nope")


# ── DenStream streaming density clusterer ─────────────────────────────────────────────────────


def test_denstream_recovers_blobs_streaming(blobs):
    x, y = blobs
    ds = betula_cluster.DenStream(eps=1.5, decay=0.001, beta=0.5, mu=4.0)
    for chunk in np.array_split(x, 5):
        ds.partial_fit(chunk)  # first chunk builds the engine, the rest stream into it
    labels = ds.predict(x)  # auto-finalizes the offline clustering on first predict
    assert ds.n_clusters_ >= 2
    assert ds.n_microclusters_ > 0
    mask = labels >= 0
    assert ari(labels[mask], y[mask]) > 0.9
    labels2 = ds.predict(x)  # already clustered → no re-cluster
    assert np.array_equal(labels, labels2)


def test_denstream_fit_predict_and_microcluster_shapes(blobs):
    x, _ = blobs
    ds = betula_cluster.DenStream(eps=1.5, decay=0.001, beta=0.5, mu=4.0)
    labels = ds.fit_predict(x)
    assert labels.shape == (len(x),)
    assert ds.microcluster_centers_.shape == (ds.n_microclusters_, x.shape[1])
    assert ds.microcluster_weights_.shape == (ds.n_microclusters_,)
    assert ds.microcluster_radii_.shape == (ds.n_microclusters_,)


def test_denstream_fit_and_explicit_cluster(blobs):
    x, _ = blobs
    ds = betula_cluster.DenStream(eps=1.5, decay=0.001, beta=0.5, mu=4.0).fit(x)
    assert ds.n_clusters_ >= 2
    ds.partial_fit(x[:100]).cluster()  # explicit streaming → cluster() path
    assert ds.n_microclusters_ > 0


def test_denstream_predict_before_fit_raises():
    with pytest.raises(AttributeError):
        betula_cluster.DenStream().predict(np.zeros((3, 2)))


def test_denstream_param_protocol():
    ds = betula_cluster.DenStream(eps=2.0)
    assert ds.get_params()["eps"] == 2.0
    ds.set_params(decay=0.5)
    assert ds.decay == 0.5
    with pytest.raises(ValueError):
        ds.set_params(nope=1)
    assert "DenStream(eps=" in repr(ds)


def test_denstream_clone_roundtrip():
    base = pytest.importorskip("sklearn.base")
    ds = betula_cluster.DenStream(eps=1.5, decay=0.1, beta=0.3, mu=8.0)
    assert base.clone(ds).get_params() == ds.get_params()


# ── DbStream streaming density clusterer (shared density) ─────────────────────────────────────────


def test_dbstream_recovers_blobs_streaming(blobs):
    x, y = blobs
    ds = betula_cluster.DbStream(r=1.5, decay=0.0005, alpha=0.1, min_weight=2.0)
    for chunk in np.array_split(x, 5):
        ds.partial_fit(chunk)
    labels = np.asarray(ds.predict(x))  # lazily finalizes the offline step
    assert ds.n_clusters_ == 4
    assigned = np.where(labels < 0, 0, labels)
    assert ari(assigned, y) > 0.9


def test_dbstream_fit_predict_and_microcluster_shapes(blobs):
    x, _ = blobs
    ds = betula_cluster.DbStream(r=1.5, decay=0.0005)
    labels = np.asarray(ds.fit_predict(x))
    assert labels.shape == (len(x),)
    nmc = ds.n_microclusters_
    assert ds.microcluster_centers_.shape == (nmc, 2)
    assert ds.microcluster_weights_.shape == (nmc,)
    assert ds.microcluster_radii_.shape == (nmc,)


def test_dbstream_shared_density_keeps_close_blobs_separate():
    # Two tight blobs whose centres are within 2r (a distance rule would merge them) but with an
    # empty gap between → zero shared density → DbStream keeps them as two clusters.
    rng = np.random.default_rng(4)
    a = rng.normal([0.0, 0.0], 0.25, (200, 2))
    b = rng.normal([2.6, 0.0], 0.25, (200, 2))
    x = np.vstack([a, b]).astype(np.float64)
    ds = betula_cluster.DbStream(r=1.5, decay=0.0005).fit(x)
    assert ds.n_clusters_ == 2


def test_dbstream_explicit_cluster(blobs):
    x, _ = blobs
    ds = betula_cluster.DbStream(r=1.5, decay=0.0005)
    ds.partial_fit(x).cluster()
    assert ds.n_clusters_ == 4


def test_dbstream_predict_before_fit_raises():
    with pytest.raises(AttributeError):
        betula_cluster.DbStream().predict(np.zeros((3, 2)))


def test_dbstream_param_protocol():
    ds = betula_cluster.DbStream(r=2.0)
    assert ds.get_params()["r"] == 2.0
    ds.set_params(alpha=0.2)
    assert ds.alpha == 0.2
    with pytest.raises(ValueError):
        ds.set_params(nope=1)
    assert "DbStream(r=" in repr(ds)


def test_dbstream_clone_roundtrip():
    base = pytest.importorskip("sklearn.base")
    ds = betula_cluster.DbStream(r=1.5, decay=0.1, alpha=0.2, min_weight=3.0)
    assert base.clone(ds).get_params() == ds.get_params()


# ── quantile sketches (betula-sketch) ─────────────────────────────────────────────────────────


def test_kll_sketch_rank_error_and_merge():
    rng = np.random.default_rng(5)
    x = rng.lognormal(0.0, 1.0, 80_000)  # skewed
    s = betula_cluster.KllSketch(k=400, seed=1)
    s.update_many(x)
    assert s.count == len(x)
    for q in (0.5, 0.9, 0.99):  # rank-error guarantee: true rank of the estimate ≈ q
        true_q = float((x <= s.quantile(q)).mean())
        assert abs(true_q - q) < 0.03
    a = betula_cluster.KllSketch(256, 1)
    b = betula_cluster.KllSketch(256, 2)
    a.update_many(np.arange(50_000, dtype=np.float64))
    b.update_many(np.arange(50_000, 100_000, dtype=np.float64))
    a.merge(b)
    assert a.count == 100_000
    assert abs(a.quantile(0.5) - 50_000) / 100_000 < 0.03


def test_kll_sketch_edges():
    s = betula_cluster.KllSketch()
    assert s.count == 0
    s.update(2.0)
    assert s.quantile(0.5) == 2.0
    assert s.rank(5.0) == 1


def test_ddsketch_relative_error_and_merge():
    rng = np.random.default_rng(7)
    x = rng.lognormal(0.0, 1.0, 80_000)  # positive, skewed → relative error shines
    s = betula_cluster.DdSketch(alpha=0.01)
    s.update_many(x)
    assert s.alpha == 0.01
    for q in (0.5, 0.9, 0.99):
        truth = float(np.quantile(x, q))
        assert abs(s.quantile(q) - truth) / truth <= 0.02
    a = betula_cluster.DdSketch(0.01)
    b = betula_cluster.DdSketch(0.01)
    a.update_many(x[: len(x) // 2])
    b.update_many(x[len(x) // 2 :])
    a.merge(b)
    assert a.count == len(x)


def test_ddsketch_errors():
    with pytest.raises(ValueError):
        betula_cluster.DdSketch(alpha=0.0)
    with pytest.raises(ValueError):
        betula_cluster.DdSketch(alpha=0.01).merge(betula_cluster.DdSketch(alpha=0.02))


# ── sparse CSR input ──────────────────────────────────────────────────────────────────────────


def _sparse_kw():
    return dict(
        n_clusters=4, feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1
    )


def test_sparse_fit_predict_matches_dense(blobs):
    sp = pytest.importorskip("scipy.sparse")
    x, y = blobs
    xs = sp.csr_matrix(x)
    dense = betula_cluster.Betula(**_sparse_kw()).fit_predict(x)
    sparse = betula_cluster.Betula(**_sparse_kw()).fit_predict(xs)
    assert np.array_equal(dense, sparse)  # the densify path is exact
    assert ari(sparse, y) > 0.9


def test_sparse_fit_then_predict(blobs):
    sp = pytest.importorskip("scipy.sparse")
    x, y = blobs
    xs = sp.csr_matrix(x)
    est = betula_cluster.Betula(**_sparse_kw()).fit(xs)
    assert est.n_clusters_ == 4
    assert est.microcluster_centers_.shape[1] == x.shape[1]
    assert ari(est.predict(xs), y) > 0.9


def test_sparse_streaming(blobs):
    sp = pytest.importorskip("scipy.sparse")
    x, y = blobs
    xs = sp.csr_matrix(x)
    est = betula_cluster.Betula(**_sparse_kw())
    for lo in range(0, x.shape[0], 600):
        est.partial_fit(xs[lo : lo + 600])
    est.partial_fit()  # finalize
    assert ari(est.predict(xs), y) > 0.9


def test_sparse_dim_mismatch_raises(blobs):
    sp = pytest.importorskip("scipy.sparse")
    x, _ = blobs
    est = betula_cluster.Betula(**_sparse_kw()).fit(sp.csr_matrix(x))
    wider = sp.csr_matrix(np.zeros((2, x.shape[1] + 5)))
    with pytest.raises(ValueError):
        est.predict(wider)


# ── O(nnz) sparse-native one-shot (fit_predict_sparse) ────────────────────────────────────────────


def _sparse_topics():
    """Two topics on disjoint high-dimensional feature blocks; returns (csr_matrix, labels)."""
    sp = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    d = 80
    rows = []
    for cols in ([0, 1, 2], [60, 61, 62]):
        for _ in range(150):
            r = np.zeros(d)
            for c in cols:
                r[c] = rng.random() + 0.5
            rows.append(r)
    y = np.array([0] * 150 + [1] * 150)
    return sp.csr_matrix(np.vstack(rows)), y


def test_fit_predict_sparse_recovers_topics():
    x, y = _sparse_topics()
    labels = np.asarray(
        betula_cluster.fit_predict_sparse(x, n_clusters=2, method="kmeans", threshold=0.5, seed=1)
    )
    assert labels.shape == (x.shape[0],)
    assert ari(labels, y) > 0.95


@pytest.mark.parametrize("method", ["gmm-full", "ward"])
def test_fit_predict_sparse_other_heads(method):
    x, y = _sparse_topics()
    labels = np.asarray(
        betula_cluster.fit_predict_sparse(x, n_clusters=2, method=method, threshold=0.5, seed=1)
    )
    assert ari(labels, y) > 0.9


def test_fit_predict_sparse_rejects_dense():
    with pytest.raises(ValueError, match="sparse"):
        betula_cluster.fit_predict_sparse(np.zeros((4, 4)))


def test_fit_predict_sparse_invalid_method():
    x, _ = _sparse_topics()
    for bad in ("hdbscan", "spectral", "leiden"):  # none is wired for the sparse O(nnz) path
        with pytest.raises(ValueError, match="method"):
            betula_cluster.fit_predict_sparse(x, method=bad)


# ── hyperparameter tuning (betula_cluster.tuning) ────────────────────────────────────────────────


def test_tune_metrics_match_reference_values():
    # Exact hand-computed values pin the metric *formulas* — 100% line coverage does not (a mutant
    # like between/(k-1) -> between*(k-1) preserves argmax and survives the argmax-only tune tests).
    # Two well-separated clusters of two points each; values verified equal to scikit-learn's.
    from betula_cluster.tuning import adjusted_rand, calinski_harabasz, davies_bouldin

    x = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 0.0], [10.0, 2.0]])
    labels = np.array([0, 0, 1, 1])
    # CH = (between/(k-1)) / (within/(N-k)) = (100/1) / (4/2) = 50
    assert abs(calinski_harabasz(x, labels) - 50.0) < 1e-9
    # DB = mean_i max_{j!=i} (s_i + s_j) / d_ij = 0.5 * (0.2 + 0.2) = 0.2
    assert abs(davies_bouldin(x, labels) - 0.2) < 1e-9
    # ARI = 1 under any label permutation of a perfect partition; ~0 for a crossed partition.
    assert abs(adjusted_rand(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) - 1.0) < 1e-9
    assert abs(adjusted_rand(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))) < 0.5


def test_dbcv_validates_density_partitions():
    pytest.importorskip("sklearn.datasets")  # make_moons has no clean NumPy one-liner
    from betula_cluster.tuning import dbcv
    from sklearn.datasets import make_blobs, make_moons

    x, y = make_blobs(n_samples=600, centers=3, cluster_std=0.4, random_state=0)
    # a correct dense partition scores well above a random one; both stay in [-1, 1].
    good = dbcv(x, y)
    bad = dbcv(x, np.random.default_rng(1).integers(0, 3, size=len(y)))
    assert -1.0 <= bad < good <= 1.0
    assert good > 0.3
    # non-convex moons: DBCV validates the correct shape (positive), unlike the convex metrics.
    xm, ym = make_moons(n_samples=400, noise=0.05, random_state=0)
    assert dbcv(xm, ym) > 0.0


def test_dbcv_edge_cases():
    from betula_cluster.tuning import dbcv

    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [0.2, 0.1],  # cluster 0
            [10.0, 10.0],
            [10.1, 10.0],  # cluster 1 (2 → MST-leaf fallback for DSC)
            [20.0, 20.0],  # cluster 2 (singleton → skipped)
            [30.0, 30.0],  # noise
        ]
    )
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, -1])
    assert -1.0 <= dbcv(x, labels) <= 1.0
    # fewer than two clusters → the worst score
    assert dbcv(x, np.zeros(len(x), dtype=int)) == -1.0
    assert dbcv(x, -np.ones(len(x), dtype=int)) == -1.0
    # subsampling path (n > sample_cap) is deterministic under a fixed seed
    big, by = _blobs_xl()
    assert dbcv(big, by, sample_cap=1500, seed=0) == dbcv(big, by, sample_cap=1500, seed=0)


def _blobs_xl():
    # NumPy-only blobs (no sklearn) so the subsampling-path test runs on the numpy-only CI matrix.
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [9, 0], [0, 9], [9, 9]], dtype=float)
    x = np.vstack([rng.normal(c, 0.5, (500, 2)) for c in centers])
    y = np.repeat(np.arange(4), 500)
    return x, y


def test_tune_dbcv_objective(blobs):
    x, _ = blobs
    result = betula_cluster.tune(x, n_clusters=4, objective="dbcv", n_trials=6, seed=0)
    assert result.best_score == max(t.score for t in result.trials)  # dbcv maximizes


def test_tune_random_returns_best(blobs):
    x, _ = blobs
    result = betula_cluster.tune(x, n_clusters=4, n_trials=6, seed=0)
    assert isinstance(result, betula_cluster.TuneResult)
    assert len(result.trials) == 6
    assert set(result.best_params) == {"max_leaves", "feature", "normalize"}
    assert np.isfinite(result.best_score)
    assert result.best_score == max(t.score for t in result.trials)  # calinski_harabasz maximizes


def test_tune_davies_bouldin_minimizes(blobs):
    x, _ = blobs
    result = betula_cluster.tune(x, n_clusters=4, objective="davies_bouldin", n_trials=6, seed=1)
    assert result.best_score == min(t.score for t in result.trials)


def test_tune_multi_objective_returns_pareto(blobs):
    x, _ = blobs
    result = betula_cluster.tune(x, n_clusters=4, n_trials=8, multi_objective=True, seed=2)
    assert result.pareto  # at least one non-dominated config
    assert all(p in result.trials for p in result.pareto)


def test_tune_ari_objective_with_labels(blobs):
    x, y = blobs
    result = betula_cluster.tune(x, n_clusters=4, y=y, objective="ari", n_trials=6, seed=0)
    assert -0.5 <= result.best_score <= 1.0


def test_tune_ari_without_labels_raises(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match="requires ground-truth"):
        betula_cluster.tune(x, n_clusters=4, objective="ari")


def test_tune_unknown_objective_raises(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match="unknown objective"):
        betula_cluster.tune(x, n_clusters=4, objective="silhouette")


def test_tune_unknown_sampler_raises(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match="unknown sampler"):
        betula_cluster.tune(x, n_clusters=4, sampler="grid")


def test_tune_custom_space_and_fixed(blobs):
    x, _ = blobs
    space = {"max_leaves": ("cat", [64, 128]), "normalize": ("cat", [False])}
    result = betula_cluster.tune(x, n_clusters=4, space=space, n_trials=4, seed=0, method="gmm")
    for tr in result.trials:
        assert tr.params["max_leaves"] in (64, 128)
        assert tr.params["normalize"] is False


def test_tune_metrics_and_degenerate():
    tuning = betula_cluster.tuning
    x = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    two = np.array([0, 0, 1, 1])
    assert tuning.calinski_harabasz(x, two) > 0
    assert tuning.davies_bouldin(x, two) >= 0
    one = np.zeros(4, dtype=int)  # a single cluster is unscoreable → sentinel
    assert tuning.calinski_harabasz(x, one) == float("-inf")
    assert tuning.davies_bouldin(x, one) == float("inf")


def test_tune_adjusted_rand():
    tuning = betula_cluster.tuning
    a = np.array([0, 0, 1, 1])
    assert tuning.adjusted_rand(a, a) == pytest.approx(1.0)
    assert tuning.adjusted_rand(a, np.array([0, 0, 0, 0])) == pytest.approx(0.0)
    # trivial identical partitions hit the maximum == expected fast path
    assert tuning.adjusted_rand(np.zeros(3, dtype=int), np.zeros(3, dtype=int)) == 1.0


def test_tune_internal_guards():
    tuning = betula_cluster.tuning
    with pytest.raises(ValueError, match="parameter spec"):
        tuning._sample(np.random.default_rng(0), ("linear", 1, 2))
    x = np.zeros((4, 2))
    assert tuning._score(x, np.zeros(4, dtype=int), None, "davies_bouldin") == float("inf")
    assert tuning._score(x, np.zeros(4, dtype=int), None, "calinski_harabasz") == float("-inf")


def test_tune_metric_extreme_scores():
    tuning = betula_cluster.tuning
    # perfectly tight clusters (zero within-cluster scatter) → Calinski-Harabasz is +inf (best)
    x = np.array([[0.0, 0.0], [0.0, 0.0], [9.0, 9.0], [9.0, 9.0]])
    assert tuning.calinski_harabasz(x, np.array([0, 0, 1, 1])) == float("inf")
    # two distinct clusters with coincident centroids → Davies-Bouldin is +inf (worst), never a
    # false 0.0 that would rank a bad clustering as perfect
    xdb = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
    assert tuning.davies_bouldin(xdb, np.array([0, 0, 1, 1])) == float("inf")


def test_tune_finalize_keeps_best_infinity():
    tuning = betula_cluster.tuning
    trial = tuning.Trial
    perfect = trial(params={"i": 1}, score=float("inf"), n_leaves=5, time_s=0.1)  # best CH
    okay = trial(params={"i": 2}, score=100.0, n_leaves=8, time_s=0.2)
    worst = trial(params={"i": 3}, score=float("-inf"), n_leaves=3, time_s=0.05)  # worst sentinel
    res = tuning._finalize([perfect, okay, worst], "calinski_harabasz", multi_objective=True)
    assert res.best_score == float("inf") and res.best_params == {"i": 1}  # was dropped pre-fix
    assert perfect in res.pareto and worst not in res.pareto
    # all-worst pool → fall back to the raw trials instead of an empty selection
    only_worst = tuning._finalize([worst], "calinski_harabasz", multi_objective=False)
    assert only_worst.best_params == {"i": 3}


def test_auto_threshold_small_data_is_noop(blobs):
    # n below the pilot cap: "auto" starts from zero exactly like the default, no double-fit.
    x, _ = blobs
    kw = dict(n_clusters=4, method="kmeans", seed=0)
    auto = betula_cluster.Betula(threshold="auto", **kw).fit(x)
    base = betula_cluster.Betula(threshold=0.0, **kw).fit(x)
    assert auto.get_params()["threshold"] == "auto"  # configured value kept verbatim (sklearn)
    assert auto._auto_threshold == 0.0  # pilot skipped
    assert auto.threshold_ == base.threshold_  # ⇒ identical resolved tree
    np.testing.assert_array_equal(auto.predict(x), base.predict(x))
    auto.fit(x)  # refit reuses the cached estimate (cache-hit path), still a no-op
    assert auto._auto_threshold == 0.0


def test_auto_threshold_warm_starts_large_n():
    # n above the pilot cap: the subsample pilot picks a real warm-start threshold, and the full
    # fit rebuilds no more than the cold (threshold=0) build while matching its clustering.
    ari = pytest.importorskip("sklearn.metrics").adjusted_rand_score
    rng = np.random.default_rng(0)
    centers = [[0, 0], [9, 0], [0, 9], [9, 9]]
    x = np.vstack([rng.normal(c, 0.5, (1500, 2)) for c in centers])
    y = np.repeat(np.arange(4), 1500)
    kw = dict(n_clusters=4, method="kmeans", max_leaves=200, seed=0)
    auto = betula_cluster.Betula(threshold="auto", **kw).fit(x)
    base = betula_cluster.Betula(threshold=0.0, **kw).fit(x)
    assert auto._auto_threshold > 0.0  # pilot on the subsample chose a positive warm start
    assert auto.threshold_ > 0.0
    assert auto.n_rebuilds_ <= base.n_rebuilds_  # warm start never rebuilds more than cold
    assert ari(y, auto.predict(x)) >= 0.9


def test_auto_threshold_fit_predict(blobs):
    x, _ = blobs
    labels = betula_cluster.Betula(
        threshold="auto", n_clusters=4, method="kmeans", seed=0
    ).fit_predict(x)
    assert len(labels) == len(x)


def test_auto_threshold_rejects_sparse():
    sp = pytest.importorskip("scipy.sparse")
    x = sp.random(60, 8, density=0.2, format="csr", random_state=0)
    with pytest.raises(ValueError, match="requires a dense array"):
        betula_cluster.Betula(threshold="auto").fit(x)


def test_auto_threshold_streaming(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(threshold="auto", n_clusters=4, method="kmeans", seed=0)
    est.partial_fit(x[:1200])  # first batch resolves + caches the estimate
    cached = est._auto_threshold
    est.partial_fit(x[1200:])  # subsequent batch keeps the same tree, no re-pilot
    assert est._auto_threshold == cached
    est.partial_fit()  # finalize the streaming clustering (sklearn-Birch style)
    assert len(est.predict(x)) == len(x)


def test_auto_threshold_constrained(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(threshold="auto", method="kmeans", n_clusters=4, seed=0)
    est.fit(x, must_link=np.array([[0, 1]]))
    assert est.get_params()["threshold"] == "auto"
    assert len(est.predict(x)) == len(x)


def test_auto_threshold_set_params_resets_cache(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(threshold="auto", n_clusters=4, method="kmeans", seed=0).fit(x)
    assert est._auto_threshold is not None
    est.set_params(n_clusters=3)
    assert est._auto_threshold is None  # a param change invalidates the pilot estimate


def test_consensus_is_stable_on_separated_blobs(blobs):
    x, y = blobs
    res = betula_cluster.consensus(x, 4, n_runs=5, method="kmeans", threshold=0.05, max_leaves=300)
    assert res.labels.shape == (len(x),) and res.confidence.shape == (len(x),)
    assert res.confidence.min() >= 0.0 and res.confidence.max() <= 1.0
    assert res.n_runs == 5
    assert ari(res.labels, y) > 0.95  # consensus recovers the true partition
    assert res.mean_confidence > 0.95  # well-separated ⇒ every insertion order agrees


def test_consensus_confidence_drops_on_overlap():
    # heavily overlapping blobs: boundary points land in different clusters across insertion orders,
    # so their per-point confidence falls below 1.
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(c, 2.0, (400, 2)) for c in ([0, 0], [3, 0], [0, 3])])
    res = betula_cluster.consensus(x, 3, n_runs=6, method="kmeans", threshold=0.1, max_leaves=300)
    assert res.confidence.min() < 1.0
    assert (res.confidence < 1.0).any()


def test_consensus_single_run_is_trivially_confident(blobs):
    x, _ = blobs
    res = betula_cluster.consensus(x, 4, n_runs=1, method="kmeans", threshold=0.05, max_leaves=300)
    assert np.all(res.confidence == 1.0)  # one run always agrees with itself


def test_consensus_rejects_bad_n_runs(blobs):
    x, _ = blobs
    with pytest.raises(ValueError, match="n_runs"):
        betula_cluster.consensus(x, 4, n_runs=0)


def test_consensus_rejects_density_method(blobs):
    x, _ = blobs
    outliers = np.array([[50.0, 50.0], [60.0, -60.0], [-70.0, 70.0]])  # guarantee HDBSCAN noise
    with pytest.raises(ValueError, match="partitional"):
        betula_cluster.consensus(np.vstack([x, outliers]), 4, method="hdbscan", threshold=0.05)


def test_consensus_parallel_matches_serial(blobs):
    x, _ = blobs
    kw = dict(n_runs=4, method="kmeans", threshold=0.1, max_leaves=300, seed=0)
    serial = betula_cluster.consensus(x, 4, n_jobs=1, **kw)
    for n_jobs in (2, -1):  # positive worker count, then all-cores (max_workers=None)
        par = betula_cluster.consensus(x, 4, n_jobs=n_jobs, **kw)
        # each run is seeded independently, so threading changes nothing but wall-clock
        np.testing.assert_array_equal(serial.labels, par.labels)
        np.testing.assert_array_equal(serial.confidence, par.confidence)


@pytest.mark.parametrize(
    "method",
    ["gmm", "gmm-full", "mppca", "vmf", "gmm-toeplitz", "gmm-toeplitz-full", "gmm-toeplitz-gs"],
)
def test_mixture_labels_are_the_maximum_posterior(method):
    """A mixture head assigns by maximum posterior; `predict` must return that partition.

    The columns of `predict_proba` are the same components `predict` chooses between, so the two
    agree by construction. Reading the label off the leaf a point routes to instead answers a
    different question — nearest microcluster — and does not weigh a component by its own
    covariance / concentration and mixing weight.
    """
    rng = np.random.default_rng(11)
    n, d = 900, 24
    t = np.arange(d)
    x = np.vstack(
        [
            rng.normal(scale=s, size=(n // 3, d)) + np.sin(t * f)[None, :] * 2.0
            for s, f in [(0.4, 0.3), (0.9, 0.8), (0.6, 1.4)]
        ]
    )
    est = betula_cluster.Betula(n_clusters=3, method=method, max_leaves=120, seed=4)
    labels = est.fit_predict(x)
    proba = est.predict_proba(x)
    assert proba.shape[0] == x.shape[0]
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(labels, proba.argmax(axis=1))
    # A silenced component is unreachable: every label predicted is one a microcluster claims, so
    # `cluster_centers_[label]` is always a real centre.
    assert np.unique(labels).max() < est.cluster_centers_.shape[0]


def test_mixture_predict_beats_the_tree_descent_on_overlapping_scales():
    """Two concentric Gaussians of very different width: a nearest-microcluster route cannot tell
    them apart (both share the same centre), while the posterior can."""
    rng = np.random.default_rng(5)
    d = 8
    tight = rng.normal(scale=0.35, size=(700, d))
    wide = rng.normal(scale=3.0, size=(700, d))
    x = np.vstack([tight, wide])
    truth = np.r_[np.zeros(700, int), np.ones(700, int)]
    from betula_cluster.tuning import adjusted_rand

    est = betula_cluster.Betula(n_clusters=2, method="gmm", max_leaves=200, seed=2).fit(x)
    assert adjusted_rand(truth, est.predict(x)) > 0.4


def test_projected_fit_still_reports_a_leaf_level_posterior():
    """With a projection the head clusters NMF codes, so it cannot score a raw row; `predict_proba`
    falls back to the row's microcluster responsibilities rather than raising."""
    rng = np.random.default_rng(7)
    x = np.abs(rng.normal(size=(600, 30))) + rng.integers(0, 3, 600)[:, None] * 2.0
    est = betula_cluster.Betula(
        n_clusters=3, method="gmm", projection="weighted-nmf", projection_dim=5, seed=0
    ).fit(x)
    proba = est.predict_proba(x)
    assert proba.shape == (600, est.microcluster_proba_.shape[1])
    assert np.allclose(proba.sum(axis=1), 1.0)


# ── WindowStream ──────────────────────────────────────────────────────────────────────────────────


def _drifting_stream(n=1200, era=30.0, span=60.0, seed=0):
    """Two eras with different structure, timestamps rising through `span`.

    Half-open on purpose: a point at exactly `t = span` opens one more frame, and every frame count
    below would then be off by one for a reason that has nothing to do with what is being tested."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, span, n, endpoint=False)
    side = np.where(np.arange(n) % 2 == 0, 0.0, 12.0)
    far = np.where(t >= era, 80.0, 0.0)
    x = np.c_[side + rng.normal(0, 0.3, n), far + rng.normal(0, 0.3, n)]
    return x, t


def test_window_stream_answers_each_era_without_seeing_the_other():
    """The point of the head: a window over the first era must not see the second. A decayed
    single-model streamer cannot do this at all — it has only a present."""
    x, t = _drifting_stream()
    ws = betula_cluster.WindowStream(frame_width=10.0, capacity=64, max_leaves=200)
    ws.partial_fit(x, t).close_frame()
    assert ws.n_frames_ == 6

    early = ws.cluster_window(0.0, 29.9, 2)
    assert early is not None
    assert np.all(np.abs(early[0][:, 1]) < 10.0), early[0]
    late = ws.cluster_window(30.0, 60.0, 2)
    assert np.all(np.abs(late[0][:, 1] - 80.0) < 10.0), late[0]


def test_window_stream_conserves_the_mass_of_the_frames_it_returns():
    x, t = _drifting_stream(n=600, span=30.0)
    ws = betula_cluster.WindowStream(frame_width=10.0, capacity=64, max_leaves=200)
    ws.partial_fit(x, t).close_frame()
    assert ws.window_moments(0.0, 30.0)["weight"] == pytest.approx(600.0)
    spans = ws.frame_spans()
    assert len(spans) == 3
    assert sum(w for _, _, w in spans) == pytest.approx(600.0)
    # Ascending, non-overlapping frames.
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_a_window_ending_inside_a_frame_gets_that_whole_frame():
    """The documented price of never subtracting, asserted rather than described: resolution is
    the frame width, and the error it costs is bounded by that width — where a snapshot
    subtraction's error is bounded by nothing."""
    x, t = _drifting_stream(n=600, span=30.0)
    ws = betula_cluster.WindowStream(frame_width=10.0, capacity=64, max_leaves=200)
    ws.partial_fit(x, t).close_frame()
    whole = ws.window_moments(0.0, 9.9)["weight"]
    reaching = ws.window_moments(0.0, 10.1)["weight"]
    assert whole == pytest.approx(200.0)
    assert reaching == pytest.approx(400.0), "a query 0.1 past the boundary took a whole frame"


def test_window_stream_refuses_a_window_it_cannot_answer():
    x, t = _drifting_stream(n=200, span=10.0)
    ws = betula_cluster.WindowStream(frame_width=10.0, max_leaves=200)
    ws.partial_fit(x, t).close_frame()
    assert ws.cluster_window(100.0, 200.0, 2) is None  # empty window
    assert ws.cluster_window(0.0, 10.0, 100_000) is None  # more clusters than micro-clusters


def test_window_stream_rejects_a_timestamp_per_row_mismatch():
    x, t = _drifting_stream(n=50, span=10.0)
    ws = betula_cluster.WindowStream(frame_width=5.0)
    with pytest.raises(ValueError, match="one timestamp per row"):
        ws._est = betula_cluster._core.WindowStream(**ws.get_params())
        ws._est.partial_fit(x, list(t[:10]))


def test_window_stream_is_not_fitted_before_it_is_fed():
    ws = betula_cluster.WindowStream(frame_width=1.0)
    with pytest.raises(AttributeError):
        _ = ws.n_frames_
    assert betula_cluster.WindowStream(frame_width=2.0).get_params()["frame_width"] == 2.0
    with pytest.raises(ValueError, match="Invalid parameter"):
        ws.set_params(nonsense=1)


def test_window_stream_set_params_discards_the_fitted_state_it_invalidates():
    """A reconfigured frame width cannot be applied to frames already closed under the old one, so
    `set_params` drops the estimator rather than mixing two geometries."""
    x, t = _drifting_stream(n=100, span=10.0)
    ws = betula_cluster.WindowStream(frame_width=5.0, max_leaves=200)
    ws.partial_fit(x, t).close_frame()
    assert ws.set_params(frame_width=2.5) is ws
    assert ws.get_params()["frame_width"] == 2.5
    with pytest.raises(AttributeError):
        _ = ws.n_frames_


def test_window_stream_repr_names_the_two_parameters_that_size_the_window():
    ws = betula_cluster.WindowStream(frame_width=3.0, capacity=32)
    assert repr(ws) == "WindowStream(frame_width=3.0, capacity=32)"


# ── export_coreset ────────────────────────────────────────────────────────────────────────────────


def _summary_cost(est, centers):
    """`sum_i (S_i + n_i d^2(mu_i, C))` over every leaf — what a coreset has to reproduce.

    `S_i = w_i r_i^2` exactly, since `microcluster_radii_` is the leaf RMS radius.
    """
    mu = np.asarray(est.microcluster_centers_, dtype=np.float64)
    w = np.asarray(est.microcluster_weights_, dtype=np.float64)
    r = np.asarray(est.microcluster_radii_, dtype=np.float64)
    d2 = (mu * mu).sum(1)[:, None] - 2.0 * mu @ centers.T + (centers * centers).sum(1)[None, :]
    return float((w * r * r).sum() + (w * np.maximum(d2, 0.0).min(1)).sum())


def test_export_coreset_scores_every_candidate_solution_within_epsilon(blobs):
    """The acceptance criterion itself: a coreset that only reproduces the solution its
    sensitivities were derived from is not a coreset, so the check sweeps candidates it has never
    seen."""
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, method="kmeans", max_leaves=800, seed=0).fit(x)
    cs = est.export_coreset(size=400, k=4)
    rng = np.random.default_rng(3)
    lo, hi = x.min(0), x.max(0)
    worst = 0.0
    for _ in range(40):
        centers = rng.uniform(lo, hi, size=(4, x.shape[1]))
        want = _summary_cost(est, centers)
        worst = max(worst, abs(cs.cost(centers) - want) / want)
    assert worst < 0.10, f"worst relative error {worst}"


def test_export_coreset_at_the_leaf_count_is_the_summary_exactly(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=200, seed=0).fit(x)
    cs = est.export_coreset(size=10_000, k=4)
    assert cs.centers.shape[0] == cs.n_leaves == est.n_leaves_
    centers = np.asarray(est.cluster_centers_, dtype=np.float64)
    assert cs.cost(centers) == pytest.approx(_summary_cost(est, centers), rel=1e-9)


def test_export_coreset_total_sensitivity_is_the_ten_plus_four_k_of_the_derivation(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=400, seed=0).fit(x)
    for k in (2, 5, 9):
        assert est.export_coreset(size=100, k=k).total_sensitivity == pytest.approx(10.0 + 4.0 * k)


def test_export_coreset_needs_only_a_tree_not_a_finalized_head(blobs):
    """The guarantee is over candidate solutions, so it cannot depend on which head was fitted —
    and the API must not pretend otherwise by demanding one."""
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=200)
    est.partial_fit(x)
    cs = est.export_coreset(size=50, k=4)
    assert cs.centers.shape[0] > 0 and cs.offset > 0.0


def test_export_coreset_without_a_size_is_the_unsampled_summary_it_always_was(blobs):
    """The zero-argument call is the pre-existing streaming summary and must stay free of the
    weighted k-means the sampled path needs — so the guarantee numbers it cannot compute are
    absent rather than faked."""
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=200, seed=0).fit(x)
    cs = est.export_coreset()
    assert np.array_equal(cs.weights, est.microcluster_weights_)
    assert cs.n_points == pytest.approx(len(x))
    assert cs.offset > 0.0
    assert cs.reference_cost is None and cs.total_sensitivity is None
    with pytest.raises(ValueError, match="needs a sampled coreset"):
        cs.summary_epsilon(1.0)


def test_summary_epsilon_grows_with_the_factor_it_is_asked_to_defend(blobs):
    """`rho` scales with `alpha`, and the bound is `4*sqrt(rho) + 4*rho` — monotone, and strictly
    positive whenever the leaves carry any within-leaf scatter at all."""
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=200, seed=0).fit(x)
    cs = est.export_coreset(size=100, k=4)
    eps = [cs.summary_epsilon(a) for a in (0.5, 1.0, 2.0)]
    assert 0.0 < eps[0] < eps[1] < eps[2]
    rho = cs.offset / cs.reference_cost
    assert eps[1] == pytest.approx(4.0 * rho**0.5 + 4.0 * rho)


def test_summary_epsilon_of_a_lossless_summary_is_zero():
    """One leaf per distinct point leaves nothing to bound: the reference cost is zero, and a
    relative error against zero is zero rather than a division."""
    x = np.zeros((100, 3))
    est = betula_cluster.Betula(n_clusters=1, threshold=0.0, max_leaves=100, seed=0).fit(x)
    cs = est.export_coreset(size=10, k=1)
    assert cs.reference_cost == pytest.approx(0.0)
    assert cs.summary_epsilon(1.0) == 0.0


def test_export_coreset_defaults_k_to_the_configured_cluster_count(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=200, seed=0).fit(x)
    assert est.export_coreset(size=100).total_sensitivity == pytest.approx(
        est.export_coreset(size=100, k=4).total_sensitivity
    )


@pytest.mark.parametrize(("size", "k"), [(0, 4), (-1, 4), (10, 0)])
def test_export_coreset_rejects_a_degenerate_request(blobs, size, k):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=200, seed=0).fit(x)
    with pytest.raises(ValueError):
        est.export_coreset(size=size, k=k)


# ── the four non-Ward linkages ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["average", "weighted", "centroid", "median"])
def test_every_linkage_recovers_well_separated_blobs(blobs, method):
    x, y = blobs
    est = betula_cluster.Betula(n_clusters=4, method=method, max_leaves=300, seed=0)
    assert ari(y, est.fit_predict(x)) > 0.95


@pytest.mark.parametrize("method", ["average", "weighted", "centroid", "median"])
def test_every_linkage_picks_its_own_k_when_asked(blobs, method):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=0, method=method, max_leaves=300, seed=0)
    labels = est.fit_predict(x)
    assert len(np.unique(labels)) == 4


@pytest.mark.parametrize("method", ["centroid", "median"])
def test_the_centroid_linkages_agree_with_scipy_point_for_point(method):
    """SciPy's `centroid` and `median` work in the squared Euclidean metric, exactly as this driver
    does, so on a one-leaf-per-point tree the two partitions must be identical -- not merely
    similar. `average` and `weighted` are excluded: SciPy applies them to *unsquared* distances,
    and the mean of squares is not the square of the mean, so a disagreement there would be a
    convention difference rather than a defect."""
    sch = pytest.importorskip("scipy.cluster.hierarchy")
    rng = np.random.default_rng(7)
    x = np.vstack([rng.normal(c, 0.7, (40, 2)) for c in ([0, 0], [6, 1], [2, 6])])
    est = betula_cluster.Betula(
        n_clusters=3, method=method, feature="spherical", threshold=0.0, max_leaves=1000, seed=0
    )
    got = est.fit_predict(x)
    want = sch.fcluster(sch.linkage(x, method=method), 3, criterion="maxclust")
    assert ari(want, got) == pytest.approx(1.0)


def test_an_unknown_linkage_name_lists_the_ones_that_exist():
    with pytest.raises(ValueError, match="'average', 'weighted', 'centroid', 'median'"):
        betula_cluster.Betula(n_clusters=3, method="upgma").fit(np.zeros((10, 2)))


def test_predict_proba_raises_without_a_posterior():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 5))
    est = betula_cluster.Betula(n_clusters=3, method="ward", seed=0).fit(x)
    with pytest.raises(ValueError, match="predict_proba is only available"):
        est._est.predict_proba(x)


# ── the leaf-budget warning ───────────────────────────────────────────────────────────────────────


def _starved(n=400, d=4, k=60, seed=0):
    """Data plus a leaf cap that forces fewer than two leaves per requested cluster."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), k


def test_leaf_budget_warning_fires_when_the_summary_cannot_carry_k():
    """Under `max_leaves < 2k` the head is asked to split a summary too coarse to hold `k`
    clusters. The message must carry both realised numbers, since the realised leaf count is what
    the head sees and it can sit below the cap."""
    x, k = _starved()
    with pytest.warns(UserWarning, match=r"leaves per cluster") as rec:
        betula_cluster.fit_predict(x, k, method="kmeans", max_leaves=40)
    text = str(rec[0].message)
    assert f"n_clusters={k}" in text
    assert "max_leaves" in text


def test_leaf_budget_warning_switches_on_exactly_at_two_leaves_per_cluster():
    """Two per cluster is the floor, not a strict inequality: the sweep in
    `local/scratch/leaves_per_k_sweep.py` puts well-separated data at its plateau there.

    The boundary is read off the *realised* leaf count rather than assumed from `max_leaves` -- the
    tree settles below its cap (91 leaves under a 100 cap here), so hard-coding the cap would test
    the interior and never the edge. Both sides are asserted, which is what makes it a boundary
    test and not two independent ones."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4000, 4)) * 30.0  # spread out, so the tree fills most of its cap
    probe = betula_cluster.Betula(n_clusters=2, method="kmeans", max_leaves=100, threshold=0.0)
    leaves = probe.fit(x).n_leaves_
    assert leaves >= 4  # the fixture is only a boundary if there is room either side

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        betula_cluster.Betula(
            n_clusters=leaves // 2, method="kmeans", max_leaves=100, threshold=0.0
        ).fit(x)
    with pytest.warns(UserWarning, match=r"leaves per cluster"):
        betula_cluster.Betula(
            n_clusters=leaves // 2 + 1, method="kmeans", max_leaves=100, threshold=0.0
        ).fit(x)


@pytest.mark.parametrize("method", ["hdbscan", "scale-space", "leiden"])
def test_leaf_budget_warning_is_silent_for_heads_that_pick_their_own_k(method):
    """`n_clusters` is ignored by these heads, so a budget stated relative to it says nothing."""
    x, k = _starved()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        betula_cluster.fit_predict(x, k, method=method, max_leaves=40)


def test_leaf_budget_warning_is_silent_for_auto_k():
    """`n_clusters=0` selects the count by BIC from the leaves it actually has. The check has no
    arm for it -- `leaves >= 2 * 0` is vacuously true -- so this pins the behaviour rather than a
    branch, and would catch a future `k.max(1)` that quietly made auto-k warn."""
    x, _ = _starved()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        betula_cluster.fit_predict(x, 0, method="gmm", max_leaves=40)


def test_leaf_budget_warning_also_fires_on_the_estimator():
    """`fit_predict` is re-exported from the extension while the estimator is wrapped, so the two
    paths reach the check differently; both must warn."""
    x, k = _starved()
    with pytest.warns(UserWarning, match=r"leaves per cluster"):
        betula_cluster.Betula(n_clusters=k, method="kmeans", max_leaves=40).fit_predict(x)


def test_leaf_budget_warning_is_silent_for_a_single_cluster():
    """`n_clusters=1` separates nothing, so a one-leaf summary answers it exactly and the floor does
    not apply. Without this arm the check would tell a caller that one leaf cannot hold one
    cluster."""
    rng = np.random.default_rng(0)
    x = rng.normal(scale=1e-9, size=(200, 3))  # collapses to a single leaf
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        betula_cluster.fit_predict(x, 1, method="kmeans")


# ── the BIRCH absorption grid ─────────────────────────────────────────────────────────────────────


def _leaf_count(absorb, x, threshold):
    est = betula_cluster.Betula(
        n_clusters=2,
        feature="diagonal",
        method="kmeans",
        absorb=absorb,
        chi2_scale=1.0,
        threshold=threshold,
        max_leaves=4000,
        seed=0,
    )
    est.fit(x)
    return est.n_leaves_


@pytest.mark.parametrize(
    "absorb", ["euclidean", "manhattan", "average", "diameter", "ward", "radius"]
)
def test_every_geometric_absorber_actually_absorbs(absorb):
    """A criterion that is wired up but never fires would still pass a quality test -- the tree
    would just fall back to one leaf per point. Each has to compress at a reachable threshold."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 4))
    assert _leaf_count(absorb, x, 0.0) > _leaf_count(absorb, x, 4.0)


def test_diameter_absorbs_at_least_as_eagerly_as_ward():
    """D3 is D4 plus both scatters over one fewer than the merged mass, so at the same threshold it
    can only ever be the larger of the two -- and a larger criterion value means a stricter gate,
    hence no fewer leaves. This pins the ordering the closed form implies rather than a number."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 4))
    ward, diameter = _leaf_count("ward", x, 2.0), _leaf_count("diameter", x, 2.0)
    assert diameter >= ward
    assert ward < len(x)  # the fixture is in the compressing regime, not one-leaf-per-point


def test_unknown_absorber_names_the_whole_set():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 3))
    with pytest.raises(ValueError, match=r"diameter"):
        betula_cluster.fit_predict(x, 2, absorb="d3")
    with pytest.raises(ValueError, match=r"diameter"):
        betula_cluster.Betula(n_clusters=2, absorb="d3").fit(x)


# ── BIRCH Phase 4 (refine=) ───────────────────────────────────────────────────────────────────────


def _kmeans_cost(x, labels):
    """Sum of squared distances to each point's own cluster mean -- the k-means objective of an
    arbitrary partition, computed here independently of anything the engine reports."""
    total = 0.0
    for c in np.unique(labels):
        block = x[labels == c]
        total += float(((block - block.mean(0)) ** 2).sum())
    return total


def _blobs(n=5000, d=10, k=6, scale=1.5, seed=0):
    """Deliberately *overlapping* blobs. Well-separated ones make the leaf summary lossless -- the
    tree recovers the exact partition and Phase 4 starts at its fixed point, so a refinement
    fixture built on them measures nothing. This is the regime where the summary costs something."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=scale, size=(k, d))
    return rng.normal(size=(n, d)) + centers[rng.integers(k, size=n)]


def test_refine_lowers_the_kmeans_objective_of_the_partition():
    """Phase 3 optimizes over the leaf summary; Phase 4 optimizes over the raw points. A coarse
    summary (30 leaves for 5000 overlapping points) leaves room, and Lloyd is monotone."""
    x = _blobs()
    kw = dict(feature="spherical", method="kmeans", max_leaves=30, seed=0)
    coarse = betula_cluster.fit_predict(x, 6, refine=0, **kw)
    refined = betula_cluster.fit_predict(x, 6, refine=20, **kw)
    assert _kmeans_cost(x, refined) < _kmeans_cost(x, coarse)


def test_refine_is_ignored_by_a_head_with_no_centre_model():
    """A mixture assigns by maximum posterior, not by nearest centre; sweeping centres would
    substitute a partition the head never fits. The labels must be byte-identical."""
    x = _blobs(seed=1)
    kw = dict(feature="diagonal", method="gmm", max_leaves=30, seed=0)
    plain = betula_cluster.fit_predict(x, 6, refine=0, **kw)
    swept = betula_cluster.fit_predict(x, 6, refine=20, **kw)
    assert np.array_equal(plain, swept)


def test_the_estimator_refines_the_rule_predict_scores_against():
    """`fit` is the in-memory entry point, so it has the rows Phase 4 needs. The refined estimator
    must both relabel its training rows and carry the change into `predict` on fresh rows."""
    x = _blobs(seed=2)
    kw = dict(n_clusters=6, feature="spherical", method="kmeans", max_leaves=30, seed=0)
    plain = betula_cluster.Betula(refine=0, **kw).fit(x)
    refined = betula_cluster.Betula(refine=20, **kw).fit(x)
    assert not np.array_equal(plain.predict(x), refined.predict(x))
    assert _kmeans_cost(x, refined.predict(x)) < _kmeans_cost(x, plain.predict(x))


def test_partial_fit_cannot_refine_and_says_so_by_leaving_the_centres_alone():
    """Streaming keeps a tree, not the data, so there is no X left to sweep. `refine` must be inert
    on that path rather than silently refining the last chunk only."""
    x = _blobs(seed=3)
    est = betula_cluster.Betula(
        n_clusters=6, feature="spherical", method="kmeans", max_leaves=30, seed=0, refine=20
    )
    for chunk in np.array_split(x, 3):
        est.partial_fit(chunk)
    est.partial_fit()
    streamed = est.predict(x)
    plain = betula_cluster.Betula(
        n_clusters=6, feature="spherical", method="kmeans", max_leaves=30, seed=0, refine=0
    ).fit(x)
    assert np.array_equal(streamed, plain.predict(x))


def test_refine_survives_the_sklearn_parameter_protocol():
    est = betula_cluster.Betula(refine=7)
    assert est.get_params()["refine"] == 7
    assert est.set_params(refine=3).get_params()["refine"] == 3


# ── projection="svd": CF-weighted PCA of the leaf summary ─────────────────────────────────────────


def _topic_rows(n=1200, d=400, k=6, seed=0):
    """Nonnegative sparse rows with a planted topic structure — the shape the text path sees, small
    enough for a test. Each row draws its non-zeros from one topic's own vocabulary slice."""
    rng = np.random.default_rng(seed)
    y = rng.integers(k, size=n)
    x = np.zeros((n, d))
    width = d // k
    for i, t in enumerate(y):
        lo = t * width
        cols = rng.integers(lo, lo + width, size=12)
        x[i, cols] += rng.random(12)
        x[i, rng.integers(d, size=3)] += 0.2 * rng.random(3)  # cross-topic noise
    return x, y


def test_svd_projection_beats_no_projection_on_planted_topics():
    """The reason the projection exists: in high-d sparse space the raw geometry is uninformative,
    and clustering CF-weighted principal codes recovers the planted structure instead."""
    ari = pytest.importorskip("sklearn.metrics").adjusted_rand_score
    x, y = _topic_rows(seed=2)
    # A coarse budget on purpose: at 100+ leaves for 1200 rows this fixture is separable either way,
    # and a test that both arms pass measures nothing.
    kw = dict(feature="spherical", method="spherical-kmeans", max_leaves=20, seed=0)
    plain = betula_cluster.fit_predict(x, 6, **kw)
    coded = betula_cluster.fit_predict(x, 6, projection="svd", projection_dim=20, **kw)
    assert ari(y, coded) > ari(y, plain)


def test_the_svd_path_labels_a_row_by_its_own_code_not_by_its_leaf():
    """The mechanism that separates `svd` from `weighted-nmf`: a PCA is a linear map, so a raw row
    is encoded and scored by the head directly. An NMF code is a per-row nonnegative least squares,
    so that path can only answer with the row's leaf's label, making the labelling constant on every
    leaf. Finding one leaf that holds two different cluster labels proves the row route ran."""
    x, _ = _topic_rows(seed=1)
    est = betula_cluster.Betula(
        n_clusters=6,
        feature="spherical",
        method="spherical-kmeans",
        max_leaves=20,
        seed=0,
        projection="svd",
        projection_dim=20,
    )
    labels = est.fit_predict(x)
    leaves = np.asarray(est.assign_microclusters(x))
    split = sum(len(np.unique(labels[leaves == leaf])) > 1 for leaf in np.unique(leaves))
    assert split > 0, "every leaf was label-constant, so rows were routed rather than encoded"


def test_the_nmf_path_stays_on_the_microcluster_route():
    """The other half of the same claim, and the reason it is not free: an NMF projection has no
    linear encoder, so its labelling is constant on every leaf by construction."""
    x, _ = _topic_rows(seed=1)
    est = betula_cluster.Betula(
        n_clusters=6,
        feature="spherical",
        method="kmeans",
        max_leaves=200,
        seed=0,
        projection="weighted-nmf",
        projection_dim=20,
    )
    labels = est.fit_predict(x)
    leaves = np.asarray(est.assign_microclusters(x))
    assert all(len(np.unique(labels[leaves == leaf])) == 1 for leaf in np.unique(leaves))


def test_svd_accepts_signed_data_that_nmf_must_reject():
    x = np.random.default_rng(0).normal(size=(300, 12))
    kw = dict(feature="spherical", method="kmeans", max_leaves=100, seed=0, projection_dim=4)
    betula_cluster.fit_predict(x, 3, projection="svd", **kw)  # must not raise
    with pytest.raises(ValueError, match=r"nonnegative"):
        betula_cluster.fit_predict(x, 3, projection="weighted-nmf", **kw)


def test_svd_components_are_an_orthonormal_basis_of_the_requested_rank():
    """PCA components are right singular vectors: orthonormal by construction. A basis that failed
    this would still produce codes, and every downstream distance would be silently skewed."""
    x, _ = _topic_rows(seed=2)
    est = betula_cluster.Betula(
        n_clusters=6,
        feature="spherical",
        method="kmeans",
        max_leaves=200,
        seed=0,
        projection="svd",
        projection_dim=8,
    )
    est.fit(x)
    v = est.components_
    assert v.shape == (8, x.shape[1])
    np.testing.assert_allclose(v @ v.T, np.eye(8), atol=1e-9)
    assert 0.0 <= est.reconstruction_err_ <= 1.0


def test_svd_predict_proba_still_agrees_with_predict():
    """`predict_proba(X).argmax(1) == predict(X)` is a documented promise. On the projected path the
    posterior has to be scored in code space, through the same encoder, or the two disagree."""
    x, _ = _topic_rows(seed=3)
    est = betula_cluster.Betula(
        n_clusters=6,
        feature="spherical",
        method="gmm",
        max_leaves=200,
        seed=0,
        projection="svd",
        projection_dim=8,
    )
    est.fit(x)
    np.testing.assert_array_equal(est.predict_proba(x).argmax(1), est.predict(x))


def test_unknown_projection_names_the_whole_set():
    x = np.random.default_rng(0).random((50, 6))
    with pytest.raises(ValueError, match=r"'svd'"):
        betula_cluster.fit_predict(x, 2, projection="pca")
    with pytest.raises(ValueError, match=r"'svd'"):
        betula_cluster.Betula(n_clusters=2, projection="pca").fit(x)


def test_a_budget_that_never_binds_warns_that_nothing_was_compressed():
    """`max_leaves >= n` with `threshold=0` builds one leaf per point: measured 3.8x the fit time at
    n=8000 and 14x at n=40000 against a binding budget, for a summary that is the input. The warning
    is what makes that price visible; it reads the realised leaf count, not the configuration."""
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.normal(size=(6000, 4)))
    with pytest.warns(UserWarning, match="one per point"):
        betula_cluster.fit_predict(x, 3, max_leaves=20_000, threshold=0.0, seed=0)


def test_a_binding_budget_does_not_warn_about_compression():
    """The control: the same call at a budget that binds compresses, so it must stay silent."""
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.normal(size=(6000, 4)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        betula_cluster.fit_predict(x, 3, max_leaves=500, threshold=0.0, seed=0)


def test_a_compressing_gmm_on_a_spherical_feature_warns_about_isotropic_scatter():
    """`Spherical::variance(_d)` ignores its argument: it returns one isotropic number for every
    dimension, because a spherical cluster feature carries a scalar scatter. A diagonal GMM adds
    that number to all `dim` component variances, so under compression a dimension with genuinely
    near-zero variance is lifted to the isotropic average. Measured on digits at x2.0 compression
    that costs ARI 0.4403 -> 0.0088 while the fitted centres stay healthy, so the mismatch is worth
    a warning rather than a silent wrong answer."""
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.normal(size=(4000, 8)))
    with pytest.warns(UserWarning, match="per-dimension covariance"):
        betula_cluster.fit_predict(
            x, 3, method="gmm", feature="spherical", max_leaves=200, threshold=0.0, seed=0
        )


def test_a_per_dimension_feature_does_not_warn_about_isotropic_scatter():
    """The control, twice over: the same compressing call on a feature that *does* carry
    per-dimension scatter must stay silent, and so must the spherical feature when the budget never
    binds — with one leaf per point there is no scatter to add, which is exactly the row where the
    measured collapse disappears."""
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.normal(size=(4000, 8)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        betula_cluster.fit_predict(
            x, 3, method="gmm", feature="full", max_leaves=200, threshold=0.0, seed=0
        )
        betula_cluster.fit_predict(
            x, 3, method="kmeans", feature="spherical", max_leaves=200, threshold=0.0, seed=0
        )


# ───────────────────────────── BregmanBetula (ADR 004) ─────────────────────────────


@pytest.fixture(scope="module")
def simplex_blobs():
    """Three groups of positive vectors: the domain KL and Itakura-Saito are defined on."""
    rng = np.random.default_rng(11)
    bases = np.array([[6.0, 1.0, 1.0], [1.0, 6.0, 1.0], [1.0, 1.0, 6.0]])
    xs, ys = [], []
    for c, b in enumerate(bases):
        xs.append(b * rng.lognormal(0.0, 0.12, (300, 3)))
        ys += [c] * 300
    return np.vstack(xs).astype(np.float64), np.array(ys)


@pytest.mark.parametrize("divergence", ["euclidean", "kl", "itakura-saito"])
@pytest.mark.parametrize("method", ["kmeans", "ward"])
def test_bregman_betula_recovers_positive_groups(simplex_blobs, divergence, method):
    x, y = simplex_blobs
    est = betula_cluster.BregmanBetula(
        n_clusters=3, divergence=divergence, method=method, max_leaves=200, seed=0
    )
    labels = est.fit_predict(x)
    assert labels.shape == (len(x),)
    assert ari(y, labels) > 0.9


def test_the_euclidean_divergence_reduces_to_the_shipped_estimator(blobs):
    """ADR 004 keeps `divergence="euclidean"` precisely so this identity is testable: squared
    Euclidean *is* a Bregman divergence, so the two estimators must agree where they overlap."""
    x, _ = blobs
    a = betula_cluster.BregmanBetula(
        n_clusters=4, divergence="euclidean", method="kmeans", max_leaves=300, seed=0
    ).fit_predict(x)
    b = betula_cluster.Betula(
        n_clusters=4, feature="spherical", method="kmeans", threshold=0.0, max_leaves=300, seed=0
    ).fit_predict(x)
    assert ari(a, b) > 0.99


def test_the_mixture_head_is_soft_and_beta_sharpens_it(simplex_blobs):
    x, y = simplex_blobs
    est = betula_cluster.BregmanBetula(
        n_clusters=3, divergence="kl", method="mixture", beta=40.0, max_leaves=200, seed=0
    )
    labels = est.fit_predict(x)
    assert ari(y, labels) > 0.9
    assert est.n_leaves_ > 0
    assert np.array_equal(est.labels_, labels)


def test_beta_is_rejected_rather_than_ignored_outside_the_mixture(simplex_blobs):
    """Silently ignoring a parameter that does nothing is how users conclude it did something."""
    x, _ = simplex_blobs
    est = betula_cluster.BregmanBetula(n_clusters=3, method="kmeans", beta=5.0)
    with pytest.raises(ValueError, match="inverse dispersion"):
        est.fit_predict(x)


@pytest.mark.parametrize(
    ("divergence", "bad", "match"),
    [
        ("kl", 0.0, "> 0"),
        ("itakura-saito", -1.0, "> 0"),
        ("logistic", 1.5, r"in \(0, 1\)"),
        ("logistic", 0.0, r"in \(0, 1\)"),
    ],
)
def test_the_domain_is_checked_before_the_engine_sees_it(divergence, bad, match):
    """`BregmanCf::push` only debug-asserts its domain, so a release build would return NaN
    instead of failing. The check has to live at the boundary, and it has to name the value."""
    x = np.full((40, 3), 0.5)
    x[7, 2] = bad
    est = betula_cluster.BregmanBetula(n_clusters=2, divergence=divergence, max_leaves=20)
    with pytest.raises(ValueError, match=match) as err:
        est.fit_predict(x)
    assert "row 7 column 2" in str(err.value)


def test_the_logistic_divergence_clusters_probabilities(blobs):
    del blobs
    rng = np.random.default_rng(3)
    lo = rng.beta(2.0, 8.0, (250, 4))
    hi = rng.beta(8.0, 2.0, (250, 4))
    x = np.vstack([lo, hi])
    y = np.array([0] * 250 + [1] * 250)
    est = betula_cluster.BregmanBetula(
        n_clusters=2, divergence="logistic", method="kmeans", max_leaves=120, seed=0
    )
    assert ari(y, est.fit_predict(x)) > 0.9


def test_bregman_betula_rejects_unknown_keywords_and_reports_before_fitting(simplex_blobs):
    x, _ = simplex_blobs
    est = betula_cluster.BregmanBetula(n_clusters=3)
    with pytest.raises(ValueError, match="unknown divergence"):
        est.set_params(divergence="hellinger").fit_predict(x)
    with pytest.raises(ValueError, match="unknown method"):
        est.set_params(divergence="kl", method="hdbscan").fit_predict(x)
    with pytest.raises(ValueError, match="Invalid parameter"):
        est.set_params(feature="full")


def test_bregman_betula_validates_its_own_arguments(simplex_blobs):
    x, _ = simplex_blobs
    with pytest.raises(ValueError, match="n_clusters"):
        betula_cluster.BregmanBetula(n_clusters=0).fit_predict(x)
    with pytest.raises(ValueError, match="beta must be positive"):
        betula_cluster.BregmanBetula(n_clusters=3, method="mixture", beta=float("inf")).fit_predict(
            x
        )
    with pytest.raises(ValueError, match="at least n_clusters rows"):
        betula_cluster.BregmanBetula(n_clusters=8).fit_predict(x[:3])


def test_bregman_betula_is_a_scikit_learn_style_estimator(simplex_blobs):
    x, _ = simplex_blobs
    est = betula_cluster.BregmanBetula(n_clusters=3, divergence="kl", max_leaves=150)
    assert est.get_params()["divergence"] == "kl"
    assert "divergence='kl'" in repr(est)
    with pytest.raises(AttributeError, match="not fitted"):
        _ = est.labels_
    with pytest.raises(AttributeError, match="not fitted"):
        _ = est.n_leaves_
    assert est.fit(x) is est
    assert est.labels_.shape == (len(x),)
    est.set_params(n_clusters=2)
    with pytest.raises(AttributeError, match="not fitted"):
        _ = est.labels_


def _gm(weights, means, covs):
    return (
        np.array(weights, dtype=np.float64),
        np.array(means, dtype=np.float64),
        np.array(covs, dtype=np.float64),
    )


def test_mixture_w2_is_zero_between_a_mixture_and_itself():
    a = _gm([0.4, 0.6], [[0.0, 0.0], [5.0, 1.0]], [[1.0, 2.0], [0.5, 0.5]])
    assert betula_cluster.mixture_w2(*a, *a) < 1e-9


def test_mixture_w2_ignores_the_order_the_components_are_listed_in():
    a = _gm([0.4, 0.6], [[0.0, 0.0], [5.0, 1.0]], [[1.0, 2.0], [0.5, 0.5]])
    b = _gm([0.6, 0.4], [[5.0, 1.0], [0.0, 0.0]], [[0.5, 0.5], [1.0, 2.0]])
    assert betula_cluster.mixture_w2(*a, *b) < 1e-9


def test_mixture_w2_of_a_common_translation_is_the_translation_length():
    means = np.array([[0.0, 0.0], [5.0, 1.0]])
    shift = np.array([3.0, 4.0])
    a = _gm([0.4, 0.6], means, [[1.0, 2.0], [0.5, 0.5]])
    b = _gm([0.4, 0.6], means + shift, [[1.0, 2.0], [0.5, 0.5]])
    assert betula_cluster.mixture_w2(*a, *b) == pytest.approx(5.0, abs=1e-9)


def test_mixture_w2_accepts_full_covariance_matrices():
    # A 90-degree rotation of an elongated Gaussian: invisible to a diagonal reading of the same
    # pair, so this also proves the (k, dim, dim) shape is not silently taking the diagonal.
    flat = np.array([[[9.0, 0.0], [0.0, 1.0]]])
    tall = np.array([[[1.0, 0.0], [0.0, 9.0]]])
    means = np.zeros((1, 2))
    rotated = betula_cluster.mixture_w2([1.0], means, flat, [1.0], means, tall)
    assert rotated > 1.0
    assert betula_cluster.mixture_w2([1.0], means, flat, [1.0], means, flat) < 1e-9


def test_mixture_w2_rejects_shapes_that_do_not_describe_a_mixture():
    a = _gm([0.5, 0.5], [[0.0], [1.0]], [[1.0], [1.0]])
    with pytest.raises(ValueError, match="different dimensions"):
        betula_cluster.mixture_w2(*a, [1.0], [[0.0, 0.0]], [[1.0, 1.0]])
    with pytest.raises(ValueError, match="means must be"):
        betula_cluster.mixture_w2([1.0, 1.0, 1.0], a[1], a[2], *a)
    with pytest.raises(ValueError, match="covariances must be"):
        betula_cluster.mixture_w2(a[0], a[1], np.ones((2, 1, 1, 1)), *a)
    with pytest.raises(ValueError, match="at least one component"):
        betula_cluster.mixture_w2(np.empty(0), np.empty((0, 1)), np.empty((0, 1)), *a)
    with pytest.raises(ValueError, match="non-positive weights"):
        betula_cluster.mixture_w2([0.0, 0.0], a[1], a[2], *a)


def test_summary_w2_is_zero_between_a_model_and_itself(blobs):
    est, _, _ = _fitted(blobs)
    assert est.summary_w2(est) < 1e-9


def test_summary_w2_reads_a_shift_of_the_data_as_the_distance_it_is(blobs):
    # The drift use: same generator, translated. The leaf summaries need not line up leaf-for-leaf,
    # so the transport is doing real work — but the density moved by exactly ‖shift‖.
    x, _ = blobs
    shift = np.array([4.0, -3.0])
    first = betula_cluster.Betula(n_clusters=4, threshold=0.5, max_leaves=200, seed=3).fit(x)
    moved = betula_cluster.Betula(n_clusters=4, threshold=0.5, max_leaves=200, seed=3).fit(
        x + shift
    )
    assert first.summary_w2(moved) == pytest.approx(5.0, rel=0.05)


def test_summary_w2_grows_with_the_drift_it_measures(blobs):
    x, _ = blobs
    base = betula_cluster.Betula(n_clusters=4, threshold=0.5, max_leaves=200, seed=3).fit(x)
    previous = 0.0
    for step in (1.0, 3.0, 9.0):
        drifted = betula_cluster.Betula(n_clusters=4, threshold=0.5, max_leaves=200, seed=3).fit(
            x + np.array([step, 0.0])
        )
        d = base.summary_w2(drifted)
        assert d > previous
        previous = d


def test_summary_w2_requires_both_models_to_be_fitted(blobs):
    est, _, _ = _fitted(blobs)
    with pytest.raises(AttributeError, match="not fitted yet"):
        est.summary_w2(betula_cluster.Betula())


def test_mapper_bhattacharyya_linkage_keeps_the_blobs_apart_where_the_centroid_rule_merges_them(
    dumbbell,
):
    # The chaining scenario: a link_scale large enough that the centroid rule links straight across
    # the sparse neck and returns a single node. Dividing the gap by the pair's own spread refuses
    # that link. A nonzero threshold is required — at threshold=0 every leaf is one point with no
    # spread, and a spread-normalised distance has nothing to normalise by.
    est = betula_cluster.Betula(
        feature="spherical", method="hdbscan", threshold=0.6, max_leaves=300
    ).fit(dumbbell)
    wide = dict(lens="coordinate", coordinate=1, resolution=1, gain=0.0, link_scale=6.0)
    assert est.mapper(link="centroid", **wide).n_nodes == 1
    assert est.mapper(link="bhattacharyya", **wide).n_nodes > 1


def test_mapper_rejects_an_unknown_linkage(dumbbell):
    est, _ = _mapped(dumbbell)
    with pytest.raises(ValueError, match="link must be"):
        est.mapper(link="hellinger")


def test_max_leaves_accepts_a_fraction_of_the_row_count(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, threshold=0.5, max_leaves=0.05, seed=0).fit(x)
    assert est.effective_max_leaves_ == math.ceil(0.05 * len(x))
    assert est.n_leaves_ <= est.effective_max_leaves_


def test_max_leaves_fraction_and_the_equivalent_integer_agree(blobs):
    x, _ = blobs
    frac = betula_cluster.Betula(n_clusters=4, threshold=0.5, max_leaves=0.1, seed=0)
    absolute = betula_cluster.Betula(
        n_clusters=4, threshold=0.5, max_leaves=math.ceil(0.1 * len(x)), seed=0
    )
    assert np.array_equal(frac.fit_predict(x), absolute.fit_predict(x))


def test_max_leaves_fraction_resolves_against_the_sparse_row_count():
    sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(0)
    x = sparse.csr_matrix((rng.random((400, 12)) > 0.7).astype(np.float64))
    est = betula_cluster.Betula(n_clusters=3, threshold=0.5, max_leaves=0.25, seed=0).fit(x)
    assert est.effective_max_leaves_ == 100


def test_max_leaves_fraction_is_undefined_for_streaming(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, max_leaves=0.05)
    with pytest.raises(ValueError, match="streaming does not have"):
        est.partial_fit(x)


@pytest.mark.parametrize("bad", [0, -3, 1.5, 0.0, "many", True])
def test_max_leaves_rejects_values_that_are_neither_a_count_nor_a_fraction(blobs, bad):
    x, _ = blobs
    with pytest.raises(ValueError, match="max_leaves must be"):
        betula_cluster.Betula(n_clusters=4, max_leaves=bad).fit(x)


def test_memory_budget_overrides_a_fractional_max_leaves(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(
        n_clusters=4, threshold=0.5, max_leaves=0.5, memory_budget_mb=0.05, seed=0
    ).fit(x)
    assert est.effective_max_leaves_ != math.ceil(0.5 * len(x))


def _split_duplicates():
    """Rows where exactly one pair is a cosine near-duplicate, and the tree splits it.

    Filler points sit on a coarse angular grid, 30 degrees apart, so no two of them are close in
    *direction* -- random filler would produce coincidental cosine pairs and the test would be
    measuring those instead. The twins are 0.5 degrees apart and far enough apart in Euclidean
    distance that a threshold-0 tree keeps them in separate leaves.
    """
    angles = np.deg2rad(np.arange(0, 360, 30))
    radii = 3.0 + 0.5 * np.arange(len(angles))
    filler = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    twin_angles = np.deg2rad([137.0, 137.5])
    twins = np.column_stack([9.0 * np.cos(twin_angles), 9.0 * np.sin(twin_angles)])
    rows = np.vstack([filler, twins])
    return rows, (len(filler), len(filler) + 1)


def test_near_duplicate_pairs_default_is_the_within_leaf_scan(blobs):
    x, _ = blobs
    est = betula_cluster.Betula(n_clusters=4, threshold=0.5, max_leaves=200, seed=0).fit(x)
    assert np.array_equal(
        est.near_duplicate_pairs(x, threshold=0.99),
        est.near_duplicate_pairs(x, threshold=0.99, neighbors=0),
    )


def test_near_duplicate_pairs_neighbors_recovers_a_pair_split_across_two_leaves():
    rows, (a, b) = _split_duplicates()
    # threshold=0 with a budget above N is one leaf per point, so *every* pair is a split pair and
    # the within-leaf scan can find nothing at all.
    est = betula_cluster.Betula(n_clusters=3, threshold=0.0, max_leaves=4000, seed=0).fit(rows)
    leaf = np.asarray(est.assign_microclusters(rows))
    assert leaf[a] != leaf[b], "the fixture stopped splitting the pair"
    assert len(est.near_duplicate_pairs(rows, threshold=0.999, neighbors=0)) == 0
    found = est.near_duplicate_pairs(rows, threshold=0.999, neighbors=1)
    assert (a, b) in {(int(i), int(j)) for _, i, j in found}


def test_near_duplicate_pairs_reports_each_pair_once_in_index_order():
    rows, _ = _split_duplicates()
    est = betula_cluster.Betula(n_clusters=3, threshold=0.0, max_leaves=4000, seed=0).fit(rows)
    found = est.near_duplicate_pairs(rows, threshold=0.9, neighbors=8)
    pairs = [(int(i), int(j)) for _, i, j in found]
    assert all(i < j for i, j in pairs)
    assert len(set(pairs)) == len(pairs)
    assert np.all(np.diff(found[:, 0]) <= 0)  # still sorted by similarity, descending


def test_near_duplicate_pairs_asking_for_more_neighbors_than_leaves_is_not_an_error():
    rows, (a, b) = _split_duplicates()
    est = betula_cluster.Betula(n_clusters=3, threshold=0.0, max_leaves=4000, seed=0).fit(rows)
    found = est.near_duplicate_pairs(rows, threshold=0.999, neighbors=10_000)
    assert (a, b) in {(int(i), int(j)) for _, i, j in found}


def test_near_duplicate_pairs_neighbors_needs_more_than_one_populated_leaf():
    # One leaf means no neighbour to expand into; the pass must be a no-op, not an index error.
    rows = np.tile([1.0, 1.0], (8, 1))
    est = betula_cluster.Betula(n_clusters=1, threshold=10.0, max_leaves=200, seed=0).fit(rows)
    assert est.n_leaves_ == 1
    assert len(est.near_duplicate_pairs(rows, threshold=0.9, neighbors=4)) == 28
