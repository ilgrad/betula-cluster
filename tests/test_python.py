"""End-to-end tests of the Python bindings (run with `pytest`).

Covers every public surface: the one-shot `fit_predict` (all heads, auto-k, f32, χ² absorption)
and the streaming `Betula` estimator, plus the error contract.
"""

import collections
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
        ("diagonal", "ward"),
    ],
)
def test_fit_predict_recovers_blobs(blobs, feature, method):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x, 4, feature=feature, method=method, threshold=0.05, max_leaves=300, seed=1
    )
    assert ari(labels, y) > 0.95


@pytest.mark.parametrize(
    "feature,method", [("diagonal", "gmm"), ("full", "gmm-full"), ("diagonal", "ward")]
)
def test_auto_k_selects_true_count_when_n_clusters_zero(blobs, feature, method):
    x, y = blobs
    labels = betula_cluster.fit_predict(
        x, n_clusters=0, feature=feature, method=method, threshold=0.05, max_leaves=300, seed=1
    )
    assert n_labels(labels) == 4
    assert ari(labels, y) > 0.95


def test_float32_reproduces_float64_on_normal_range(blobs):
    x, y = blobs
    kw = dict(feature="diagonal", method="gmm", threshold=0.05, max_leaves=300, seed=1)
    l64 = betula_cluster.fit_predict(x, 4, **kw)
    l32 = betula_cluster.fit_predict(x.astype(np.float32), 4, **kw)
    assert np.asarray(l32).dtype == np.int64
    assert ari(l64, y) > 0.95 and ari(l32, y) > 0.95
    assert ari(l32, l64) > 0.99  # f32 path agrees with f64 on moderate-range data


@pytest.mark.parametrize("absorb", ["euclidean", "chi2"])
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


def test_summary_reports_structure(blobs):
    est, _, _ = _fitted(blobs)
    s = est.summary()
    assert s["n_samples"] == 2400
    assert s["n_clusters"] == 4
    assert s["n_microclusters"] == est.n_leaves_
    assert s["mean_microcluster_radius"] >= 0


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
    with pytest.raises(ValueError, match="method"):
        betula_cluster.fit_predict_sparse(x, method="hdbscan")
