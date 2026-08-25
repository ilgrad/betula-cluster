"""Unsupervised, memory-aware hyperparameter tuning for betula-cluster.

A dependency-free random search (NumPy only) over the CF-representation knobs, scored by an internal
metric — or by ARI when you have labels. The multi-objective mode returns the
**quality / memory / speed Pareto front**, matching betula's design goals: accurate, small, fast.

Optuna is an *optional* accelerator (``sampler="optuna"``, ``pip install 'betula-cluster[tune]'``):
its TPE / NSGA-II samplers replace the built-in random search. Nothing here is a runtime
dependency — the default path needs only NumPy.

    from betula_cluster import tune
    result = tune(X, n_clusters=8)                       # maximize Calinski-Harabasz
    result = tune(X, n_clusters=8, multi_objective=True) # accuracy / memory / speed Pareto
    best = result.best_params
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── internal cluster-quality metrics (NumPy only; no scikit-learn at runtime) ────────────────────


def calinski_harabasz(x: np.ndarray, labels: np.ndarray) -> float:
    """Variance-ratio criterion (higher is better). Noise labels (``-1``) are dropped."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    keep = labels >= 0
    x, lab = x[keep], labels[keep]
    groups = np.unique(lab)
    k, n = len(groups), len(x)
    if k < 2 or n <= k:
        return float("-inf")
    mean = x.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in groups:
        xc = x[lab == c]
        muc = xc.mean(axis=0)
        between += len(xc) * float(np.sum((muc - mean) ** 2))
        within += float(np.sum((xc - muc) ** 2))
    if within <= 0.0:  # perfectly tight clusters → best possible score
        return float("inf")
    return (between / (k - 1)) / (within / (n - k))


def davies_bouldin(x: np.ndarray, labels: np.ndarray) -> float:
    """Davies-Bouldin index (lower is better). Noise labels (``-1``) are dropped."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    keep = labels >= 0
    x, lab = x[keep], labels[keep]
    groups = np.unique(lab)
    k = len(groups)
    if k < 2:
        return float("inf")
    centroids = np.array([x[lab == c].mean(axis=0) for c in groups])
    scatter = np.array(
        [
            float(np.sqrt(np.mean(np.sum((x[lab == c] - centroids[i]) ** 2, axis=1))))
            for i, c in enumerate(groups)
        ]
    )
    total = 0.0
    for i in range(k):
        worst = 0.0
        for j in range(k):
            if i == j:
                continue
            sep = float(np.sqrt(np.sum((centroids[i] - centroids[j]) ** 2)))
            # coincident centroids of *distinct* clusters are maximally bad, not maximally good
            ratio = (scatter[i] + scatter[j]) / sep if sep > 0.0 else float("inf")
            worst = max(worst, ratio)
        total += worst
    return total / k


def adjusted_rand(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index (higher is better; 1.0 = identical partitions). NumPy-only."""
    a = np.asarray(labels_true)
    b = np.asarray(labels_pred)
    groups_a, ia = np.unique(a, return_inverse=True)
    groups_b, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((len(groups_a), len(groups_b)), dtype=np.int64)
    np.add.at(cont, (ia.ravel(), ib.ravel()), 1)

    def comb2(v: np.ndarray) -> np.ndarray:
        return v * (v - 1) // 2

    sum_ij = int(comb2(cont).sum())
    sum_a = int(comb2(cont.sum(axis=1)).sum())
    sum_b = int(comb2(cont.sum(axis=0)).sum())
    total = len(a) * (len(a) - 1) // 2
    expected = sum_a * sum_b / total if total else 0.0
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0
    return (sum_ij - expected) / (maximum - expected)


def _mst_prim(w: np.ndarray) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Minimum spanning tree of a dense symmetric weight matrix by Prim, `O(m²)`. Returns the edges
    `(i, j, weight)` and each node's MST degree."""
    m = w.shape[0]
    in_tree = np.zeros(m, dtype=bool)
    in_tree[0] = True
    best = w[0].copy()
    parent = np.zeros(m, dtype=int)
    edges: list[tuple[int, int, float]] = []
    deg = np.zeros(m)
    for _ in range(m - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        i = int(parent[j])
        edges.append((i, j, float(w[i, j])))
        deg[i] += 1
        deg[j] += 1
        in_tree[j] = True
        upd = (~in_tree) & (w[j] < best)
        best[upd] = w[j][upd]
        parent[upd] = j
    return edges, deg


def dbcv(x: np.ndarray, labels: np.ndarray, *, sample_cap: int = 1500, seed: int = 0) -> float:
    """Density-Based Clustering Validation (Moulavi et al. 2014) in ``[-1, 1]``, higher is better.

    Unlike Calinski-Harabasz / Davies-Bouldin (convex, centroid-scatter metrics that *penalise*
    correct non-convex partitions), DBCV validates variable-density non-convex clusters — the right
    internal metric for the HDBSCAN-CF and DbStream heads. Computed over a random subsample to bound
    the ``O(m²)`` all-points-core-distance; noise points (label ``-1``) lower the score.
    """
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    if x.shape[0] > sample_cap:
        keep = np.random.default_rng(seed).choice(x.shape[0], sample_cap, replace=False)
        x, labels = x[keep], labels[keep]
    n, d = x.shape
    clusters = [int(c) for c in np.unique(labels) if c >= 0]
    if len(clusters) < 2:
        return -1.0
    dist = np.sqrt(np.maximum(((x[:, None] - x[None]) ** 2).sum(-1), 0.0))
    members = {c: np.where(labels == c)[0] for c in clusters}
    # All-points-core-distance in log space: d_core = (mean_{y≠x} (1/dist)^d)^(-1/d).
    core = np.zeros(n)
    for ids in members.values():
        if len(ids) < 2:
            continue
        sub = dist[np.ix_(ids, ids)].copy()
        np.fill_diagonal(sub, np.inf)  # self → inf distance → 0 contribution to the mean
        with np.errstate(divide="ignore"):
            log_terms = -d * np.log(sub)  # diagonal → −∞ (excluded from the log-sum-exp)
        mx = log_terms.max(axis=1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(log_terms - mx).sum(axis=1))
        core[ids] = np.exp(-(lse - np.log(len(ids) - 1)) / d)
    mreach = np.maximum(dist, np.maximum(core[:, None], core[None]))
    score = 0.0
    for c, ids in members.items():
        if len(ids) < 2:
            continue  # a singleton cluster contributes 0 validity (weight · 0)
        edges, deg = _mst_prim(mreach[np.ix_(ids, ids)])
        internal = [wt for (i, j, wt) in edges if deg[i] > 1 and deg[j] > 1]
        dsc = max(internal) if internal else max(wt for _, _, wt in edges)
        other = np.concatenate([members[o] for o in clusters if o != c])
        dspc = float(mreach[np.ix_(ids, other)].min())
        denom = max(dspc, dsc, 1e-300)  # positive by construction for distinct points
        validity = (dspc - dsc) / denom
        score += (len(ids) / n) * validity  # weight by |Ci| / |all objects| (noise penalises)
    return float(score)


_METRICS = {
    "calinski_harabasz": calinski_harabasz,
    "davies_bouldin": davies_bouldin,
    "dbcv": dbcv,
}
_MAXIMIZE = {"calinski_harabasz": True, "davies_bouldin": False, "dbcv": True, "ari": True}
_WORST = {
    "calinski_harabasz": float("-inf"),
    "davies_bouldin": float("inf"),
    "dbcv": -1.0,
    "ari": float("-inf"),
}


# ── results ──────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Trial:
    """One evaluated configuration: its params and the three objectives (quality, memory, speed)."""

    params: dict[str, Any]
    score: float
    n_leaves: int
    time_s: float


@dataclass
class TuneResult:
    """The outcome of :func:`tune`: the best single configuration, every trial, and — for
    ``multi_objective=True`` — the non-dominated (quality / memory / speed) Pareto front."""

    best_params: dict[str, Any]
    best_score: float
    trials: list[Trial]
    pareto: list[Trial] = field(default_factory=list)


# ── search ─────────────────────────────────────────────────────────────────────────────────────


def _default_space(n_clusters: int) -> dict[str, tuple]:
    return {
        "max_leaves": ("int_log", max(2 * n_clusters, 16), max(64 * n_clusters, 2048)),
        "feature": ("cat", ["spherical", "diagonal", "full"]),
        "normalize": ("cat", [False, True]),
    }


def _sample(rng: np.random.Generator, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "int_log":
        return round(float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2])))))
    if kind == "cat":
        return spec[1][int(rng.integers(len(spec[1])))]
    raise ValueError(
        f"unknown parameter spec {spec!r}; use ('int_log', lo, hi) or ('cat', [values])"
    )


def _n_labels(labels: np.ndarray) -> int:
    return int(np.sum(np.unique(labels) >= 0))


def _score(x: np.ndarray, labels: np.ndarray, y: np.ndarray | None, objective: str) -> float:
    if _n_labels(labels) < 2:
        return _WORST[objective]
    if objective == "ari":
        assert y is not None  # tune() guarantees labels are present for objective='ari'
        return adjusted_rand(y, labels)
    return _METRICS[objective](x, labels)


def _evaluate(
    x: np.ndarray,
    n_clusters: int,
    params: dict,
    y: np.ndarray | None,
    objective: str,
    seed: int,
    fixed: dict,
) -> Trial:
    from . import Betula  # lazy: avoids an import cycle with the package __init__

    start = time.perf_counter()
    est = Betula(n_clusters=n_clusters, seed=seed, **params, **fixed)
    labels = np.asarray(est.fit_predict(x))
    elapsed = time.perf_counter() - start
    return Trial(
        params=params,
        score=_score(x, labels, y, objective),
        n_leaves=len(est.microcluster_centers_),
        time_s=elapsed,
    )


def _pareto(trials: list[Trial], objective: str) -> list[Trial]:
    """Non-dominated front over (quality, −memory, −time), all recast as higher-is-better."""
    sign = 1.0 if _MAXIMIZE[objective] else -1.0
    pts = [(sign * t.score, -float(t.n_leaves), -t.time_s) for t in trials]
    front = []
    for i, a in enumerate(pts):
        dominated = any(
            all(b[d] >= a[d] for d in range(3)) and any(b[d] > a[d] for d in range(3))
            for j, b in enumerate(pts)
            if j != i
        )
        if not dominated:
            front.append(trials[i])
    return front


def _finalize(trials: list[Trial], objective: str, multi_objective: bool) -> TuneResult:
    # drop only the worst sentinel, so a good-direction +inf (a perfect clustering) still wins
    pool = [t for t in trials if t.score != _WORST[objective]] or trials
    pick = max if _MAXIMIZE[objective] else min
    best = pick(pool, key=lambda t: t.score)
    return TuneResult(
        best_params=best.params,
        best_score=best.score,
        trials=trials,
        pareto=_pareto(pool, objective) if multi_objective else [],
    )


def tune(
    x: np.ndarray,
    n_clusters: int,
    *,
    space: dict[str, tuple] | None = None,
    y: np.ndarray | None = None,
    objective: str = "calinski_harabasz",
    n_trials: int = 30,
    sampler: str = "random",
    multi_objective: bool = False,
    seed: int = 0,
    **fixed: Any,
) -> TuneResult:
    """Search betula's CF-representation knobs for the best clustering of ``x`` into ``n_clusters``.

    Parameters
    ----------
    x, n_clusters
        Data and the number of clusters to fit each trial with.
    space
        Search space ``{param: spec}`` where each ``spec`` is ``("int_log", lo, hi)`` or
        ``("cat", [values])``. Defaults to sweeping ``max_leaves``, ``feature`` and ``normalize``.
    y
        Ground-truth labels; required only for ``objective="ari"``.
    objective
        ``"calinski_harabasz"`` (default, higher better), ``"davies_bouldin"`` (lower better),
        ``"dbcv"`` (density-based, higher better — use for the HDBSCAN-CF / DbStream density heads,
        where the convex metrics mislead), or ``"ari"`` (needs ``y``).
    n_trials, seed
        Search budget and RNG seed.
    sampler
        ``"random"`` (NumPy, no extra deps) or ``"optuna"`` (needs ``betula-cluster[tune]``):
        TPE for single-objective, NSGA-II for the ``multi_objective`` Pareto front.
    multi_objective
        If true, also return the (quality / memory=``n_leaves`` / speed=fit-time) Pareto front.
    **fixed
        Extra ``Betula`` keyword arguments held constant across trials (e.g. ``method="gmm"``).

    Returns
    -------
    TuneResult
        ``best_params`` / ``best_score`` / ``trials`` (and ``pareto`` when ``multi_objective``).
    """
    x = np.asarray(x, dtype=np.float64)
    if objective not in _MAXIMIZE:
        raise ValueError(f"unknown objective {objective!r}; choose one of {sorted(_MAXIMIZE)}")
    if objective == "ari" and y is None:
        raise ValueError("objective='ari' requires ground-truth labels y")
    space = space or _default_space(n_clusters)
    if sampler == "optuna":  # pragma: no cover - optional Optuna backend
        return _tune_optuna(
            x, n_clusters, space, y, objective, n_trials, multi_objective, seed, fixed
        )
    if sampler != "random":
        raise ValueError(f"unknown sampler {sampler!r}; use 'random' or 'optuna'")
    rng = np.random.default_rng(seed)
    trials = [
        _evaluate(
            x, n_clusters, {k: _sample(rng, s) for k, s in space.items()}, y, objective, seed, fixed
        )
        for _ in range(n_trials)
    ]
    return _finalize(trials, objective, multi_objective)


def _tune_optuna(  # pragma: no cover - optional Optuna backend
    x: np.ndarray,
    n_clusters: int,
    space: dict[str, tuple],
    y: np.ndarray | None,
    objective: str,
    n_trials: int,
    multi_objective: bool,
    seed: int,
    fixed: dict,
) -> TuneResult:
    try:
        import optuna  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sampler='optuna' requires Optuna — install with: pip install 'betula-cluster[tune]'"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[Trial] = []

    def objective_fn(trial: Any) -> Any:
        params = {}
        for name, spec in space.items():
            if spec[0] == "int_log":
                params[name] = trial.suggest_int(name, spec[1], spec[2], log=True)
            else:
                params[name] = trial.suggest_categorical(name, spec[1])
        evaluated = _evaluate(x, n_clusters, params, y, objective, seed, fixed)
        trials.append(evaluated)
        if multi_objective:
            return evaluated.score, float(evaluated.n_leaves), evaluated.time_s
        return evaluated.score

    # Best sampler per task (both seeded for reproducibility):
    #   - single-objective -> TPE: Optuna's strongest general sampler, sample-efficient even at a
    #     low trial budget; the right default when you optimise one metric (CH / DB / ARI).
    #   - multi-objective  -> NSGA-II: Optuna's default for Pareto search, an evolutionary sampler
    #     that evolves the whole (quality / memory / speed) front. It needs the budget to span
    #     several generations, so keep the population well under n_trials (a population >= the
    #     budget degenerates to one random generation). betula fits are cheap -- run many trials.
    quality_dir = "maximize" if _MAXIMIZE[objective] else "minimize"
    if multi_objective:
        pop = max(4, min(50, n_trials // 4))  # ~4+ generations at the given budget
        sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=pop)
        study = optuna.create_study(
            directions=[quality_dir, "minimize", "minimize"], sampler=sampler
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=quality_dir, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)
    return _finalize(trials, objective, multi_objective)


# ── A-BIRCH: a gap-statistic estimate of the absorption threshold ───────────────────────────────


@dataclass(frozen=True)
class ThresholdEstimate:
    """A gap-statistic estimate of the CF-tree absorption threshold, with its own caveats.

    ``assumptions`` lists, in plain words, every condition of Lorbeer et al. (2018) that the sample
    violates. A non-empty list does not make ``threshold`` useless — it makes it advisory, which is
    the only way this number is ever meant to be read.
    """

    threshold: float
    n_clusters: int
    radius_ratio: float
    separation: float
    assumptions: list[str]


def _kmeans_sample(
    x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25, n_init: int = 3
):
    """Best of `n_init` Lloyd runs from k-means++ seeds, by dispersion.

    The restarts are not a refinement: the gap statistic compares `log W_k` across `k`, so a single
    unlucky seed at one `k` inflates that `W_k` and the rule stops one cluster early. Small samples
    only — this is `O(n k d)` per pass.
    """
    runs = []
    for _ in range(n_init):
        labels, centers = _lloyd(x, k, rng, iters)
        runs.append((_dispersion(x, labels, centers), labels, centers))
    best = min(runs, key=lambda run: run[0])
    return best[1], best[2]


def _lloyd(x: np.ndarray, k: int, rng: np.random.Generator, iters: int):
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        total = d2.sum()
        pick = (
            rng.integers(len(x))
            if total <= 0
            else np.searchsorted(np.cumsum(d2 / total), rng.random())
        )
        centers.append(x[min(int(pick), len(x) - 1)])
    c = np.array(centers)
    labels = np.zeros(len(x), dtype=np.intp)
    for _ in range(iters):
        new = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = x[m].mean(0)
    return labels, c


def _dispersion(x: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """Pooled within-cluster sum of squares, the `W_k` the gap statistic compares against."""
    return float(sum(((x[labels == j] - centers[j]) ** 2).sum() for j in range(len(centers))))


def estimate_threshold(
    X: np.ndarray,
    *,
    k_max: int = 10,
    n_refs: int = 10,
    sample_size: int = 1000,
    seed: int = 0,
) -> ThresholdEstimate:
    """Estimate an absorption ``threshold`` from a sample, A-BIRCH style (Lorbeer et al. 2018).

    Picks ``k`` by the gap statistic (Tibshirani, Walther & Hastie 2001) against uniform references
    over the sample's bounding box, then reports the median cluster RMS radius as the threshold that
    would let the tree keep those clusters apart.

    The method assumes well-separated, near-spherical clusters of comparable size. Those are checked
    rather than assumed: the returned ``assumptions`` names each one that fails, and the estimate is
    advisory in every case — ``max_leaves`` remains the knob that binds.
    """
    rows = np.asarray(X, dtype=np.float64)
    if rows.ndim != 2 or len(rows) < 2:
        raise ValueError(
            f"estimate_threshold needs a 2-D sample of at least 2 rows, got {rows.shape}"
        )
    rng = np.random.default_rng(seed)
    if len(rows) > sample_size:
        rows = rows[rng.choice(len(rows), sample_size, replace=False)]
    k_top = max(2, min(k_max, len(rows) - 1))
    lo, hi = rows.min(0), rows.max(0)
    # Validate the one condition every step below relies on, rather than guarding each of them: a
    # sample with no spread has no dispersion to compare against a reference, no cluster radius to
    # report, and no separation to measure.
    if not np.any(hi > lo):
        raise ValueError(
            "estimate_threshold needs a sample with some spread; every row is identical"
        )

    gaps, sks = [], []
    for k in range(1, k_top + 1):
        labels, centers = _kmeans_sample(rows, k, rng)
        logw = np.log(max(_dispersion(rows, labels, centers), 1e-300))
        refs = []
        for _ in range(n_refs):
            ref = rng.uniform(lo, hi, rows.shape)
            rl, rc = _kmeans_sample(ref, k, rng)
            refs.append(np.log(max(_dispersion(ref, rl, rc), 1e-300)))
        gaps.append(float(np.mean(refs) - logw))
        sks.append(float(np.std(refs) * np.sqrt(1.0 + 1.0 / n_refs)))

    # One-standard-error rule against the *best* k, not against k+1. Tibshirani's original form —
    # the first k with `gap(k) >= gap(k+1) - s(k+1)` — stops at the first dip in the gap curve, and
    # the curve is not monotone: on four blobs 40 sigma apart it reads gap(2) = -0.186 against
    # gap(3) = -0.274, so the rule returns k = 2 while the true k = 4 sits at gap 3.379. Taking the
    # simplest k within one standard error of the maximum keeps the "prefer fewer clusters" spirit
    # and still finds it, and it returns k = 1 on structureless data, where every gap ties.
    best = int(np.argmax(gaps))
    cut = gaps[best] - sks[best]
    undominated = [k for k in range(1, k_top + 1) if gaps[k - 1] >= cut]
    local = [k for k in undominated if k == k_top or gaps[k - 1] >= gaps[k] - sks[k]]
    chosen = local[0] if local else undominated[0]

    labels, centers = _kmeans_sample(rows, chosen, rng)
    radii = np.array(
        [
            np.sqrt(((rows[labels == j] - centers[j]) ** 2).sum(1).mean())
            if (labels == j).any()
            else 0.0
            for j in range(chosen)
        ]
    )
    # `tiny` rather than a branch on zero: a cluster with no radius is one whose points coincide,
    # and both readings that follow from that are the ones we want — its size differs from its
    # neighbours' by an unbounded factor, and it is unboundedly well separated from them.
    tiny = np.finfo(np.float64).tiny
    ratio = float(radii.max() / max(float(radii.min()), tiny))
    separation = float("inf")
    if chosen > 1:
        pair = np.sqrt(((centers[:, None, :] - centers[None]) ** 2).sum(-1))
        pair[np.diag_indices(chosen)] = np.inf
        i, j = np.unravel_index(np.argmin(pair), pair.shape)
        separation = float(pair[i, j] / max(float(radii[i] + radii[j]), tiny))

    notes = []
    if chosen == 1:
        notes.append(
            "the gap statistic found no structure (k = 1): the estimate is the whole "
            "sample's radius"
        )
    if chosen == k_top:
        notes.append(
            f"k hit the search ceiling k_max={k_max}: the true k may be larger, so the "
            "threshold is an over-estimate"
        )
    if ratio > 2.0:
        notes.append(
            f"cluster radii differ by {ratio:.1f}x: A-BIRCH assumes comparable sizes, and "
            "one global threshold cannot serve both"
        )
    if separation < 2.0:
        notes.append(
            f"the closest pair is only {separation:.2f} combined radii apart: A-BIRCH "
            "assumes well-separated clusters"
        )
    return ThresholdEstimate(
        threshold=float(np.median(radii)),
        n_clusters=int(chosen),
        radius_ratio=ratio,
        separation=separation,
        assumptions=notes,
    )
