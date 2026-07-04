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


_METRICS = {"calinski_harabasz": calinski_harabasz, "davies_bouldin": davies_bouldin}
_MAXIMIZE = {"calinski_harabasz": True, "davies_bouldin": False, "ari": True}
_WORST = {"calinski_harabasz": float("-inf"), "davies_bouldin": float("inf"), "ari": float("-inf")}


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
        ``"calinski_harabasz"`` (default, higher better), ``"davies_bouldin"`` (lower better), or
        ``"ari"`` (needs ``y``).
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
