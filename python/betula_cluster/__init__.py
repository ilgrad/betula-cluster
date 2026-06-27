"""Fast, numerically stable BETULA clustering with a Rust core.

The compiled engine lives in :mod:`betula_cluster._core`. ``fit_predict`` is re-exported verbatim;
``Betula`` is a thin, scikit-learn-compatible estimator around the engine. Keeping the estimator in
Python (rather than exposing the ``#[pyclass]`` directly) is what makes ``sklearn.base.clone`` /
``Pipeline`` / ``GridSearchCV`` work: those rely on ``get_params`` returning the *same* objects the
constructor was handed, which a compiled getter (returning freshly built Python objects) cannot do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# `_core` is the compiled Rust extension — opaque to source-level type checkers; the public API is
# typed via `__init__.pyi` (validated against the runtime by `mypy.stubtest`).
from ._core import Betula as _CoreBetula  # type: ignore
from ._core import DbStream as _CoreDbStream  # type: ignore
from ._core import DdSketch, KllSketch, fit_predict  # type: ignore
from ._core import DenStream as _CoreDenStream  # type: ignore
from ._core import KPrototypes as _CoreKPrototypes  # type: ignore
from ._core import fit_predict_sparse as _core_fit_predict_sparse  # type: ignore

__all__ = [
    "Betula",
    "Coreset",
    "DbStream",
    "DdSketch",
    "DenStream",
    "KPrototypes",
    "KllSketch",
    "MapperGraph",
    "fit_predict",
    "fit_predict_sparse",
]


@dataclass(frozen=True)
class MapperGraph:
    """A Mapper topological-skeleton graph over a fitted model's leaf microclusters.

    Each node is a connected group of microclusters inside one cover bin; edges link nodes that
    share microclusters (from the cover overlap). ``branch_points`` are nodes where the shape splits
    (degree ≥ 3); ``bridges`` index the ``edges`` whose removal would disconnect the graph — thin
    links between otherwise separate regions (e.g. leakage between topics in an embedding).
    """

    node_members: list[np.ndarray]  # microcluster indices per node
    node_mass: np.ndarray  # (n_nodes,) total mass per node
    node_bin: np.ndarray  # (n_nodes,) cover bin per node
    node_lens: np.ndarray  # (n_nodes,) mean lens value per node
    node_centroids: np.ndarray  # (n_nodes, dim) mass-weighted centroid per node
    edges: np.ndarray  # (n_edges, 3): columns (node_a, node_b, shared microcluster count)
    branch_points: np.ndarray  # node indices with degree ≥ 3
    bridges: np.ndarray  # indices into `edges` that are bridges

    @property
    def n_nodes(self) -> int:
        return len(self.node_members)

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def to_networkx(self):
        """Build a ``networkx.Graph`` (requires ``networkx``); nodes carry mass/bin/lens/centroid,
        edges carry ``weight`` and a boolean ``bridge`` flag."""
        import importlib

        try:
            nx = importlib.import_module("networkx")  # optional dependency, resolved at call time
        except ImportError as exc:  # pragma: no cover - optional visualization dependency
            raise ImportError(
                "MapperGraph.to_networkx() requires networkx (`pip install networkx`)"
            ) from exc

        g = nx.Graph()
        for i in range(self.n_nodes):
            g.add_node(
                i,
                mass=float(self.node_mass[i]),
                bin=int(self.node_bin[i]),
                lens=float(self.node_lens[i]),
                centroid=self.node_centroids[i],
            )
        bridge_set = set(self.bridges.tolist())
        for e, row in enumerate(self.edges):
            g.add_edge(int(row[0]), int(row[1]), weight=int(row[2]), bridge=e in bridge_set)
        return g


@dataclass(frozen=True)
class Coreset:
    """A weighted-point coreset: the CF-tree leaf microclusters as ``(centers, weights, radii)``.

    Each row is one microcluster — a numerically stable summary of the points absorbed into it. The
    set is bounded by ``max_leaves`` and built in a single streaming pass, so fitting a weighted
    clustering / classifier on it is competitive with fitting on the full data at a fraction of the
    cost.
    """

    centers: np.ndarray  # (n_microclusters, dim) mass-weighted centroids
    weights: np.ndarray  # (n_microclusters,) effective point mass
    radii: np.ndarray  # (n_microclusters,) RMS radius sqrt(ssd / weight)

    @property
    def n_points(self) -> float:
        """Total mass (≈ number of points summarized)."""
        return float(self.weights.sum())


def _farthest_point_order(points: np.ndarray, k: int) -> np.ndarray:
    """Greedy farthest-point sampling order (start from the centroid-nearest point), length ≤ k."""
    n = len(points)
    start = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    chosen = [start]
    dist = np.linalg.norm(points - points[start], axis=1)
    while len(chosen) < min(k, n):
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen, dtype=np.int64)


def _dim_of(X) -> int | None:
    """Feature dimension of a 2-D input, or ``None`` (unknown / not 2-D, incl. ``X is None``) — used
    to size the memory budget. ``np.asarray(None)`` is 0-D, so it falls through to ``None``."""
    a = np.asarray(X)
    return int(a.shape[1]) if a.ndim == 2 else None


def _constraint_pairs(c) -> np.ndarray:
    """Normalize a constraint argument to an ``(m, 2)`` int64 array of row-index pairs; ``None``
    becomes an empty set."""
    if c is None:
        return np.empty((0, 2), dtype=np.int64)
    a = np.asarray(c, dtype=np.int64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("must_link / cannot_link must be an (m, 2) array of row-index pairs")
    return a


def _bytes_per_leaf(feature: str, dim: int) -> int:
    """Approximate resident bytes per CF-tree leaf for ``feature`` at ``dim`` (mean/scatter arrays +
    CF / node / Vec overhead). Used only to translate ``memory_budget_mb`` into ``max_leaves`` — a
    rough target for the tree's resident size, not an exact accounting."""
    base = {
        "spherical": 8 * dim + 16,
        "diagonal": 16 * dim + 16,
        "full": 8 * dim + 4 * dim * (dim + 1),  # mean + packed upper-triangular scatter
        "fd": 16 * dim + 16,
    }.get(feature, 16 * dim + 16)
    return base + 96


def _budget_max_leaves(budget_mb: float, dim: int, feature: str, branching: int) -> int:
    """Translate a memory budget (MiB) into a ``max_leaves`` cap for the resident CF-tree."""
    derived = int(budget_mb * 1_048_576 / _bytes_per_leaf(feature, dim))
    return max(branching + 1, min(derived, 10_000_000))


def _to_csr(X):
    """If ``X`` is a scipy sparse matrix, return its CSR arrays as
    ``(data_f64, indices_i64, indptr_i64, n_features)`` (the dense matrix is never materialized);
    return ``None`` for dense input. Duck-typed, so scipy is not a hard dependency."""
    if not hasattr(X, "tocsr"):
        return None
    m = X.tocsr()
    return (
        np.ascontiguousarray(m.data, dtype=np.float64),
        np.ascontiguousarray(m.indices, dtype=np.int64),
        np.ascontiguousarray(m.indptr, dtype=np.int64),
        int(m.shape[1]),
    )


def fit_predict_sparse(
    X, n_clusters=8, method="kmeans", threshold=0.0, max_leaves=2048, max_iter=100, seed=0
):
    """One-shot ``O(nnz)`` clustering of a ``scipy.sparse`` matrix.

    Summarises rows into spherical micro-clusters touching only the non-zeros (a flat leader pass
    bounded by ``max_leaves``), clusters those with a parametric head (``kmeans`` / ``gmm`` /
    ``gmm-full`` / ``ward``), and labels each row by its nearest micro-cluster. For very
    high-dimensional sparse data this avoids the ``O(d)``-per-row cost of the dense path. It uses
    the expanded squared-distance form for speed and so does **not** carry the dense path's
    cancellation-free guarantee (accurate for sparse rows far from the dense centroid; see the
    library docs). Returns one ``int64`` label per row.
    """
    csr = _to_csr(X)
    if csr is None:
        raise ValueError(
            "fit_predict_sparse requires a scipy.sparse matrix (use fit_predict for dense input)"
        )
    return _core_fit_predict_sparse(
        *csr,
        n_clusters=n_clusters,
        method=method,
        threshold=threshold,
        max_leaves=max_leaves,
        max_iter=max_iter,
        seed=seed,
    )


# Defaults mirror `_core.Betula.__new__`; order defines `get_params` / `__repr__` order.
_DEFAULTS = {
    "n_clusters": 8,
    "feature": "diagonal",
    "method": "gmm",
    "threshold": 0.0,
    "branching": 32,
    "leaf_cap": 32,
    "max_leaves": 2000,
    "max_iter": 100,
    "min_samples": 5,
    "min_cluster_size": 5,
    "seed": 0,
    "distance": "euclidean",
    "absorb": "euclidean",
    "chi2_p": 0.95,
    "chi2_scale": 0.0,
    "decay": 1.0,
    "normalize": False,
    "huber_k": None,
    "memory_budget_mb": None,
}
_PARAM_NAMES = tuple(_DEFAULTS)


class Betula:
    """Streaming, scikit-learn-style BETULA estimator.

    Parameters are validated lazily — when the engine is built at ``fit`` / ``partial_fit`` time —
    following the scikit-learn convention that ``__init__`` only records its arguments verbatim.
    """

    def __init__(
        self,
        n_clusters=8,
        feature="diagonal",
        method="gmm",
        threshold=0.0,
        branching=32,
        leaf_cap=32,
        max_leaves=2000,
        max_iter=100,
        min_samples=5,
        min_cluster_size=5,
        seed=0,
        distance="euclidean",
        absorb="euclidean",
        chi2_p=0.95,
        chi2_scale=0.0,
        decay=1.0,
        normalize=False,
        huber_k=None,
        memory_budget_mb=None,
    ):
        self.n_clusters = n_clusters
        self.feature = feature
        self.method = method
        self.threshold = threshold
        self.branching = branching
        self.leaf_cap = leaf_cap
        self.max_leaves = max_leaves
        self.max_iter = max_iter
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.seed = seed
        self.distance = distance
        self.absorb = absorb
        self.chi2_p = chi2_p
        self.chi2_scale = chi2_scale
        self.decay = decay
        self.normalize = normalize
        # Robust insertion: clamp each point to within ``huber_k`` per-dim stds of its target
        # microcluster before folding it in, so outliers cannot stretch a centroid/radius. ``None``
        # disables it. Most useful for streaming, where re-fitting on cleaned data is not an option.
        self.huber_k = huber_k
        # When set, max_leaves is derived from this budget (+ dim + feature) at fit time: a target
        # for the CF-tree resident size (MiB), not total RSS. Most useful for streaming.
        self.memory_budget_mb = memory_budget_mb
        self._est = None
        self._effective_max_leaves = max_leaves

    # ── scikit-learn parameter protocol ──────────────────────────────────────────────────────
    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in _PARAM_NAMES}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in _DEFAULTS:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator Betula. "
                    f"Valid parameters are: {sorted(_PARAM_NAMES)}."
                )
            setattr(self, key, value)
        self._est = None  # params changed → any prior fit is stale
        return self

    # ── fit / predict ────────────────────────────────────────────────────────────────────────
    def _build(self, dim=None):
        # `memory_budget_mb` is a wrapper-only knob: strip it and, when set (and the dimension is
        # known), translate it into the `max_leaves` the engine actually uses.
        params = {k: getattr(self, k) for k in _PARAM_NAMES if k != "memory_budget_mb"}
        if self.memory_budget_mb is not None and dim is not None:
            params["max_leaves"] = _budget_max_leaves(
                self.memory_budget_mb, dim, self.feature, self.branching
            )
        self._effective_max_leaves = params["max_leaves"]
        return _CoreBetula(**params)

    def fit(self, X, y=None, must_link=None, cannot_link=None):
        # `X`: a dense float array or a scipy.sparse matrix (CSR-routed; never fully densified).
        # `must_link` / `cannot_link`: optional (m, 2) row-index pairs → semi-supervised COP-KMeans.
        if must_link is not None or cannot_link is not None:
            return self._fit_constrained(X, must_link, cannot_link)
        csr = _to_csr(X)
        if csr is not None:
            est = self._build(csr[3])
            est.fit_csr(*csr)
        else:
            est = self._build(_dim_of(X))
            est.fit(X)
        self._est = est
        return self

    def fit_predict(self, X, y=None, must_link=None, cannot_link=None):
        if must_link is not None or cannot_link is not None:
            self._fit_constrained(X, must_link, cannot_link)
            return np.asarray(self.predict(X))
        csr = _to_csr(X)
        if csr is not None:
            est = self._build(csr[3])
            labels = est.fit_predict_csr(*csr)
        else:
            est = self._build(_dim_of(X))
            labels = est.fit_predict(X)
        self._est = est
        return labels

    def _fit_constrained(self, X, must_link, cannot_link):
        # Semi-supervised COP-KMeans. Constraints are row-index pairs into X, honoured at the
        # microcluster granularity: a cannot-link between two points the tree compressed into one
        # leaf is reported as infeasible (lower ``threshold`` to keep such points separable).
        if self.method != "kmeans":
            raise ValueError("constraints (must_link / cannot_link) require method='kmeans'")
        if _to_csr(X) is not None:
            raise ValueError("constrained clustering requires a dense array, not a sparse matrix")
        ml = _constraint_pairs(must_link)
        cl = _constraint_pairs(cannot_link)
        est = self._build(_dim_of(X))
        est.fit_constrained(X, ml, cl)
        self._est = est
        return self

    def partial_fit(self, X=None, y=None):
        csr = None if X is None else _to_csr(X)
        if csr is not None:
            if self._est is None:
                self._est = self._build(csr[3])
            self._est.partial_fit_csr(*csr)
        else:
            if self._est is None:
                self._est = self._build(_dim_of(X))
            self._est.partial_fit(X)
        return self

    def predict(self, X):
        if self._est is None:
            raise ValueError("This Betula instance is not fitted yet; call 'fit' first.")
        csr = _to_csr(X)
        if csr is not None:
            return self._est.predict_csr(*csr)
        return self._est.predict(X)

    # ── fitted attributes ────────────────────────────────────────────────────────────────────
    @property
    def n_clusters_(self):
        if self._est is None:
            raise AttributeError("This Betula instance is not fitted yet.")
        return self._est.n_clusters_

    @property
    def n_leaves_(self):
        if self._est is None:
            raise AttributeError("This Betula instance is not fitted yet.")
        return self._est.n_leaves_

    @property
    def n_rebuilds_(self):
        """How many times the CF-tree rebuilt under the leaf bound; high ⇒ thrashing."""
        return self._require_fit().n_rebuilds_

    @property
    def threshold_(self):
        """Current CF-tree absorption threshold (grows as it rebuilds)."""
        return self._require_fit().threshold_

    @property
    def effective_max_leaves_(self):
        """The ``max_leaves`` actually used: derived from ``memory_budget_mb`` if set, else
        configured."""
        self._require_fit()
        return self._effective_max_leaves

    # ── persistence ──────────────────────────────────────────────────────────────────────────
    def save(self, path):
        if self._est is None:
            raise ValueError("This Betula instance is not fitted yet; nothing to save.")
        self._est.save(path)

    @classmethod
    def load(cls, path):
        est = _CoreBetula.load(path)
        obj = cls(**est.get_params())
        obj._est = est
        obj._effective_max_leaves = obj.max_leaves  # the (already resolved) cap baked into the tree
        return obj

    # ── inspectability: dataset structure, not just labels ───────────────────────────────────
    def _require_fit(self):
        if self._est is None:
            raise AttributeError("This Betula instance is not fitted yet.")
        return self._est

    @property
    def microcluster_centers_(self):
        """Leaf (microcluster) centroids — ``(n_microclusters, dim)``."""
        return self._require_fit().microcluster_centers_

    @property
    def microcluster_weights_(self):
        """Leaf effective point mass — ``(n_microclusters,)``."""
        return self._require_fit().microcluster_weights_

    @property
    def microcluster_radii_(self):
        """Leaf RMS radius — ``(n_microclusters,)``."""
        return self._require_fit().microcluster_radii_

    @property
    def cluster_centers_(self):
        """Macro-cluster centroids — ``(n_clusters, dim)``; requires a finalized clustering."""
        return self._require_fit().cluster_centers_

    @property
    def cluster_radii_(self):
        """Macro-cluster RMS radius — ``(n_clusters,)``; requires a finalized clustering."""
        return self._require_fit().cluster_radii_

    @property
    def cluster_sizes_(self):
        """Macro-cluster total point mass — ``(n_clusters,)``; requires a finalized clustering."""
        return self._require_fit().cluster_sizes_

    def assign_microclusters(self, X):
        """Nearest leaf index per row (matches ``microcluster_centers_`` order)."""
        return self._require_fit().assign_microclusters(X)

    def outlier_scores(self, X):
        """Per-row distance to its assigned cluster centroid / that cluster's RMS radius."""
        return self._require_fit().outlier_scores(X)

    def summary(self):
        """A compact dict describing the dataset's structure (microclusters + macro clusters)."""
        est = self._require_fit()
        radii = est.microcluster_radii_
        weights = est.microcluster_weights_
        info = {
            "n_samples": round(float(weights.sum())),
            "n_microclusters": int(est.n_leaves_),
            "mean_microcluster_radius": float(radii.mean()) if radii.size else 0.0,
        }
        if est.n_clusters_ > 0:  # clustering has been finalized
            sizes = est.cluster_sizes_
            cradii = est.cluster_radii_
            info["n_clusters"] = int(est.n_clusters_)
            info["largest_cluster_size"] = round(float(sizes.max())) if sizes.size else 0
            info["mean_cluster_radius"] = float(cradii.mean()) if cradii.size else 0.0
        return info

    def find_outliers(self, X, top_k=100):
        """Row indices of the ``top_k`` most outlying points (highest score first)."""
        scores = self.outlier_scores(X)
        return np.argsort(scores)[::-1][:top_k]

    def sample_representatives(self, X, k=5):
        """For each cluster, the row indices of the ``k`` points nearest its centroid."""
        centers = self.cluster_centers_
        labels = np.asarray(self.predict(X))
        rows = np.asarray(X)
        reps = {}
        for c in range(centers.shape[0]):
            members = np.flatnonzero(labels == c)
            if members.size == 0:  # pragma: no cover - empty component (gap in label values)
                continue
            d = np.linalg.norm(rows[members] - centers[c], axis=1)
            reps[c] = members[np.argsort(d)[:k]]
        return reps

    def find_near_duplicates(self, X, radius):
        """Groups (row-index arrays) of points sharing a microcluster tighter than ``radius``."""
        leaf = np.asarray(self.assign_microclusters(X))
        radii = self.microcluster_radii_
        groups = []
        for j in np.flatnonzero(radii < radius):
            members = np.flatnonzero(leaf == j)
            if members.size >= 2:
                groups.append(members)
        return groups

    def mapper(
        self,
        lens="density",
        resolution=10,
        gain=0.3,
        link_scale=1.0,
        min_node_mass=0.0,
        density_k=5,
        coordinate=0,
    ):
        """Build a Mapper topological-skeleton :class:`MapperGraph` over the fitted microclusters.

        TDA Mapper specialised to BETULA: a ``lens`` filter (``"density"`` | ``"radius"`` |
        ``"l2norm"`` | ``"coordinate"`` | ``"eccentricity"``) is covered by ``resolution`` bins
        overlapping by ``gain``; microclusters in a bin are single-linked at ``link_scale`` × the
        bin's median nearest-neighbour gap; one node per (bin, component).
        It surfaces non-convex structure, branch points and bridges (topic leakage) over the
        ``M << N`` microclusters — an exploration tool, not a partition. Build the model first.
        """
        d = self._require_fit().mapper(
            lens=lens,
            resolution=resolution,
            gain=gain,
            link_scale=link_scale,
            min_node_mass=min_node_mass,
            density_k=density_k,
            coordinate=coordinate,
        )
        return MapperGraph(
            node_members=[np.asarray(m, dtype=np.int64) for m in d["node_members"]],
            node_mass=d["node_mass"],
            node_bin=d["node_bin"],
            node_lens=d["node_lens"],
            node_centroids=d["node_centroids"],
            edges=d["edges"],
            branch_points=d["branch_points"],
            bridges=d["bridges"],
        )

    def mapper_stability(self, resolutions=None, **mapper_kwargs):
        """Sweep Mapper ``resolution`` and report how the topology persists across scale.

        Returns a list of dicts (one per resolution) with ``resolution``, ``n_nodes``, ``n_edges``,
        ``n_branch_points``, ``n_bridges``, ``n_components`` (β₀, connected components) and
        ``n_loops`` (β₁ = edges − nodes + components, the number of independent cycles). Features
        constant across many resolutions are real structure; ones that flicker are binning
        artefacts — the Mapper analogue of a persistence diagram, without cross-scale node matching.

        ``resolutions`` defaults to ``range(4, 30, 2)``; ``mapper_kwargs`` (``lens``, ``gain``,
        ``link_scale`` …) pass straight through to :meth:`mapper`. Build the model first.
        """
        self._require_fit()
        if "resolution" in mapper_kwargs:
            raise ValueError(
                "`resolution` is the swept axis of mapper_stability; pass `resolutions=` (a "
                "sequence) instead, with the other Mapper options as keyword arguments."
            )
        if resolutions is None:
            resolutions = range(4, 30, 2)
        rows = []
        for r in resolutions:
            g = self.mapper(resolution=int(r), **mapper_kwargs)
            parent = list(range(g.n_nodes))

            def find(x, parent=parent):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b, _w in g.edges:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[ra] = rb
            components = len({find(i) for i in range(g.n_nodes)})
            rows.append(
                {
                    "resolution": int(r),
                    "n_nodes": g.n_nodes,
                    "n_edges": g.n_edges,
                    "n_branch_points": int(g.branch_points.shape[0]),
                    "n_bridges": int(g.bridges.shape[0]),
                    "n_components": components,
                    "n_loops": max(0, g.n_edges - g.n_nodes + components),
                }
            )
        return rows

    # ── coreset / soft assignment / diagnostics ──────────────────────────────────────────────────
    def export_coreset(self):
        """The CF-tree leaves as a weighted-point :class:`Coreset` (centers, weights, radii) — a
        compact streaming summary of all data seen. Fit any weighted clustering / model on it rather
        than the raw points. Requires a built model."""
        est = self._require_fit()
        return Coreset(
            centers=est.microcluster_centers_,
            weights=est.microcluster_weights_,
            radii=est.microcluster_radii_,
        )

    @property
    def microcluster_proba_(self):
        """Per-microcluster GMM soft responsibilities ``(n_microclusters, k)``. GMM heads only."""
        return self._require_fit().microcluster_proba_

    def predict_proba(self, X):
        """Per-point soft assignment, shape ``(n, n_components)``.

        The **GMM** heads return the true posterior responsibilities (routed via each point's
        microcluster). **k-means / Ward / HDBSCAN** return a heuristic ``softmax(−d²/2τ²)`` over the
        cluster centroids (``τ`` = mean cluster radius) — a confidence *proxy*, **not** a calibrated
        posterior. Columns are component indices aligned with :meth:`predict`."""
        est = self._require_fit()
        if self.method in ("gmm", "gmm-full"):
            leaf_proba = est.microcluster_proba_
            leaves = np.asarray(est.assign_microclusters(X))
            return leaf_proba[leaves]
        centers = np.asarray(est.cluster_centers_, dtype=np.float64)
        rows = np.asarray(X, dtype=np.float64)
        d2 = (
            (rows * rows).sum(1)[:, None]
            + (centers * centers).sum(1)[None, :]
            - 2.0 * rows @ centers.T
        )
        np.maximum(d2, 0.0, out=d2)
        radii = est.cluster_radii_
        tau = max(float(radii.mean()), 1e-12)
        logits = -d2 / (2.0 * tau * tau)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def assignment_confidence(self, X):
        """Per-point confidence in ``[0, 1]`` = the max soft-assignment probability (see
        :meth:`predict_proba`); low values flag boundary / ambiguous points."""
        return self.predict_proba(X).max(axis=1)

    def diagnostics(self):
        """A richer structural report than :meth:`summary` — compression, microcluster-radius
        percentiles, rebuild count, and (once finalized) cluster mass spread."""
        est = self._require_fit()
        radii = est.microcluster_radii_
        weights = est.microcluster_weights_
        n = float(weights.sum())
        nlv = int(est.n_leaves_)
        info = {
            "n_samples": round(n),
            "n_microclusters": nlv,
            "compression_ratio": n / nlv,
            "n_rebuilds": int(est.n_rebuilds_),
            "threshold": float(est.threshold_),
            "microcluster_radius_p50": float(np.percentile(radii, 50)),
            "microcluster_radius_p90": float(np.percentile(radii, 90)),
            "microcluster_radius_p99": float(np.percentile(radii, 99)),
        }
        if est.n_clusters_ > 0:
            sizes = est.cluster_sizes_
            info["n_clusters"] = int(est.n_clusters_)
            info["cluster_mass_min"] = float(sizes.min())
            info["cluster_mass_median"] = float(np.median(sizes))
            info["cluster_mass_max"] = float(sizes.max())
            info["mean_cluster_radius"] = float(est.cluster_radii_.mean())
        return info

    def representatives(self, X, cluster_id, method="medoid", k=5):
        """Row indices of ``k`` representatives of ``cluster_id``. ``method``: ``medoid`` (nearest
        centroid), ``boundary`` (farthest in-cluster), ``outlier`` (highest outlier score),
        ``diverse`` (farthest-point sampling). Empty if the cluster has no predicted members."""
        rows = np.asarray(X, dtype=np.float64)
        members = np.flatnonzero(np.asarray(self.predict(X)) == cluster_id)
        if members.size == 0:
            return np.array([], dtype=np.int64)
        if method == "outlier":
            order = np.argsort(np.asarray(self.outlier_scores(X))[members])[::-1]
        elif method == "diverse":
            order = _farthest_point_order(rows[members], k)
        else:
            d = np.linalg.norm(rows[members] - self.cluster_centers_[cluster_id], axis=1)
            if method == "medoid":
                order = np.argsort(d)
            elif method == "boundary":
                order = np.argsort(d)[::-1]
            else:
                raise ValueError("method must be 'medoid', 'boundary', 'outlier' or 'diverse'")
        return members[order[:k]]

    def cluster_profile(self, cluster_id):
        """A JSON-able profile of a macro-cluster (size, radius, center, nearest clusters) — feed to
        an LLM to name it. Geometry only; no data pass needed."""
        centers = self.cluster_centers_
        d = np.linalg.norm(centers - centers[cluster_id], axis=1)
        d[cluster_id] = np.inf
        nearest = np.argsort(d)[:3]
        return {
            "cluster_id": int(cluster_id),
            "size": round(float(self.cluster_sizes_[cluster_id])),
            "radius": float(self.cluster_radii_[cluster_id]),
            "center": centers[cluster_id].tolist(),
            "nearest_clusters": [
                {"cluster_id": int(j), "distance": float(d[j])}
                for j in nearest
                if np.isfinite(d[j])
            ],
        }

    # ── drift monitoring / active learning ───────────────────────────────────────────────────────
    def snapshot(self):
        """A JSON-able snapshot of the current cluster geometry (centers / sizes / radii) for drift
        monitoring across time. Requires a finalized clustering; compare two with
        :meth:`compare_snapshots`."""
        est = self._require_fit()
        return {
            "n_clusters": int(est.n_clusters_),
            "n_microclusters": int(est.n_leaves_),
            "centers": est.cluster_centers_.tolist(),
            "sizes": est.cluster_sizes_.tolist(),
            "radii": est.cluster_radii_.tolist(),
        }

    @staticmethod
    def compare_snapshots(before, after):
        """Drift report between two :meth:`snapshot` dicts. Each ``after`` cluster is matched to its
        nearest ``before`` centroid; reports the centroid shift (absolute and in ``after``-radius
        units) and the mass ratio per match, plus the cluster counts and the worst shift. Both
        snapshots must come from finalized models with ≥ 1 cluster."""
        cb = np.asarray(before["centers"], dtype=np.float64)
        ca = np.asarray(after["centers"], dtype=np.float64)
        sb = np.asarray(before["sizes"], dtype=np.float64)
        sa = np.asarray(after["sizes"], dtype=np.float64)
        ra = np.asarray(after["radii"], dtype=np.float64)
        matches = []
        for j in range(len(ca)):
            d = np.linalg.norm(cb - ca[j], axis=1)
            i = int(np.argmin(d))
            scale = ra[j] if ra[j] > 0 else 1.0
            matches.append(
                {
                    "after": j,
                    "before": i,
                    "centroid_shift": float(d[i]),
                    "centroid_shift_radii": float(d[i] / scale),
                    "mass_ratio": float(sa[j] / sb[i]),
                }
            )
        return {
            "n_clusters_before": int(before["n_clusters"]),
            "n_clusters_after": int(after["n_clusters"]),
            "matches": matches,
            "max_centroid_shift_radii": max(
                (m["centroid_shift_radii"] for m in matches), default=0.0
            ),
        }

    def active_learning_batch(self, X, n=100, strategy="uncertain"):
        """Row indices of the ``n`` most informative points to review/label. ``strategy``:
        ``uncertain`` (lowest :meth:`assignment_confidence`) or ``outlier`` (highest
        :meth:`outlier_scores`) — for human-in-the-loop curation / labeling."""
        if strategy == "uncertain":
            score = -self.assignment_confidence(X)
        elif strategy == "outlier":
            score = np.asarray(self.outlier_scores(X))
        else:
            raise ValueError("strategy must be 'uncertain' or 'outlier'")
        return np.argsort(score)[::-1][:n]

    def __repr__(self):
        changed = ", ".join(
            f"{k}={getattr(self, k)!r}" for k in _PARAM_NAMES if getattr(self, k) != _DEFAULTS[k]
        )
        return f"Betula({changed})"


_DENSTREAM_PARAMS = ("eps", "decay", "beta", "mu")


class DenStream:
    """Streaming **DenStream** density clusterer (Cao et al., SDM 2006) over fading micro-clusters.

    For evolving streams where old data should fade: feed chunks with :meth:`partial_fit`, then
    :meth:`predict` (which finalizes the offline clustering on first call) — or both at once with
    :meth:`fit` / :meth:`fit_predict`. ``eps`` is the micro-cluster radius (tune to the data scale),
    ``decay`` the fading rate λ, and ``beta`` × ``mu`` the promotion/pruning weight (must exceed 1).
    Spherical micro-clusters, ``float64``; ``-1`` labels are noise.
    """

    def __init__(self, eps=1.0, decay=0.25, beta=0.2, mu=10.0):
        self.eps = eps
        self.decay = decay
        self.beta = beta
        self.mu = mu
        self._est = None
        self._need_cluster = False

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in _DENSTREAM_PARAMS}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in _DENSTREAM_PARAMS:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator DenStream. "
                    f"Valid parameters are: {sorted(_DENSTREAM_PARAMS)}."
                )
            setattr(self, key, value)
        self._est = None
        return self

    def _build(self):
        return _CoreDenStream(**self.get_params())

    def _require_fit(self):
        if self._est is None:
            raise AttributeError("This DenStream instance is not fitted yet.")
        return self._est

    def partial_fit(self, X, y=None):
        """Stream a chunk of points into the fading micro-clusters."""
        if self._est is None:
            self._est = self._build()
        self._est.partial_fit(X)
        self._need_cluster = True  # offline labels are now stale
        return self

    def cluster(self):
        """Run the offline step (label the potential micro-clusters) over what has streamed."""
        self._require_fit().cluster()
        self._need_cluster = False
        return self

    def fit(self, X, y=None):
        est = self._build()
        est.fit(X)
        self._est = est
        self._need_cluster = False
        return self

    def fit_predict(self, X, y=None):
        est = self._build()
        labels = est.fit_predict(X)
        self._est = est
        self._need_cluster = False
        return labels

    def predict(self, X):
        """Label rows by their nearest potential micro-cluster (``-1`` = noise); finalizes the
        offline clustering first if points have streamed since the last :meth:`cluster`."""
        est = self._require_fit()
        if self._need_cluster:
            est.cluster()
            self._need_cluster = False
        return est.predict(X)

    @property
    def n_clusters_(self):
        return self._require_fit().n_clusters_

    @property
    def n_microclusters_(self):
        """Number of potential (cluster-eligible) micro-clusters."""
        return self._require_fit().n_microclusters_

    @property
    def microcluster_centers_(self):
        return self._require_fit().microcluster_centers_

    @property
    def microcluster_weights_(self):
        """Potential micro-cluster weights, faded to the current stream time."""
        return self._require_fit().microcluster_weights_

    @property
    def microcluster_radii_(self):
        return self._require_fit().microcluster_radii_

    def __repr__(self):
        return f"DenStream(eps={self.eps}, decay={self.decay}, beta={self.beta}, mu={self.mu})"


_DBSTREAM_PARAMS = ("r", "decay", "alpha", "min_weight")


class DbStream:
    """Streaming **DBSTREAM** density clusterer (Hahsler & Bolaños, 2016) over fading micros.

    Like :class:`DenStream` it fades old data and marks ``-1`` as noise, but it connects
    micro-clusters by **shared density** — the mass of points within radius ``r`` of *both* — rather
    than by mere proximity. This recovers arbitrarily-shaped clusters (chained overlapping
    micro-clusters) and, unlike a distance rule, keeps two close-but-disconnected dense regions
    apart (an empty gap means zero shared density). ``r`` is the radius, ``decay`` the fading rate,
    ``alpha`` the shared-density bridge threshold (a pair links when their overlap mass exceeds
    ``alpha * min_weight``), and ``min_weight`` the weight a micro-cluster needs to form a cluster.
    """

    def __init__(self, r=1.0, decay=0.01, alpha=0.1, min_weight=2.0):
        self.r = r
        self.decay = decay
        self.alpha = alpha
        self.min_weight = min_weight
        self._est = None
        self._need_cluster = False

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in _DBSTREAM_PARAMS}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in _DBSTREAM_PARAMS:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator DbStream. "
                    f"Valid parameters are: {sorted(_DBSTREAM_PARAMS)}."
                )
            setattr(self, key, value)
        self._est = None
        return self

    def _build(self):
        return _CoreDbStream(**self.get_params())

    def _require_fit(self):
        if self._est is None:
            raise AttributeError("This DbStream instance is not fitted yet.")
        return self._est

    def partial_fit(self, X, y=None):
        """Stream a chunk of points into the fading micro-clusters."""
        if self._est is None:
            self._est = self._build()
        self._est.partial_fit(X)
        self._need_cluster = True  # offline labels are now stale
        return self

    def cluster(self):
        """Run the offline step (label micro-clusters via the shared-density graph)."""
        self._require_fit().cluster()
        self._need_cluster = False
        return self

    def fit(self, X, y=None):
        est = self._build()
        est.fit(X)
        self._est = est
        self._need_cluster = False
        return self

    def fit_predict(self, X, y=None):
        est = self._build()
        labels = est.fit_predict(X)
        self._est = est
        self._need_cluster = False
        return labels

    def predict(self, X):
        """Label rows by their nearest micro-cluster within ``r`` (``-1`` = noise); finalizes the
        offline clustering first if points have streamed since the last :meth:`cluster`."""
        est = self._require_fit()
        if self._need_cluster:
            est.cluster()
            self._need_cluster = False
        return est.predict(X)

    @property
    def n_clusters_(self):
        return self._require_fit().n_clusters_

    @property
    def n_microclusters_(self):
        return self._require_fit().n_microclusters_

    @property
    def microcluster_centers_(self):
        return self._require_fit().microcluster_centers_

    @property
    def microcluster_weights_(self):
        """Micro-cluster weights, faded to the current stream time."""
        return self._require_fit().microcluster_weights_

    @property
    def microcluster_radii_(self):
        return self._require_fit().microcluster_radii_

    def __repr__(self):
        return (
            f"DbStream(r={self.r}, decay={self.decay}, alpha={self.alpha}, "
            f"min_weight={self.min_weight})"
        )


_KPROTOTYPES_PARAMS = (
    "n_clusters",
    "categorical",
    "gamma",
    "threshold",
    "max_leaves",
    "max_iter",
    "n_init",
    "seed",
)


class KPrototypes:
    """k-prototypes clustering of **mixed numeric + categorical** data (Huang, 1997).

    ``categorical`` lists the integer-coded categorical column indices of ``X``; the rest are
    numeric. Distance is ``||Δnum||² + gamma · (categorical mismatch)``; ``gamma`` defaults to half
    the mean numeric standard deviation (Huang's heuristic) when ``None``. Rows are summarised into
    bounded mixed micro-clusters (a flat leader pass capped at ``max_leaves``) before clustering, so
    memory stays bounded. Both numeric and categorical columns are required; ``float64``.
    """

    def __init__(
        self,
        n_clusters=8,
        categorical=(),
        gamma=None,
        threshold=0.0,
        max_leaves=2048,
        max_iter=100,
        n_init=4,
        seed=0,
    ):
        self.n_clusters = n_clusters
        self.categorical = categorical
        self.gamma = gamma
        self.threshold = threshold
        self.max_leaves = max_leaves
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed
        self._est = None

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in _KPROTOTYPES_PARAMS}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in _KPROTOTYPES_PARAMS:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator KPrototypes. "
                    f"Valid parameters are: {sorted(_KPROTOTYPES_PARAMS)}."
                )
            setattr(self, key, value)
        self._est = None
        return self

    def _build(self):
        params = self.get_params()
        params["categorical"] = list(params["categorical"])  # the engine expects a list[int]
        return _CoreKPrototypes(**params)

    def _require_fit(self):
        if self._est is None:
            raise AttributeError("This KPrototypes instance is not fitted yet.")
        return self._est

    def fit(self, X, y=None):
        est = self._build()
        est.fit(X)
        self._est = est
        return self

    def fit_predict(self, X, y=None):
        est = self._build()
        labels = est.fit_predict(X)
        self._est = est
        return labels

    def predict(self, X):
        """Label rows by their nearest mixed micro-cluster."""
        return self._require_fit().predict(X)

    @property
    def n_clusters_(self):
        return self._require_fit().n_clusters_

    @property
    def cluster_centroids_(self):
        """Numeric cluster centroids — ``(n_clusters, n_numeric)``."""
        return self._require_fit().cluster_centroids_

    @property
    def cluster_modes_(self):
        """Categorical cluster modes — ``(n_clusters, n_categorical)`` integer codes."""
        return self._require_fit().cluster_modes_

    def __repr__(self):
        return f"KPrototypes(n_clusters={self.n_clusters}, categorical={list(self.categorical)})"
