"""Fast, numerically stable BETULA clustering with a Rust core.

The compiled engine lives in :mod:`betula_cluster._core`. ``fit_predict`` is re-exported verbatim;
``Betula`` is a thin, scikit-learn-compatible estimator around the engine. Keeping the estimator in
Python (rather than exposing the ``#[pyclass]`` directly) is what makes ``sklearn.base.clone`` /
``Pipeline`` / ``GridSearchCV`` work: those rely on ``get_params`` returning the *same* objects the
constructor was handed, which a compiled getter (returning freshly built Python objects) cannot do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np

# `_core` is the compiled Rust extension — opaque to source-level type checkers; the public API is
# typed via `__init__.pyi` (validated against the runtime by `mypy.stubtest`).
from ._core import Betula as _CoreBetula  # type: ignore
from ._core import BregmanBetula as _CoreBregmanBetula  # type: ignore
from ._core import DbStream as _CoreDbStream  # type: ignore
from ._core import DdSketch, KllSketch, fit_predict  # type: ignore
from ._core import DenStream as _CoreDenStream  # type: ignore
from ._core import KPrototypes as _CoreKPrototypes  # type: ignore
from ._core import WindowStream as _CoreWindowStream  # type: ignore
from ._core import fit_predict_sparse as _core_fit_predict_sparse  # type: ignore
from ._core import mixture_w2 as _core_mixture_w2  # type: ignore
from .tuning import GapCurve, ThresholdEstimate, TuneResult, estimate_threshold, gap_statistic, tune

try:
    __version__ = version("betula-cluster")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "Betula",
    "BregmanBetula",
    "ConsensusResult",
    "Coreset",
    "DbStream",
    "DdSketch",
    "DenStream",
    "GapCurve",
    "KPrototypes",
    "KllSketch",
    "MapperGraph",
    "ReachabilityPlot",
    "ThresholdEstimate",
    "TuneResult",
    "WindowStream",
    "__version__",
    "consensus",
    "estimate_threshold",
    "fit_predict",
    "fit_predict_sparse",
    "gap_statistic",
    "mixture_w2",
    "tune",
]


@dataclass(frozen=True)
class ReachabilityPlot:
    """An OPTICS reachability plot over a fitted model's leaf microclusters.

    ``reachability[i]`` belongs to ``order[i]``, and ``order[0]``'s entry is ``inf`` — nothing
    reached the first leaf. ``core_distances`` and ``weights`` are in the leaves' own indexing, not
    the sweep's. Plot ``reachability`` against its position and clusters appear as valleys, with a
    peak at the distance you would have to walk to leave one and enter the next.

    It is not an approximation of what ``method="hdbscan"`` does: OPTICS with no cutoff is Prim's
    algorithm on the mutual-reachability graph, so this is the *same* spanning tree that head takes
    its hierarchy from, written in a different order. Every peak is a merge height in that
    hierarchy, and :meth:`labels_at` reproduces its cut exactly.

    One position per **leaf**, so a valley's width is a leaf count, not a point count. Read
    ``weights`` before reading the width: three leaves can hold a hundred thousand points.
    """

    order: np.ndarray  # (n_leaves,) leaf indices in sweep order
    reachability: np.ndarray  # (n_leaves,) aligned with `order`; [0] is inf
    core_distances: np.ndarray  # (n_leaves,) per leaf, leaf indexing
    weights: np.ndarray  # (n_leaves,) mass per leaf, leaf indexing

    def labels_at(self, eps: float) -> np.ndarray:
        r"""DBSCAN\* at ``eps``, per leaf, read off the plot; ``-1`` is noise.

        Ankerst et al.'s ExtractDBSCAN-Clustering: walk the order and start a segment wherever the
        reachability rises above ``eps``, then drop every leaf whose core distance exceeds it. The
        segments are the connected components of the mutual-reachability tree under ``eps``, so this
        is DBSCAN\* on the summary and not a lookalike — there are no border points, which is the
        ``*`` in the name.
        """
        eps = float(eps)
        labels = np.full(len(self.order), -1, dtype=np.int64)
        segment = np.cumsum(np.r_[np.zeros(1, dtype=np.int64), self.reachability[1:] > eps])
        labels[self.order] = segment
        labels[self.core_distances > eps] = -1
        live = np.unique(labels[labels >= 0])
        remap = np.full(int(segment[-1]) + 1, -1, dtype=np.int64)
        remap[live] = np.arange(len(live))
        return np.where(labels >= 0, remap[labels], -1)


@dataclass(frozen=True)
class MapperGraph:
    """A Mapper topological-skeleton graph over a fitted model's leaf microclusters.

    Each node is a connected group of microclusters inside one cover bin; edges link nodes that
    share microclusters (from the cover overlap). ``branch_points`` are nodes where the shape splits
    (degree ≥ 3); ``bridges`` index the ``edges`` whose removal would disconnect the graph — thin
    links between otherwise separate regions (e.g. leakage between topics in an embedding).
    ``edge_overlap`` is a per-edge Bhattacharyya coefficient in ``(0, 1]`` from the two nodes'
    pooled diagonal-Gaussian summaries: a bridge across a sparse neck scores lower than an edge
    inside one dense blob, so links are weighted by distributional overlap, not a bare shared count.
    """

    node_members: list[np.ndarray]  # microcluster indices per node
    node_mass: np.ndarray  # (n_nodes,) total mass per node
    node_bin: np.ndarray  # (n_nodes,) cover bin per node
    node_lens: np.ndarray  # (n_nodes,) mean lens value per node
    node_centroids: np.ndarray  # (n_nodes, dim) mass-weighted centroid per node
    edges: np.ndarray  # (n_edges, 3): columns (node_a, node_b, shared microcluster count)
    edge_overlap: np.ndarray  # (n_edges,) Bhattacharyya overlap ∈ (0, 1] per edge
    branch_points: np.ndarray  # node indices with degree ≥ 3
    bridges: np.ndarray  # indices into `edges` that are bridges
    persistence_overlap: np.ndarray  # (k, 2) 0-D diagram over 1-Bhattacharyya (inf = essential)
    persistence_lens: np.ndarray  # (k, 2) 0-D diagram over the lens sublevel (inf = essential)

    @property
    def n_nodes(self) -> int:
        return len(self.node_members)

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def persistence(self, filtration: str = "overlap", finite_only: bool = False) -> np.ndarray:
        """The nerve's 0-D persistence diagram as ``(k, 2)`` births/deaths, sorted by persistence.

        ``filtration="overlap"`` (default) filters by the ``1 − edge_overlap`` gap — a finite bar's
        death is the Bhattacharyya depth of a bottleneck, a ranked upgrade of the boolean bridges;
        ``"lens"`` is the lens sublevel diagram (flares of the shape). Essential (component) classes
        carry ``np.inf`` in the death column; ``finite_only=True`` drops them.
        """
        if filtration == "overlap":
            diag = self.persistence_overlap
        elif filtration == "lens":
            diag = self.persistence_lens
        else:
            raise ValueError(f"filtration must be 'overlap' or 'lens', got {filtration!r}")
        return diag[np.isfinite(diag[:, 1])] if finite_only else diag

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
            g.add_edge(
                int(row[0]),
                int(row[1]),
                weight=int(row[2]),
                overlap=float(self.edge_overlap[e]),
                bridge=e in bridge_set,
            )
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
    offset: float = 0.0  # Δ = Σᵢ Sᵢ over *every* leaf, sampled or not
    reference_cost: float | None = None  # ĉost of the α-approximate solution; None if unsampled
    total_sensitivity: float | None = None  # S = Σᵢ sᵢ ≈ 10 + 4k; None if unsampled
    n_leaves: int | None = None  # leaves the tree held before sampling; None if unsampled

    @property
    def n_points(self) -> float:
        """Total mass (≈ number of points summarized)."""
        return float(self.weights.sum())

    def cost(self, centers) -> float:
        """``Σⱼ wⱼ·d²(xⱼ, C) + offset`` — this coreset's estimate of the summary's k-means cost.

        A method rather than a formula in the docs because ``offset`` is easy to forget and
        forgetting it understates every cost by the same constant, which looks like nothing until
        two costs from different coresets are compared.
        """
        c = np.asarray(centers, dtype=np.float64)
        x = np.asarray(self.centers, dtype=np.float64)
        d2 = (x * x).sum(1)[:, None] - 2.0 * x @ c.T + (c * c).sum(1)[None, :]
        np.maximum(d2, 0.0, out=d2)
        return float((np.asarray(self.weights, dtype=np.float64) * d2.min(1)).sum() + self.offset)

    def summary_epsilon(self, alpha: float) -> float:
        """Relative summarization error ``4√ρ + 4ρ`` at ``ρ = alpha · offset / reference_cost``.

        ``alpha`` is required, not defaulted. ``reference_cost`` is the cost of an α-approximate
        solution and therefore **upper**-bounds ``OPT_k``, so ``offset / reference_cost``
        *under*-states the true ``ρ = Δ/OPT_k`` and ``summary_epsilon(1.0)`` is an optimistic
        reading rather than a certificate. The shipped seeding is k-means++ with greedy trials,
        whose ``O(log k)`` guarantee holds in expectation, not for the run in hand — so pass the
        factor you can defend.

        Covers the summary only; sampling error sits on top of it and is what ``size`` buys down.
        """
        if self.reference_cost is None:
            raise ValueError(
                "summary_epsilon needs a sampled coreset — call export_coreset(size=…)"
            )
        if not (self.reference_cost > 0.0):
            return 0.0
        rho = max(alpha * self.offset / self.reference_cost, 0.0)
        return 4.0 * rho**0.5 + 4.0 * rho


@dataclass(frozen=True)
class ConsensusResult:
    """Consensus of several insertion-order-permuted clusterings + a per-point stability score.

    The CF-tree is sensitive to insertion order; clustering ``n_runs`` random permutations and
    voting turns that into a measurable quantity. ``labels`` is the majority label per point (input
    order); ``confidence`` is the fraction of runs agreeing with it — low on an unstable boundary,
    high where every insertion order groups the point the same way.
    """

    labels: np.ndarray  # (n,) consensus label per point
    confidence: np.ndarray  # (n,) in [0, 1] — fraction of runs agreeing with the consensus label
    n_runs: int

    @property
    def mean_confidence(self) -> float:
        """Mean per-point stability across the input — a scalar robustness summary in [0, 1]."""
        return float(self.confidence.mean())


def _align_labels(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Relabel ``labels`` into ``reference``'s id space by maximum cluster overlap so votes from
    different runs refer to the same clusters. Not a strict bijection — when two runs genuinely
    disagree, the merge surfaces as reduced consensus, which is the point."""
    ref_ids, ref_pos = np.unique(reference, return_inverse=True)
    lab_ids, lab_pos = np.unique(labels, return_inverse=True)
    contingency = np.zeros((lab_ids.size, ref_ids.size), dtype=np.int64)
    np.add.at(contingency, (lab_pos, ref_pos), 1)
    best = ref_ids[contingency.argmax(axis=1)]  # lab cluster → best-overlap ref cluster
    lut = np.zeros(int(lab_ids.max()) + 1, dtype=np.int64)
    lut[lab_ids] = best
    return lut[labels]


def consensus(
    X, n_clusters: int, *, n_runs: int = 5, seed: int = 0, n_jobs: int = 1, **fit_kwargs
) -> ConsensusResult:
    """Cluster ``X`` under ``n_runs`` random insertion-order permutations; return the consensus
    labelling and a per-point stability score (see :class:`ConsensusResult`).

    Extra keyword arguments are forwarded to :func:`fit_predict` (``feature`` / ``method`` /
    ``threshold`` / …). Intended for the partitional heads (``kmeans`` / ``gmm`` / ``ward`` /
    ``spectral``) at a fixed ``n_clusters``; density heads (``hdbscan``) emit noise / variable
    counts the vote cannot align, and are rejected. ``n_jobs`` runs the (independent) permutations
    in parallel threads — the Rust core releases the GIL, so this scales — with `<0` meaning all
    cores; each run is seeded independently, so the result is identical regardless of ``n_jobs``.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    x = np.asarray(X)
    n = x.shape[0]

    def run(r: int) -> np.ndarray:
        perm = np.random.default_rng([seed, r]).permutation(n)  # independent per run, order-free
        labels_perm = np.asarray(fit_predict(x[perm], n_clusters, seed=seed + r, **fit_kwargs))
        if labels_perm.min() < 0:
            raise ValueError("consensus requires a partitional method (got noise labels < 0)")
        original = np.empty(n, dtype=np.int64)
        original[perm] = labels_perm  # undo the permutation → labels back in the input's order
        return original

    if n_jobs == 1:
        runs = [run(r) for r in range(n_runs)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as pool:
            runs = list(pool.map(run, range(n_runs)))
    reference = runs[0]
    aligned = np.array([reference] + [_align_labels(r, reference) for r in runs[1:]])
    counts = np.zeros((n, int(aligned.max()) + 1), dtype=np.int64)
    idx = np.arange(n)
    for row in aligned:
        counts[idx, row] += 1
    return ConsensusResult(
        labels=counts.argmax(axis=1), confidence=counts.max(axis=1) / n_runs, n_runs=n_runs
    )


def _isotropic_variances(est) -> np.ndarray:
    """Leaf RMS radii as the per-coordinate variances of an isotropic Gaussian: ``R² / d``."""
    centers = np.asarray(est.microcluster_centers_, dtype=np.float64)
    radii = np.asarray(est.microcluster_radii_, dtype=np.float64)
    dim = centers.shape[1]
    return np.repeat((radii**2 / dim)[:, None], dim, axis=1)


def mixture_w2(
    weights_a,
    means_a,
    covariances_a,
    weights_b,
    means_b,
    covariances_b,
) -> float:
    """Mixture-Wasserstein ``MW2`` between two fitted Gaussian mixtures. Lower is closer.

    Delon & Desolneux (SIAM J. Imaging Sci. 13(2), 2020) restrict the transport plan to be itself a
    Gaussian mixture, which turns the distance into a closed-form ``W2`` per component pair (the
    Bures metric) plus a small exact transport over the ``k_a × k_b`` grid. It needs no labels, no
    shared sample and no common ``k``: two fits of different sizes, on different data, are directly
    comparable.

    ``covariances`` may be ``(k, dim)`` per-coordinate variances or ``(k, dim, dim)`` full matrices,
    the two shapes :class:`sklearn.mixture.GaussianMixture` uses for ``covariance_type="diag"`` and
    ``"full"``. The parameters are passed rather than an estimator so that a mixture fitted
    elsewhere — sklearn, ELKI, the same model an hour earlier — can be compared without conversion.

    Two uses: as a **drift** metric between the same model at two times (see
    :meth:`Betula.summary_w2`), and as a **cross-implementation** agreement number, which is sharper
    than an ARI between two labellings because it separates a parameter difference from a tie broken
    the other way at a boundary.

    The value is an upper bound on the true ``W2`` between the two densities, not that ``W2``
    itself; the two coincide when the optimal unrestricted coupling happens to be a Gaussian
    mixture, as it is for a pure translation.
    """
    return float(
        _core_mixture_w2(
            np.ascontiguousarray(weights_a, dtype=np.float64),
            np.ascontiguousarray(means_a, dtype=np.float64),
            np.ascontiguousarray(covariances_a, dtype=np.float64),
            np.ascontiguousarray(weights_b, dtype=np.float64),
            np.ascontiguousarray(means_b, dtype=np.float64),
            np.ascontiguousarray(covariances_b, dtype=np.float64),
        )
    )


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


def _rows_of(X, csr) -> int:
    """Row count of the input, for resolving a fractional ``max_leaves`` against it.

    ``len`` rather than ``shape[0]`` so a nested list works the same as an array; the CSR row count
    is one less than the length of ``indptr``.
    """
    return len(csr[2]) - 1 if csr is not None else len(X)


def _absolute_max_leaves(max_leaves, n_rows: int | None) -> int:
    """``max_leaves`` as the engine takes it: a count, or a fraction of ``N`` resolved against it.

    One parameter carries both forms, discriminated by range rather than by a second flag: a value
    strictly between 0 and 1 is a fraction (ELKI's ``-cftree.maxleaves`` convention, whose
    default is ``0.05``), anything else is an absolute count and must be an integer. No value could
    mean either, so no precedence rule is needed between them.
    """
    if isinstance(max_leaves, float) and 0.0 < max_leaves < 1.0:
        if n_rows is None:
            raise ValueError(
                "a fractional max_leaves needs the row count, which streaming does not have: "
                "pass an absolute integer to partial_fit, or fit() on an array."
            )
        return max(1, math.ceil(max_leaves * n_rows))
    if isinstance(max_leaves, bool) or not isinstance(max_leaves, int) or max_leaves < 1:
        raise ValueError(
            "max_leaves must be an integer >= 1 (an absolute leaf cap) or a float in (0, 1) "
            f"(a fraction of the row count); got {max_leaves!r}"
        )
    return max_leaves


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
    X,
    n_clusters=8,
    method="kmeans",
    threshold=0.0,
    max_leaves=2048,
    max_iter=100,
    seed=0,
    projection="none",
    projection_dim=64,
    projection_max_iter=100,
    auto_k_max=0,
):
    """One-shot ``O(nnz)`` clustering of a ``scipy.sparse`` matrix.

    Summarises rows into spherical micro-clusters touching only the non-zeros (a flat leader pass
    bounded by ``max_leaves``), clusters those with a parametric head (``kmeans`` / ``gmm`` /
    ``gmm-full`` / ``ward``), and labels each row by the head's own point rule. For very
    high-dimensional sparse data this avoids the ``O(d)``-per-row cost of the dense path. It uses
    the expanded squared-distance form for speed and so does **not** carry the dense path's
    cancellation-free guarantee (accurate for sparse rows far from the dense centroid; see the
    library docs). Returns one ``int64`` label per row.

    The centre-based heads (``kmeans`` / ``xmeans`` / ``spherical-kmeans``) label each row by its
    nearest cluster centroid, the rule the dense path uses; ``kmedoids`` uses the same scan against
    the micro-clusters it chose as medoids, which are not the cluster means. ``gmm`` and ``vmf``
    label by maximum
    posterior: their kernels split over the support of a row, so the density costs ``O(nnz)`` rather
    than the ``O(d)`` this path exists to avoid. The rest — the agglomerative heads (``ward`` /
    ``average`` / ``weighted`` / ``centroid`` / ``median``) and ``gmm-full`` / ``gmm-toeplitz*`` /
    ``mppca``, whose densities read every coordinate — keep the label of the micro-cluster the
    summarisation put the row in.

    That last group is where the flat summary shows. Once the leader budget is spent the pass has no
    proximity gate left, so on a 6 000 × 4 000 block-topic corpus at ``max_leaves=2048`` it force-
    assigns 3 952 of 6 000 rows into 544 leaders; a head that models each micro-cluster separately
    inherits those mixed leaders, and ``ward`` reads ARI 0.068 against the dense tree's 0.987 at the
    same compression. Raising ``threshold`` does not help (measured flat from 0.5 to 14, worse
    above), because near-orthogonal sparse rows give the gate nothing to merge. Prefer a
    centre-based head on this path, or reduce first with ``projection="svd"``.

    ``projection="svd"`` turns this into the one-call reduce-then-cluster pipeline for text: the
    leaf summary is reduced to ``projection_dim`` CF-weighted principal directions, the head
    clusters the codes, and each row is labelled by its own code (encoded from its non-zeros).
    Clustering the raw high-dimensional geometry directly is the thing to avoid here — see
    ``docs/USAGE.md``. ``auto_k_max`` bounds the ``n_clusters=0`` search; ``0`` keeps the default.
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
        projection=projection,
        projection_dim=projection_dim,
        projection_max_iter=projection_max_iter,
        auto_k_max=auto_k_max,
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
    "balance": None,
    "resolution": 1.0,
    "covariance_weight": 0.0,
    "tangent_weight": 0.0,
    "tangent_rank": 2,
    "projection": "none",
    "projection_dim": 64,
    "projection_max_iter": 100,
    "refine": 0,
    "rank": 2,
    "fuzzifier": 2.0,
    "graph_degree": 0,
    "auto_k_max": 0,
    "memory_budget_mb": None,
}
_PARAM_NAMES = tuple(_DEFAULTS)

# Directional heads (`spherical-kmeans` / `vmf` / `watson`) cluster points on the unit sphere, so
# the engine is always built with ``normalize=True`` for them. ``self.normalize`` is left as the
# user set it — ``get_params`` stays verbatim so ``sklearn.base.clone`` / ``set_params``
# round-trip unchanged.
_DIRECTIONAL_METHODS = ("spherical-kmeans", "vmf", "watson")

# `threshold="auto"` pilot: fit this many points (max of the two) to estimate a warm-start
# threshold. Oversampling the leaf budget makes the subsample crowd the tree to `max_leaves`, so
# the threshold it converges to transfers to the full fit.
_AUTO_PILOT_CAP_FACTOR = 8
_AUTO_PILOT_MIN = 4000


class Betula:
    """Streaming, scikit-learn-style BETULA estimator.

    Parameters are validated lazily — when the engine is built at ``fit`` / ``partial_fit`` time —
    following the scikit-learn convention that ``__init__`` only records its arguments verbatim.

    ``threshold`` accepts a non-negative float (the CF absorption radius, ``0.0`` grows it from
    scratch) or ``"auto"``: a subsample pilot then estimates a warm-start threshold so the full fit
    starts near-converged instead of thrashing rebuilds up from zero. ``"auto"`` is dense-only.
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
        balance=None,
        resolution=1.0,
        covariance_weight=0.0,
        tangent_weight=0.0,
        tangent_rank=2,
        projection="none",
        projection_dim=64,
        projection_max_iter=100,
        refine=0,
        rank=2,
        fuzzifier=2.0,
        graph_degree=0,
        auto_k_max=0,
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
        # Mass-balanced leaf budget: no leaf may hold more than ``balance`` × (n / max_leaves) of
        # the mass, enforced at absorption and at compaction, with ``max_leaves`` still a hard
        # bound. ``None`` (default) is the purely geometric budget, where one dense region can take
        # the whole tree — worth +0.58 ARI on a size-imbalanced fixture, mixed on well-spread data.
        self.balance = balance
        # Leiden resolution γ (only method="leiden" / "leiden-cpm"): higher ⇒ more, smaller
        # communities. The modularity objective has a resolution limit; "leiden-cpm" does not.
        self.resolution = resolution
        # Covariance-aware Leiden weight β (method="leiden" / "leiden-cpm" with feature="full"):
        # >0 adds a log-Euclidean shape term to the microcluster affinity, so communities agree in
        # both centroid and covariance. 0 disables it (plain centroid affinity).
        self.covariance_weight = covariance_weight
        # Tangent-aware Leiden weight γ + subspace rank (method="leiden" / "leiden-cpm",
        # feature="full"): >0 adds a Grassmann term over each microcluster's rank-`tangent_rank`
        # principal subspace, separating crossing/adjacent manifolds. 0 disables it.
        self.tangent_weight = tangent_weight
        self.tangent_rank = tangent_rank
        # Phase-3 CF-weighted NMF reduction for **nonnegative** data (TF-IDF / counts / spectra).
        # "weighted-nmf" factorizes leaf centroids into `projection_dim` nonnegative codes over the
        # M ≪ N microclusters, then the head clusters the codes; "none" disables it.
        self.projection = projection
        self.projection_dim = projection_dim
        # The factorizer's own sweep budget. Separate from `max_iter` (the clustering head's): the
        # two converge at different rates, and one shared number made a larger head budget silently
        # pay for NMF sweeps too.
        self.projection_max_iter = projection_max_iter
        # BIRCH Phase 4: Lloyd sweeps over the raw rows after the leaf clustering, warm-started from
        # the CF centres. Centroid heads only (kmeans / xmeans / spherical-kmeans), dense in-memory
        # `fit` / `fit_predict` only — a Lloyd sweep is the k-means update, so it would move a
        # `kmedoids` centre off the data and out of its own objective. A better objective is not
        # automatically a better partition. 0 = off.
        self.refine = refine
        # Subspace rank q of the MPPCA head (method="mppca"): each component's covariance is
        # W Wᵀ + σ²I with W of rank q, clamped to at most dim - 1. Ignored by every other head.
        self.rank = rank
        # Fuzzifier m > 1 of the fuzzy c-means head (method="fuzzy-cmeans"): the exponent that
        # decides how soft the memberships are. m -> 1 is k-means, large m sends every membership
        # to 1/k. Ignored by every other head.
        self.fuzzifier = fuzzifier
        # Out-degree of the proximity graph the hdbscan head takes its MST over. 0 (default) uses
        # the exact complete graph, which is O(m²) in the leaf count and is what makes a large
        # max_leaves unaffordable there. Any positive value is a floor: the head raises it to
        # whatever degree min_samples needs in leaves. Ignored by every other head.
        self.graph_degree = graph_degree
        # Ceiling the n_clusters=0 search is bounded by; 0 keeps the per-head default. The sweeps
        # refit at every candidate k, so their cost is quadratic in it and the default stays at 20;
        # ward/average/weighted/centroid/median cut one dendrogram and xmeans stops on its own test,
        # so those are already bounded only by the leaf count. Raising it warns nothing and costs
        # time; leaving it too low is silent, which is why a selection that lands on the ceiling
        # raises a UserWarning.
        self.auto_k_max = auto_k_max
        # When set, max_leaves is derived from this budget (+ dim + feature) at fit time: a target
        # for the CF-tree resident size (MiB), not total RSS. Most useful for streaming.
        self.memory_budget_mb = memory_budget_mb
        self._est = None
        self._effective_max_leaves = max_leaves
        # Cache for the pilot-estimated threshold when `threshold="auto"`; reset on `set_params`.
        self._auto_threshold = None

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
        self._auto_threshold = None  # …and the pilot estimate no longer matches the params
        return self

    # ── fit / predict ────────────────────────────────────────────────────────────────────────
    def _resolve_max_leaves(self, dim, n_rows=None):
        # `memory_budget_mb` is a wrapper-only knob: when set (and the dimension is known),
        # translate it into the `max_leaves` the engine actually uses. It wins over `max_leaves` in
        # either form, because it is the harder constraint — a budget the tree must fit inside.
        if self.memory_budget_mb is not None and dim is not None:
            return _budget_max_leaves(self.memory_budget_mb, dim, self.feature, self.branching)
        return _absolute_max_leaves(self.max_leaves, n_rows)

    def _build(self, dim=None, threshold_override=None, n_rows=None):
        params = {k: getattr(self, k) for k in _PARAM_NAMES if k != "memory_budget_mb"}
        params["max_leaves"] = self._resolve_max_leaves(dim, n_rows)
        # `threshold="auto"` is resolved to a float by the caller (see `_resolve_auto`); the engine
        # only ever receives a concrete threshold.
        if threshold_override is not None:
            params["threshold"] = threshold_override
        self._effective_max_leaves = params["max_leaves"]
        if self.method in _DIRECTIONAL_METHODS:
            params["normalize"] = True
        return _CoreBetula(**params)

    def _resolve_auto(self, X, csr):
        """Resolve ``threshold="auto"`` to a float warm-start threshold; ``None`` when not auto.

        Pilots a bounded subsample through a ``threshold=0`` tree with the same ``max_leaves`` and
        reads the threshold it converges to, so the full fit starts near-converged instead of
        thrashing rebuilds up from zero. The estimate is cached for refits / streaming batches.
        """
        if self.threshold != "auto":
            return None
        if csr is not None:
            raise ValueError("threshold='auto' requires a dense array, not a sparse matrix")
        if self._auto_threshold is None:
            self._auto_threshold = self._pilot_threshold(np.asarray(X))
        return self._auto_threshold

    def _pilot_threshold(self, X):
        dim = X.shape[1] if X.ndim == 2 else 1
        max_leaves = self._resolve_max_leaves(dim, X.shape[0])
        cap = max(_AUTO_PILOT_CAP_FACTOR * max_leaves, _AUTO_PILOT_MIN)
        n = X.shape[0]
        if n <= cap:
            # Small data: growing from zero is already cheap (rebuilds fold O(leaves), not O(n)),
            # and a full-data pilot would just double the work — skip it and start at zero.
            return 0.0
        rng = np.random.default_rng(self.seed)
        sub = X[rng.choice(n, cap, replace=False)]
        params = {k: getattr(self, k) for k in _PARAM_NAMES if k != "memory_budget_mb"}
        params["threshold"] = 0.0
        params["max_leaves"] = max_leaves
        if self.method in _DIRECTIONAL_METHODS:
            params["normalize"] = True
        pilot = _CoreBetula(**params)
        pilot.fit(sub)
        return float(pilot.threshold_)

    def _check_projection_input(self, X, csr):
        # CF-weighted NMF is defined only for nonnegative data; reject signed input rather than
        # silently shifting it (a shift changes angles / cosine geometry). The factorization runs on
        # the leaf centroids, which the sparse path builds the same way, so CSR is accepted too —
        # only the stored values can be negative, implicit zeros are already nonnegative.
        if not str(self.projection).startswith("weighted-nmf"):
            return
        values = csr[0] if csr is not None else np.asarray(X)
        if bool((values < 0).any()):
            raise ValueError(
                "projection='weighted-nmf' requires nonnegative data (X >= 0). For signed "
                "embeddings use method='vmf'/'spherical-kmeans' or reduce with PCA/TruncatedSVD."
            )

    def fit(self, X, y=None, must_link=None, cannot_link=None):
        """Fit the CF-tree on ``X`` and cluster it; returns ``self`` (scikit-learn style).

        Args:
            X: dense float32/float64 array or a ``scipy.sparse`` CSR matrix (never densified).
            y: ignored; present for scikit-learn API compatibility.
            must_link: optional ``(m, 2)`` row-index pairs forced into the same cluster.
            cannot_link: optional ``(m, 2)`` row-index pairs forced apart. Any constraint switches
                the run to semi-supervised COP-KMeans (``method="kmeans"``, dense input only).
        """
        if must_link is not None or cannot_link is not None:
            return self._fit_constrained(X, must_link, cannot_link)
        csr = _to_csr(X)
        self._check_projection_input(X, csr)
        override = self._resolve_auto(X, csr)
        rows = _rows_of(X, csr)
        if csr is not None:
            est = self._build(csr[3], override, rows)
            est.fit_csr(*csr)
        else:
            est = self._build(_dim_of(X), override, rows)
            est.fit(X)
        self._est = est
        return self

    def fit_predict(self, X, y=None, must_link=None, cannot_link=None):
        """Fit on ``X``; return one ``int64`` label per row (``-1`` = noise, HDBSCAN head).

        Takes the same ``X`` / ``must_link`` / ``cannot_link`` as :meth:`fit`.
        """
        if must_link is not None or cannot_link is not None:
            self._fit_constrained(X, must_link, cannot_link)
            return np.asarray(self.predict(X))
        csr = _to_csr(X)
        self._check_projection_input(X, csr)
        override = self._resolve_auto(X, csr)
        rows = _rows_of(X, csr)
        if csr is not None:
            est = self._build(csr[3], override, rows)
            labels = est.fit_predict_csr(*csr)
        else:
            est = self._build(_dim_of(X), override, rows)
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
        override = self._resolve_auto(X, None)
        est = self._build(_dim_of(X), override, _rows_of(X, None))
        est.fit_constrained(X, ml, cl)
        self._est = est
        return self

    def partial_fit(self, X=None, y=None):
        csr = None if X is None else _to_csr(X)
        override = self._resolve_auto(X, csr) if self._est is None else None
        if csr is not None:
            if self._est is None:
                self._est = self._build(csr[3], override)
            self._est.partial_fit_csr(*csr)
        else:
            if self._est is None:
                self._est = self._build(_dim_of(X), override)
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
    def components_(self):
        """NMF parts ``H`` — ``(projection_dim, dim)``, rows unit-L2, ordered by descending energy.

        Every leaf code is a nonnegative combination of these rows, so a row reads directly as a
        "topic" over the input features. Requires a ``"weighted-nmf"`` / ``"weighted-nmf-kl"``
        ``projection``.
        """
        return self._require_fit().components_

    @property
    def reconstruction_err_(self):
        """Relative reconstruction error of the projection, ``‖X̃ − W H‖_F / ‖X̃‖_F``.

        Measured over the leaf centroid matrix the factorizer actually fits — how much of the
        compressed data ``projection_dim`` parts explain. Requires a projection.
        """
        return self._require_fit().reconstruction_err_

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

    def outlier_scores(self, X, metric="radius"):
        """Per-row deviation from the assigned cluster centroid, over that cluster's own spread.

        ``metric="radius"`` (the default) divides by the cluster's scalar RMS radius. That radius is
        the *trace* of the pooled covariance, so an elongated cluster's short axis is judged by the
        length of its long one — measured on sheared 6-D clusters, ROC-AUC 0.596, and identical at
        ``feature="diagonal"`` and ``feature="full"`` because the per-dimension scatter is never
        consulted.

        ``metric="mahalanobis"`` whitens the deviation by the cluster's full pooled covariance
        instead — the parallel-axis pooling of the leaves' own scatter and their spread about the
        centroid, whose trace *is* that RMS radius. The two are therefore calibrated: where the
        pooled covariance is isotropic they return the same number, so the refinement moves a score
        only where the cluster has a shape. Off-diagonal terms are the point — a cluster can be
        elongated along a direction that is not a coordinate axis, and a per-dimension variance
        cannot see that — so it costs ``O(k·d³)`` once plus ``O(d²)`` per row against the scalar
        path's ``O(d)``, which is worth watching in high dimension.
        """
        return self._require_fit().outlier_scores(X, metric)

    def tree_report(self, X=None, **estimate_kwargs):
        """Why the CF-tree looks the way it does — leaf budget, mass concentration, threshold.

        Answers "why is my tree collapsing?" with the numbers that decide it: how much of the
        leaf budget was spent, how much of the *mass* one leaf ended up holding, and how wide that
        leaf is against a typical one. A near-full budget with a single leaf carrying most of the
        points is the size-imbalance pathology of scikit-learn's Birch issue #22854 — the tree spent
        its leaves resolving the sparse minority and merged the dense majority into one summary.

        Whether that costs anything is a separate question, and the heavy leaf's *radius* answers
        it: a dense blob that really is point-like is summarized faithfully by one tight leaf, while
        a heavy leaf as wide as a typical one is a merged region whose internal structure is gone.

        Pass ``X`` to add an A-BIRCH threshold estimate (:func:`~betula_cluster.tuning
        .estimate_threshold`, Lorbeer et al. 2018) alongside the threshold actually in use. That
        estimate is **advisory**: it assumes well-separated, near-spherical clusters of comparable
        size, and ``diagnosis`` names every one of those assumptions the data breaks. ``max_leaves``
        remains the knob that binds — the threshold is what the rebuild derives from it.
        """
        est = self._require_fit()
        weights = np.asarray(est.microcluster_weights_, dtype=np.float64)
        centers = np.asarray(est.microcluster_centers_, dtype=np.float64)
        mass = float(weights.sum())
        # The row count is what a fractional `max_leaves` resolves against, and after a fit the leaf
        # mass *is* that count — so the report can answer for a streaming fit too.
        budget = self._resolve_max_leaves(centers.shape[1] if centers.size else None, round(mass))
        radii = np.asarray(est.microcluster_radii_, dtype=np.float64)
        top1 = float(weights.max() / mass) if weights.size and mass > 0 else 0.0
        fill = float(est.n_leaves_) / budget if budget else 0.0
        spread = radii[radii > 0.0]
        heaviest = float(radii[int(np.argmax(weights))]) if weights.size else 0.0
        width = heaviest / float(np.median(spread)) if spread.size else 0.0
        report = {
            "n_leaves": int(est.n_leaves_),
            "max_leaves": int(budget),
            "fill": fill,
            "threshold": float(est.threshold_),
            "heaviest_leaf_mass_fraction": top1,
            "heaviest_leaf_width": width,
            "leaf_mass_quantiles": {
                q: float(np.quantile(weights, q / 100.0)) for q in (50, 90, 99, 100)
            }
            if weights.size
            else {},
            "diagnosis": [],
        }
        if fill >= 0.9 and top1 >= 0.25:
            note = (
                f"the leaf budget is {fill:.0%} spent and one leaf holds {top1:.0%} of the mass: "
                "the tree spent its leaves resolving the sparse part of the data and merged the "
                "dense part (scikit-learn Birch #22854). "
            )
            # The cut is measured on `bench/size_imbalance.py`'s own positive/negative control pair,
            # medians of seeds 0/1/2 at budgets 250 and 4000: the `structured` core (two clusters
            # 2.0 apart inside the heavy leaf) gives 0.53 and 0.75, the `flat` core (nothing inside
            # to lose) gives 0.17 and 0.27.
            if width >= 0.4:
                note += (
                    f"That leaf is {width:.2f}x as wide as a typical one, so it is a merged region "
                    "rather than a point-like blob and the structure inside it is unrecoverable. "
                    "Raise max_leaves, or set balance= to allocate leaves by mass, not geometry."
                )
            else:
                note += (
                    f"That leaf is only {width:.2f}x as wide as a typical one, so the dense "
                    "part is genuinely point-like and one leaf summarizes it faithfully — "
                    "this costs nothing unless you expect structure inside it."
                )
            report["diagnosis"].append(note)
        if weights.size and float(np.quantile(weights, 0.5)) <= 1.0 and fill < 0.5:
            report["diagnosis"].append(
                "half the leaves hold a single point and the budget is under half spent: the "
                "threshold is too small to compress anything, so the summary is the data."
            )
        if X is not None:
            estimate = estimate_threshold(X, **estimate_kwargs)
            report["suggested_threshold"] = estimate.threshold
            report["suggested_n_clusters"] = estimate.n_clusters
            report["diagnosis"].extend(estimate.assumptions)
            if estimate.threshold > 0 and report["threshold"] > 2.0 * estimate.threshold:
                report["diagnosis"].append(
                    f"the tree settled at threshold {report['threshold']:.3g}, over twice the "
                    f"{estimate.threshold:.3g} a sample of the data suggests: leaves are absorbing "
                    "points from more than one cluster."
                )
        return report

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

    def validity(self):
        """Internal validity indices of the fitted partition, scored on the leaf summary.

        Returns ``calinski_harabasz`` (higher is better), ``davies_bouldin`` (lower is better) and
        ``medoid_silhouette`` (higher is better, capped at 1). All three cost
        ``O(n_leaves · k · d)`` rather than the ``O(N²)`` an exact silhouette over the points would
        — the sum of squared distances inside a leaf is ``S_i + n_i‖μ_i − c‖²`` exactly, so no
        point ever has to be revisited.

        Caveats worth reading before using any of them to choose ``k``: Calinski–Harabasz is exact
        but undefined at ``k = 1``; Davies–Bouldin is the RMS-dispersion variant, not the classical
        mean-distance one; the medoid silhouette is the index of the summary, not of the points, and
        no richer leaf model would change that — two point sets with identical cluster features can
        have different pairwise distances, so the exact silhouette is not recoverable from any
        summary in this family (``research/RESULTS-cf-boundary.md``). None of the three can report
        "there is no structure here" — for that, fit with ``n_clusters=0`` on a mixture head and let
        BIC answer.
        """
        ch, db, ms = self._require_fit().validity_()
        return {
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "medoid_silhouette": ms,
        }

    def summary_mmd(self, X, *, bandwidth=None):
        """Kernel distance between the leaf summary and the raw sample ``X``. Lower is better.

        A label-free, head-independent fidelity number: the leaves are read as the Gaussian mixture
        ``Σ (n_i/N)·N(μ_i, s_i I)`` with ``s_i = S_i/(n_i·d)`` and compared to ``X`` by maximum mean
        discrepancy under a Gaussian kernel, in closed form — no sampling from the surrogate. Unlike
        :meth:`validity` it needs no labels, no ``k`` and no head, so a tree built with
        ``partial_fit`` alone can be scored.

        ``bandwidth`` is the kernel's ``h``; ``None`` takes the median heuristic on ``X``, which
        makes values comparable across leaf budgets of the *same* data and meaningless across
        different data.

        Use it to find the leaf budget past which more leaves buy nothing: it flattens where the
        clustering stops changing, which ``mean_sq_radius`` does not. It is **not** monotone in the
        budget — very coarse leaves can model a smooth blob better than two-point ones — and it
        costs ``O(n_leaves² + n_leaves·N + N²)`` kernel evaluations, so it is a diagnostic to run at
        a few budgets rather than something to put in a loop.
        """
        arr = np.ascontiguousarray(X, dtype=np.float64)
        return float(self._require_fit().summary_mmd_(arr, bandwidth))

    def summary_w2(self, other) -> float:
        """Mixture-Wasserstein distance between this model's leaf summary and ``other``'s.

        The drift metric: fit at ``t1``, fit at ``t2``, and read how far the summary moved. It needs
        no labels, no shared points and not even the same number of leaves, which is what an
        ARI-over-time cannot do. Both leaf summaries are read through the same isotropic surrogate
        :meth:`summary_mmd` uses — leaf ``i`` is ``N(μ_i, (R_i²/d)·I)`` with mass ``n_i`` — so the
        two diagnostics agree on what a leaf is.

        Zero means the two summaries describe the same density; the value is in the units of the
        data, so it is comparable across runs on the same feature scale and meaningless across
        different ones.
        """
        mine, theirs = self._require_fit(), other._require_fit()
        return mixture_w2(
            mine.microcluster_weights_,
            mine.microcluster_centers_,
            _isotropic_variances(mine),
            theirs.microcluster_weights_,
            theirs.microcluster_centers_,
            _isotropic_variances(theirs),
        )

    def find_outliers(self, X, top_k=100, metric="radius"):
        """Row indices of the ``top_k`` most outlying points (highest score first).

        ``metric`` is passed through to :meth:`outlier_scores`.
        """
        scores = np.asarray(self.outlier_scores(X, metric))
        k = min(top_k, scores.size)
        if k <= 0:
            return np.empty(0, dtype=np.intp)
        # O(N) partition for the top-k, then sort only those k — vs a full O(N log N) sort of all N.
        idx = np.argpartition(scores, scores.size - k)[scores.size - k :]
        return idx[np.argsort(scores[idx])[::-1]]

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

    def near_duplicate_pairs(self, X, threshold=0.9, *, neighbors=0):
        """Scored near-duplicate row pairs by exact cosine similarity, blocked by microcluster.

        The CF-tree blocks rows into leaves in ``O(N)``; within each (small) leaf, exact pairwise
        cosine is computed and pairs scoring ``>= threshold`` are kept. The cost is
        ``~O(N * leaf_size)`` -- the scalable counterpart to an ``O(N^2)`` all-pairs scan, and the
        scored complement to :meth:`find_near_duplicates` (which returns unscored groups). Returns
        an ``(m, 3)`` ``float64`` array of ``[cos_sim, i, j]`` with ``i < j``, sorted by similarity
        descending.

        Recall is bounded by the blocking: a duplicate pair whose two rows were absorbed into
        *different* leaves is invisible to a within-leaf scan. ``neighbors`` buys that recall back
        without giving up the blocking — each leaf is also scored against its ``neighbors`` nearest
        leaves by centroid distance, which is the geometry that split the pair in the first place.
        ``0`` (the default) is the within-leaf scan unchanged. The added cost is
        ``O(M^2 d)`` for the leaf-neighbour search plus ``~neighbors x`` the scoring, both still
        sub-quadratic in ``N``; a coarser tree (smaller ``max_leaves``) is the other lever, trading
        speed for recall by widening the blocks themselves.
        """
        leaf = np.asarray(self.assign_microclusters(X))
        rows = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        unit = rows / np.where(norms > 0.0, norms, 1.0)  # guard zero-norm rows (no NaN)
        members = {int(lf): np.flatnonzero(leaf == lf) for lf in np.unique(leaf)}
        blocks = [(m, m) for m in members.values() if m.size >= 2]
        for a, b in self._adjacent_leaves(members, neighbors):
            blocks.append((members[a], members[b]))
        scores: list = []
        lo: list = []
        hi: list = []
        for left, right in blocks:
            sim = unit[left] @ unit[right].T
            if left is right:
                iu, ju = np.triu_indices(left.size, k=1)
            else:
                iu, ju = np.meshgrid(np.arange(left.size), np.arange(right.size), indexing="ij")
                iu, ju = iu.ravel(), ju.ravel()
            s = sim[iu, ju]
            keep = s >= threshold
            if not keep.any():
                continue
            i, j = left[iu[keep]], right[ju[keep]]
            scores.append(s[keep])
            lo.append(np.minimum(i, j))
            hi.append(np.maximum(i, j))
        if not scores:
            return np.empty((0, 3), dtype=np.float64)
        s = np.concatenate(scores)
        i = np.concatenate(lo)
        j = np.concatenate(hi)
        order = np.argsort(s)[::-1]
        return np.column_stack([s[order], i[order], j[order]]).astype(np.float64)

    def _adjacent_leaves(self, members, neighbors):
        """Unordered pairs of populated leaves that are among each other's nearest by centroid.

        One `(M, M)` distance matrix rather than a per-leaf scan: `M` is the leaf count, which is
        bounded by `max_leaves` and independent of `N`, so this stays sub-quadratic in the data.
        """
        if neighbors <= 0 or len(members) < 2:
            return []
        ids = np.array(sorted(members))
        centers = np.asarray(self.microcluster_centers_, dtype=np.float64)[ids]
        sq = (centers**2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (centers @ centers.T)
        np.fill_diagonal(d2, np.inf)
        take = min(neighbors, len(ids) - 1)
        nearest = np.argpartition(d2, take - 1, axis=1)[:, :take]
        pairs = {
            (int(ids[a]), int(ids[b])) if ids[a] < ids[b] else (int(ids[b]), int(ids[a]))
            for a, row in enumerate(nearest)
            for b in row
        }
        return sorted(pairs)

    def mapper(
        self,
        lens="density",
        resolution=10,
        gain=0.3,
        link_scale=1.0,
        min_node_mass=0.0,
        density_k=5,
        coordinate=0,
        link="centroid",
    ):
        """Build a Mapper topological-skeleton :class:`MapperGraph` over the fitted microclusters.

        TDA Mapper specialised to BETULA: a ``lens`` filter (``"density"`` | ``"radius"`` |
        ``"l2norm"`` | ``"coordinate"`` | ``"eccentricity"``) is covered by ``resolution`` bins
        overlapping by ``gain``; microclusters in a bin are single-linked at ``link_scale`` × the
        bin's median nearest-neighbour gap; one node per (bin, component).
        It surfaces non-convex structure, branch points and bridges (topic leakage) over the
        ``M << N`` microclusters — an exploration tool, not a partition. Build the model first.

        ``link`` chooses what "close" means inside a bin. ``"centroid"`` (the default) is the
        Euclidean distance between leaf centroids and sees only the gap. ``"bhattacharyya"`` divides
        that gap by the pair's own spread, using the second moments the cluster features already
        carry, which is what stops a thin bridge of sparse microclusters from chaining two dense
        regions together: on a dumbbell fixture the centroid rule merges the two lobes at
        ``link_scale=3`` on all three seeds and the Bhattacharyya rule merges neither, at any
        ``link_scale`` measured. It is not free — a leaf that holds one point has no spread, so
        every bridge of singletons stays fragmented whether or not it was real structure.
        """
        d = self._require_fit().mapper(
            lens=lens,
            resolution=resolution,
            gain=gain,
            link_scale=link_scale,
            min_node_mass=min_node_mass,
            density_k=density_k,
            coordinate=coordinate,
            link=link,
        )
        return MapperGraph(
            node_members=[np.asarray(m, dtype=np.int64) for m in d["node_members"]],
            node_mass=d["node_mass"],
            node_bin=d["node_bin"],
            node_lens=d["node_lens"],
            node_centroids=d["node_centroids"],
            edges=d["edges"],
            edge_overlap=d["edge_overlap"],
            branch_points=d["branch_points"],
            bridges=d["bridges"],
            persistence_overlap=d["persistence_overlap"],
            persistence_lens=d["persistence_lens"],
        )

    def reachability(self, min_samples=5, graph_degree=0):
        """Build the OPTICS :class:`ReachabilityPlot` over the fitted microclusters.

        A density *diagnostic*, not a partition: it answers "what does the density structure look
        like" rather than "which cluster is this in". ``min_samples`` and ``graph_degree`` mean what
        they do for ``method="hdbscan"`` — pass the values that fit used, or the plot describes a
        different neighbourhood than the head did. Build the model first.

        Reads the same mutual-reachability spanning tree the density head does, so
        :meth:`ReachabilityPlot.labels_at` is that head's hierarchy cut at a height, exactly.
        """
        d = self._require_fit().reachability(min_samples=min_samples, graph_degree=graph_degree)
        return ReachabilityPlot(
            order=d["order"],
            reachability=d["reachability"],
            core_distances=d["core_distances"],
            weights=d["weights"],
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
    def export_coreset(self, size=None, k=None, seed=None):
        """The leaf summary as a weighted-point :class:`Coreset`, optionally sampled down to
        ``size`` points with a provable ``(k, ε)`` guarantee. Requires a built tree only —
        ``partial_fit`` is enough, since which head this estimator fitted does not enter it.

        With ``size=None`` this returns every leaf at its own mass: the streaming summary, exactly
        as before, in one ``O(n_leaves)`` pass. Pass a ``size`` and the leaves are subsampled by
        **sensitivity sampling** (Feldman & Langberg 2011), which costs one weighted k-means over
        the leaves and fills in ``reference_cost`` / ``total_sensitivity``.

        The error has two independent halves, and neither is folded into the other.

        **Summarization**, present in both modes. With ``Δ = offset = Σᵢ Sᵢ``, the summary's cost
        ``ĉost(C) = Σᵢ (Sᵢ + nᵢ‖μᵢ − C‖²)`` satisfies, for every candidate ``C`` and every ``k``::

            0 ≤ ĉost(C) − cost(C) ≤ 4·√(Δ · cost(C)) + 4·Δ

        — a relative error of ``4√ρ + 4ρ`` at ``ρ = Δ/cost(C)``, and ``cost(C) ≥ OPT_k`` bounds it
        uniformly. :meth:`Coreset.summary_epsilon` evaluates it, and makes you name the ``α`` it
        assumes. This is what makes the word "coreset" here a claim rather than a label.

        **Sampling**, present only when ``size`` is given. ``ĉost(C) = Δ + Σᵢ nᵢ‖μᵢ − C‖²`` and
        ``Δ`` does not depend on ``C``, so the sample only has to be a coreset of the weighted set
        ``{(μᵢ, nᵢ)}`` — ``offset`` carries the constant instead of losing it. Sensitivity sampling
        attains the optimal worst-case size ``Õ(k·ε⁻²·min(√k, ε⁻²))``, matching the STOC 2022 lower
        bound, and ``Õ(k/ε²)`` on stable instances (arXiv 2405.01339).

        ``size`` at or above the leaf count returns every leaf exactly, with no sampling error,
        rather than a noisy redraw of something already held exactly. ``k`` defaults to
        ``n_clusters`` and ``seed`` to this estimator's.
        """
        est = self._require_fit()
        if size is None:
            w = np.asarray(est.microcluster_weights_, dtype=np.float64)
            r = np.asarray(est.microcluster_radii_, dtype=np.float64)
            return Coreset(
                centers=est.microcluster_centers_,
                weights=est.microcluster_weights_,
                radii=est.microcluster_radii_,
                offset=float((w * r * r).sum()),
                n_leaves=int(w.size),
            )
        if k is None:
            k = self.n_clusters if self.n_clusters and self.n_clusters > 0 else 8
        if k < 1:
            raise ValueError("k must be >= 1")
        if size < 1:
            raise ValueError("size must be >= 1")
        pts, weights, offset, ref, sens, n_leaves, radii = est.export_coreset_(
            int(k), int(size), int(self.seed if seed is None else seed)
        )
        return Coreset(
            centers=pts,
            weights=weights,
            radii=radii,
            offset=offset,
            reference_cost=ref,
            total_sensitivity=sens,
            n_leaves=n_leaves,
        )

    @property
    def microcluster_proba_(self):
        """Per-microcluster GMM soft responsibilities ``(n_microclusters, k)``. GMM heads only."""
        return self._require_fit().microcluster_proba_

    def predict_proba(self, X):
        """Per-point soft assignment, shape ``(n, n_components)``.

        The **GMM**, **vMF**, **Watson** and **Toeplitz** (``gmm-toeplitz`` / ``-full`` / ``-gs``)
        heads score the point under the fitted mixture, so ``predict_proba(X).argmax(1)`` is exactly
        :meth:`predict`. **fuzzy-cmeans** returns its own memberships
        ``u_j ∝ d_j^{−1/(m−1)}``, which also argmax to :meth:`predict` but are a partition of
        unity over the centres, **not** a posterior — no density is fitted. **k-means / x-means /
        Ward / HDBSCAN** return a heuristic
        ``softmax(−d²/2τ²)`` over the cluster centroids (``τ`` = mean cluster radius) — a confidence
        *proxy*, **not** a calibrated posterior. Columns are component indices aligned with
        :meth:`predict`."""
        est = self._require_fit()
        if self.method in (
            "gmm",
            "gmm-full",
            "mppca",
            "vmf",
            "watson",
            "gmm-toeplitz",
            "gmm-toeplitz-full",
            "gmm-toeplitz-gs",
            "fuzzy-cmeans",
        ):
            return est.predict_proba(X)
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


_WINDOWSTREAM_PARAMS = (
    "frame_width",
    "capacity",
    "max_micros",
    "threshold",
    "max_leaves",
    "branching",
    "leaf_cap",
    "seed",
)


class WindowStream:
    """Streaming **windowed** clusterer: ask "cluster the window ``[t0, t1]``" of a live stream.

    This is the CluStream question (Aggarwal et al., VLDB 2003) answered without CluStream's
    mechanism. CluStream keeps snapshots and **subtracts** the one at ``t0`` from the one at ``t1``;
    cluster-feature additivity makes that exact in real arithmetic and badly conditioned in floating
    point, and the condition number is ``S_AB/S_B`` — the scatter ratio, not the point-count ratio.
    The two agree on a stationary stream and diverge under drift, which is exactly when a window
    query is worth asking. On a two-half fixture with a **mass ratio of 2.0**, where any guard
    written on point counts sees nothing, the subtracted scatter comes back wrong by a factor of
    6155.

    So this keeps micro-clusters **per frame** instead, and a window is a *sum* of frames — every
    combination is the stable merge. The trade runs the other way and is worth knowing: a window
    resolves only to the frame boundary, so a query ending inside a frame receives that whole frame.
    That error is bounded by ``frame_width``. The subtraction's is bounded by nothing.

    Unlike :class:`DenStream`, which fades old data into one current model, this retains history:
    you can ask about last Tuesday. Older frames are merged pairwise as ``capacity`` fills, so
    resolution coarsens with age and never with recency.

    >>> ws = WindowStream(frame_width=3600.0, capacity=48)
    >>> ws.partial_fit(X, timestamps)     # doctest: +SKIP
    >>> ws.close_frame()                  # doctest: +SKIP
    >>> centers, masses, inertia = ws.cluster_window(t0, t1, k=5)   # doctest: +SKIP
    """

    def __init__(
        self,
        frame_width=1.0,
        capacity=64,
        max_micros=200,
        threshold=0.0,
        max_leaves=2000,
        branching=32,
        leaf_cap=32,
        seed=0,
    ):
        self.frame_width = frame_width
        self.capacity = capacity
        self.max_micros = max_micros
        self.threshold = threshold
        self.max_leaves = max_leaves
        self.branching = branching
        self.leaf_cap = leaf_cap
        self.seed = seed
        self._est = None

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in _WINDOWSTREAM_PARAMS}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in _WINDOWSTREAM_PARAMS:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator WindowStream. "
                    f"Valid parameters are: {sorted(_WINDOWSTREAM_PARAMS)}."
                )
            setattr(self, key, value)
        self._est = None
        return self

    def _require_fit(self):
        if self._est is None:
            raise AttributeError("This WindowStream instance is not fitted yet.")
        return self._est

    def partial_fit(self, X, t, y=None):
        """Stream a chunk of rows with their timestamps (one per row, or one scalar for all)."""
        rows = np.asarray(X, dtype=np.float64)
        times = np.broadcast_to(np.asarray(t, dtype=np.float64), (len(rows),))
        if self._est is None:
            self._est = _CoreWindowStream(**self.get_params())
        self._est.partial_fit(rows, [float(v) for v in times])
        return self

    def close_frame(self):
        """Close the frame currently being filled. Call before querying the most recent data."""
        self._require_fit().close_frame()
        return self

    @property
    def n_frames_(self):
        """Closed frames retained."""
        return self._require_fit().n_frames

    def frame_spans(self):
        """``(t_min, t_max, weight)`` per closed frame, oldest first."""
        return self._require_fit().frame_spans()

    def window_moments(self, t0, t1):
        """``{"weight", "mean", "ssd"}`` of the frames reaching into ``[t0, t1]``."""
        w, mean, ssd = self._require_fit().window_moments(float(t0), float(t1))
        return {"weight": w, "mean": np.asarray(mean, dtype=np.float64), "ssd": ssd}

    def cluster_window(self, t0, t1, k, max_iter=100):
        """``(centers, cluster_masses, inertia)`` for the window, or ``None`` if it holds fewer
        than ``k`` micro-clusters — a question the summary cannot answer rather than guess at."""
        got = self._require_fit().cluster_window(float(t0), float(t1), int(k), int(max_iter))
        if got is None:
            return None
        centers, masses, inertia = got
        return (
            np.asarray(centers, dtype=np.float64),
            np.asarray(masses, dtype=np.float64),
            inertia,
        )

    def __repr__(self):
        return f"WindowStream(frame_width={self.frame_width}, capacity={self.capacity})"


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
    def drift_(self):
        """Drift diagnostic over the routing distance — ``{alarms, last_alarm, distance, window}``.

        The detector (ADWIN; Bifet & Gavaldà, SDM 2007) watches one number per streamed point: how
        far the point landed from the nearest micro-cluster, in units of ``eps``. Stationary
        data sits near 1 by construction; a distribution moving into space the model does not cover
        sends it far higher, and an *alarm* is raised when that shift is larger than chance explains
        at δ = 0.002. ``distance`` is the mean over the adaptive window, ``window`` its size (it
        collapses on a change and regrows while the stream is stationary), ``last_alarm`` the stream
        time of the most recent report.

        **It reports; it does not act.** An alarm prunes nothing, promotes nothing and relabels
        nothing — what to do about a change is the caller's policy. ``decay`` and this answer
        different questions: ``decay`` sets how fast the model forgets, on a fixed schedule, whether
        or not anything changed; the detector says whether anything did. An early alarm is the model
        warming up, not drift.
        """
        return self._require_fit().drift_

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
    def drift_(self):
        """Drift diagnostic over the routing distance — ``{alarms, last_alarm, distance, window}``.

        The detector (ADWIN; Bifet & Gavaldà, SDM 2007) watches one number per streamed point: how
        far the point landed from the nearest micro-cluster, in units of ``r``. Stationary
        data sits near 1 by construction; a distribution moving into space the model does not cover
        sends it far higher, and an *alarm* is raised when that shift is larger than chance explains
        at δ = 0.002. ``distance`` is the mean over the adaptive window, ``window`` its size (it
        collapses on a change and regrows while the stream is stationary), ``last_alarm`` the stream
        time of the most recent report.

        **It reports; it does not act.** An alarm prunes nothing, promotes nothing and relabels
        nothing — what to do about a change is the caller's policy. ``decay`` and this answer
        different questions: ``decay`` sets how fast the model forgets, on a fixed schedule, whether
        or not anything changed; the detector says whether anything did. An early alarm is the model
        warming up, not drift.
        """
        return self._require_fit().drift_

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


_BREGMAN_PARAMS = (
    "n_clusters",
    "divergence",
    "method",
    "beta",
    "threshold",
    "branching",
    "leaf_cap",
    "max_leaves",
    "max_iter",
    "n_init",
    "seed",
)

_BREGMAN_DOMAIN = {
    "kl": "every value must be > 0",
    "itakura-saito": "every value must be > 0",
    "logistic": "every value must be in (0, 1)",
}


class BregmanBetula:
    """CF-tree clustering in a **Bregman geometry** rather than squared Euclidean.

    A second estimator rather than a ``divergence=`` keyword on :class:`Betula`, because the two
    axes are orthogonal and collapsing them makes meaningless combinations writable — a Gaussian
    head reading a Bregman information as if it were a variance, a chi-squared gate applying a
    variance prior to a quantity that is not one. See ``docs/adr/004-bregman-public-api.md``.

    ``divergence`` picks the geometry: ``"kl"`` for distributions on the simplex,
    ``"itakura-saito"`` for spectra (scale-invariant), ``"logistic"`` for probabilities, and
    ``"euclidean"`` for the squared-Euclidean case, which is a Bregman divergence too and makes
    this estimator reduce to the shipped one.

    ``method`` picks the head over the leaves: ``"kmeans"`` (Bregman k-means, Banerjee et al.),
    ``"ward"`` (Bregman-Ward HAC by Anderberg — see ``docs/adr/002``), or ``"mixture"`` (soft
    Bregman mixture by variational EM).

    ``beta`` is the mixture's **inverse dispersion**: the model is
    ``p(x | k) ∝ exp(−beta · d_φ(x, μ_k)) · b_φ(x)``, and ``beta = 1`` is Banerjee's soft Bregman
    clustering exactly. Separation is measured in *nats of divergence*, not in coordinates — under
    a scale-invariant divergence like Itakura–Saito, centres that look far apart can be a fraction
    of a nat apart, and the mixture will correctly report that they overlap. Raise ``beta`` until
    the responsibilities are as sharp as the application needs. It is rejected rather than ignored
    when ``method`` is not ``"mixture"``.

    The domain is checked here, before any value reaches the engine: KL and Itakura–Saito need
    ``x > 0``, logistic needs ``x`` in ``(0, 1)``. ``float64``.
    """

    def __init__(
        self,
        n_clusters=8,
        divergence="kl",
        method="kmeans",
        beta=1.0,
        threshold=0.0,
        branching=50,
        leaf_cap=50,
        max_leaves=2048,
        max_iter=100,
        n_init=4,
        seed=0,
    ):
        self.n_clusters = n_clusters
        self.divergence = divergence
        self.method = method
        self.beta = beta
        self.threshold = threshold
        self.branching = branching
        self.leaf_cap = leaf_cap
        self.max_leaves = max_leaves
        self.max_iter = max_iter
        self.n_init = n_init
        self.seed = seed
        self._est = None

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in _BREGMAN_PARAMS}

    def set_params(self, **params):
        for key, value in params.items():
            if key not in _BREGMAN_PARAMS:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator BregmanBetula. "
                    f"Valid parameters are: {sorted(_BREGMAN_PARAMS)}."
                )
            setattr(self, key, value)
        self._est = None
        return self

    def _build(self):
        if self.method != "mixture" and self.beta != 1.0:
            raise ValueError(
                f"beta is the mixture's inverse dispersion and does nothing under "
                f"method={self.method!r}; drop it or set method='mixture'."
            )
        return _CoreBregmanBetula(**self.get_params())

    def _require_fit(self):
        if self._est is None:
            raise AttributeError("This BregmanBetula instance is not fitted yet.")
        return self._est

    def fit(self, X, y=None):
        self.fit_predict(X)
        return self

    def fit_predict(self, X, y=None):
        est = self._build()
        labels = est.fit_predict(X)
        self._est = est
        return labels

    @property
    def labels_(self):
        """Training-row labels from the last fit."""
        return self._require_fit().labels

    @property
    def n_leaves_(self):
        """Leaf count of the fitted tree — how much of ``max_leaves`` the geometry used."""
        return self._require_fit().n_leaves

    def __repr__(self):
        return (
            f"BregmanBetula(n_clusters={self.n_clusters}, divergence={self.divergence!r}, "
            f"method={self.method!r})"
        )
