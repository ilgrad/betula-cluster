"""Type stubs for the `betula_cluster` package (Rust `_core` engine + Python estimator)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, final

import numpy as np
from numpy.typing import NDArray

from .tuning import ThresholdEstimate as ThresholdEstimate
from .tuning import TuneResult as TuneResult
from .tuning import estimate_threshold as estimate_threshold
from .tuning import tune as tune

__all__ = [
    "Betula",
    "BregmanBetula",
    "ConsensusResult",
    "Coreset",
    "DbStream",
    "DdSketch",
    "DenStream",
    "KPrototypes",
    "KllSketch",
    "MapperGraph",
    "ThresholdEstimate",
    "TuneResult",
    "WindowStream",
    "__version__",
    "consensus",
    "estimate_threshold",
    "fit_predict",
    "fit_predict_sparse",
    "mixture_w2",
    "tune",
]

__version__: str

_FloatArray = NDArray[np.float64] | NDArray[np.float32]
_Feature = Literal["spherical", "diagonal", "full", "fd"]
_Projection = Literal["none", "weighted-nmf", "weighted-nmf-kl", "svd"]
_Method = Literal[
    "kmeans",
    "gmm",
    "gmm-full",
    "mppca",
    "ward",
    "average",
    "weighted",
    "centroid",
    "median",
    "spectral",
    "leiden",
    "leiden-cpm",
    "spherical-kmeans",
    "vmf",
    "gmm-toeplitz",
    "gmm-toeplitz-full",
    "gmm-toeplitz-gs",
    "hdbscan",
    "scale-space",
]
_Distance = Literal["euclidean", "manhattan", "ward", "average"]
_Absorb = Literal[
    "euclidean", "manhattan", "average", "diameter", "ward", "radius", "chi2", "subspace"
]
_Lens = Literal["density", "radius", "l2norm", "coordinate", "eccentricity"]
_Link = Literal["centroid", "bhattacharyya"]
_OutlierMetric = Literal["radius", "mahalanobis"]
_Repr = Literal["medoid", "boundary", "outlier", "diverse"]
_Strategy = Literal["uncertain", "outlier"]

@dataclass(frozen=True)
class Coreset:
    """A weighted-point coreset: the CF-tree leaf microclusters as (centers, weights, radii)."""

    centers: NDArray[np.float64]
    weights: NDArray[np.float64]
    radii: NDArray[np.float64]
    offset: float = ...
    reference_cost: float | None = ...
    total_sensitivity: float | None = ...
    n_leaves: int | None = ...
    @property
    def n_points(self) -> float: ...
    def cost(self, centers: _FloatArray) -> float: ...
    def summary_epsilon(self, alpha: float) -> float: ...

@dataclass(frozen=True)
class ConsensusResult:
    """Consensus of insertion-order-permuted clusterings + a per-point stability score."""

    labels: NDArray[np.int64]
    confidence: NDArray[np.float64]
    n_runs: int
    @property
    def mean_confidence(self) -> float: ...

def consensus(
    X: _FloatArray,
    n_clusters: int,
    *,
    n_runs: int = ...,
    seed: int = ...,
    n_jobs: int = ...,
    **fit_kwargs: object,
) -> ConsensusResult: ...

@dataclass(frozen=True)
class MapperGraph:
    """A Mapper topological-skeleton graph over a fitted model's leaf microclusters."""

    node_members: list[NDArray[np.int64]]
    node_mass: NDArray[np.float64]
    node_bin: NDArray[np.int64]
    node_lens: NDArray[np.float64]
    node_centroids: NDArray[np.float64]
    edges: NDArray[np.int64]
    edge_overlap: NDArray[np.float64]
    branch_points: NDArray[np.int64]
    bridges: NDArray[np.int64]
    persistence_overlap: NDArray[np.float64]
    persistence_lens: NDArray[np.float64]
    @property
    def n_nodes(self) -> int: ...
    @property
    def n_edges(self) -> int: ...
    def persistence(
        self, filtration: str = ..., finite_only: bool = ...
    ) -> NDArray[np.float64]: ...
    def to_networkx(self) -> Any: ...

def fit_predict(
    data: _FloatArray,
    n_clusters: int = ...,
    feature: _Feature = ...,
    method: _Method = ...,
    threshold: float = ...,
    branching: int = ...,
    leaf_cap: int = ...,
    max_leaves: int = ...,
    max_iter: int = ...,
    min_samples: int = ...,
    min_cluster_size: int = ...,
    seed: int = ...,
    distance: _Distance = ...,
    absorb: _Absorb = ...,
    chi2_p: float = ...,
    chi2_scale: float = ...,
    n_jobs: int = ...,
    normalize: bool = ...,
    resolution: float = ...,
    covariance_weight: float = ...,
    tangent_weight: float = ...,
    tangent_rank: int = ...,
    projection: _Projection = ...,
    projection_dim: int = ...,
    projection_max_iter: int = ...,
    refine: int = ...,
    rank: int = ...,
    graph_degree: int = ...,
    balance: float | None = ...,
) -> NDArray[np.int64]:
    """Cluster ``data`` in one shot and return per-point integer labels (``-1`` = noise)."""

def fit_predict_sparse(
    X: Any,
    n_clusters: int = ...,
    method: _Method = ...,
    threshold: float = ...,
    max_leaves: int = ...,
    max_iter: int = ...,
    seed: int = ...,
    projection: _Projection = ...,
    projection_dim: int = ...,
    projection_max_iter: int = ...,
) -> NDArray[np.int64]:
    """One-shot O(nnz) clustering of a scipy.sparse matrix; one int64 label per row."""

def mixture_w2(
    weights_a: _FloatArray,
    means_a: _FloatArray,
    covariances_a: _FloatArray,
    weights_b: _FloatArray,
    means_b: _FloatArray,
    covariances_b: _FloatArray,
) -> float:
    """Mixture-Wasserstein MW2 between two fitted Gaussian mixtures; lower is closer."""

class Betula:
    """Streaming, scikit-learn-style BETULA estimator."""

    def __init__(
        self,
        n_clusters: int = ...,
        feature: _Feature = ...,
        method: _Method = ...,
        threshold: float | Literal["auto"] = ...,
        branching: int = ...,
        leaf_cap: int = ...,
        max_leaves: int | float = ...,
        max_iter: int = ...,
        min_samples: int = ...,
        min_cluster_size: int = ...,
        seed: int = ...,
        distance: _Distance = ...,
        absorb: _Absorb = ...,
        chi2_p: float = ...,
        chi2_scale: float = ...,
        decay: float = ...,
        normalize: bool = ...,
        huber_k: float | None = ...,
        balance: float | None = ...,
        resolution: float = ...,
        covariance_weight: float = ...,
        tangent_weight: float = ...,
        tangent_rank: int = ...,
        projection: _Projection = ...,
        projection_dim: int = ...,
        projection_max_iter: int = ...,
        refine: int = ...,
        rank: int = ...,
        graph_degree: int = ...,
        memory_budget_mb: float | None = ...,
    ) -> None: ...
    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> Betula: ...
    def fit(
        self, X: _FloatArray, y: Any = ..., must_link: Any = ..., cannot_link: Any = ...
    ) -> Betula: ...
    def fit_predict(
        self, X: _FloatArray, y: Any = ..., must_link: Any = ..., cannot_link: Any = ...
    ) -> NDArray[np.int64]: ...
    def partial_fit(self, X: _FloatArray | None = ..., y: Any = ...) -> Betula:
        """Absorb a chunk; call with no argument to finalize the global clustering."""

    def predict(self, X: _FloatArray) -> NDArray[np.int64]: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> Betula: ...
    @property
    def n_clusters_(self) -> int: ...
    @property
    def n_leaves_(self) -> int: ...
    @property
    def n_rebuilds_(self) -> int: ...
    @property
    def threshold_(self) -> float: ...
    @property
    def effective_max_leaves_(self) -> int: ...
    # ── inspectability ────────────────────────────────────────────────────────────────────────
    @property
    def microcluster_centers_(self) -> NDArray[np.float64]: ...
    @property
    def microcluster_weights_(self) -> NDArray[np.float64]: ...
    @property
    def microcluster_radii_(self) -> NDArray[np.float64]: ...
    @property
    def components_(self) -> NDArray[np.float64]: ...
    @property
    def reconstruction_err_(self) -> float: ...
    @property
    def cluster_centers_(self) -> NDArray[np.float64]: ...
    @property
    def cluster_radii_(self) -> NDArray[np.float64]: ...
    @property
    def cluster_sizes_(self) -> NDArray[np.float64]: ...
    def assign_microclusters(self, X: _FloatArray) -> NDArray[np.int64]: ...
    def outlier_scores(
        self, X: _FloatArray, metric: _OutlierMetric = ...
    ) -> NDArray[np.float64]: ...
    def tree_report(
        self, X: _FloatArray | None = ..., **estimate_kwargs: Any
    ) -> dict[str, Any]: ...
    def summary(self) -> dict[str, float]: ...
    def validity(self) -> dict[str, float]: ...
    def summary_mmd(self, X: _FloatArray, *, bandwidth: float | None = ...) -> float: ...
    def summary_w2(self, other: Betula) -> float: ...
    def find_outliers(
        self, X: _FloatArray, top_k: int = ..., metric: _OutlierMetric = ...
    ) -> NDArray[np.int64]: ...
    def sample_representatives(
        self, X: _FloatArray, k: int = ...
    ) -> dict[int, NDArray[np.int64]]: ...
    def find_near_duplicates(self, X: _FloatArray, radius: float) -> list[NDArray[np.int64]]: ...
    def near_duplicate_pairs(
        self, X: _FloatArray, threshold: float = ..., *, neighbors: int = ...
    ) -> NDArray[np.float64]: ...
    def mapper(
        self,
        lens: _Lens = ...,
        resolution: int = ...,
        gain: float = ...,
        link_scale: float = ...,
        min_node_mass: float = ...,
        density_k: int = ...,
        coordinate: int = ...,
        link: _Link = ...,
    ) -> MapperGraph: ...
    def mapper_stability(
        self, resolutions: Sequence[int] | None = ..., **mapper_kwargs: Any
    ) -> list[dict[str, int]]: ...
    # ── coreset / soft assignment / diagnostics ──────────────────────────────────────────────────
    @property
    def microcluster_proba_(self) -> NDArray[np.float64]: ...
    def export_coreset(
        self, size: int | None = ..., k: int | None = ..., seed: int | None = ...
    ) -> Coreset: ...
    def predict_proba(self, X: _FloatArray) -> NDArray[np.float64]: ...
    def assignment_confidence(self, X: _FloatArray) -> NDArray[np.float64]: ...
    def diagnostics(self) -> dict[str, float]: ...
    def representatives(
        self, X: _FloatArray, cluster_id: int, method: _Repr = ..., k: int = ...
    ) -> NDArray[np.int64]: ...
    def cluster_profile(self, cluster_id: int) -> dict[str, Any]: ...
    # ── drift monitoring / active learning ────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]: ...
    @staticmethod
    def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]: ...
    def active_learning_batch(
        self, X: _FloatArray, n: int = ..., strategy: _Strategy = ...
    ) -> NDArray[np.int64]: ...

class WindowStream:
    frame_width: float
    capacity: int
    max_micros: int
    threshold: float
    max_leaves: int
    branching: int
    leaf_cap: int
    seed: int
    def __init__(
        self,
        frame_width: float = ...,
        capacity: int = ...,
        max_micros: int = ...,
        threshold: float = ...,
        max_leaves: int = ...,
        branching: int = ...,
        leaf_cap: int = ...,
        seed: int = ...,
    ) -> None: ...
    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> WindowStream: ...
    def partial_fit(
        self, X: _FloatArray, t: float | Sequence[float] | _FloatArray, y: Any = ...
    ) -> WindowStream: ...
    def close_frame(self) -> WindowStream: ...
    @property
    def n_frames_(self) -> int: ...
    def frame_spans(self) -> list[tuple[float, float, float]]: ...
    def window_moments(self, t0: float, t1: float) -> dict[str, Any]: ...
    def cluster_window(
        self, t0: float, t1: float, k: int, max_iter: int = ...
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None: ...

class DenStream:
    """Streaming DenStream density clusterer over fading micro-clusters."""

    def __init__(
        self, eps: float = ..., decay: float = ..., beta: float = ..., mu: float = ...
    ) -> None: ...
    def get_params(self, deep: bool = ...) -> dict[str, float]: ...
    def set_params(self, **params: Any) -> DenStream: ...
    def partial_fit(self, X: _FloatArray, y: Any = ...) -> DenStream: ...
    def cluster(self) -> DenStream: ...
    def fit(self, X: _FloatArray, y: Any = ...) -> DenStream: ...
    def fit_predict(self, X: _FloatArray, y: Any = ...) -> NDArray[np.int64]: ...
    def predict(self, X: _FloatArray) -> NDArray[np.int64]: ...
    @property
    def n_clusters_(self) -> int: ...
    @property
    def n_microclusters_(self) -> int: ...
    @property
    def microcluster_centers_(self) -> NDArray[np.float64]: ...
    @property
    def microcluster_weights_(self) -> NDArray[np.float64]: ...
    @property
    def microcluster_radii_(self) -> NDArray[np.float64]: ...

class DbStream:
    """Streaming DBSTREAM density clusterer (shared-density connectivity)."""

    def __init__(
        self, r: float = ..., decay: float = ..., alpha: float = ..., min_weight: float = ...
    ) -> None: ...
    def get_params(self, deep: bool = ...) -> dict[str, float]: ...
    def set_params(self, **params: Any) -> DbStream: ...
    def partial_fit(self, X: _FloatArray, y: Any = ...) -> DbStream: ...
    def cluster(self) -> DbStream: ...
    def fit(self, X: _FloatArray, y: Any = ...) -> DbStream: ...
    def fit_predict(self, X: _FloatArray, y: Any = ...) -> NDArray[np.int64]: ...
    def predict(self, X: _FloatArray) -> NDArray[np.int64]: ...
    @property
    def n_clusters_(self) -> int: ...
    @property
    def n_microclusters_(self) -> int: ...
    @property
    def microcluster_centers_(self) -> NDArray[np.float64]: ...
    @property
    def microcluster_weights_(self) -> NDArray[np.float64]: ...
    @property
    def microcluster_radii_(self) -> NDArray[np.float64]: ...

class BregmanBetula:
    """CF-tree clustering in a Bregman geometry rather than squared Euclidean."""

    def __init__(
        self,
        n_clusters: int = ...,
        divergence: str = ...,
        method: str = ...,
        beta: float = ...,
        threshold: float = ...,
        branching: int = ...,
        leaf_cap: int = ...,
        max_leaves: int = ...,
        max_iter: int = ...,
        n_init: int = ...,
        seed: int = ...,
    ) -> None: ...
    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> BregmanBetula: ...
    def fit(self, X: _FloatArray, y: Any = ...) -> BregmanBetula: ...
    def fit_predict(self, X: _FloatArray, y: Any = ...) -> NDArray[np.int64]: ...
    @property
    def labels_(self) -> NDArray[np.int64]: ...
    @property
    def n_leaves_(self) -> int: ...

class KPrototypes:
    """k-prototypes clustering of mixed numeric + categorical data (Huang, 1997)."""

    def __init__(
        self,
        n_clusters: int = ...,
        categorical: Sequence[int] = ...,
        gamma: float | None = ...,
        threshold: float = ...,
        max_leaves: int = ...,
        max_iter: int = ...,
        n_init: int = ...,
        seed: int = ...,
    ) -> None: ...
    def get_params(self, deep: bool = ...) -> dict[str, Any]: ...
    def set_params(self, **params: Any) -> KPrototypes: ...
    def fit(self, X: _FloatArray, y: Any = ...) -> KPrototypes: ...
    def fit_predict(self, X: _FloatArray, y: Any = ...) -> NDArray[np.int64]: ...
    def predict(self, X: _FloatArray) -> NDArray[np.int64]: ...
    @property
    def n_clusters_(self) -> int: ...
    @property
    def cluster_centroids_(self) -> NDArray[np.float64]: ...
    @property
    def cluster_modes_(self) -> NDArray[np.int64]: ...

@final
class KllSketch:
    """Streaming KLL quantile sketch (rank-error guarantee)."""

    def __new__(cls, k: int = ..., seed: int = ...) -> KllSketch: ...
    def update(self, x: float) -> None:
        """Add one value."""

    def update_many(self, data: NDArray[np.float64]) -> None:
        """Add every value of a 1-D array."""

    def merge(self, other: KllSketch) -> None:
        """Merge another KLL sketch in; this one then summarizes the union of both streams."""

    def quantile(self, q: float) -> float:
        """Estimated ``q``-quantile (``q`` in [0, 1]); exact at the endpoints, NaN if empty."""

    def quantiles(self, qs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimated quantiles for an array of ``q`` values, in one pass over the built CDF."""

    def rank(self, value: float) -> int:
        """Estimated number of stored values ``<= value``."""

    @property
    def count(self) -> int:
        """Total number of values added."""

    @property
    def min(self) -> float:
        """Smallest value seen (NaN if empty)."""

    @property
    def max(self) -> float:
        """Largest value seen (NaN if empty)."""

@final
class DdSketch:
    """Streaming DDSketch quantile sketch (relative-error guarantee)."""

    def __new__(cls, alpha: float = ..., max_bins: int = ...) -> DdSketch: ...
    def update(self, x: float) -> None:
        """Add one value."""

    def update_many(self, data: NDArray[np.float64]) -> None:
        """Add every value of a 1-D array."""

    def merge(self, other: DdSketch) -> None:
        """Merge another DDSketch in; this one then summarizes the union of both streams."""

    def quantile(self, q: float) -> float:
        """Estimated ``q``-quantile (``q`` in [0, 1]); exact at the endpoints, NaN if empty."""

    def quantiles(self, qs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Estimated quantiles for an array of ``q`` values."""

    @property
    def count(self) -> int:
        """Total number of values added."""

    @property
    def alpha(self) -> float:
        """Relative accuracy ``alpha`` (smaller = tighter quantiles, more buckets)."""

    @property
    def min(self) -> float:
        """Smallest value seen (NaN if empty)."""

    @property
    def max(self) -> float:
        """Largest value seen (NaN if empty)."""
