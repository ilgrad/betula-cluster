"""Reproduce one published number per census rival before its rows are read.

A rival run badly is a claim nobody should believe. For each package this runs the example its own
documentation publishes, with the documented call, and checks the documented output; the census
section of `bench/RESULTS.md` cites the result. lumbermark, EVoC, DelTriC and CoreSPECT publish no
numeric output, so they have no row here, and that is stated where their rows are read. The
CluProp paper publishes a number and no code, so its row reruns the recipe on the paper's data.

    uv run --no-sync --with scikit-learn --with classixclustering --with denmune \\
        --with finch-clust --with genieclust --with hdbscan --with fast-hdbscan \\
        --with phenograph --with dbscan --with pyod --with bico --with river \\
        --with pynndescent --with python-igraph python bench/rival_reproductions.py

Two examples need the authors' data files, fetched once into scikit-learn's data home: `jain`
from the clustering benchmark suite (genieclust, DenMune) and FINCH's `gestalt_clusters.txt`.
"""

from __future__ import annotations

import contextlib
import gzip
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import get_data_home, make_blobs
from sklearn.metrics import adjusted_rand_score

SOURCES = {
    "jain.data.gz": "https://raw.githubusercontent.com/gagolews/clustering-data-v1/master/sipu/jain.data.gz",
    "jain.labels0.gz": "https://raw.githubusercontent.com/gagolews/clustering-data-v1/master/sipu/jain.labels0.gz",
    "gestalt_clusters.txt": "https://raw.githubusercontent.com/ssarfraz/FINCH-Clustering/master/data/toy_data/gestalt_clusters.txt",
}


def fetch(name: str) -> Path:
    path = Path(get_data_home()) / "betula_bench" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCES[name], timeout=60) as r:
            path.write_bytes(r.read())
    return path


def jain():
    with gzip.open(fetch("jain.data.gz"), "rt") as f:
        x = np.loadtxt(f, ndmin=2)
    with gzip.open(fetch("jain.labels0.gz"), "rt") as f:
        y = np.loadtxt(f, dtype=np.intp) - 1
    return x, y


def classix():
    """classix docs, clustering_tutorial.rst: 163 groups merged into 14 clusters, 7 after minPts=99."""
    from classix import CLASSIX

    x, _ = make_blobs(n_samples=1000, centers=10, n_features=2, cluster_std=1, random_state=42)
    clx = CLASSIX(sorting="pca", group_merging="distance", radius=0.1, verbose=0, minPts=99)
    return 7, len(np.unique(clx.fit(x).labels_))


def genie():
    """genieclust docs, weave/basics.html: ARI 1.0 on jain with the true count."""
    import genieclust

    x, y = jain()
    labels = genieclust.Genie(n_clusters=len(np.unique(y))).fit_predict(x)
    return 1.0, round(adjusted_rand_score(y, labels), 6)


def denmune():
    """DenMune docs, examples/2D_shapes_datasets.html: jain at k_nearest=15, ARI 1.0, 2 clusters."""
    from denmune import DenMune

    x, y = jain()
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        labels, _ = DenMune(train_data=x, k_nearest=15).fit_predict(
            show_analyzer=False, show_noise=True, show_plots=False
        )
    return 1.0, round(adjusted_rand_score(y, np.asarray(labels["train"])), 6)


def finch():
    """FINCH notebooks/Basic_usage.ipynb: partitions of 92, 29, 14 and 12 clusters on gestalt."""
    from finch import FINCH

    data = np.genfromtxt(fetch("gestalt_clusters.txt"))[:, :2]
    _, num_clust, _ = FINCH(data, distance="euclidean", verbose=False)
    return [92, 29, 14, 12], [int(k) for k in num_clust]


def hdbscan_blobs():
    """hdbscan docs, basic_hdbscan.html: three clusters on make_blobs(2000, 10 features).

    The example draws without a seed; seed 0 is ours, and the published count is what is checked.
    """
    import hdbscan

    x, _ = make_blobs(n_samples=2000, n_features=10, random_state=0)
    return 3, int(hdbscan.HDBSCAN().fit(x).labels_.max()) + 1


def plscan():
    """fast_hdbscan, the PLSCAN docstring: one cluster layer on make_blobs(1000, random_state=42)."""
    import fast_hdbscan

    x, _ = make_blobs(1000, random_state=42)
    clusterer = fast_hdbscan.PLSCAN()
    clusterer.fit_predict(x)
    return 1, len(clusterer.cluster_layers_)


def optics():
    """scikit-learn, the OPTICS docstring: [0, 0, 0, 1, 1, 1]."""
    from sklearn.cluster import OPTICS

    x = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]])
    return [0, 0, 0, 1, 1, 1], OPTICS(min_samples=2).fit(x).labels_.tolist()


def _dbscan_blobs():
    """The fixture of scikit-learn's `plot_dbscan.py`, which the `dbscan` README copies."""
    from sklearn.preprocessing import StandardScaler

    x, y = make_blobs(
        n_samples=750, centers=[[1, 1], [-1, -1], [1, -1]], cluster_std=0.4, random_state=0
    )
    return StandardScaler().fit_transform(x), y


def sklearn_dbscan():
    """scikit-learn, plot_dbscan.py as rendered for 1.9.1: 3 clusters, 18 noise points, ARI 0.952."""
    from sklearn.cluster import DBSCAN

    x, y = _dbscan_blobs()
    labels = DBSCAN(eps=0.3, min_samples=10).fit(x).labels_
    found = len(set(labels)) - (1 if -1 in labels else 0)
    return (3, 18, 0.952), (
        found,
        int((labels == -1).sum()),
        round(adjusted_rand_score(y, labels), 3),
    )


def dbscan_pkg():
    """dbscan (Wang, Gu, Shun), the README example: the plot's title reads 3 clusters."""
    from dbscan import DBSCAN

    x, _ = _dbscan_blobs()
    labels, _ = DBSCAN(x, eps=0.3, min_samples=10)
    return 3, len(set(labels.tolist())) - (1 if -1 in labels else 0)


def pyod_knn():
    """PyOD README, the `KNN` quick start: ROC 0.9992 and precision @ rank n 0.95 on training data."""
    from pyod.models.knn import KNN
    from pyod.utils.data import generate_data
    from pyod.utils.utility import precision_n_scores
    from sklearn.metrics import roc_auc_score

    x, _, y, _ = generate_data(
        n_train=200, n_test=100, n_features=2, contamination=0.1, random_state=42
    )
    s = KNN().fit(x).decision_scores_
    return (0.9992, 0.95), (round(roc_auc_score(y, s), 4), round(precision_n_scores(y, s), 4))


def phenograph():
    """PhenoGraph examples/example-using-random-data-points.ipynb: three communities, Q = 0.664858.

    Louvain runs unseeded there, so the count and the modularity to two decimals are checked.
    """
    import phenograph

    x, _ = make_blobs(
        n_samples=150,
        cluster_std=[0.8, 1, 1],
        centers=[(-2, -2), (-3, 3), (2, 2)],
        n_features=2,
        random_state=42,
    )
    communities, _, q = phenograph.cluster(x, k=30)
    return (3, 0.66), (len(np.unique(communities)), round(float(q), 2))


def minibatch_kmeans():
    """scikit-learn, the MiniBatchKMeans docstring: centres (3.375, 3) and (0.75, 0.5) after two
    `partial_fit` batches, and `predict` [1, 0]."""
    from sklearn.cluster import MiniBatchKMeans

    x = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 0], [4, 4]])
    x = np.vstack([x, [[4, 5], [0, 1], [2, 2], [3, 2], [5, 5], [1, -1]]])
    km = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=6, n_init="auto")
    km.partial_fit(x[0:6]).partial_fit(x[6:12])
    centres = tuple(tuple(round(float(v), 3) for v in c) for c in km.cluster_centers_)
    return (((3.375, 3.0), (0.75, 0.5)), [1, 0]), (centres, km.predict([[0, 0], [4, 4]]).tolist())


def birch():
    """scikit-learn, the Birch docstring: `predict` [0, 0, 0, 1, 1, 1] with `n_clusters=None`."""
    from sklearn.cluster import Birch

    x = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    return [0, 0, 0, 1, 1, 1], Birch(n_clusters=None).fit(x).predict(x).tolist()


def bico():
    """BICO README, the first example: its first cluster centre, printed to eight decimals."""
    from bico import BICO

    np.random.seed(42)
    data = np.random.rand(10000, 10)
    model = BICO(n_clusters=3, random_state=0, fit_coreset=True).fit(data)
    want = (0.46892639, 0.41968333, 0.47302945, 0.51782955, 0.39390839)
    want += (0.56209413, 0.4481691, 0.49521457, 0.31394509, 0.5104331)
    return want, tuple(round(float(v), 8) for v in model.cluster_centers_[0])


def river_dbstream():
    """River, the DBSTREAM docstring: `predict_one` 0 and 1 on its two probes, two clusters."""
    from river import cluster

    points = [[1, 0.5], [1, 0.625], [1, 0.75], [1, 1.125], [1, 1.5], [1, 1.75]]
    points += [[4, 1.5], [4, 2.25], [4, 2.5], [4, 3], [4, 3.25], [4, 3.5]]
    model = cluster.DBSTREAM(
        clustering_threshold=1.5,
        fading_factor=0.05,
        cleanup_interval=4,
        intersection_factor=0.5,
        minimum_weight=1,
    )
    for row in points:
        model.learn_one(dict(enumerate(row)))
    got = [model.predict_one(dict(enumerate(q))) for q in ([1, 2], [5, 2])]
    return [0, 1, 2], [*got, model._n_clusters]


def river_denstream():
    """River, the DenStream docstring: `predict_one` 0, 2 and 1 on its three probes, three clusters."""
    from river import cluster

    points = [[-1, -0.5], [-1, -0.625], [-1, -0.75], [-1, -1], [-1, -1.125], [-1, -1.25]]
    points += [[-1.5, -0.5], [-1.5, -0.625], [-1.5, -0.75], [-1.5, -1], [-1.5, -1.125]]
    points += [[-1.5, -1.25], [1, 1.5], [1, 1.75], [1, 2], [4, 1.25], [4, 1.5], [4, 2.25]]
    points += [[4, 2.5], [4, 3], [4, 3.25], [4, 3.5], [4, 3.75], [4, 4]]
    model = cluster.DenStream(
        decaying_factor=0.01, beta=0.5, mu=2.5, epsilon=1.0, n_samples_init=10
    )
    for row in points:
        model.learn_one(dict(enumerate(row)))
    got = [model.predict_one(dict(enumerate(q))) for q in ([-1, -2], [5, 4], [1, 1])]
    return [0, 2, 1, 3], [*got, model.n_clusters]


def cluprop():
    """CluProp (arXiv 2605.00390 v2), Table 1: ARI 0.87 ± 0.01 and 12 ± 1 clusters on raw
    MNIST-70k, cosine, k = 8. The paper publishes no code; `rivals._cluprop` is the recipe as its
    text describes it, with the three choices it leaves open made as that function says. Seed 0."""
    from rivals import _cluprop, load

    x, y = load("mnist", 70_000, 0)
    labels = _cluprop(x, 0)
    ari, k = adjusted_rand_score(y, labels), len(np.unique(labels))
    return (
        "ARI 0.87 ± 0.01, 12 ± 1 clusters",
        (round(ari, 3), k),
        abs(ari - 0.87) <= 0.01 and abs(k - 12) <= 1,
    )


CHECKS = {
    "CLASSIX 1.5.3": classix,
    "genieclust 1.3.0": genie,
    "DenMune 1.17.1": denmune,
    "FINCH 0.2.3": finch,
    "hdbscan 0.8.44": hdbscan_blobs,
    "fast_hdbscan 0.3.4 (PLSCAN)": plscan,
    "scikit-learn 1.9.1 OPTICS": optics,
    "PhenoGraph 1.5.7": phenograph,
    "scikit-learn 1.9.1 DBSCAN": sklearn_dbscan,
    "dbscan 1.0.0": dbscan_pkg,
    "PyOD 3.6.8 KNN": pyod_knn,
    "scikit-learn 1.9.1 MiniBatchKMeans": minibatch_kmeans,
    "scikit-learn 1.9.1 Birch": birch,
    "BICO 0.1.8": bico,
    "River 0.27.0 DBSTREAM": river_dbstream,
    "River 0.27.0 DenStream": river_denstream,
    "CluProp recipe (pynndescent 0.7.0, python-igraph 1.0.0)": cluprop,
}


# A mismatch traced to the documentation, not to the call: stable across runs, and present with the
# documented call verbatim. Each is reported, not hidden, and is not counted as a harness failure.
STALE = {
    # The finest partition reproduces; the coarser ones are 25, 7, 2 with either metric's call
    # as published (cosine gives 132, 40, 11, 3): the notebook's output does not match this release.
    "FINCH 0.2.3": "the finest partition (92) reproduces; the coarser ones do not",
    # Three layers (33, 7 and 3 clusters) on 5 of 5 runs, with and without `reproducible=True`, and
    # at the commit that added the example: its output was never what the code returns.
    # The harness runs `fast_hdbscan.HDBSCAN`, whose documentation publishes plots only.
    "fast_hdbscan 0.3.4 (PLSCAN)": "the docstring's output is wrong: 3 layers on every run",
}


def main() -> int:
    failed = 0
    print("| rival | published | reproduced | |")
    print("|---|---|---|---|")
    for name, check in CHECKS.items():
        # A check whose published number carries a spread says itself whether it lands inside.
        want, got, *within = check()
        if within[0] if within else want == got:
            verdict = "yes"
        elif name in STALE:
            verdict = f"no — {STALE[name]}"
        else:
            verdict = "**NO**"
            failed += 1
        print(f"| {name} | {want} | {got} | {verdict} |", flush=True)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
