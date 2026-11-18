"""betula's density and hierarchy heads against the specialists people use for the same job.

`bench/comprehensive.py` pairs `hdbscan` with scikit-learn's HDBSCAN, and
`bench/external_baselines.py` adds `fast_hdbscan` on one blobs fixture. The other methods users
pick when the number of clusters is unknown, or when clusters are not convex, had no row:

| family | rivals |
|---|---|
| density, count chosen | `hdbscan` (scikit-learn-contrib), `fast_hdbscan`, scikit-learn `OPTICS` and `DBSCAN` (at its defaults and at a k-distance `eps`), the `dbscan` package, CLASSIX, DenMune, FINCH |
| hierarchy, count given | genieclust (`Genie`), lumbermark |
| count chosen under a bound | scikit-learn `BayesianGaussianMixture` |
| count chosen, newer | DelTriC (`deltric`), CoreSPECT (`cplearn`) |

against `hdbscan`, `scale-space`, `leiden` on the unit and the rows graph and on the rows graph
with `resolution="stable"` (count chosen), `xmeans`
and the `gmm` and `gmm-full` BIC sweeps (count chosen under a bound) and `dc-median`, `ward`,
`leiden` on the rows graph with `resolution="auto"`, `gmm-full`, `kmeans` (count given). One more
told-count row, `betula-gmm-full-merge`, is a recipe rather than a head: the `gmm-full` BIC fit
merged down to the count on its row posteriors. It is printed, and is not a candidate for betula's
arm on the board. A method that takes an upper bound on
the count gets `K_BOUND` = 20, the same for all, and never the true count; the board files it with
the methods that choose the count. The fixtures
are the main bench's quality fixtures at 30 000 rows (standardised; `blobs`, `aniso`, `varied`,
`moons`, `circles`, `highdim`) and `digits`. Every method runs at its documented defaults, except
that a minimum cluster size, where a method has one, is the same for all: `min_points(n)`, as in
`bench/external_baselines.py`. Scores are over all rows with noise as one more label; the noise
share is recorded beside them.

None of the rivals is a project dependency:

    uv run --no-sync --with scikit-learn --with pandas --with hdbscan --with fast-hdbscan \\
        --with classixclustering --with denmune --with finch-clust --with genieclust \\
        --with lumbermark --with pynndescent --with dbscan --with deltric --with cplearn \\
        python bench/density_rivals.py

Writes `bench/results_density_rivals.csv` (schema of `bench/rivals.py`, `version` included).
`--arms=a,b` re-runs only those arms and replaces their rows in the existing CSV.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _provenance import versions
from _worker import Peak, cap_memory
from rivals import corespect, deltric, score

SEEDS = (0, 1, 2)
K_BOUND = 20
TIMEOUT = 1800.0
N = 30_000
DATASETS = ("blobs", "aniso", "varied", "moons", "circles", "highdim", "digits")


def min_points(n: int) -> int:
    return max(20, n // 400)


def load(dataset: str):
    if dataset == "digits":
        from _fixtures import fetch

        x, y, k = fetch("digits")
        from sklearn.preprocessing import StandardScaler

        return StandardScaler().fit_transform(x), y, k
    from comprehensive import gen_dataset

    return gen_dataset(dataset, N, seed=0)


def _betula(x, k, seed, **kw):
    import betula_cluster as bc

    return bc.Betula(n_clusters=k, threshold=0.0, seed=seed, **kw).fit_predict(x)


def _betula_bounded(x, seed, **kw):
    import betula_cluster as bc

    return bc.Betula(threshold=0.0, seed=seed, **kw).fit_predict(x)


def _gmm_full_merge(x, k, seed):
    """The BIC fit, its components merged to `k` on the row posteriors by the entropy criterion of
    Baudry et al. 2010. A recipe on the public API, not a head. A sweep that stops below `k` leaves
    nothing to merge, and the recipe falls back to the fit told `k`."""
    import betula_cluster as bc

    est = bc.Betula(n_clusters=0, method="gmm-full", threshold=0.0, seed=seed).fit(x)
    p = np.asarray(est.predict_proba(x), dtype=np.float64)
    if p.shape[1] < k:
        return _betula(x, k, seed, method="gmm-full")

    def h(q):
        return -(q * np.log(np.clip(q, 1e-300, None))).sum()

    while p.shape[1] > k:
        pairs = [(a, b) for a in range(p.shape[1]) for b in range(a + 1, p.shape[1])]
        a, b = max(pairs, key=lambda ab: h(p[:, ab[0]]) + h(p[:, ab[1]]) - h(p[:, ab].sum(1)))
        p[:, a] += p[:, b]
        p = np.delete(p, b, axis=1)
    return p.argmax(1)


def _bgmm(x, seed):
    from sklearn.mixture import BayesianGaussianMixture

    return BayesianGaussianMixture(n_components=K_BOUND, random_state=seed).fit_predict(x)


def _dbscan(x):
    from dbscan import DBSCAN

    # The README's call, at its default `eps`; the package supports at most 20 dimensions.
    labels, _ = DBSCAN(x, min_samples=min_points(len(x)))
    return labels


def kdist_params(x) -> tuple[int, float]:
    """`(minPts, eps)` as the DBSCAN literature derives them from the data, since no default fits.

    `minPts = 2·d` (Schubert et al., TODS 2017); `eps` at the elbow of the sorted distances to the
    `minPts`-th neighbour, the point farthest below the chord of the normalised curve (Kneedle's
    difference curve). The rule is ours to state, not a package's default.
    """
    from sklearn.neighbors import NearestNeighbors

    min_pts = 2 * x.shape[1]
    dist = NearestNeighbors(n_neighbors=min_pts).fit(x).kneighbors(x)[0][:, -1]
    curve = np.sort(dist)
    span = curve[-1] - curve[0]
    t = np.linspace(0.0, 1.0, len(curve))
    eps = float(curve[np.argmax(t - (curve - curve[0]) / span)]) if span > 0 else float(curve[0])
    return min_pts, eps


def _dbscan_kdist(x):
    from sklearn.cluster import DBSCAN

    min_pts, eps = kdist_params(x)
    return DBSCAN(eps=eps, min_samples=min_pts).fit_predict(x)


def _classix(x):
    from classix import CLASSIX

    return CLASSIX(minPts=min_points(len(x)), verbose=0).fit(x).labels_


def _denmune(x):
    import contextlib
    import tempfile

    from denmune import DenMune

    # DenMune writes `_temp_2d`, `solution.txt` and `tmp/` into the working directory, and its
    # `show_plots=True` default calls `plt.show()`, which blocks on a window until it is closed.
    # Data that is not 2-D it first embeds with its own t-SNE, and clusters the embedding.
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        labels, _ = DenMune(train_data=x, k_nearest=39).fit_predict(
            show_analyzer=False, show_noise=True, show_plots=False
        )
    return np.asarray(labels["train"])


def _finch(x):
    from finch import FINCH

    # FINCH's default metric is cosine ("recommended for real-world data"); its own toy-data
    # example passes `distance="euclidean"`, which is the geometry of every fixture here.
    c, _, _ = FINCH(x, distance="euclidean", verbose=False)
    return c[:, -1]  # the coarsest partition FINCH proposes


def _genie(x, k):
    import genieclust

    return genieclust.Genie(n_clusters=k).fit_predict(x)


def _lumbermark(x, k):
    import lumbermark

    return lumbermark.Lumbermark(n_clusters=k).fit_predict(x)


# name → (told_k, fit(x, k, seed) → labels)
ARMS = {
    "betula-hdbscan": (
        False,
        lambda x, k, s: _betula(x, k, s, method="hdbscan", min_cluster_size=min_points(len(x))),
    ),
    "betula-scale-space": (False, lambda x, k, s: _betula(x, k, s, method="scale-space")),
    "betula-leiden": (False, lambda x, k, s: _betula_bounded(x, s, method="leiden")),
    "betula-xmeans": (
        False,
        lambda x, k, s: _betula_bounded(x, s, method="xmeans", n_clusters=K_BOUND),
    ),
    # `n_clusters=0` is the BIC sweep, whose default ceiling `auto_k_max` is K_BOUND.
    "betula-gmm-bic": (False, lambda x, k, s: _betula_bounded(x, s, method="gmm", n_clusters=0)),
    "betula-gmm-full-bic": (
        False,
        lambda x, k, s: _betula_bounded(x, s, method="gmm-full", n_clusters=0),
    ),
    "betula-leiden-rows": (
        False,
        lambda x, k, s: _betula_bounded(x, s, method="leiden", leaf_graph="rows"),
    ),
    "betula-leiden-rows-stable": (
        False,
        lambda x, k, s: _betula_bounded(
            x, s, method="leiden", leaf_graph="rows", resolution="stable"
        ),
    ),
    "betula-dc-median": (True, lambda x, k, s: _betula(x, k, s, method="dc-median")),
    "betula-ward": (True, lambda x, k, s: _betula(x, k, s, method="ward")),
    "betula-leiden-rows-auto": (
        True,
        lambda x, k, s: _betula(x, k, s, method="leiden", leaf_graph="rows", resolution="auto"),
    ),
    "betula-gmm-full": (True, lambda x, k, s: _betula(x, k, s, method="gmm-full")),
    "betula-kmeans": (True, lambda x, k, s: _betula(x, k, s, method="kmeans")),
    # A recipe, not a head: `bench/scoreboard.py` keeps it out of the choice of betula's arm.
    "betula-gmm-full-merge": (True, lambda x, k, s: _gmm_full_merge(x, k, s)),
    "hdbscan": (
        False,
        lambda x, k, s: (
            __import__("hdbscan").HDBSCAN(min_cluster_size=min_points(len(x))).fit_predict(x)
        ),
    ),
    "fast-hdbscan": (
        False,
        lambda x, k, s: (
            __import__("fast_hdbscan").HDBSCAN(min_cluster_size=min_points(len(x))).fit_predict(x)
        ),
    ),
    "sklearn-optics": (
        False,
        lambda x, k, s: (
            __import__("sklearn.cluster", fromlist=["OPTICS"])
            .OPTICS(min_samples=min_points(len(x)))
            .fit_predict(x)
        ),
    ),
    "sklearn-dbscan": (
        False,
        lambda x, k, s: (
            __import__("sklearn.cluster", fromlist=["DBSCAN"])
            .DBSCAN(min_samples=min_points(len(x)))
            .fit_predict(x)
        ),
    ),
    "dbscan": (False, lambda x, k, s: _dbscan(x)),
    "sklearn-dbscan-kdist": (False, lambda x, k, s: _dbscan_kdist(x)),
    "sklearn-bgmm": (False, lambda x, k, s: _bgmm(x, s)),
    "deltric": (False, lambda x, k, s: deltric(x, s)),
    "cplearn-corespect": (False, lambda x, k, s: corespect(x, s)),
    "classix": (False, lambda x, k, s: _classix(x)),
    "denmune": (False, lambda x, k, s: _denmune(x)),
    "finch": (False, lambda x, k, s: _finch(x)),
    "genieclust-genie": (True, lambda x, k, s: _genie(x, k)),
    "lumbermark": (True, lambda x, k, s: _lumbermark(x, k)),
}


# The libraries each rival arm runs, written into its rows' `version` column. A betula arm runs
# betula.
RIVAL_DISTS = {
    "hdbscan": ("hdbscan",),
    "fast-hdbscan": ("fast-hdbscan",),
    "sklearn-optics": ("scikit-learn",),
    "sklearn-dbscan": ("scikit-learn",),
    "dbscan": ("dbscan",),
    "sklearn-dbscan-kdist": ("scikit-learn",),
    "sklearn-bgmm": ("scikit-learn",),
    "deltric": ("deltric", "umap-learn"),
    "cplearn-corespect": ("cplearn",),
    "classix": ("classixclustering",),
    "denmune": ("denmune",),
    "finch": ("finch-clust",),
    "genieclust-genie": ("genieclust",),
    "lumbermark": ("lumbermark",),
}
assert set(RIVAL_DISTS) == {a for a in ARMS if not a.startswith("betula")}


# Rivals with a random draw, run on every seed: a random state, or numpy's global generator.
SEEDED = {"sklearn-bgmm", "deltric", "cplearn-corespect"}


def one(arm: str, dataset: str, seed: int) -> dict:
    cap_memory()
    x, y, k = load(dataset)
    x = np.ascontiguousarray(x, dtype=np.float64)
    peak = Peak()
    t0 = time.perf_counter()
    labels = ARMS[arm][1](x, k, seed)
    dt = time.perf_counter() - t0
    dists = ("betula-cluster",) if arm.startswith("betula") else RIVAL_DISTS[arm]
    return {
        "time_s": dt,
        "rss_mb": peak.mb(),
        **score(np.asarray(y), labels),
        "version": versions(*dists),
    }


def main(argv: list[str]) -> int:
    import pandas as pd

    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    csv = HERE / "results_density_rivals.csv"
    # Kept rows pass through as text: pandas 2 parses some floats one ulp off, and a re-run of
    # one arm would then rewrite every other arm's row.
    kept = (
        pd.read_csv(csv, dtype=str, keep_default_na=False)
        if only and csv.exists()
        else pd.DataFrame()
    )
    if only:
        unknown = sorted(set(only) - set(ARMS))
        if unknown:
            raise SystemExit(f"unknown arms: {unknown}")
        kept = kept[~kept["arm"].isin(only)] if len(kept) else kept
    ready = (
        subprocess.run(
            ["bash", str(Path.home() / ".claude/scripts/bench-ready.sh")], capture_output=True
        ).returncode
        == 0
    )
    rows = []
    for dataset in DATASETS:
        n = len(load(dataset)[0])
        for arm, (told_k, _) in ARMS.items():
            if only and arm not in only:
                continue
            # A deterministic rival gives the same labels on every seed; one run says it all.
            for seed in SEEDS if arm.startswith("betula") or arm in SEEDED else SEEDS[:1]:
                cmd = [sys.executable, str(Path(__file__)), "--one", arm, dataset, str(seed)]
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
                    tail = out.stdout.strip().splitlines()
                    res = (
                        json.loads(tail[-1])
                        if out.returncode == 0 and tail
                        else {"error": (out.stderr.strip().splitlines() or ["no output"])[-1][:160]}
                    )
                except subprocess.TimeoutExpired:
                    res = {"error": f"timeout {TIMEOUT:.0f} s"}
                timed_out = res.get("error", "").startswith("timeout")
                row = {"dataset": dataset, "n": n, "arm": arm, "told_k": told_k, "seed": seed}
                row |= {"dnf": res["error"]} if "error" in res else res
                row["timing_valid"] = ready
                rows.append(row)
                shown = res.get("error") or (
                    f"ARI {res['ari']:.3f}  k={res['n_clusters']:<4} noise {res['noise_share']:.2f}"
                    f"  {res['time_s']:7.2f} s  {res['rss_mb']:6.0f} MB"
                )
                print(f"{dataset:8s} {arm:20s} seed={seed}  {shown}", flush=True)
                pd.concat([kept, pd.DataFrame(rows)]).to_csv(csv, index=False)
                if timed_out:
                    break  # the other seeds would time out too; the DNF row stands for them
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2], sys.argv[3], int(sys.argv[4]))))
    else:
        sys.exit(main(sys.argv[1:]))
