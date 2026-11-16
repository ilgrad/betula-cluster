"""An outside implementation for every betula head that had none to be measured against.

`bench/comprehensive.py` pairs the heads scikit-learn also ships (k-means, GMM, Ward, HDBSCAN).
Four head families had no outside reference at all: k-medoids, k-prototypes on mixed data, fuzzy
c-means, and the Toeplitz-covariance mixture for stationary signals. This harness gives each one the
implementation its users would otherwise reach for, on the same rows and seeds:

| betula head | reference | data |
|---|---|---|
| `kmedoids` | `kmedoids.KMedoids` (FasterPAM, by the algorithm's authors) | `digits` |
| `fuzzy-cmeans` | `skfuzzy.cmeans` | `digits` |
| `kmeans` | scikit-learn `KMeans`, `BisectingKMeans` | `digits` |
| `KPrototypes` | `kmodes.KPrototypes` (Huang init) | covtype, 10 000 rows: 10 numeric columns + wilderness area and soil type as two categorical columns |
| `gmm-toeplitz`, `gmm-toeplitz-full` | tslearn `KShape`; aeon `KASBA`; scikit-learn diagonal `GaussianMixture` | three AR processes, 30 windows each, length 64 (`bench/toeplitz_ar_mixture.py`) |
| `KPrototypes` with no numeric column (k-modes) | `kmodes.KModes` | soybean, the 47 × 35 categorical table of kmodes' own example |

Every reference is told the cluster count, as the betula head is. The rows are fixed; the seed
varies only the methods' initialisation. Same subprocess isolation, scoring and CSV schema as
`bench/rivals.py`, so `bench/scoreboard.py` reads both files the same way.

Where both sides optimise one objective, the `objective` column holds it, computed from the labels
alone with each cluster's best representative: the within-cluster sum of squares (k-means), the sum
of distances to the best medoid (k-medoids), the sum of mismatches to the per-column mode (k-modes).
Lower is better. Two methods that land on the same partition get the same value.

None of the references is a project dependency:

    uv run --no-sync --with scikit-learn --with pandas --with kmedoids --with scikit-fuzzy \\
        --with kmodes --with tslearn --with 'aeon>=1.6' python bench/head_references.py [--arms=a,b]

Unpinned, `aeon` next to pandas 3.1 resolves to `aeon 0.0.0`, an empty placeholder release:
aeon 1.6.0 requires pandas below 3.1.

Writes `bench/results_head_refs.csv`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _worker import Peak, cap_memory
from rivals import score

SEEDS = (0, 1, 2)
TIMEOUT = 1800.0


def load(dataset: str):
    """`(X, y, extra)`: the rows are fixed, whatever the seed."""
    if dataset == "digits":
        from sklearn.datasets import load_digits

        x, y = load_digits(return_X_y=True)
        return np.asarray(x, dtype=np.float64), y, {}
    if dataset == "covtype-mixed":
        from sklearn.datasets import fetch_covtype

        d = fetch_covtype()
        pick = np.random.default_rng(0).choice(len(d.data), 10_000, replace=False)
        raw, y = d.data[pick], d.target[pick] - 1
        num = raw[:, :10]
        num = (num - num.mean(0)) / num.std(0)
        wilderness = raw[:, 10:14].argmax(1)
        soil = raw[:, 14:54].argmax(1)
        x = np.column_stack([num, wilderness, soil]).astype(np.float64)
        return x, y, {"categorical": (10, 11)}
    if dataset == "soybean":
        path = _fetch(
            "soybean.csv",
            "https://raw.githubusercontent.com/nicodv/kmodes/0.12.2/examples/soybean.csv",
        )
        x = np.genfromtxt(path, dtype=int, delimiter=",")[:, :-1].astype(np.float64)
        y = np.genfromtxt(path, dtype=str, delimiter=",", usecols=(35,))
        return x, np.unique(y, return_inverse=True)[1], {"categorical": tuple(range(x.shape[1]))}
    if dataset == "ar-mixture":
        from toeplitz_ar_mixture import SPECS, ar_windows, make_mixture

        x, y = make_mixture(ar_windows, SPECS, 64, 30, seed=0)
        return x, y, {}
    raise ValueError(dataset)


def _fetch(name: str, url: str) -> Path:
    from sklearn.datasets import get_data_home

    path = Path(get_data_home()) / "betula_bench" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            path.write_bytes(r.read())
    return path


def _betula(x, k, seed, **kw):
    import betula_cluster as bc

    return bc.Betula(n_clusters=k, threshold=0.0, seed=seed, **kw).fit_predict(x)


def _kprototypes_betula(x, k, seed, extra):
    import betula_cluster as bc

    return bc.KPrototypes(n_clusters=k, categorical=extra["categorical"], seed=seed).fit_predict(x)


def _kprototypes_kmodes(x, k, seed, extra):
    from kmodes.kprototypes import KPrototypes

    model = KPrototypes(n_clusters=k, init="Huang", n_init=4, random_state=seed)
    return model.fit_predict(x, categorical=list(extra["categorical"]))


def _kmodes(x, k, seed):
    from kmodes.kmodes import KModes

    return KModes(n_clusters=k, random_state=seed).fit_predict(x.astype(int))


def _kasba(x, k, seed):
    from aeon.clustering import KASBA

    return KASBA(n_clusters=k, random_state=seed).fit_predict(x[:, None, :])


def _kmedoids(x, k, seed):
    import kmedoids

    return kmedoids.KMedoids(
        k, metric="euclidean", method="fasterpam", random_state=seed
    ).fit_predict(x)


def _skfuzzy(x, k, seed):
    import skfuzzy

    _, u, *_ = skfuzzy.cmeans(x.T, k, m=2.0, error=1e-5, maxiter=300, seed=seed)
    return u.argmax(0)


def _kshape(x, k, seed):
    from tslearn.clustering import KShape

    return KShape(n_clusters=k, random_state=seed).fit_predict(x[:, :, None])


def _sk(name, x, k, seed):
    from sklearn import cluster, mixture

    if name == "kmeans":
        return cluster.KMeans(k, n_init=4, random_state=seed).fit_predict(x)
    if name == "bisecting":
        return cluster.BisectingKMeans(k, random_state=seed).fit_predict(x)
    return mixture.GaussianMixture(
        k, covariance_type="diag", n_init=4, random_state=seed
    ).fit_predict(x)


# name → (dataset, fit(x, k, seed, extra) → labels)
ARMS = {
    "betula-kmedoids": ("digits", lambda x, k, s, e: _betula(x, k, s, method="kmedoids")),
    "kmedoids (fasterpam)": ("digits", lambda x, k, s, e: _kmedoids(x, k, s)),
    "betula-fuzzy-cmeans": ("digits", lambda x, k, s, e: _betula(x, k, s, method="fuzzy-cmeans")),
    "skfuzzy-cmeans": ("digits", lambda x, k, s, e: _skfuzzy(x, k, s)),
    "betula-kmeans": ("digits", lambda x, k, s, e: _betula(x, k, s, method="kmeans")),
    "sklearn-kmeans": ("digits", lambda x, k, s, e: _sk("kmeans", x, k, s)),
    "sklearn-bisecting-kmeans": ("digits", lambda x, k, s, e: _sk("bisecting", x, k, s)),
    "betula-kprototypes": ("covtype-mixed", _kprototypes_betula),
    "kmodes-kprototypes": ("covtype-mixed", _kprototypes_kmodes),
    "betula-kmodes": ("soybean", _kprototypes_betula),
    "kmodes-kmodes": ("soybean", lambda x, k, s, e: _kmodes(x, k, s)),
    "betula-gmm-toeplitz": (
        "ar-mixture",
        lambda x, k, s, e: _betula(x, k, s, method="gmm-toeplitz", feature="full"),
    ),
    "betula-gmm-toeplitz-full": (
        "ar-mixture",
        lambda x, k, s, e: _betula(x, k, s, method="gmm-toeplitz-full", feature="full"),
    ),
    "tslearn-kshape": ("ar-mixture", lambda x, k, s, e: _kshape(x, k, s)),
    "aeon-kasba": ("ar-mixture", lambda x, k, s, e: _kasba(x, k, s)),
    "sklearn-gmm-diag": ("ar-mixture", lambda x, k, s, e: _sk("gmm-diag", x, k, s)),
}


# The objective an arm optimises, where its reference optimises the same one.
OBJECTIVE = {
    "betula-kmeans": "sse",
    "sklearn-kmeans": "sse",
    "sklearn-bisecting-kmeans": "sse",
    "betula-kmedoids": "medoid",
    "kmedoids (fasterpam)": "medoid",
    "betula-kmodes": "modes",
    "kmodes-kmodes": "modes",
}


def objective(kind: str, x: np.ndarray, labels: np.ndarray) -> float:
    """The partition's cost with each cluster's best representative; lower is better."""
    total = 0.0
    for c in np.unique(labels):
        m = x[labels == c]
        if kind == "sse":
            total += float(((m - m.mean(0)) ** 2).sum())
        elif kind == "medoid":
            d = np.sqrt(np.maximum(((m[:, None, :] - m[None, :, :]) ** 2).sum(-1), 0.0))
            total += float(d.sum(0).min())
        elif kind == "modes":
            mode = np.array([_mode(col) for col in m.T])
            total += float((m != mode).sum())
        else:
            raise ValueError(kind)
    return total


def _mode(col: np.ndarray) -> float:
    """The most frequent value; the smallest one on a tie, as `np.unique` orders them."""
    values, counts = np.unique(col, return_counts=True)
    return values[int(np.argmax(counts))]


def one(arm: str, seed: int) -> dict:
    cap_memory()
    dataset, fit = ARMS[arm]
    x, y, extra = load(dataset)
    k = len(np.unique(y))
    peak = Peak()
    t0 = time.perf_counter()
    labels = np.asarray(fit(x, k, seed, extra))
    dt = time.perf_counter() - t0
    out = {"time_s": dt, "rss_mb": peak.mb(), **score(y, labels)}
    if arm in OBJECTIVE:
        out["objective"] = objective(OBJECTIVE[arm], x, labels)
    return out


def main(argv: list[str]) -> int:
    import pandas as pd

    # `--arms=a,b` re-runs only those arms and replaces their rows in the existing CSV.
    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    csv = HERE / "results_head_refs.csv"
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
    for arm, (dataset, _) in ARMS.items():
        if only and arm not in only:
            continue
        n = len(load(dataset)[0])
        for seed in SEEDS:
            cmd = [sys.executable, str(Path(__file__)), "--one", arm, str(seed)]
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
            row = {"dataset": dataset, "n": n, "arm": arm, "told_k": True, "seed": seed}
            row |= {"dnf": res["error"]} if "error" in res else res
            row["timing_valid"] = ready
            rows.append(row)
            shown = res.get("error") or (
                f"ARI {res['ari']:.3f}  AMI {res['ami']:.3f}  {res['time_s']:7.2f} s"
                f"  {res['rss_mb']:6.0f} MB"
            )
            print(f"{dataset:14s} {arm:28s} seed={seed}  {shown}", flush=True)
            pd.concat([kept, pd.DataFrame(rows)]).to_csv(csv, index=False)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2], int(sys.argv[3]))))
    else:
        sys.exit(main(sys.argv[1:]))
