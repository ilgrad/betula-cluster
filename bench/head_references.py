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
| `gmm-toeplitz`, `gmm-toeplitz-full` | tslearn `KShape`; scikit-learn diagonal `GaussianMixture` | three AR processes, 30 windows each, length 64 (`bench/toeplitz_ar_mixture.py`) |

Every reference is told the cluster count, as the betula head is. The rows are fixed; the seed
varies only the methods' initialisation. Same subprocess isolation, scoring and CSV schema as
`bench/rivals.py`, so `bench/scoreboard.py` reads both files the same way.

None of the references is a project dependency:

    uv run --no-sync --with scikit-learn --with pandas --with kmedoids --with scikit-fuzzy \\
        --with kmodes --with tslearn python bench/head_references.py [--arms=a,b]

Writes `bench/results_head_refs.csv`.
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
    if dataset == "ar-mixture":
        from toeplitz_ar_mixture import SPECS, ar_windows, make_mixture

        x, y = make_mixture(ar_windows, SPECS, 64, 30, seed=0)
        return x, y, {}
    raise ValueError(dataset)


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
    "betula-gmm-toeplitz": (
        "ar-mixture",
        lambda x, k, s, e: _betula(x, k, s, method="gmm-toeplitz", feature="full"),
    ),
    "betula-gmm-toeplitz-full": (
        "ar-mixture",
        lambda x, k, s, e: _betula(x, k, s, method="gmm-toeplitz-full", feature="full"),
    ),
    "tslearn-kshape": ("ar-mixture", lambda x, k, s, e: _kshape(x, k, s)),
    "sklearn-gmm-diag": ("ar-mixture", lambda x, k, s, e: _sk("gmm-diag", x, k, s)),
}


def one(arm: str, seed: int) -> dict:
    cap_memory()
    dataset, fit = ARMS[arm]
    x, y, extra = load(dataset)
    k = len(np.unique(y))
    peak = Peak()
    t0 = time.perf_counter()
    labels = fit(x, k, seed, extra)
    dt = time.perf_counter() - t0
    return {"time_s": dt, "rss_mb": peak.mb(), **score(y, labels)}


def main(argv: list[str]) -> int:
    import pandas as pd

    # `--arms=a,b` re-runs only those arms and replaces their rows in the existing CSV.
    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    csv = HERE / "results_head_refs.csv"
    kept = pd.read_csv(csv) if only and csv.exists() else pd.DataFrame()
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
