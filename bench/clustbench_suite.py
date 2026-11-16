"""betula's told-k heads on Gagolewski's clustering benchmark suite (v1.2.0), against the published
label vectors of every method in `clustering-results-v1/original`, with the suite's own metric:
normalised clustering accuracy (NCA), the maximum over a dataset's alternative reference partitions,
reference noise points ignored.

Neither the data nor the published results are redistributed here (both are AGPL-3.0). Clone them:

    git clone --depth 1 https://github.com/gagolews/clustering-data-v1
    git clone --depth 1 https://github.com/gagolews/clustering-results-v1

    uv run --no-sync --with clustering-benchmarks python bench/clustbench_suite.py run <root> \\
        > scores.jsonl
    uv run --no-sync --with pandas python bench/clustbench_suite.py summary scores.jsonl

`run` scores every battery; `summary` prints one configuration per method over the whole suite and
writes betula's own rows to `bench/results_clustbench.csv`, each with the betula version that scored
it (`version`; rows written before the column existed leave it empty).

Data as the framework prescribes: `clustbench.preprocess_data` under seed 123. Each head is told
every K a dataset's reference partitions name, runs once at seed 0, and a head that returns another
count than K is absent for that K, as the framework treats the published methods.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from _provenance import versions

BATTERIES = ("wut", "fcps", "graves", "uci", "mnist", "sipu", "other")
HEADS = {
    "betula-kmeans": {"method": "kmeans"},
    "betula-gmm": {"method": "gmm"},
    "betula-ward": {"method": "ward"},
    "betula-spectral": {"method": "spectral"},
    "betula-dc-median": {"method": "dc-median"},
    "betula-leiden-auto": {"method": "leiden", "resolution": "auto"},
}


def nca(y_true, y_pred):
    # genieclust 1.3.0 raises unless both vectors hold the same number of labels, which a
    # K-partition restricted to the reference's non-noise points need not. 1.1.6 and 1.2.0 score it
    # with the missing clusters as zero columns; a square confusion matrix gives 1.3.0 that value.
    import genieclust

    k = int(max(y_true.max(), y_pred.max()))
    c = np.zeros((k, k))
    np.add.at(c, (y_true - 1, y_pred - 1), 1)
    return genieclust.compare_partitions.normalized_clustering_accuracy(c)


def run(root: Path) -> None:
    import betula_cluster as bc
    import clustbench

    data, results = root / "clustering-data-v1", root / "clustering-results-v1" / "original"

    def emit(**rec):
        print(json.dumps(rec), flush=True)

    for battery in BATTERIES:
        for name in clustbench.get_dataset_names(battery, path=data):
            b = clustbench.load_dataset(battery, name, path=data, preprocess=False)
            np.random.seed(123)
            x = np.ascontiguousarray(clustbench.preprocess_data(b.data), dtype=np.float64)
            ks = sorted({int(k) for k in b.n_clusters if k >= 2})
            if not ks:
                continue
            published = clustbench.load_results("*", battery, name, ks, path=results)
            for method, res in published.items():
                s = clustbench.get_score(b.labels, res, metric=nca, warn_if_missing=False)
                if np.isfinite(s):
                    emit(battery=battery, dataset=name, method=method, score=float(s))
            for head, kw in HEADS.items():
                res = {}
                for k in ks:
                    lab = np.asarray(
                        bc.Betula(n_clusters=k, threshold=0.0, seed=0, **kw).fit_predict(x)
                    )
                    found = len(np.unique(lab[lab >= 0]))
                    if found != k or (lab < 0).any():
                        emit(battery=battery, dataset=name, method=head, k=k, mismatch=found)
                        continue
                    res[k] = lab + 1  # clustbench labels clusters 1..K
                if res:
                    s = clustbench.get_score(b.labels, res, metric=nca, warn_if_missing=False)
                    emit(
                        battery=battery,
                        dataset=name,
                        method=head,
                        score=float(s),
                        version=versions("betula-cluster"),
                    )


def summary(paths: list[str]) -> None:
    import pandas as pd

    rows = [json.loads(line) for path in paths for line in Path(path).read_text().splitlines()]
    df = pd.DataFrame([r for r in rows if "score" in r])
    df = df.drop_duplicates(["battery", "dataset", "method"], keep="last")
    ours = df[df["method"].str.startswith("betula")].sort_values(["battery", "dataset", "method"])
    ours.rename(columns={"score": "nca"}).to_csv(
        Path(__file__).with_name("results_clustbench.csv"), index=False, float_format="%.6f"
    )
    missed = pd.DataFrame([r for r in rows if "mismatch" in r])

    # One configuration per method for the whole suite, as one betula head is: the mean over the
    # batteries of the battery mean, for every method scored in all of them.
    by_battery = df.groupby(["method", "battery"])["score"].mean().unstack().dropna()
    suite = by_battery.mean(axis=1).sort_values(ascending=False)
    print(f"{len(suite)} methods scored in every battery; the first 12 and every betula head:")
    for i, (m, v) in enumerate(suite.items(), 1):
        if i <= 12 or m.startswith("betula"):
            n_missed = 0 if missed.empty else int((missed["method"] == m).sum())
            note = f"  (another count than K on {n_missed} dataset-K pairs)" if n_missed else ""
            print(f"{i:4d}  {m:24s} {v:.3f}{note}")

    head = next(m for m in suite.index if m.startswith("betula"))
    wide = df.pivot_table(index=["battery", "dataset"], columns="method", values="score")
    print(f"\n{head}, paired on every dataset both scored (a gap under 0.02 is not counted):")
    named = ["Genie_G0.3", "ITM", "sklearn_gm", "sklearn_kmeans", "fastcluster_ward"]
    for m in [m for m in suite.index[:12] if not m.startswith("betula")] + named:
        both = wide[[head, m]].dropna()
        ahead = int((both[head] > both[m] + 0.02).sum())
        behind = int((both[m] > both[head] + 0.02).sum())
        print(
            f"  {m:24s} {both[m].mean():.3f} against {both[head].mean():.3f} on {len(both)}:"
            f" ahead on {ahead}, behind on {behind}"
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run(Path(sys.argv[2]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "summary":
        summary(sys.argv[2:])
    else:
        sys.exit(__doc__)
