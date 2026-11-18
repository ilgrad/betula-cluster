"""GPU rivals on this machine's RTX 3080 Laptop (8 GB): cuML's k-means, UMAP + k-means and
UMAP + HDBSCAN on the rows and seeds of `bench/rivals.py`; cuML's HDBSCAN on the raw rows; FAISS's
and fastkmeans's k-means; and cuML's HDBSCAN and DBSCAN on the density fixtures of
`bench/density_rivals.py`.

A different hardware budget: a GPU arm is never paired with a CPU cell and never enters
`bench/scoreboard.py` (which reads named tables only). Quality is comparable across the two tables,
since the rows, the seeds and the scoring are the CPU harnesses' own; wall time is recorded with
`timing_valid` from `bench-ready.sh`, and device memory is the peak `nvidia-smi` reading above the
reading before the arm starts, every 20 ms, CUDA context included; host RSS as `rivals.py` records
it.

| arm | where | settings |
|---|---|---|
| `cuml-kmeans`, `cuml-umap+kmeans`, `cuml-umap+hdbscan` | MNIST 20 000 and 70 000, news | the CPU table's UMAP and HDBSCAN settings |
| `cuml-hdbscan` | the same boards, and the density fixtures | the raw rows; cuML's defaults on the boards, the shared minimum cluster size on the fixtures, as the CPU `hdbscan` arm |
| `cuml-dbscan-kdist` | the density fixtures | `density_rivals.kdist_params`, the parameters of the CPU `sklearn-dbscan-kdist` arm; on 784 pixels its `minPts = 2·d` would be 1 568, so it has no board row |
| `faiss-gpu-kmeans` | the boards | the CPU FAISS arm of `bench/external_baselines.py` with `gpu=True` |
| `fastkmeans`, `fastkmeans (use_triton=False)` | the boards | its defaults, which take its Triton kernel on this card; and its PyTorch path |

None of these is a dependency of anything here. cuML lives in one scratch environment, and FAISS and
fastkmeans (on PyTorch) in another, since the two stacks pin different CUDA libraries:

    uv venv --python 3.14 venv-gpu
    uv pip install --python venv-gpu/bin/python 'cuml-cu12==26.10.*' scikit-learn pandas
    venv-gpu/bin/python bench/gpu_rivals.py --arms=cuml-kmeans,cuml-hdbscan,...
    uv venv --python 3.14 venv-gpu-torch
    uv pip install --python venv-gpu-torch/bin/python torch fastkmeans faiss-gpu scikit-learn pandas
    venv-gpu-torch/bin/python bench/gpu_rivals.py --arms=faiss-gpu-kmeans,fastkmeans,...

Before any row (or alone, with `--check`) it reproduces the documented examples of the libraries the
chosen arms run: the `cuml.KMeans` docstring (labels [0, 0, 1, 1], centres (1, 1.5) and (3.5, 2.5))
and the `cuml.DBSCAN` docstring (labels [0, 1, 2]). cuML's HDBSCAN docstring has no example; it runs
the `hdbscan` documentation's blobs (3 clusters) instead. FAISS and fastkmeans publish no numeric
output. Writes `bench/results_gpu_rivals.csv` (schema of `rivals.py`, plus `device_mb`); `version`
names the release of the library the arm ran, and rows written before the column existed leave it
empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _provenance import versions
from _worker import Peak
from rivals import SEEDS, TIMEOUT, score
from rivals import load as load_board

BOARDS = [("mnist", 20_000), ("news-minilm", 17_901), ("mnist", 70_000)]
DENSITY = ("blobs", "aniso", "varied", "moons", "circles", "highdim", "digits")


def load(dataset, n, seed):
    if dataset in DENSITY:
        from density_rivals import load as load_fixture

        x, y = load_fixture(dataset)[:2]
        return np.ascontiguousarray(x, dtype=np.float64), np.asarray(y)
    return load_board(dataset, n, seed)


def _umap_kw(dataset):
    # The CPU table's UMAP settings (`rivals._umap`): BERTopic's for text, the UMAP docs' for MNIST.
    if dataset == "news-minilm":
        return {"n_neighbors": 15, "min_dist": 0.0, "n_components": 5, "metric": "cosine"}
    return {"n_neighbors": 30, "min_dist": 0.0, "n_components": 2}


def _kmeans(z, k, seed):
    from cuml import KMeans

    return KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(z)


def _umap(x, seed, dataset):
    from cuml import UMAP

    return UMAP(random_state=seed, **_umap_kw(dataset)).fit_transform(x)


def _hdbscan(z, dataset):
    from cuml.cluster.hdbscan import HDBSCAN

    kw = (
        {"min_cluster_size": 10}
        if dataset == "news-minilm"
        else {"min_samples": 10, "min_cluster_size": 500}
    )
    return HDBSCAN(**kw).fit_predict(z)


def _hdbscan_raw(x, dataset):
    from cuml.cluster.hdbscan import HDBSCAN
    from density_rivals import min_points

    kw = {"min_cluster_size": min_points(len(x))} if dataset in DENSITY else {}
    return HDBSCAN(**kw).fit_predict(x)


def _dbscan_kdist(x):
    from cuml import DBSCAN
    from density_rivals import kdist_params

    min_pts, eps = kdist_params(x)
    return DBSCAN(eps=eps, min_samples=min_pts).fit_predict(x)


def _faiss_kmeans(x, k, seed):
    import faiss

    # `bench/external_baselines.py`'s FAISS arm at its defaults, on the device.
    x = np.ascontiguousarray(x, dtype=np.float32)
    km = faiss.Kmeans(x.shape[1], k, niter=25, nredo=1, seed=seed, gpu=True)
    km.train(x)
    return km.index.search(x, 1)[1].ravel()


def _fastkmeans(x, k, seed, **kw):
    from fastkmeans import FastKMeans

    x = np.ascontiguousarray(x, dtype=np.float32)
    return FastKMeans(d=x.shape[1], k=k, seed=seed, **kw).fit_predict(x)


# name → (told_k, where, fit(x, k, seed, dataset) → labels, the libraries it runs). Names mirror the
# CPU tables' with a prefix. `where` is "boards", "density" or "both".
ARMS = {
    "cuml-kmeans": (True, "boards", lambda x, k, s, d: _kmeans(x, k, s), ("cuml-cu12",)),
    "cuml-umap+kmeans": (
        True,
        "boards",
        lambda x, k, s, d: _kmeans(_umap(x, s, d), k, s),
        ("cuml-cu12",),
    ),
    "cuml-umap+hdbscan": (
        False,
        "boards",
        lambda x, k, s, d: _hdbscan(_umap(x, s, d), d),
        ("cuml-cu12",),
    ),
    "cuml-hdbscan": (False, "both", lambda x, k, s, d: _hdbscan_raw(x, d), ("cuml-cu12",)),
    "cuml-dbscan-kdist": (False, "density", lambda x, k, s, d: _dbscan_kdist(x), ("cuml-cu12",)),
    "faiss-gpu-kmeans": (True, "boards", lambda x, k, s, d: _faiss_kmeans(x, k, s), ("faiss-gpu",)),
    "fastkmeans": (
        True,
        "boards",
        lambda x, k, s, d: _fastkmeans(x, k, s),
        ("fastkmeans", "torch"),
    ),
    "fastkmeans (use_triton=False)": (
        True,
        "boards",
        lambda x, k, s, d: _fastkmeans(x, k, s, use_triton=False),
        ("fastkmeans", "torch"),
    ),
}

# Arms whose result does not move with the seed: one run says it all.
DETERMINISTIC = {"cuml-hdbscan", "cuml-dbscan-kdist"}


class DevicePeak:
    """Peak device memory above the reading taken before the arm initialises CUDA, so the CUDA
    context counts, as a CPU arm's import stack counts in its RSS. `nvidia-smi` streams a reading
    every 20 ms; an allocation shorter than that can be missed."""

    def __init__(self):
        self._smi = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-lms",
                "20",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._smi.stdout is not None
        self.base = self.peak = float(self._smi.stdout.readline())
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        assert self._smi.stdout is not None
        for line in self._smi.stdout:
            self.peak = max(self.peak, float(line))

    def mb(self) -> float:
        time.sleep(0.1)  # one more reading after the arm returns
        self._smi.terminate()
        self._thread.join()
        return self.peak - self.base


def _as_host(labels):
    return np.asarray(labels.get() if hasattr(labels, "get") else labels)


def check(libraries) -> None:
    """The documented examples of the libraries in `libraries` (distribution names)."""
    if "cuml-cu12" not in libraries:
        return
    from cuml import DBSCAN, KMeans
    from cuml.cluster.hdbscan import HDBSCAN
    from sklearn.datasets import make_blobs

    a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]], dtype=np.float32)
    km = KMeans(n_clusters=2, n_init="auto", random_state=1).fit(a)
    labels = _as_host(km.labels_).tolist()
    centres = sorted(map(tuple, np.round(_as_host(km.cluster_centers_), 6).tolist()))
    assert labels in ([0, 0, 1, 1], [1, 1, 0, 0]), labels
    assert centres == [(1.0, 1.5), (3.5, 2.5)], centres
    print("cuml.KMeans docstring example reproduced:", labels, centres)
    # The docstring builds these three rows as a cuDF frame of three columns.
    a = np.asarray([[1.0, 4.0, 4.0], [2.0, 2.0, 2.0], [5.0, 1.0, 1.0]], dtype=np.float32)
    labels = _as_host(DBSCAN(eps=1.0, min_samples=1).fit(a).labels_).tolist()
    assert labels == [0, 1, 2], labels
    print("cuml.DBSCAN docstring example reproduced:", labels)
    x, _ = make_blobs(n_samples=2000, n_features=10, random_state=0)
    clusters = int(_as_host(HDBSCAN().fit(x).labels_).max()) + 1
    assert clusters == 3, clusters
    print("cuml HDBSCAN on the hdbscan documentation's blobs: 3 clusters, as published")


def one(arm: str, dataset: str, n: int, seed: int) -> dict:
    # No `cap_memory()`: CUDA reserves far more address space than the CPU harness's RLIMIT_AS.
    x, y = load(dataset, n, seed)
    k = len(np.unique(y))
    device, host = DevicePeak(), Peak()
    t0 = time.perf_counter()
    labels = _as_host(ARMS[arm][2](x, k, seed, dataset))
    dt = time.perf_counter() - t0
    return {
        "time_s": dt,
        "rss_mb": host.mb(),
        "device_mb": device.mb(),
        **score(y, labels),
        "version": versions(*ARMS[arm][3]),
        "n": len(x),
    }


def main(argv: list[str]) -> int:
    import pandas as pd
    from density_rivals import N

    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    unknown = sorted(set(only or ()) - set(ARMS))
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    csv = HERE / "results_gpu_rivals.csv"
    # Kept rows pass through as text: pandas 2 parses some floats one ulp off, and a re-run of
    # one arm would then rewrite every other arm's row.
    kept = (
        pd.read_csv(csv, dtype=str, keep_default_na=False)
        if only and csv.exists()
        else pd.DataFrame()
    )
    if only:
        kept = kept[~kept["arm"].isin(only)] if len(kept) else kept
    check({lib for arm in only or ARMS for lib in ARMS[arm][3]})
    ready = (
        subprocess.run(
            ["bash", str(Path.home() / ".claude/scripts/bench-ready.sh")], capture_output=True
        ).returncode
        == 0
    )
    plan = [(d, n, "boards") for d, n in BOARDS] + [(d, N, "density") for d in DENSITY]
    rows = []
    for dataset, n, kind in plan:
        for arm, (told_k, where, _, _) in ARMS.items():
            if (only and arm not in only) or where not in (kind, "both"):
                continue
            for seed in SEEDS[:1] if arm in DETERMINISTIC else SEEDS:
                cmd = [
                    sys.executable,
                    str(Path(__file__)),
                    "--one",
                    arm,
                    dataset,
                    str(n),
                    str(seed),
                ]
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
                    tail = out.stdout.strip().splitlines()
                    # CUDA appends a debugging hint after the error; the reason is the last
                    # line that names an error.
                    err = out.stderr.strip().splitlines() or ["no output"]
                    named = [line for line in err if "Error: " in line]
                    res = (
                        json.loads(tail[-1])
                        if out.returncode == 0 and tail
                        else {"error": (named or err)[-1][:160]}
                    )
                except subprocess.TimeoutExpired:
                    res = {"error": f"timeout {TIMEOUT:.0f} s"}
                # A density fixture's row count is the fixture's own (`digits` has 1 797).
                rows_n = res.pop("n", None) or n
                row = {"dataset": dataset, "n": rows_n, "arm": arm, "told_k": told_k, "seed": seed}
                row |= {"dnf": res["error"]} if "error" in res else res
                row["timing_valid"] = ready
                rows.append(row)
                shown = res.get("error") or (
                    f"ARI {res['ari']:.3f}  k={res['n_clusters']:<4} noise {res['noise_share']:.2f}"
                    f"  {res['rss_mb']:6.0f} MB RSS  {res['device_mb']:6.0f} MB on device"
                )
                print(f"{dataset:12s} n={n:<6} {arm:30s} seed={seed}  {shown}", flush=True)
                pd.concat([kept, pd.DataFrame(rows)]).to_csv(csv, index=False)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        a = sys.argv[2:]
        print(json.dumps(one(a[0], a[1], int(a[2]), int(a[3]))))
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        import importlib.util

        check({"cuml-cu12"} if importlib.util.find_spec("cuml") else set())
    else:
        sys.exit(main(sys.argv[1:]))
