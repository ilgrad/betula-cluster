"""GPU rivals on this machine's RTX 3080 Laptop (8 GB): cuML's k-means, UMAP + k-means and
UMAP + HDBSCAN on the rows and seeds of `bench/rivals.py`.

A different hardware budget: a GPU arm is never paired with a CPU cell and never enters
`bench/scoreboard.py` (which reads named tables only). Quality is comparable across the two tables,
since the rows, the seeds and the scoring are `bench/rivals.py`'s own; wall time is recorded with
`timing_valid` from `bench-ready.sh`, and device memory is the peak `nvidia-smi` reading above the
reading before the arm starts, every 20 ms, CUDA context included; host RSS as `rivals.py` records
it.

cuML is not a dependency of anything here. It lives in a scratch environment:

    uv venv --python 3.14 venv-gpu
    uv pip install --python venv-gpu/bin/python 'cuml-cu12==26.10.*' scikit-learn pandas
    venv-gpu/bin/python bench/gpu_rivals.py [--arms=a,b]

Before any row (or alone, with `--check`) it reproduces the `cuml.KMeans` docstring example
(labels [0, 0, 1, 1], centres (1, 1.5) and (3.5, 2.5)). Writes `bench/results_gpu_rivals.csv`
(schema of `rivals.py`, plus `device_mb`).
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

from _worker import Peak
from rivals import SEEDS, TIMEOUT, load, score

PLAN = [("mnist", 20_000), ("news-minilm", 17_901), ("mnist", 70_000)]


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


# name → (told_k, fit(x, k, seed, dataset) → labels); names mirror the CPU table's with a prefix.
ARMS = {
    "cuml-kmeans": (True, lambda x, k, s, d: _kmeans(x, k, s)),
    "cuml-umap+kmeans": (True, lambda x, k, s, d: _kmeans(_umap(x, s, d), k, s)),
    "cuml-umap+hdbscan": (False, lambda x, k, s, d: _hdbscan(_umap(x, s, d), d)),
}


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


def check() -> None:
    from cuml import KMeans

    a = np.asarray([[1.0, 1.0], [1.0, 2.0], [3.0, 2.0], [4.0, 3.0]], dtype=np.float32)
    km = KMeans(n_clusters=2, n_init="auto", random_state=1).fit(a)
    labels = _as_host(km.labels_).tolist()
    centres = sorted(map(tuple, np.round(_as_host(km.cluster_centers_), 6).tolist()))
    assert labels in ([0, 0, 1, 1], [1, 1, 0, 0]), labels
    assert centres == [(1.0, 1.5), (3.5, 2.5)], centres
    print("cuml.KMeans docstring example reproduced:", labels, centres)


def one(arm: str, dataset: str, n: int, seed: int) -> dict:
    # No `cap_memory()`: CUDA reserves far more address space than the CPU harness's RLIMIT_AS.
    x, y = load(dataset, n, seed)
    k = len(np.unique(y))
    device, host = DevicePeak(), Peak()
    t0 = time.perf_counter()
    labels = _as_host(ARMS[arm][1](x, k, seed, dataset))
    dt = time.perf_counter() - t0
    return {"time_s": dt, "rss_mb": host.mb(), "device_mb": device.mb(), **score(y, labels)}


def main(argv: list[str]) -> int:
    import pandas as pd

    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    csv = HERE / "results_gpu_rivals.csv"
    kept = pd.read_csv(csv) if only and csv.exists() else pd.DataFrame()
    if only:
        kept = kept[~kept["arm"].isin(only)] if len(kept) else kept
    check()
    ready = (
        subprocess.run(
            ["bash", str(Path.home() / ".claude/scripts/bench-ready.sh")], capture_output=True
        ).returncode
        == 0
    )
    rows = []
    for dataset, n in PLAN:
        for arm, (told_k, _) in ARMS.items():
            if only and arm not in only:
                continue
            for seed in SEEDS:
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
                    res = (
                        json.loads(tail[-1])
                        if out.returncode == 0 and tail
                        else {"error": (out.stderr.strip().splitlines() or ["no output"])[-1][:160]}
                    )
                except subprocess.TimeoutExpired:
                    res = {"error": f"timeout {TIMEOUT:.0f} s"}
                row = {"dataset": dataset, "n": n, "arm": arm, "told_k": told_k, "seed": seed}
                row |= {"dnf": res["error"]} if "error" in res else res
                row["timing_valid"] = ready
                rows.append(row)
                shown = res.get("error") or (
                    f"ARI {res['ari']:.3f}  k={res['n_clusters']:<4} noise {res['noise_share']:.2f}"
                    f"  {res['rss_mb']:6.0f} MB RSS  {res['device_mb']:6.0f} MB on device"
                )
                print(f"{dataset:12s} n={n:<6} {arm:20s} seed={seed}  {shown}", flush=True)
                pd.concat([kept, pd.DataFrame(rows)]).to_csv(csv, index=False)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        a = sys.argv[2:]
        print(json.dumps(one(a[0], a[1], int(a[2]), int(a[3]))))
    elif len(sys.argv) > 1 and sys.argv[1] == "--check":
        check()
    else:
        sys.exit(main(sys.argv[1:]))
