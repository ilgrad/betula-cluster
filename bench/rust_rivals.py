"""betula's Rust API against the Rust clustering crates: linfa-clustering, clump and `hdbscan`.

The crates.io audience calls betula from Rust, where scikit-learn is not the comparison. The rival
crates live in `bench/rust_rivals/`, a crate of its own outside betula-cluster's manifest, so none of
them enters its dependency tree. Each run is one process: this script writes the rows to a CSV, the
binary prints one label per row, and the score and the peak RSS of that process are recorded here.

| contest | arms |
|---|---|
| told k | betula `kmeans`, `gmm` (diagonal) and `gmm-full` through `Model::fit`, linfa `KMeans` and `GaussianMixtureModel` (full covariance), clump `Kmeans` |
| k not given | the `hdbscan` crate, clump `Hdbscan` |

Fixtures, minimum cluster size and scoring are `bench/density_rivals.py`'s, so betula's own
`hdbscan` head is read from that table: its Rust API is not a one-call function. Every crate runs at
its documented defaults; betula as `docs/USAGE.md`'s Rust section builds it. Seeded arms run on
seeds 0, 1, 2; the deterministic ones once. Quality and peak RSS only.

    uv run --no-sync --with scikit-learn --with pandas python bench/rust_rivals.py [--arms=a,b]

Writes `bench/results_rust_rivals.csv` (schema of `bench/rivals.py`).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from density_rivals import DATASETS, N, load, min_points
from rivals import SEEDS, TIMEOUT, score

CRATE = HERE / "rust_rivals"
TARGET = HERE.parent / "target" / "rust-rivals"
BINARY = TARGET / "release" / "rust-rivals"
MEM_CAP = 10 * 1024**3

# name → (told_k, seeded)
ARMS = {
    "betula-kmeans": (True, True),
    "betula-gmm": (True, True),
    "betula-gmm-full": (True, True),
    "linfa-kmeans": (True, True),
    "linfa-gmm": (True, True),
    "clump-kmeans": (True, True),
    "hdbscan-crate": (False, False),
    "clump-hdbscan": (False, False),
}


def build() -> None:
    manifest = str(CRATE / "Cargo.toml")
    cmd = ["cargo", "build", "--release", "--manifest-path", manifest, "--target-dir", str(TARGET)]
    subprocess.run(cmd, check=True)


def _cap():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, MEM_CAP))


def run(arm: str, rows: Path, k: int, seed: int, mcs: int) -> dict:
    """One process; its peak RSS is the `VmHWM` it reports, not `wait4`'s `ru_maxrss`, which
    keeps this parent's high-water mark from before the fork."""
    with tempfile.TemporaryFile() as out:
        child = subprocess.Popen(
            [str(BINARY), arm, str(rows), str(k), str(seed), str(mcs)],
            stdout=out,
            stderr=subprocess.PIPE,
            preexec_fn=_cap,
        )
        start = time.perf_counter()
        while True:
            pid, status, _ = os.wait4(child.pid, os.WNOHANG)
            if pid:
                break
            if time.perf_counter() - start > TIMEOUT:
                child.kill()
                os.wait4(child.pid, 0)
                return {"error": f"timeout {TIMEOUT:.0f} s"}
            time.sleep(0.05)
        elapsed = time.perf_counter() - start
        if os.waitstatus_to_exitcode(status) != 0:
            err = child.stderr.read().decode().strip().splitlines() if child.stderr else []
            return {"error": (err or [f"exit {os.waitstatus_to_exitcode(status)}"])[-1][:160]}
        out.seek(0)
        labels = np.loadtxt(out, dtype=np.int64)
    assert child.stderr is not None
    hwm = child.stderr.read().decode().split("VmHWM")[-1].split()
    return {"time_s": elapsed, "rss_mb": int(hwm[0]) / 1024, "labels": labels}


def main(argv: list[str]) -> int:
    import pandas as pd

    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    build()
    csv = HERE / "results_rust_rivals.csv"
    kept = pd.read_csv(csv) if only and csv.exists() else pd.DataFrame()
    if only and len(kept):
        kept = kept[~kept["arm"].isin(only)]
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for dataset in DATASETS:
            x, y = load(dataset)[:2]
            path = Path(scratch) / f"{dataset}.csv"
            np.savetxt(path, x, delimiter=",", fmt="%.17g")
            k, mcs = len(np.unique(y)), min_points(len(x))
            for arm, (told_k, seeded) in ARMS.items():
                if only and arm not in only:
                    continue
                for seed in SEEDS if seeded else SEEDS[:1]:
                    res = run(arm, path, k, seed, mcs)
                    row = {"dataset": dataset, "n": len(x), "arm": arm, "told_k": told_k}
                    row |= {"seed": seed, "timing_valid": False}
                    if "error" in res:
                        row["dnf"] = res["error"]
                        shown = res["error"]
                    else:
                        s = score(y, res["labels"])
                        row |= {"time_s": res["time_s"], "rss_mb": res["rss_mb"], **s}
                        shown = (
                            f"ARI {s['ari']:.3f}  k={s['n_clusters']:<4} noise "
                            f"{s['noise_share']:.2f}  {res['rss_mb']:7.0f} MB"
                        )
                    rows.append(row)
                    print(f"{dataset:8s} {arm:14s} seed={seed}  {shown}", flush=True)
                    pd.concat([kept, pd.DataFrame(rows)]).to_csv(csv, index=False)
    return 0


if __name__ == "__main__":
    assert N == 30_000
    sys.exit(main(sys.argv[1:]))
