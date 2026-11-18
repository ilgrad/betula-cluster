"""betula's Rust API against the Rust clustering crates: linfa-clustering, clump, `hdbscan`,
anofox-ml-cluster, scirs2-cluster and fuel.

The crates.io audience calls betula from Rust, where scikit-learn is not the comparison. The rival
crates live in `bench/rust_rivals/`, a crate of its own outside betula-cluster's manifest, so none of
them enters its dependency tree. Each run is one process: this script writes the rows to a CSV, the
binary prints one label per row, and the score and the peak RSS of that process are recorded here.

| contest | arms |
|---|---|
| told k | betula `kmeans`, `gmm` (diagonal) and `gmm-full` through `Model::fit`, linfa `KMeans` and `GaussianMixtureModel` (full covariance), clump `Kmeans`, fuel `kmeans`, the `Birch` of anofox-ml-cluster and of scirs2-cluster |
| k not given | the `hdbscan` crate, clump `Hdbscan`, fuel `hdbscan`, linfa `Dbscan` and `Optics`, clump `Dbscan` |

fuel (Schubert; AGPL-3.0) is not on crates.io: the bench crate takes it from its git mirror, and it
never enters betula-cluster's dependency tree.

Fixtures, minimum cluster size and scoring are `bench/density_rivals.py`'s, so betula's own
`hdbscan` head is read from that table: its Rust API is not a one-call function. Every crate runs at
its documented defaults; betula as `docs/USAGE.md`'s Rust section builds it, on the leaf feature the
Python wrapper builds for each head (`Diagonal` for the two mixtures, which it refuses on
`Spherical`). Two parameters have no default that fits, and get a rule instead. DBSCAN's radius is
absolute (linfa's default is 1e-4, clump's has none), so both `Dbscan` arms take
`density_rivals.kdist_params`, the k-distance `minPts` and `eps` of the `sklearn-dbscan-kdist` arm,
and the three implementations meet on the same parameters. linfa's `Optics` returns the augmented
ordering, not labels: it runs at the shared minimum cluster size, and its labels are scikit-learn's
`cluster_optics_xi` at the defaults scikit-learn's `OPTICS` extracts with (`xi = 0.05`, the same
`min_samples`), without the predecessor correction, since linfa reports no predecessors.

Seeded arms run on seeds 0, 1, 2; the deterministic ones once. Quality, peak RSS, and the heap
allocations of the fit (`allocs`, `alloc_mb`: every call into the Rust allocator from the start to
the end of the fit, and the bytes it asked for; what a C library allocates itself is not seen).

    uv run --no-sync --with scikit-learn --with pandas python bench/rust_rivals.py [--arms=a,b]
    uv run --no-sync --with scikit-learn --with pandas python bench/rust_rivals.py --check

`--check`, also run before any row, reproduces documented examples: scikit-learn's Birch docstring
through anofox's `Birch`, which documents itself as that estimator's mirror, and scirs2's `birch`
doc example. linfa and fuel publish no numeric output, so their arms run the examples of other
packages' documentation instead: scikit-learn's DBSCAN example through linfa's and clump's `Dbscan`,
the scikit-learn OPTICS docstring through linfa's `Optics` and the extraction above, and the blobs
of the `hdbscan` documentation through fuel's `hdbscan`.

Writes `bench/results_rust_rivals.csv` (schema of `bench/rivals.py`, plus `allocs` and `alloc_mb`);
`version` names the crate each arm ran, as the build's `Cargo.lock` resolved it (a crate from git
with its commit), and rows written before the column existed leave it empty.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from density_rivals import DATASETS, N, kdist_params, load, min_points
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
    "fuel-kmeans": (True, True),
    "anofox-birch": (True, False),
    # The documented `birch()` returns each row's leaf entry as its label, not the cluster the
    # global step gives that entry; `-mapped` maps it through `Birch::extract_clusters`.
    "scirs2-birch": (True, False),
    "scirs2-birch-mapped": (True, False),
    "hdbscan-crate": (False, False),
    "clump-hdbscan": (False, False),
    "fuel-hdbscan": (False, False),
    "linfa-dbscan": (False, False),
    "clump-dbscan": (False, False),
    "linfa-optics": (False, False),
}

# Arms that take the k-distance `(minPts, eps)` in place of the minimum cluster size.
KDIST = {"linfa-dbscan", "clump-dbscan"}


# The crate each arm runs, written into its rows' `version` column.
CRATES = {
    "betula-kmeans": "betula-cluster",
    "betula-gmm": "betula-cluster",
    "betula-gmm-full": "betula-cluster",
    "linfa-kmeans": "linfa-clustering",
    "linfa-gmm": "linfa-clustering",
    "clump-kmeans": "clump",
    "fuel-kmeans": "fuel",
    "anofox-birch": "anofox-ml-cluster",
    "scirs2-birch": "scirs2-cluster",
    "scirs2-birch-mapped": "scirs2-cluster",
    "hdbscan-crate": "hdbscan",
    "clump-hdbscan": "clump",
    "fuel-hdbscan": "fuel",
    "linfa-dbscan": "linfa-clustering",
    "clump-dbscan": "clump",
    "linfa-optics": "linfa-clustering",
}
assert set(CRATES) == set(ARMS)


def crate_versions() -> dict[str, str]:
    """`crate version` for every arm's crate, as the lockfile `build()` just wrote resolved it."""
    lock = tomllib.loads((CRATE / "Cargo.lock").read_text())
    resolved: dict[str, list[str]] = {}
    for package in lock["package"]:
        # A crate taken from a repository is named by its commit; its version field never moves.
        source = package.get("source", "")
        commit = "+" + source.rpartition("#")[2][:7] if source.startswith("git+") else ""
        resolved.setdefault(package["name"], []).append(package["version"] + commit)
    rev = os.environ.get("BETULA_GIT_REV", "")
    out = {}
    for arm, crate in CRATES.items():
        (v,) = resolved[crate]  # one version per arm's crate, or the row could not name it
        out[arm] = f"{crate} {v}+{rev}" if crate == "betula-cluster" and rev else f"{crate} {v}"
    return out


def build() -> None:
    manifest = str(CRATE / "Cargo.toml")
    cmd = ["cargo", "build", "--release", "--manifest-path", manifest, "--target-dir", str(TARGET)]
    subprocess.run(cmd, check=True)


def _cap():
    resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, MEM_CAP))


def optics_labels(ordering: np.ndarray, min_samples: int) -> np.ndarray:
    """Labels from linfa's `index reachability core` lines, by scikit-learn's xi extraction."""
    from sklearn.cluster import cluster_optics_xi

    order = ordering[:, 0].astype(np.intp)
    reach = np.empty(len(order))
    reach[order] = ordering[:, 1]
    labels, _ = cluster_optics_xi(
        reachability=reach,
        predecessor=np.full(len(order), -1),
        ordering=order,
        min_samples=min_samples,
        predecessor_correction=False,
    )
    return labels


def run(arm: str, rows: Path, k: int, seed: int, params: tuple) -> dict:
    """One process; its peak RSS is the `VmHWM` it reports, not `wait4`'s `ru_maxrss`, which
    keeps this parent's high-water mark from before the fork. `params` is `(min_cluster_size,)`,
    or `(minPts, eps)` for a `KDIST` arm."""
    with tempfile.TemporaryFile() as out:
        child = subprocess.Popen(
            [str(BINARY), arm, str(rows), str(k), str(seed), *map(str, params)],
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
        if arm == "linfa-optics":
            labels = optics_labels(np.loadtxt(out, ndmin=2), params[0])
        else:
            labels = np.loadtxt(out, dtype=np.int64)
    assert child.stderr is not None
    lines = child.stderr.read().decode().splitlines()
    keys = ("fit_s", "allocs", "alloc_bytes", "VmHWM")
    report = {k: v for k, _, v in (ln.partition(" ") for ln in lines) if k in keys}
    return {
        "time_s": elapsed,
        "fit_s": float(report["fit_s"]),
        "rss_mb": int(report["VmHWM"].split()[0]) / 1024,
        "allocs": int(report["allocs"]),
        "alloc_mb": int(report["alloc_bytes"]) / 1024**2,
        "labels": labels,
    }


def check() -> int:
    """The documented examples of the module docstring, as a table; returns how many differ."""
    from rival_reproductions import _dbscan_blobs
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score

    def through(arm, x, params):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "rows.csv"
            np.savetxt(path, x, delimiter=",", fmt="%.17g")
            res = run(arm, path, 0, 0, params)
        if "error" in res:
            raise RuntimeError(f"{arm}: {res['error']}")
        return res["labels"]

    out = subprocess.run([str(BINARY), "check"], capture_output=True, text=True, check=True)
    found = [tuple(line.split("\t")) for line in out.stdout.splitlines()]
    x, y = _dbscan_blobs()
    for arm in ("linfa-dbscan", "clump-dbscan"):
        labels = through(arm, x, (10, 0.3))
        clusters = len(set(labels.tolist()) - {-1})
        got = (clusters, int((labels == -1).sum()), round(adjusted_rand_score(y, labels), 3))
        found.append((arm, "(3, 18, 0.952)", str(got)))
    x = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], dtype=float)
    found.append(
        ("linfa-optics", "[0, 0, 0, 1, 1, 1]", str(through("linfa-optics", x, (2,)).tolist()))
    )
    x, _ = make_blobs(n_samples=2000, n_features=10, random_state=0)
    found.append(("fuel-hdbscan", "3", str(int(through("fuel-hdbscan", x, (5,)).max()) + 1)))
    print("| arm | published | reproduced | |")
    print("|---|---|---|---|")
    for name, published, got in found:
        print(f"| {name} | {published} | {got} | {'yes' if published == got else '**NO**'} |")
    return sum(published != got for _, published, got in found)


def main(argv: list[str]) -> int:
    import pandas as pd

    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--arms=")), None)
    build()
    if check():
        raise SystemExit("a documented example did not reproduce; no row is written")
    version = crate_versions()
    csv = HERE / "results_rust_rivals.csv"
    # Kept rows pass through as text: pandas 2 parses some floats one ulp off, and a re-run of
    # one arm would then rewrite every other arm's row.
    kept = (
        pd.read_csv(csv, dtype=str, keep_default_na=False)
        if only and csv.exists()
        else pd.DataFrame()
    )
    if only and len(kept):
        kept = kept[~kept["arm"].isin(only)]
    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for dataset in DATASETS:
            x, y = load(dataset)[:2]
            path = Path(scratch) / f"{dataset}.csv"
            np.savetxt(path, x, delimiter=",", fmt="%.17g")
            k, mcs = len(np.unique(y)), min_points(len(x))
            kdist = kdist_params(np.ascontiguousarray(x, dtype=np.float64))
            for arm, (told_k, seeded) in ARMS.items():
                if only and arm not in only:
                    continue
                for seed in SEEDS if seeded else SEEDS[:1]:
                    res = run(arm, path, k, seed, kdist if arm in KDIST else (mcs,))
                    row = {"dataset": dataset, "n": len(x), "arm": arm, "told_k": told_k}
                    row |= {"seed": seed, "timing_valid": False, "version": version[arm]}
                    if "error" in res:
                        row["dnf"] = res["error"]
                        shown = res["error"]
                    else:
                        s = score(y, res["labels"])
                        row |= {"time_s": res["time_s"], "rss_mb": res["rss_mb"], **s}
                        row |= {"allocs": res["allocs"], "alloc_mb": res["alloc_mb"]}
                        shown = (
                            f"ARI {s['ari']:.3f}  k={s['n_clusters']:<4} noise "
                            f"{s['noise_share']:.2f}  {res['rss_mb']:7.0f} MB"
                            f"  {res['allocs']:>9d} allocs"
                        )
                    rows.append(row)
                    print(f"{dataset:8s} {arm:20s} seed={seed}  {shown}", flush=True)
                    pd.concat([kept, pd.DataFrame(rows)]).to_csv(csv, index=False)
    return 0


if __name__ == "__main__":
    assert N == 30_000
    if sys.argv[1:] == ["--check"]:
        build()
        sys.exit(check())
    sys.exit(main(sys.argv[1:]))
