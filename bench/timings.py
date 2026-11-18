"""Wall time for the arms a published sentence needs, on a quiet machine: wave 3 of the plan.

The quality harnesses record a time beside every score, but on whatever machine was at hand, one arm
after another, with a first call that pays for imports, numba's compilation and a CUDA context. This
harness re-times their arms under one protocol, and adds what no harness times: labelling new rows,
a stream over a million rows, input larger than memory, a first call with an empty numba cache. It
records times, peak memory and the score, the last as a check that the work timed is the work
scored elsewhere:

- It refuses to start unless `~/.claude/scripts/bench-ready.sh` passes. `--force` runs anyway and
  marks every row `timing_valid=False`. `--dry` starts every case once to see that it runs: one
  round, every input capped at `DRY_ROWS` rows, every row invalid, the CSV under `BENCH_DATA`.
- One process per case (an arm on a dataset at one size). It loads the rows, then fits twice:
  `cold_s` is the first fit, with imports, JIT compilation and device set-up, and `warm_s` the
  second, the one a verdict judges (`bench/scoreboard.py`). A user who fits once pays the first.
  The `scale` stage fits once: at a million rows the set-up is noise against a fit of minutes.
- Each case has its own round count (`Case.rounds`): three where a ratio may come out under 3×, one
  where every slice is further apart. Each round shuffles the cases still due with its own seed, so
  drift spreads over every arm instead of landing on the last ones; the round is the seed. The
  stages of one invocation share its rounds.
- A control, a fixed NumPy SVD that no arm can change, runs once before the first case of the
  invocation and after each case. `control_drift` is its change against that first reading, and a
  row whose control moved by more than `DRIFT` is `timing_valid=False`. Rows of two invocations
  were judged against two bases: a session runs one invocation per environment.
- Every arm that takes a thread count gets the same one, `--threads` (8): OpenMP, BLAS, rayon and
  numba through the environment, the rest through `_provenance.JOBS`. A seeded UMAP on the
  0.5-compatible layout runs on one thread whatever it is given (umap-learn 0.6 keeps the threads
  of a seeded fit on its own layout only); its row says `threads=1` and why, in `note`. CoreSPECT
  keeps numba on one thread (`NUMBA_ONE_THREAD`: cplearn 0.2.1 fails above one), noted. The
  `threads` stage times betula on one thread and an unseeded UMAP 0.6 where the two meet.
- `vmhwm_mb` is the process's peak resident set (`VmHWM`, `_worker.Peak`) from the moment the rows
  are loaded. `n_init` is the restart count every scikit-learn k-means of the fit resolved to, read
  off the estimator: a ratio against k-means is proportional to it.
- Each row records the CPU temperature before and after; GPU rows also record the device's clocks,
  temperature and power draw, and its peak memory (`device_mb`). Each invocation adds its
  environment (governor, power source, thread counts, package versions, uptime, the control base,
  the `bench-ready.sh` report before and after) to a JSON beside the CSV, under the `run` id its
  rows carry.

Stages, by the task of the plan each serves (`--stages=a,b`):

| stage | task | what it times |
|---|---|---|
| `rivals` | F-15 | `rivals.py`, MNIST 20 000 and 70 000, news-MiniLM: the page's heads, rivals |
| `density` | F-15 | `density_rivals.py` on blobs, moons, highdim and digits |
| `heads` | F-15 | k-prototypes and the Toeplitz mixture against their references |
| `rust` | F-15 | `rust_rivals.py`'s binary on three density fixtures, timed inside it |
| `stream` | F-15, F-17.2 | news-MiniLM, 5 checkpoints: betula's stream, its refit, a UMAP refit |
| `predict` | F-18.1 | labels for 10 000 new MNIST rows, four ways; `Betula.load` |
| `threads` | F-18.2 | betula's MNIST heads on one thread; an unseeded UMAP 0.6 at 70 000 rows |
| `budget` | F-18.3 | the told-count head at 1 000 … 16 000 leaves: MNIST-70k, news, arxiv-1m |
| `sparse` | F-18.4 | the sparse route at 256 … 2 048 leaves, raw and `svd`, and scikit-learn |
| `outliers` | F-18.5 | `outlier_scores`, IsolationForest and LOF at 10 000 … 160 000 rows |
| `numba-cold` | F-18.6 | EVoC, UMAP, PaCMAP and fast_hdbscan, each with an empty numba cache |
| `rust-1m` | F-18.7 | `Model::fit`, linfa's `KMeans` and mixture at 1 000 000 rows |
| `external` | F-18.8 | `external_baselines.py`'s contests, in both row orders |
| `scale` | F-16 | arxiv-1m at 200 000 rows (the pricing pass), then 1 000 000; one fit |
| `stream-arxiv` | F-17.3 | arxiv-1m in ten checkpoints: betula's stream and its refit; one pass |
| `out-of-core` | F-17.1 | a 43 GB file, chunked: betula, `MiniBatchKMeans`, `Birch` streaming |
| `gpu` | F-20 | `gpu_rivals.py`: the boards, the density fixtures, arxiv-1m at a million rows |

F-19's build profile is `bench/profile_build.py`. Cut from the campaign prepared at `567301c`, with
the plan's reasons: the `aniso`, `varied` and `circles` fixtures; betula arms no recommendation
names; and every rival cell whose committed median ARI is under `NO_STRUCTURE`, read from the
quality CSV of its stage. At a million rows a case that did not finish at 200 000 rows is recorded
as not finished without being run.

`out-of-core` reads `synthetic-28000000.npy`: 28 000 000 unit vectors in 384 dimensions (MiniLM's
width) around 50 random directions, written once under `BENCH_DATA` (default
`~/.cache/betula-bench`, never `/tmp`, a RAM-backed tmpfs here). The stage refuses to start when the
disk has less than the file plus 10 GB free, and deletes the file afterwards unless `--keep`.

Each stage runs in the environment its arms need: the `uv run --no-sync --with …` list of the module
the arms come from, so an arm runs the versions it was scored with; cuML and the PyTorch stack of
`gpu_rivals.py` in its two environments. DelTriC and CoreSPECT were scored with cplearn 0.2.1, whose
pins take umap-learn back to 0.5 and scikit-learn to 1.7: they run in an environment of their own
(`--skip-arms` keeps them out of the stage's main one). The session script runs one invocation per
stage and environment:

    python bench/timings.py --stages=rivals,predict [--arms=a,b | --skip-arms=a,b] \\
        [--datasets=a,b] [--n=a,b] [--threads=8] [--rounds=N] [--csv=PATH] [--force | --dry] \\
        [--keep]
    python bench/timings.py --stages=out-of-core --prepare   # write data, build binaries, stop
    python bench/timings.py --stages=rivals,density --price  # cases, processes, estimated hours

Writes `bench/results_timings.csv`: a case's rows replace the earlier rows of the same stage, arm,
dataset and size, so stages run in separate environments add up to one table.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CSV = HERE / "results_timings.csv"
READY = Path.home() / ".claude/scripts/bench-ready.sh"
DATA = Path(os.environ.get("BENCH_DATA", Path.home() / ".cache/betula-bench"))
DRIFT = 0.05
THREADS = 8
DRY_ROWS = 100_000
# A two-hour fit, or the session's memory cap, is "did not finish" with the reason.
TIMEOUT = dict.fromkeys(("scale", "stream-arxiv", "out-of-core", "budget", "rust-1m"), 7200.0)
TIMEOUT_DEFAULT = 3600.0
# A rival under this median ARI on a slice found nothing there, and its time backs no sentence.
NO_STRUCTURE = 0.01
# A rival this far from betula's head on its contest, either way, gets one round; one whose
# committed fit is under `SECONDS` keeps three (`_rival_rounds`).
FACTOR, SECONDS = 3.0, 30.0
# The text a seeded UMAP warns with when it drops to one thread.
SEEDED_UMAP = "overridden to 1 by setting random_state"

COLUMNS = (
    "stage",
    "dataset",
    "n",
    "arm",
    "round",
    "seed",
    "threads",
    "n_init",
    "seen",
    "cold_s",
    "warm_s",
    "vmhwm_mb",
    "device_mb",
    "ari",
    "roc_auc",
    "dnf",
    "process_s",
    "control_drift",
    "timing_valid",
    "temp_before",
    "temp_after",
    "gpu_before",
    "gpu_after",
    "version",
    "note",
    "run",
)

ARXIV = "arxiv-1m"
SCALE_SIZES = (200_000, 1_000_000)
SCALE_LEAVES = 16_000
BUDGETS = (1_000, 2_000, 4_000, 8_000, 16_000)
OOC_ROWS, OOC_CHUNK, OOC_K, OOC_PROBE, OOC_DIM = 28_000_000, 200_000, 50, 200_000, 384
# `predict` fits on 60 000 MNIST rows at five rows a leaf (`rivals.BUDGET`'s rule) and labels the
# other 10 000.
PREDICT_NEW, PREDICT_LEAVES = 10_000, 12_000


@dataclass(frozen=True)
class Case:
    stage: str
    arm: str
    dataset: str
    n: int
    rounds: int = 3
    threads: int | None = None  # None: the invocation's `--threads`

    def key(self) -> tuple[str, str, str, int]:
        return (self.stage, self.arm, self.dataset, self.n)


# ── which cases each stage runs ──────────────────────────────────────────────────────────────


class Committed(NamedTuple):
    """A cell of a committed quality table: medians over its seeds, of finished runs."""

    ari: float
    time_s: float  # a first fit on whatever machine was at hand: a price, never a result


def _committed(table: str) -> dict[tuple[str, int, str], Committed]:
    """`(dataset, n, arm)` → `Committed`, for the cut and round rules below."""
    import pandas as pd

    df = pd.read_csv(HERE / f"{table}.csv")
    if "dnf" in df.columns:
        df = df[df["dnf"].isna()]
    med = df.groupby(["dataset", "n", "arm"])[["ari", "time_s"]].median()
    return {(str(d), int(n), str(a)): Committed(*map(float, v)) for (d, n, a), v in med.iterrows()}


def _found(committed: dict, dataset: str, n: int, arm: str) -> bool:
    """A rival keeps its case unless its committed median ARI there is under `NO_STRUCTURE`. A
    betula arm, and a cell no committed row covers, always stay."""
    cell = committed.get((dataset, n, arm))
    return arm.startswith("betula") or cell is None or cell.ari >= NO_STRUCTURE


def _rival_rounds(rival: Committed | None, head: Committed | None) -> int:
    """Three rounds for a rival within `FACTOR` of betula's head on its contest, by their committed
    first fits; one when it is further away, either way, since three rounds are what a ratio near
    the 1.5× of `bench/scoreboard.py`'s `LEVEL` needs and a wider one needs one. A rival whose fit
    takes seconds keeps three: its committed first fit holds an import."""
    if rival is None or head is None or not rival.time_s >= SECONDS:
        return 3
    ratio = rival.time_s / head.time_s
    return 1 if ratio >= FACTOR or ratio <= 1 / FACTOR else 3


# `rivals`, per dataset: the betula arms timed (the contests' heads and the rows of the page), and
# the rivals timed at three rounds, the plan's list; every other rival gets one.
RIVALS = {
    "mnist": (
        ("betula-kmeans", "betula-leiden", "betula-leiden-rows", "betula-leiden-rows-auto"),
        ("sklearn-kmeans", "pacmap+kmeans", "umap-0.6+kmeans", "evoc"),
    ),
    "news-minilm": (
        ("betula-kmeans", "betula-leiden-rows", "betula-leiden-rows-auto", "betula-vmf-shared"),
        ("sklearn-kmeans", "pacmap+kmeans", "evoc"),
    ),
}
RIVALS_SLICES = (("mnist", 20_000), ("mnist", 70_000), ("news-minilm", 17_901))

DENSITY_FIXTURES = ("blobs", "moons", "highdim", "digits")
# betula's head on each density contest (told the count or not), the board's choice since F-09.
DENSITY_HEAD = {True: "betula-leiden-rows-auto", False: "betula-hdbscan"}
DENSITY_BETULA = (
    "betula-hdbscan",
    "betula-dc-median",
    "betula-gmm-bic",
    "betula-gmm-full-bic",
    "betula-leiden-rows",
    "betula-leiden-rows-stable",
    "betula-leiden-rows-auto",
    "betula-gmm-full",
    "betula-kmeans",
    "betula-gmm-full-merge",
)
# One round, on these fixtures only: the cost sentence each backs ("tens of seconds against a
# tenth") needs one cell per shape.
DENSITY_ONE = {
    "betula-scale-space": ("moons", "highdim"),
    "sklearn-optics": ("moons", "highdim"),
    "denmune": ("moons", "digits"),
    "finch": DENSITY_FIXTURES,
}

HEADS = (
    "betula-kprototypes",
    "kmodes-kprototypes",
    "betula-gmm-toeplitz",
    "tslearn-kshape",
    "aeon-kasba",
)

PREDICT = (
    "betula-kmeans",
    "betula-leiden-rows-auto",
    "sklearn-kmeans",
    "umap-transform",
    "hdbscan-approximate-predict",
    "betula-load",
)
UNSEEDED_UMAP = "umap-0.6+kmeans (unseeded)"
SCALE_ARMS = (
    "betula-kmeans",
    "betula-vmf-shared",
    "betula-leiden-rows",
    "betula-leiden-rows-auto",
    "sklearn-kmeans",
    "minibatch-kmeans",
    "faiss-kmeans-pp",
    "umap-0.6",
    "evoc",
)
SCALE_DISTS = {
    "sklearn-kmeans": ("scikit-learn",),
    "minibatch-kmeans": ("scikit-learn",),
    "faiss-kmeans-pp": ("faiss-cpu",),
    "evoc": ("evoc",),
}
GPU_MILLION = ("cuml-kmeans", "cuml-umap+kmeans", "cuml-umap+hdbscan", "faiss-gpu-kmeans")
RUST_MILLION = {
    "betula-kmeans": 3,
    "betula-gmm": 3,
    "betula-gmm-full": 3,
    "linfa-kmeans": 3,
    "linfa-gmm": 1,
}


def _runs_on(where: str, dataset: str, n: int) -> bool:
    """`rivals.ARMS`'s `where`, read as `bench/rivals.py` reads it."""
    import rivals

    if where == "embeddings":
        return dataset in rivals.EMBEDDINGS
    return where in ("all", dataset, f"{dataset}-{n}")


def _rivals_cases() -> list[Case]:
    import rivals

    seen = _committed("results_rivals")
    out = []
    for dataset, n in RIVALS_SLICES:
        betula, three = RIVALS[dataset]
        assert set(betula + three) <= set(rivals.ARMS), betula + three
        for arm, (_, where, _) in rivals.ARMS.items():
            if not _runs_on(where, dataset, n) or not _found(seen, dataset, n, arm):
                continue
            if arm in betula or not arm.startswith("betula"):
                out.append(Case("rivals", arm, dataset, n, 3 if arm in betula + three else 1))
    return out


def _density_cases() -> list[Case]:
    import density_rivals as dr

    assert set(DENSITY_BETULA) | set(DENSITY_ONE) <= set(dr.ARMS)
    seen = _committed("results_density_rivals")
    out = []
    for dataset in DENSITY_FIXTURES:
        n = len(dr.load(dataset)[1])
        for arm, (told, _) in dr.ARMS.items():
            if arm in DENSITY_ONE:
                if dataset in DENSITY_ONE[arm] and _found(seen, dataset, n, arm):
                    out.append(Case("density", arm, dataset, n, 1))
            elif arm in DENSITY_BETULA:
                out.append(Case("density", arm, dataset, n, 3))
            elif not arm.startswith("betula") and _found(seen, dataset, n, arm):
                head = seen.get((dataset, n, DENSITY_HEAD[told]))
                rounds = _rival_rounds(seen.get((dataset, n, arm)), head)
                out.append(Case("density", arm, dataset, n, rounds))
    return out


def _heads_cases() -> list[Case]:
    import head_references as hr

    sizes: dict[str, int] = {}
    out = []
    for arm in HEADS:
        dataset = hr.ARMS[arm][0]
        if dataset not in sizes:
            sizes[dataset] = len(hr.load(dataset)[1])
        out.append(Case("heads", arm, dataset, sizes[dataset], 1))
    return out


def _rust_cases() -> list[Case]:
    """`digits` for the arms told the count, `moons` and `highdim` for the ones that choose it, and
    the `hdbscan` crate on all three. A crate told the count is set against betula's `kmeans`
    through `Model::fit`; one that chooses it against betula's `hdbscan` head, a Python row of the
    `density` table, since its Rust API is not one call."""
    import density_rivals as dr
    import rust_rivals as rr

    seen = _committed("results_rust_rivals")
    heads = _committed("results_density_rivals")
    out = []
    for dataset in ("digits", "moons", "highdim"):
        n = len(dr.load(dataset)[1])
        for arm, (told, _) in rr.ARMS.items():
            if not (told == (dataset == "digits") or arm == "hdbscan-crate"):
                continue
            if arm.startswith("betula"):
                out.append(Case("rust", arm, dataset, n, 3))
            elif _found(seen, dataset, n, arm):
                head = (
                    seen.get((dataset, n, "betula-kmeans"))
                    if told
                    else heads.get((dataset, n, "betula-hdbscan"))
                )
                out.append(
                    Case("rust", arm, dataset, n, _rival_rounds(seen.get((dataset, n, arm)), head))
                )
    return out


def _stream_cases() -> list[Case]:
    from embed_news import cached

    n = len(cached()[1])
    return [
        Case("stream", "betula-stream", "news-minilm", n, 3),
        Case("stream", "betula-refit", "news-minilm", n, 3),
        Case("stream", "umap+kmeans", "news-minilm", n, 1),
    ]


def _threads_cases() -> list[Case]:
    out = [
        Case("threads", arm, "mnist", n, 3, threads=1)
        for n in (20_000, 70_000)
        for arm in ("betula-leiden-rows-auto", "betula-leiden-rows")
    ]
    return [*out, Case("threads", UNSEEDED_UMAP, "mnist", 70_000, 3)]


def _budget_cases() -> list[Case]:
    from embed_news import cached

    slices = (("mnist", 70_000), ("news-minilm", len(cached()[1])), (ARXIV, 1_000_000))
    return [
        Case("budget", f"betula-leiden-rows-auto ({leaves} leaves)", d, n, 1 if d == ARXIV else 3)
        for d, n in slices
        for leaves in BUDGETS
    ]


def _sparse_cases() -> list[Case]:
    from _worker import load_sparse

    n = load_sparse("20news")[0].shape[0]
    arms = [
        f"{name} ({leaves} leaves)"
        for name in ("betula-sparse", "betula-svd")
        for leaves in (256, 512, 1024, 2048)
    ]
    return [Case("sparse", arm, "20news", n, 3) for arm in (*arms, "sklearn-kmeans", "sklearn-svd")]


def _outlier_cases() -> list[Case]:
    import outliers

    sizes = [len(outliers.FIXTURES["blobs+uniform"](0, n=n)[1]) for n in (10_000, 40_000, 160_000)]
    arms = ("betula", "sklearn-iforest", "sklearn-lof")
    return [Case("outliers", arm, "blobs+uniform", n, 3) for n in sizes for arm in arms]


def _numba_cold_cases() -> list[Case]:
    import density_rivals as dr

    out = [
        Case("numba-cold", arm, "mnist", 20_000, 3)
        for arm in ("evoc", "umap-0.6+kmeans", "pacmap+kmeans")
    ]
    return [*out, Case("numba-cold", "fast-hdbscan", "blobs", len(dr.load("blobs")[1]), 3)]


def _external_cases() -> list[Case]:
    import external_baselines as eb

    cases = {
        Case("external", method, spec["dataset"] + ("" if order == "label" else "-shuffled"), n)
        for spec in eb.CONTESTS.values()
        for n in spec["sizes"]
        for order in eb.ORDERS
        for method in spec["methods"]
    }
    return sorted(cases, key=Case.key)


def _scale_cases() -> list[Case]:
    return [
        Case("scale", arm, ARXIV, n, 3 if arm.startswith("betula") else 1)
        for n in SCALE_SIZES
        for arm in SCALE_ARMS
    ]


def _stream_arxiv_cases() -> list[Case]:
    return [
        Case("stream-arxiv", arm, ARXIV, 1_000_000, 1) for arm in ("betula-stream", "betula-refit")
    ]


def _out_of_core_cases() -> list[Case]:
    arms = ("betula-vmf-shared", "minibatch-kmeans", "birch")
    return [Case("out-of-core", arm, "synthetic-384", OOC_ROWS, 1) for arm in arms]


def _gpu_cases() -> list[Case]:
    import density_rivals as dr
    import gpu_rivals as g

    q = _committed("results_gpu_rivals")
    out = []
    plan = [(d, n, "boards") for d, n in g.BOARDS]
    plan += [(d, len(dr.load(d)[1]), "density") for d in g.DENSITY]
    for dataset, n, kind in plan:
        for arm, (_, where, _, _) in g.ARMS.items():
            if where in (kind, "both") and _found(q, dataset, n, arm):
                out.append(Case("gpu", arm, dataset, n, 3))
    return [*out, *(Case("gpu", arm, ARXIV, 1_000_000, 1) for arm in GPU_MILLION)]


STAGES: dict[str, Callable[[], list[Case]]] = {
    "rivals": _rivals_cases,
    "density": _density_cases,
    "heads": _heads_cases,
    "rust": _rust_cases,
    "stream": _stream_cases,
    "predict": lambda: [Case("predict", arm, "mnist", 70_000, 3) for arm in PREDICT],
    "threads": _threads_cases,
    "budget": _budget_cases,
    "sparse": _sparse_cases,
    "outliers": _outlier_cases,
    "numba-cold": _numba_cold_cases,
    "rust-1m": lambda: [
        Case("rust-1m", a, "highdim", 1_000_000, r) for a, r in RUST_MILLION.items()
    ],
    "external": _external_cases,
    "scale": _scale_cases,
    "stream-arxiv": _stream_arxiv_cases,
    "out-of-core": _out_of_core_cases,
    "gpu": _gpu_cases,
}


# ── data written once ────────────────────────────────────────────────────────────────────────


def synthetic(rows: int) -> tuple[Path, Path]:
    """A float32 `.npy` of `rows` unit vectors around `OOC_K` directions and its labels, written
    once in chunks so that no step holds more than one chunk."""
    DATA.mkdir(parents=True, exist_ok=True)
    xp, yp = DATA / f"synthetic-{rows}.npy", DATA / f"synthetic-{rows}-labels.npy"
    if xp.exists() and yp.exists():
        return xp, yp
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((OOC_K, OOC_DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    x = np.lib.format.open_memmap(xp.with_suffix(".tmp"), "w+", np.float32, (rows, OOC_DIM))
    y = rng.integers(0, OOC_K, rows).astype(np.int32)
    for a in range(0, rows, OOC_CHUNK):
        b = min(rows, a + OOC_CHUNK)
        block = centres[y[a:b]] + 0.06 * rng.standard_normal((b - a, OOC_DIM), dtype=np.float32)
        x[a:b] = block / np.linalg.norm(block, axis=1, keepdims=True)
    x.flush()
    del x
    np.save(yp, y)
    xp.with_suffix(".tmp").rename(xp)
    return xp, yp


def _blobs_rows(dataset: str, n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """`_worker.gen`'s fixture with its labels, as `bench/external_baselines.py` builds them."""
    from _worker import gen

    x, k = gen(dataset, n)
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    return x, np.repeat(np.arange(k), sizes), k


# ── the child: one case ──────────────────────────────────────────────────────────────────────


@dataclass
class Prepared:
    """One case with everything but the timed call done."""

    fit: Callable[[], np.ndarray]
    truth: np.ndarray | None
    dists: tuple[str, ...]
    score: str = "ari"  # "roc_auc" where `fit` returns outlier scores


def _dists(table: dict[str, tuple[str, ...]], arm: str) -> tuple[str, ...]:
    return ("betula-cluster",) if arm.startswith("betula") else table[arm]


def _betula_fit(x, k: int, seed: int, **kw) -> np.ndarray:
    import betula_cluster as bc

    return bc.Betula(n_clusters=k, threshold=0.0, seed=seed, **kw).fit_predict(x)


def _leaves(arm: str) -> int:
    """The leaf count in an arm named `name (N leaves)`."""
    return int(arm.rpartition("(")[2].split()[0])


ROWS_AUTO = {"method": "leiden", "leaf_graph": "rows", "resolution": "auto"}
SCALE_BETULA = {
    "betula-kmeans": {"method": "kmeans"},
    "betula-vmf-shared": {"method": "vmf", "kappa": "shared"},
    "betula-leiden-rows": {"method": "leiden", "leaf_graph": "rows"},
    "betula-leiden-rows-auto": ROWS_AUTO,
}


def _rivals_arm(case: Case, seed: int) -> Prepared:
    import rivals

    x, y = rivals.load(case.dataset, case.n, seed)
    k = len(np.unique(y))
    if case.arm == UNSEEDED_UMAP:
        import umap

        def fit():
            params = rivals.umap_params(case.dataset)
            z = umap.UMAP(compatibility_layout=False, **params).fit_transform(x)
            return rivals._kmeans(z, k, seed)

        return Prepared(fit, y, ("umap-learn", "pynndescent", "scikit-learn"))
    arm = rivals.ARMS[case.arm][2]
    return Prepared(
        lambda: arm(x, k, seed, case.dataset, case.n), y, _dists(rivals.RIVAL_DISTS, case.arm)
    )


def _density_arm(case: Case, seed: int) -> Prepared:
    import density_rivals as dr

    x, y, k = dr.load(case.dataset)
    x = np.ascontiguousarray(x, dtype=np.float64)
    arm = dr.ARMS[case.arm][1]
    return Prepared(lambda: arm(x, k, seed), np.asarray(y), _dists(dr.RIVAL_DISTS, case.arm))


def _sparse_arm(case: Case, seed: int) -> Prepared:
    from _worker import load_sparse, reduce_dims

    x, y, k = load_sparse(case.dataset)
    name = case.arm.partition(" (")[0]

    def fit():
        import betula_cluster as bc
        from sklearn.cluster import KMeans

        if name == "betula-sparse":
            return bc.fit_predict_sparse(
                x, n_clusters=k, method="kmeans", max_leaves=_leaves(case.arm), seed=seed
            )
        if name == "betula-svd":
            return bc.fit_predict_sparse(
                x,
                n_clusters=k,
                method="spherical-kmeans",
                max_leaves=_leaves(case.arm),
                seed=seed,
                projection="svd",
                projection_dim=50,
            )
        z = reduce_dims(x, "svd", 50) if name == "sklearn-svd" else x
        return KMeans(k, n_init=3, random_state=seed).fit_predict(z)

    return Prepared(fit, y, ("betula-cluster",) if name.startswith("betula") else ("scikit-learn",))


def _external_arm(case: Case, seed: int) -> Prepared:
    import external_baselines as eb

    base, _, order = case.dataset.partition("-")
    x, truth, k = _blobs_rows(base, case.n)
    if order == "shuffled":
        perm = np.random.default_rng(1000 + seed).permutation(case.n)
        x, truth = x[perm], truth[perm]
    lib = eb.library(case.arm)
    dist = {
        "betula_cluster": "betula-cluster",
        "faiss": "faiss-cpu",
        "fast_hdbscan": "fast-hdbscan",
    }
    return Prepared(lambda: eb.fit(case.arm, x, k, seed), truth, (dist.get(lib, "scikit-learn"),))


def _scale_arm(case: Case, seed: int) -> Prepared:
    import rivals

    x, y = rivals.load(case.dataset, case.n, seed)
    k = len(np.unique(y))
    if case.arm in SCALE_BETULA:
        kw = SCALE_BETULA[case.arm]
        return Prepared(
            lambda: _betula_fit(x, k, seed, max_leaves=SCALE_LEAVES, **kw), y, ("betula-cluster",)
        )

    def fit():
        if case.arm == "sklearn-kmeans":
            return rivals._kmeans(x, k, seed)
        if case.arm == "minibatch-kmeans":
            from sklearn.cluster import MiniBatchKMeans

            return MiniBatchKMeans(k, random_state=seed).fit_predict(x)
        if case.arm == "faiss-kmeans-pp":
            import external_baselines as eb

            return eb.fit("faiss-kmeans-pp", x, k, seed)
        if case.arm == "evoc":
            return rivals._evoc(x, seed)
        raise ValueError(case.arm)

    return Prepared(fit, y, SCALE_DISTS[case.arm])


def _prepare(case: Case, seed: int) -> Prepared:
    """The rows are loaded here, outside every timed call."""
    stage = case.stage
    if stage in ("rivals", "threads") or (stage == "numba-cold" and case.dataset == "mnist"):
        return _rivals_arm(case, seed)
    if stage in ("density", "numba-cold"):
        return _density_arm(case, seed)
    if stage == "heads":
        import head_references as hr

        x, y, extra = hr.load(case.dataset)
        k = len(np.unique(y))
        arm = hr.ARMS[case.arm][1]
        return Prepared(
            lambda: arm(x, k, seed, extra), np.asarray(y), _dists(hr.REFERENCE_DISTS, case.arm)
        )
    if stage == "budget":
        import rivals

        x, y = rivals.load(case.dataset, case.n, seed)
        k, leaves = len(np.unique(y)), _leaves(case.arm)
        return Prepared(
            lambda: _betula_fit(x, k, seed, max_leaves=leaves, **ROWS_AUTO), y, ("betula-cluster",)
        )
    if stage == "sparse":
        return _sparse_arm(case, seed)
    if stage == "outliers":
        import outliers

        x, y = outliers.FIXTURES[case.dataset](seed, n=case.n)
        detector = outliers.DETECTORS[case.arm]
        return Prepared(lambda: detector(x, seed), y, outliers.DISTS[case.arm], "roc_auc")
    if stage == "external":
        return _external_arm(case, seed)
    if stage == "scale":
        return _scale_arm(case, seed)
    if stage == "gpu":
        import gpu_rivals as g

        x, y = g.load(case.dataset, case.n, seed)
        k = len(np.unique(y))
        _, _, arm, dists = g.ARMS[case.arm]
        # The labels come back to the host inside the timed call, as the input went to the device.
        return Prepared(lambda: g._as_host(arm(x, k, seed, case.dataset)), y, dists)
    raise ValueError(stage)


def _score(truth, labels, kind: str = "ari") -> float:
    from sklearn.metrics import adjusted_rand_score, roc_auc_score

    if kind == "roc_auc":
        return float(roc_auc_score(truth, labels))
    labels = np.asarray(labels)
    return float(adjusted_rand_score(truth, np.where(labels < 0, labels.max() + 1, labels)))


def _trace_kmeans(starts: set[int]) -> None:
    """Record the `n_init` every later scikit-learn k-means fit in this process resolves to."""
    from sklearn.cluster import KMeans, MiniBatchKMeans

    for cls in (KMeans, MiniBatchKMeans):

        def traced(est, *args, _fit=cls.fit, **kw):
            out = _fit(est, *args, **kw)
            starts.add(int(est._n_init))
            return out

        cls.fit = traced  # type: ignore[method-assign]


class Fits:
    """Times `count` calls of a fit, and what the fits said about themselves: the restart count of
    every scikit-learn k-means, and a seeded UMAP dropping to one thread.

    The tracer goes in once `sklearn.cluster` is loaded, so a fit that never imports it carries no
    extra module in its peak memory; before a single fit it goes in only for a rival, whose stack
    imports scikit-learn anyway.
    """

    def __init__(self) -> None:
        self.starts: set[int] = set()
        self.traced = False
        self.one_thread = False

    def trace(self) -> None:
        if not self.traced:
            with contextlib.suppress(ImportError):  # no scikit-learn in this interpreter
                _trace_kmeans(self.starts)
            self.traced = True

    def run(
        self, fit: Callable, count: int, trace_first: bool = False
    ) -> tuple[list[float], object]:
        times, out = [], None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(count):
                if "sklearn.cluster" in sys.modules or (i == 0 and trace_first):
                    self.trace()
                t0 = time.perf_counter()
                out = fit()
                times.append(time.perf_counter() - t0)
        self.one_thread |= any(SEEDED_UMAP in str(w.message) for w in caught)
        return times, out

    def facts(self) -> dict:
        out: dict = {}
        if self.starts:
            out["n_init"] = "/".join(map(str, sorted(self.starts)))
        if self.one_thread:
            out |= {"threads": 1, "note": "seeded UMAP: the layout ran on one thread"}
        return out


def child(case: Case, seed: int) -> dict:
    import _provenance

    if "BENCH_JOBS" in os.environ:
        _provenance.JOBS = int(os.environ["BENCH_JOBS"])
    if case.stage in ("stream", "stream-arxiv"):
        return child_stream(case, seed)
    if case.stage == "out-of-core":
        return child_out_of_core(case, seed)
    if case.stage == "predict":
        return child_predict(case, seed)
    if case.stage == "scale" and case.arm == "umap-0.6":
        return child_layout(case, seed)
    from _worker import Peak

    prep = _prepare(case, seed)
    device = None
    if case.stage == "gpu":
        import gpu_rivals as g

        device = g.DevicePeak()
    peak = Peak()
    fits = Fits()
    count = 1 if case.stage == "scale" else 2
    times, labels = fits.run(prep.fit, count, trace_first=not case.arm.startswith("betula"))
    out = {"cold_s": times[0], "warm_s": times[1] if count > 1 else None, "vmhwm_mb": peak.mb()}
    if device is not None:
        out["device_mb"] = device.mb()
    out[prep.score] = _score(prep.truth, labels, prep.score)
    from _provenance import versions

    return out | fits.facts() | {"version": versions(*prep.dists)}


def child_predict(case: Case, seed: int) -> dict:
    """Labelling rows the model never saw. The model is fitted on 60 000 MNIST rows outside the
    timed call, which labels the other 10 000; the first call can build what a later one reuses.
    `betula-load` times reading a saved `leiden` rows model back from disk."""
    import betula_cluster as bc
    import rivals
    from _provenance import versions
    from _worker import Peak

    x, y = rivals.load(case.dataset, case.n, seed)
    k = len(np.unique(y))
    order = np.random.default_rng(0).permutation(len(x))
    train, new, truth = x[order[:-PREDICT_NEW]], x[order[-PREDICT_NEW:]], y[order[-PREDICT_NEW:]]
    fits = Fits()
    labels = None
    path = DATA / f"predict-{os.getpid()}.betula"
    if case.arm.startswith("betula"):
        kw = {"method": "kmeans"} if case.arm == "betula-kmeans" else ROWS_AUTO
        est = bc.Betula(n_clusters=k, threshold=0.0, seed=seed, max_leaves=PREDICT_LEAVES, **kw)
        est.fit(train)
        dists: tuple[str, ...] = ("betula-cluster",)
        if case.arm == "betula-load":
            DATA.mkdir(parents=True, exist_ok=True)
            est.save(str(path))

            def call():
                return bc.Betula.load(str(path))
        else:
            call = lambda: est.predict(new)  # noqa: E731
    elif case.arm == "sklearn-kmeans":
        from sklearn.cluster import KMeans

        fits.trace()
        km = KMeans(k, n_init=4, random_state=seed).fit(train)  # `rivals._kmeans`'s restarts
        call, dists = (lambda: km.predict(new)), ("scikit-learn",)
    else:
        import umap

        params = rivals.umap_params(case.dataset)
        reducer = umap.UMAP(random_state=seed, compatibility_layout=False, **params).fit(train)
        dists = ("umap-learn", "pynndescent")
        if case.arm == "umap-transform":
            call = lambda: reducer.transform(new)  # noqa: E731
        else:
            import hdbscan

            # The UMAP docs' MNIST recipe, kept for prediction; the new rows' layout is given.
            clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500, prediction_data=True)
            clusterer.fit(reducer.embedding_)
            z = reducer.transform(new)
            call, dists = (lambda: hdbscan.approximate_predict(clusterer, z)[0]), ("hdbscan",)
    peak = Peak()
    times, got = fits.run(call, 2)
    out = {"cold_s": times[0], "warm_s": times[1], "vmhwm_mb": peak.mb(), "seen": len(new)}
    if case.arm == "betula-load":
        labels = got.predict(new)  # type: ignore[union-attr]
        path.unlink()
    elif case.arm != "umap-transform":
        labels = got
    if labels is not None:
        out["ari"] = _score(truth, labels)
    return out | fits.facts() | {"version": versions(*dists)}


def child_layout(case: Case, seed: int) -> dict:
    """The UMAP 0.6 layout fitted once, then k-means and HDBSCAN on it: three rows, each time the
    layout's plus its clusterer's, so the million-row layout is not fitted twice."""
    import rivals
    import umap
    from _provenance import versions
    from _worker import Peak

    x, y = rivals.load(case.dataset, case.n, seed)
    k = len(np.unique(y))
    peak = Peak()
    fits = Fits()
    params = rivals.umap_params(case.dataset)
    (t_layout,), z = fits.run(
        lambda: umap.UMAP(random_state=seed, compatibility_layout=False, **params).fit_transform(x),
        1,
    )
    vm_layout = peak.mb()
    (t_km,), km = fits.run(lambda: rivals._kmeans(z, k, seed), 1, trace_first=True)
    vm_km = peak.mb()
    (t_hd,), hd = fits.run(lambda: rivals._hdbscan(z, min_cluster_size=10), 1)
    facts = fits.facts()
    umap_v = versions("umap-learn", "pynndescent")
    parts = [
        {"arm": "umap-0.6 layout", "cold_s": t_layout, "vmhwm_mb": vm_layout, "version": umap_v},
        {
            "arm": "umap-0.6+kmeans",
            "cold_s": t_layout + t_km,
            "vmhwm_mb": vm_km,
            "ari": _score(y, km),
            "version": versions("umap-learn", "pynndescent", "scikit-learn"),
        },
        {
            "arm": "umap-0.6+hdbscan (mcs=10)",
            "cold_s": t_layout + t_hd,
            "vmhwm_mb": peak.mb(),
            "ari": _score(y, hd),
            "version": versions("umap-learn", "pynndescent", "hdbscan"),
        },
    ]
    parts[1]["n_init"] = facts.pop("n_init", None)
    return {"parts": [facts | p for p in parts]}


@dataclass(frozen=True)
class Stream:
    fractions: tuple[float, ...]
    leaves: int
    passes: int  # two: the second is the warm one; at a million rows one


def child_stream(case: Case, seed: int) -> dict:
    """A stream in checkpoints, one row each. The betula stream's time at a checkpoint is its
    `partial_fit` on the new chunk, the global step and a `predict` of every row seen; a refit arm
    fits from scratch on every row seen."""
    import betula_cluster as bc
    from _provenance import versions
    from _worker import Peak

    if case.dataset == "news-minilm":
        import stream_vs_refit as sr

        x, y = sr.cached()
        x = np.ascontiguousarray(x, dtype=np.float64)
        spec = Stream(sr.FRACTIONS, sr.LEAVES, 2)
    else:
        import rivals

        x, y = rivals.load(case.dataset, case.n, seed)  # float32: a float64 copy would be 6 GB
        spec = Stream(tuple((i + 1) / 10 for i in range(10)), SCALE_LEAVES, 1)
    perm = np.random.default_rng(2000 + seed).permutation(len(x))
    x, y = x[perm], y[perm]
    k = len(np.unique(y))
    cuts = [int(f * len(x)) for f in spec.fractions]

    def vmf():
        return bc.Betula(
            method="vmf",
            kappa="shared",
            n_clusters=k,
            max_leaves=spec.leaves,
            threshold=0.0,
            seed=seed,
        )

    peak = Peak()
    fits = Fits()
    passes, aris = [], []
    for p in range(spec.passes):
        est, start, times = vmf(), 0, []
        for cp in cuts:
            if case.arm == "betula-stream":

                def step(est=est, a=start, b=cp):
                    est.partial_fit(x[a:b])
                    est.partial_fit()
                    return est.predict(x[:b])
            elif case.arm == "betula-refit":
                step = lambda b=cp: vmf().fit_predict(x[:b])  # noqa: E731
            else:
                import stream_vs_refit as sr

                step = lambda b=cp: sr.refit(case.arm, x[:b], seed)  # noqa: E731
            (t,), labels = fits.run(step, 1)
            times.append(t)
            start = cp
            if p == spec.passes - 1:
                aris.append(_score(y[:cp], labels))
        passes.append(times)
    vm = peak.mb()
    dists = ("betula-cluster",) if case.arm.startswith("betula") else ("umap-learn", "scikit-learn")
    common = fits.facts() | {"vmhwm_mb": vm, "version": versions(*dists)}
    parts = []
    for i, cp in enumerate(cuts):
        warm = passes[1][i] if spec.passes > 1 else None
        parts.append(common | {"seen": cp, "cold_s": passes[0][i], "warm_s": warm, "ari": aris[i]})
    return {"parts": parts}


def child_out_of_core(case: Case, seed: int) -> dict:
    """`partial_fit` in chunks over a file larger than RAM, read with plain reads: a memory map
    would put the file's pages in the resident set, and `VmHWM` would count the page cache."""
    from _provenance import versions
    from _worker import Peak

    xp, yp = synthetic(case.n)
    peak = Peak()
    readers = {
        (1, 0): np.lib.format.read_array_header_1_0,
        (2, 0): np.lib.format.read_array_header_2_0,
    }
    if case.arm == "betula-vmf-shared":
        import betula_cluster as bc

        est = bc.Betula(
            method="vmf",
            kappa="shared",
            n_clusters=OOC_K,
            max_leaves=SCALE_LEAVES,
            threshold=0.0,
            seed=seed,
        )
        learn, finish, dists = est.partial_fit, (lambda: est.partial_fit()), ("betula-cluster",)
    elif case.arm == "minibatch-kmeans":
        from sklearn.cluster import MiniBatchKMeans

        est = MiniBatchKMeans(OOC_K, random_state=seed)

        def learn(block):  # one call per mini-batch, as `stream_vs_refit.MiniBatchStream` streams
            for a in range(0, len(block), est.batch_size):
                est.partial_fit(block[a : a + est.batch_size])

        finish, dists = (lambda: None), ("scikit-learn",)
    else:
        from sklearn.cluster import Birch

        # The tree alone during the stream, as `bench/_worker.py` streams it: with a count set,
        # every chunk would rerun the global step over every subcluster.
        est = Birch(n_clusters=None)
        learn, dists = est.partial_fit, ("scikit-learn",)

        def finish():
            est.set_params(n_clusters=OOC_K)
            est.partial_fit()

    t0 = time.perf_counter()
    with open(xp, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, _, dtype = readers[version](f)
        for _ in range(0, shape[0], OOC_CHUNK):
            block = np.fromfile(f, dtype=dtype, count=OOC_CHUNK * shape[1]).reshape(-1, shape[1])
            learn(block)
        finish()
    total = time.perf_counter() - t0
    vm = peak.mb()
    probe = np.load(xp, mmap_mode="r")[:OOC_PROBE]
    truth = np.load(yp, mmap_mode="r")[:OOC_PROBE]
    labels = est.predict(np.ascontiguousarray(probe))
    out = {"cold_s": total, "vmhwm_mb": vm, "seen": shape[0], "ari": _score(truth, labels)}
    return out | {"version": versions(*dists), "note": f"ARI on the first {len(truth)} rows"}


# ── the parent: rounds, control, environment ─────────────────────────────────────────────────


def control() -> float:
    a = np.random.default_rng(0).standard_normal((1000, 1000))
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        np.linalg.svd(a)
        reps.append(time.perf_counter() - t0)
    return float(np.median(reps))


def cpu_temp() -> float | None:
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if (hw / "name").read_text().strip() == "k10temp":
            return int((hw / "temp1_input").read_text()) / 1000
    zones = [int(p.read_text()) / 1000 for p in Path("/sys/class/thermal").glob("*/temp")]
    return max(zones) if zones else None


def gpu_state() -> str | None:
    query = "clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def environment(ready_report: str) -> dict:
    from importlib import metadata

    packages = {}
    for name in (
        "betula-cluster",
        "numpy",
        "scikit-learn",
        "umap-learn",
        "pynndescent",
        "hdbscan",
        "evoc",
        "fast-hdbscan",
        "leidenalg",
        "python-igraph",
        "pacmap",
        "openTSNE",
        "PhenoGraph",
        "faiss-cpu",
        "faiss-gpu",
        "numba",
        "cuml-cu12",
        "torch",
        "fastkmeans",
        "genieclust",
    ):
        with contextlib.suppress(metadata.PackageNotFoundError):
            packages[name] = metadata.version(name)
    ac = [p for p in Path("/sys/class/power_supply").glob("*") if (p / "online").exists()]
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": sha,
        "kernel": platform.release(),
        "cpu": next(
            (
                ln.split(":", 1)[1].strip()
                for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")
            ),
            None,
        ),
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "platform_profile": _read("/sys/firmware/acpi/platform_profile"),
        "on_mains": {p.name: _read(str(p / "online")) for p in ac},
        "uptime_h": float(Path("/proc/uptime").read_text().split()[0]) / 3600,
        "temp_start": cpu_temp(),
        "dimm_layout": "not read (needs root)",
        "placement": "not read (needs root)",
        "threads": {
            v: os.environ.get(v)
            for v in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "python": sys.version.split()[0],
        "packages": packages,
        "gpu": gpu_state(),
        "bench_ready": ready_report,
    }


# Arms whose numba kernels run on one thread whatever `--threads` says, with the row's note.
# cplearn 0.2.1 under numba 0.63 returns out-of-range indices (an IndexError) whenever numba
# runs more than one thread; at one numba thread it finishes, whatever OMP_NUM_THREADS is.
NUMBA_ONE_THREAD = {
    "cplearn-corespect": "CoreSPECT: numba on one thread (cplearn 0.2.1 fails above one)",
}


def _thread_env(threads: int) -> dict[str, str]:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    names += ("RAYON_NUM_THREADS", "NUMBA_NUM_THREADS", "BENCH_JOBS")
    return dict.fromkeys(names, str(threads))


def run_case(case: Case, seed: int, threads: int) -> dict:
    """One child process; a case that runs past its time limit or is killed is "did not finish"."""
    if case.stage in ("rust", "rust-1m"):
        return run_rust(case, seed)
    cmd = [sys.executable, str(Path(__file__)), "--child", case.stage, case.arm]
    cmd += [case.dataset, str(case.n), str(seed)]
    env = os.environ | _thread_env(threads)
    if case.arm in NUMBA_ONE_THREAD:
        env["NUMBA_NUM_THREADS"] = "1"
    cache = None
    if case.stage == "numba-cold":
        cache = DATA / "numba-cold" / f"{os.getpid()}-{time.monotonic_ns()}"
        cache.mkdir(parents=True)
        env["NUMBA_CACHE_DIR"] = str(cache)
    limit = TIMEOUT.get(case.stage, TIMEOUT_DEFAULT)
    t0 = time.perf_counter()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=limit, env=env)
    except subprocess.TimeoutExpired:
        return {"dnf": f"did not finish: over {limit:.0f} s"}
    finally:
        if cache is not None:
            shutil.rmtree(cache)
    process_s = time.perf_counter() - t0
    tail = out.stdout.strip().splitlines()
    if out.returncode == -9:
        return {"dnf": "did not finish: killed by SIGKILL (the memory cap)", "process_s": process_s}
    if out.returncode != 0 or not tail:
        last = (out.stderr.strip().splitlines() or ["no output"])[-1]
        return {"dnf": f"rc={out.returncode}: {last[:160]}", "process_s": process_s}
    return json.loads(tail[-1]) | {"process_s": process_s}


_RUST_ROWS: dict[tuple[str, int], tuple[Path, np.ndarray, tuple, tuple]] = {}


def _rust_rows(case: Case) -> tuple[Path, np.ndarray, tuple, tuple]:
    """The case's rows as the CSV the binary reads, written once under `BENCH_DATA`, with the
    labels and the two parameter tuples (`rust_rivals.run`'s `params`)."""
    from density_rivals import kdist_params, min_points

    key = (case.dataset, case.n)
    if key not in _RUST_ROWS:
        if case.stage == "rust":
            import density_rivals as dr

            x, y = dr.load(case.dataset)[:2]
        else:
            x, y, _ = _blobs_rows(case.dataset, case.n)
        x = np.ascontiguousarray(x, dtype=np.float64)
        path = DATA / f"rust-{case.dataset}-{len(x)}.csv"
        if not path.exists():
            DATA.mkdir(parents=True, exist_ok=True)
            np.savetxt(path.with_suffix(".tmp"), x, delimiter=",", fmt="%.17g")
            path.with_suffix(".tmp").rename(path)
        kdist = kdist_params(x) if case.stage == "rust" else ()
        _RUST_ROWS[key] = (path, np.asarray(y), (min_points(len(x)),), kdist)
    return _RUST_ROWS[key]


def run_rust(case: Case, seed: int) -> dict:
    """Two runs of the binary; each prints the seconds its fit took, CSV parsing excluded."""
    import rust_rivals as rr

    path, y, mcs, kdist = _rust_rows(case)
    params = kdist if case.arm in rr.KDIST else mcs
    runs = []
    for _ in range(2):
        res = rr.run(case.arm, path, len(np.unique(y)), seed, params)
        if "error" in res:
            return {"dnf": res["error"]}
        runs.append(res)
    return {
        "cold_s": runs[0]["fit_s"],
        "warm_s": runs[1]["fit_s"],
        "vmhwm_mb": runs[1]["rss_mb"],
        "ari": _score(y, runs[1]["labels"]),
        "process_s": runs[1]["time_s"],
        "version": rr.crate_versions()[case.arm],
    }


def _carried_dnf(
    failed: dict[tuple[str, str, str], list[tuple[int, str]]], case: Case
) -> str | None:
    """At a million rows, a case that did not finish at a smaller size of the same stage is not
    run again: the two-hour limit would only be spent twice. `failed` holds every unfinished
    `(n, reason)` per `(stage, arm, dataset)`, from the CSV and from this run."""
    if case.stage != "scale":
        return None
    smaller = [
        (n, why) for n, why in failed.get((case.stage, case.arm, case.dataset), []) if n < case.n
    ]
    if not smaller:
        return None
    n, why = min(smaller)
    return f"not run: did not finish at {n} rows ({why})"


def _price(todo: list[Case]) -> None:
    """Cases, processes and an estimate per stage from the committed first-fit times (`time_s`) of
    the quality harness each stage re-times: rounds × (two fits + 3 s of start-up and control). An
    estimate on a busy machine's numbers, never a measurement; a case with no committed time at its
    size is counted apart."""
    import pandas as pd

    source = {
        "rivals": "results_rivals",
        "threads": "results_rivals",
        "density": "results_density_rivals",
        "heads": "results_head_refs",
        "rust": "results_rust_rivals",
        "gpu": "results_gpu_rivals",
    }
    times: dict[str, dict] = {}
    for stage, table in source.items():
        df = pd.read_csv(HERE / f"{table}.csv")
        times[stage] = df.groupby(["dataset", "n", "arm"])["time_s"].median().to_dict()
    ext = pd.read_csv(HERE / "results_external.csv")
    ext["dataset"] = ext["dataset"] + ext["order"].map({"label": "", "shuffled": "-shuffled"})
    times["external"] = ext.set_index(["dataset", "n", "method"])["time_s"].to_dict()
    print(f"{'stage':13s} {'cases':>5s} {'processes':>9s} {'priced':>6s} {'estimate h':>10s}")
    total = 0.0
    for stage in dict.fromkeys(c.stage for c in todo):
        mine = [c for c in todo if c.stage == stage]
        seconds, priced = 0.0, 0
        for c in mine:
            t = times.get(stage, {}).get((c.dataset, c.n, c.arm))
            if t is not None and not np.isnan(t):
                priced += 1
                seconds += c.rounds * (2 * t + 3.0)
        total += seconds
        processes = sum(c.rounds for c in mine)
        print(f"{stage:13s} {len(mine):5d} {processes:9d} {priced:6d} {seconds / 3600:10.2f}")
    print(f"priced total {total / 3600:.2f} h; unpriced cases are not in it")


def _options(argv: list[str]) -> tuple[dict[str, str], set[str]]:
    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    flags = {a[2:] for a in argv if a.startswith("--") and "=" not in a}
    known = {"stages", "arms", "skip-arms", "datasets", "n", "rounds", "threads", "csv"}
    unknown = (set(opts) - known) | (flags - {"force", "dry", "keep", "prepare", "price"})
    if unknown:
        raise SystemExit(f"unknown options: {sorted(unknown)}")
    return opts, flags


def main(argv: list[str]) -> int:
    import pandas as pd

    opts, flags = _options(argv)
    stages = [s for s in opts.get("stages", "").split(",") if s]
    unknown = sorted(set(stages) - set(STAGES))
    if not stages or unknown:
        raise SystemExit(f"name the stages (unknown: {unknown}): --stages=" + ",".join(STAGES))
    dry = "dry" in flags
    threads = int(opts.get("threads", THREADS))
    rounds_cap = 1 if dry else int(opts.get("rounds", 99))
    arms = set(opts["arms"].split(",")) if "arms" in opts else None
    skip = set(opts.get("skip-arms", "").split(","))
    datasets = set(opts["datasets"].split(",")) if "datasets" in opts else None
    sizes = {int(v) for v in opts["n"].split(",")} if "n" in opts else None
    csv = Path(opts["csv"]) if "csv" in opts else (DATA / "dry" / CSV.name if dry else CSV)
    env_json = csv.with_name(csv.stem + "_env.json")

    todo = [
        c
        for s in stages
        for c in STAGES[s]()
        if (arms is None or c.arm in arms)
        and c.arm not in skip
        and (datasets is None or c.dataset in datasets)
        and (sizes is None or c.n in sizes)
    ]
    if not todo:
        raise SystemExit("no case matches the filters")
    todo = [replace(c, rounds=min(c.rounds, rounds_cap)) for c in todo]
    if dry:
        todo = list(dict.fromkeys(replace(c, n=min(c.n, DRY_ROWS)) for c in todo))
    if "price" in flags:
        _price(todo)
        return 0
    if "out-of-core" in stages:
        rows = max(c.n for c in todo if c.stage == "out-of-core")
        free = shutil.disk_usage(DATA if DATA.exists() else DATA.parent).free
        need = rows * OOC_DIM * 4 + 10e9
        if not (DATA / f"synthetic-{rows}.npy").exists() and free < need:
            raise SystemExit(f"out-of-core needs {need / 1e9:.0f} GB free under {DATA}")
        synthetic(rows)
    if any(c.stage in ("rust", "rust-1m") for c in todo):
        import rust_rivals as rr

        rr.build()
    if "prepare" in flags:
        return 0

    ready = subprocess.run(["bash", str(READY)], capture_output=True, text=True)
    forced = "force" in flags or dry
    if ready.returncode != 0 and not forced:
        print(ready.stdout + ready.stderr)
        print("bench-ready.sh failed: not measuring. --force runs anyway, rows marked invalid.")
        return 1
    os.environ.update(_thread_env(threads))  # the Rust binaries inherit the parent's environment
    run_id = datetime.now(UTC).isoformat(timespec="seconds")
    base = control()
    env = environment(ready.stdout) | {
        "stages": stages,
        "filters": {
            k: opts[k] for k in ("arms", "skip-arms", "datasets", "n", "rounds") if k in opts
        },
        "dry": dry,
        "threads_given": threads,
        "control_base_s": base,
    }
    stored = json.loads(env_json.read_text()) if env_json.exists() else {}
    stored[run_id] = env
    csv.parent.mkdir(parents=True, exist_ok=True)
    env_json.write_text(json.dumps(stored, indent=2) + "\n")
    print(
        f"{len(todo)} cases, {sum(c.rounds for c in todo)} processes; bench-ready "
        f"rc={ready.returncode}; {threads} threads; rows to {csv}",
        flush=True,
    )

    # Kept rows pass through as text, as in `bench/rivals.py`: pandas parses some floats one ulp
    # off, and a re-run of one case would then rewrite every other case's rows.
    kept = pd.read_csv(csv, dtype=str, keep_default_na=False) if csv.exists() else pd.DataFrame()
    replaced: set[tuple[str, str, str, str]] = set()
    rows: list[dict] = []
    failed: dict[tuple[str, str, str], list[tuple[int, str]]] = {}
    for r in kept.to_dict("records") if len(kept) else []:
        if r["dnf"]:
            failed.setdefault((r["stage"], r["arm"], r["dataset"]), []).append(
                (int(r["n"]), r["dnf"])
            )
    for rnd in range(max(c.rounds for c in todo)):
        order = [c for c in todo if c.rounds > rnd]
        random.Random(rnd).shuffle(order)
        for case in order:
            threads_here = case.threads or threads
            before, gpu_before = cpu_temp(), gpu_state() if case.stage == "gpu" else None
            carried = _carried_dnf(failed, case)
            res = {"dnf": carried} if carried else run_case(case, rnd, threads_here)
            after, gpu_after = cpu_temp(), gpu_state() if case.stage == "gpu" else None
            drift = control() / base - 1
            row = {
                "stage": case.stage,
                "dataset": case.dataset,
                "n": case.n,
                "arm": case.arm,
                "round": rnd,
                "seed": rnd,
                "threads": threads_here,
                "control_drift": drift,
                "temp_before": before,
                "temp_after": after,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "timing_valid": ready.returncode == 0 and not dry and abs(drift) <= DRIFT,
                "note": "dry run" if dry else None,
                "run": run_id,
            }
            if case.stage == "numba-cold":
                row["note"] = "; ".join(filter(None, (row["note"], "empty numba cache")))
            if case.arm in NUMBA_ONE_THREAD:
                row["note"] = "; ".join(filter(None, (row["note"], NUMBA_ONE_THREAD[case.arm])))
            if res.get("dnf"):
                key = (case.stage, case.arm, case.dataset)
                failed.setdefault(key, []).append((case.n, res["dnf"]))
            for part in res.pop("parts", None) or [{}]:
                new = row | res | part
                notes = (row["note"], res.get("note"), part.get("note"))
                new["note"] = "; ".join(dict.fromkeys(filter(None, notes))) or None
                extra = set(new) - set(COLUMNS)
                if extra:
                    raise RuntimeError(f"{case}: columns {sorted(extra)} are not in COLUMNS")
                rows.append(new)
                replaced.add((case.stage, new["arm"], case.dataset, str(case.n)))
                _show(rnd, new, drift)
            replaced.add((case.stage, case.arm, case.dataset, str(case.n)))
            if len(kept):
                key = zip(kept["stage"], kept["arm"], kept["dataset"], kept["n"], strict=True)
                kept = kept[[k not in replaced for k in key]]
            table = pd.DataFrame(rows).reindex(columns=list(COLUMNS))
            pd.concat([kept, table]).to_csv(csv, index=False)

    stored = json.loads(env_json.read_text())
    stored[run_id] |= {
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "temp_end": cpu_temp(),
        "bench_ready_after": subprocess.run(
            ["bash", str(READY)], capture_output=True, text=True
        ).stdout,
    }
    env_json.write_text(json.dumps(stored, indent=2) + "\n")
    if "out-of-core" in stages and "keep" not in flags:
        for p in DATA.glob("synthetic-*"):
            p.unlink()
    return 0


def _show(rnd: int, row: dict, drift: float) -> None:
    def num(v, unit):
        return "n/a" if v is None else f"{v:.2f} {unit}"

    if row.get("dnf"):
        shown = row["dnf"]
    else:
        score = row.get("ari") if row.get("ari") is not None else row.get("roc_auc")
        shown = f"cold {num(row.get('cold_s'), 's')}  warm {num(row.get('warm_s'), 's')}"
        shown += f"  score {num(score, '')}  VmHWM {num(row.get('vmhwm_mb'), 'MB')}"
        if row.get("n_init"):
            shown += f"  n_init={row['n_init']}"
        if row.get("seen"):
            shown += f"  seen {row['seen']}"
    print(
        f"r{rnd} {row['stage']:12s} {row['dataset']:16s} {row['n']:<8} {row['arm']:42.42s} "
        f"t={row['threads']} {shown}  control {drift:+.1%}",
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, stage, arm, dataset, n, seed = sys.argv
        print(json.dumps(child(Case(stage, arm, dataset, int(n)), int(seed))))
    else:
        sys.exit(main(sys.argv[1:]))
