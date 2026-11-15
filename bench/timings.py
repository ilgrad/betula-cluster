"""Wall time for every rival arm on a quiet machine: BL-2, BL-3 and the GPU table's time half.

The quality harnesses record a time beside every score, but on whatever machine was at hand, one arm
after another, with a first call that pays for imports, numba's compilation and a CUDA context. This
harness re-times their arms under the bench's protocol and records times, peak memory and one ARI
as a check that the work done is the work scored elsewhere:

- It refuses to start unless `~/.claude/scripts/bench-ready.sh` passes. `--force` runs anyway, for a
  smoke test, and marks every row `timing_valid=False`.
- One process per case (an arm on a dataset). It loads the rows, then fits twice: `cold_s` is the
  first fit, with imports, JIT compilation and device set-up, and `warm_s` the second. Both are
  kept: a user who fits once pays the first. At a million rows the set-up is noise against a fit
  of minutes, so `scale` fits once.
- Cases run in rounds. Each round shuffles them with its own seed, so drift during the campaign
  spreads over every arm instead of landing on the last ones. The round is the seed.
- A control, a fixed NumPy SVD that no arm can change, runs once before the first case and after
  each case. `control_drift` is its change against the campaign's first reading; a row whose
  control moved by more than `DRIFT` is `timing_valid=False`.
- Each row records the CPU temperature before and after; GPU rows also record the device's clock,
  temperature and power draw. The run writes its environment (governor, power source, thread
  counts, package versions, uptime, the `bench-ready.sh` report) to a JSON beside the CSV.

Stages (`--stages=a,b`); each runs in the environment its rivals need, listed in the module the
arms come from:

| stage | arms | task |
|---|---|---|
| `rivals` | `rivals.py`: MNIST 20 000 and 70 000 rows, news-MiniLM | IN-11; BL-2 at 70 000 rows |
| `density` | `density_rivals.py` | IN-12, IN-14 |
| `heads` | `head_references.py` | IN-13 |
| `stream` | `stream_vs_refit.py`'s arms, each checkpoint timed: a `partial_fit` step against a refit | IN-17 |
| `scale` | betula, UMAP + k-means, UMAP + HDBSCAN, EVoC, scikit-learn k-means at 1 000 000 rows | BL-2 |
| `out-of-core` | betula `partial_fit` over a memory-mapped file larger than this machine's RAM | BL-2 |
| `rust` | `rust_rivals.py`'s binary; the time is taken inside it, around the fit | IN-18 |
| `gpu` | `gpu_rivals.py`, run with an interpreter that has cuML | IN-19 |

`scale` and `out-of-core` read a synthetic set: no real embedding set of a million rows is cached
here. Rows are unit vectors in 384 dimensions (MiniLM's width) drawn around 50 random directions,
written once to `BENCH_DATA` (default `~/.cache/betula-bench`, never `/tmp`, a RAM-backed tmpfs on
this machine). The out-of-core file holds 28 000 000 rows, 43 GB; its stage refuses to start when
the disk has less than that free plus 10 GB, and deletes the file afterwards unless `--keep`.

    uv run --no-sync --with scikit-learn --with pandas [the stage's --with list] \\
        python bench/timings.py --stages=rivals [--rounds=3] [--arms=a,b] [--datasets=a,b]

Writes `bench/results_timings.csv`: rows from a run replace the earlier rows of the same stage, arm,
dataset and size, so stages run in separate environments add up to one table.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CSV = HERE / "results_timings.csv"
ENV_JSON = HERE / "results_timings_env.json"
READY = Path.home() / ".claude/scripts/bench-ready.sh"
DATA = Path(os.environ.get("BENCH_DATA", Path.home() / ".cache/betula-bench"))
DRIFT = 0.05
TIMEOUT = {"scale": 7200.0, "out-of-core": 7200.0}
TIMEOUT_DEFAULT = 3600.0

SCALE_ROWS, SCALE_K, SCALE_LEAVES = 1_000_000, 50, 16_000
OOC_ROWS, OOC_CHUNK = 28_000_000, 200_000
DIM = 384


@dataclass(frozen=True)
class Case:
    stage: str
    arm: str
    dataset: str
    n: int


# ── data for the scale stages ────────────────────────────────────────────────────────────────


def synthetic(rows: int) -> tuple[Path, Path]:
    """A float32 `.npy` of `rows` unit vectors around `SCALE_K` directions and its labels, written
    once in chunks so that no step holds more than one chunk."""
    DATA.mkdir(parents=True, exist_ok=True)
    xp, yp = DATA / f"synthetic-{rows}.npy", DATA / f"synthetic-{rows}-labels.npy"
    if xp.exists() and yp.exists():
        return xp, yp
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((SCALE_K, DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    x = np.lib.format.open_memmap(xp.with_suffix(".tmp"), "w+", np.float32, (rows, DIM))
    y = rng.integers(0, SCALE_K, rows).astype(np.int32)
    for a in range(0, rows, OOC_CHUNK):
        b = min(rows, a + OOC_CHUNK)
        block = centres[y[a:b]] + 0.06 * rng.standard_normal((b - a, DIM), dtype=np.float32)
        x[a:b] = block / np.linalg.norm(block, axis=1, keepdims=True)
    x.flush()
    del x
    np.save(yp, y)
    xp.with_suffix(".tmp").rename(xp)
    return xp, yp


# ── what each stage runs ─────────────────────────────────────────────────────────────────────


def _scale_arms():
    import rivals

    def betula(**kw):
        def fit(x, k, seed):
            import betula_cluster as bc

            est = bc.Betula(max_leaves=SCALE_LEAVES, threshold=0.0, seed=seed, n_clusters=k, **kw)
            return est.fit_predict(x)

        return fit

    def umap_then(cluster):
        def fit(x, k, seed):
            return cluster(rivals._umap(x, seed, "news-minilm"), k, seed)

        return fit

    return {
        "betula-kmeans": betula(method="kmeans"),
        "betula-vmf-shared": betula(method="vmf", kappa="shared"),
        "betula-leiden-rows-auto": betula(method="leiden", leaf_graph="rows", resolution="auto"),
        "betula-hdbscan": betula(method="hdbscan"),
        "sklearn-kmeans": lambda x, k, seed: rivals._kmeans(x, k, seed),
        "umap+kmeans": umap_then(rivals._kmeans),
        "bertopic default (umap+hdbscan mcs=10)": umap_then(
            lambda z, k, seed: rivals._hdbscan(z, min_cluster_size=10)
        ),
        "evoc": lambda x, k, seed: rivals._evoc(x, seed),
    }


def cases(stage: str) -> list[Case]:
    if stage == "rivals":
        import rivals

        plan = [("mnist", 20_000), ("news-minilm", 17_901), ("mnist", 70_000)]
        return [
            Case(stage, arm, d, n)
            for d, n in plan
            for arm, (_, where, _) in rivals.ARMS.items()
            if where in ("both", d)
        ]
    if stage == "density":
        import density_rivals as dr

        return [Case(stage, arm, d, dr.N) for d in dr.DATASETS for arm in dr.ARMS]
    if stage == "heads":
        import head_references as hr

        return [Case(stage, arm, d, len(hr.load(d)[0])) for arm, (d, _) in hr.ARMS.items()]
    if stage == "stream":
        return [
            Case(stage, arm, "news-minilm", 17_901)
            for arm in (
                "betula-stream",
                "betula-refit",
                "sklearn-kmeans",
                "umap+kmeans",
                "umap+hdbscan",
                "evoc",
            )
        ]
    if stage == "scale":
        return [Case(stage, arm, "synthetic-384", SCALE_ROWS) for arm in _scale_arms()]
    if stage == "out-of-core":
        return [Case(stage, "betula-vmf-shared", "synthetic-384", OOC_ROWS)]
    if stage == "rust":
        import density_rivals as dr
        import rust_rivals as rr

        return [Case(stage, arm, d, dr.N) for d in dr.DATASETS for arm in rr.ARMS]
    if stage == "gpu":
        import gpu_rivals as g

        return [Case(stage, arm, d, n) for d, n in g.PLAN for arm in g.ARMS]
    raise SystemExit(f"unknown stage {stage}")


def _prepare(case: Case, seed: int):
    """`(truth, fit)` for one case: the rows are loaded here, outside every timed call."""
    if case.stage == "rivals":
        import rivals

        x, y = rivals.load(case.dataset, case.n, seed)
        k = len(np.unique(y))
        return y, lambda: rivals.ARMS[case.arm][2](x, k, seed, case.dataset, case.n)
    if case.stage == "density":
        import density_rivals as dr

        x, y, k = dr.load(case.dataset)
        x = np.ascontiguousarray(x, dtype=np.float64)
        return np.asarray(y), lambda: dr.ARMS[case.arm][1](x, k, seed)
    if case.stage == "heads":
        import head_references as hr

        x, y, extra = hr.load(case.dataset)
        k = len(np.unique(y))
        return np.asarray(y), lambda: hr.ARMS[case.arm][1](x, k, seed, extra)
    if case.stage == "scale":
        xp, yp = synthetic(case.n)
        x, y = np.load(xp), np.load(yp)
        fit = _scale_arms()[case.arm]
        return y, lambda: fit(x, SCALE_K, seed)
    if case.stage == "gpu":
        import gpu_rivals as g

        x, y = g.load(case.dataset, case.n, seed)
        k = len(np.unique(y))
        return y, lambda: g._as_host(g.ARMS[case.arm][1](x, k, seed, case.dataset))
    raise ValueError(case.stage)


# ── the child: one case, two fits ────────────────────────────────────────────────────────────


def _ari(y, labels) -> float:
    from sklearn.metrics import adjusted_rand_score

    labels = np.asarray(labels)
    return float(adjusted_rand_score(y, np.where(labels < 0, labels.max() + 1, labels)))


def child(case: Case, seed: int) -> dict:
    from _worker import Peak

    if case.stage == "stream":
        return child_stream(case.arm, seed)
    if case.stage == "out-of-core":
        return child_out_of_core(seed)
    y, fit = _prepare(case, seed)
    device = None
    if case.stage == "gpu":
        import gpu_rivals as g

        device = g.DevicePeak()
    peak = Peak()
    times = []
    for _ in range(1 if case.stage == "scale" else 2):
        t0 = time.perf_counter()
        labels = fit()
        times.append(time.perf_counter() - t0)
    out = {"cold_s": times[0], "warm_s": times[-1] if len(times) > 1 else None}
    out |= {"rss_mb": peak.mb(), "ari": _ari(y, labels)}
    if device is not None:
        out["device_mb"] = device.mb()
    return out


def child_stream(arm: str, seed: int) -> dict:
    """Every checkpoint of `stream_vs_refit.py`, the stream run twice in this process. The betula
    stream's time at a checkpoint is its `partial_fit` on the new chunk, the global step and a
    `predict` of every row seen; every other arm refits on every row seen."""
    import stream_vs_refit as sr

    x, y = sr.cached()
    x = np.ascontiguousarray(x, dtype=np.float64)
    perm = np.random.default_rng(2000 + seed).permutation(len(x))
    x, y = x[perm], y[perm]
    cuts = [int(f * len(x)) for f in sr.FRACTIONS]

    def vmf():
        import betula_cluster as bc

        return bc.Betula(
            method="vmf",
            kappa="shared",
            n_clusters=sr.K,
            max_leaves=sr.LEAVES,
            threshold=0.0,
            seed=seed,
        )

    def refit(xs):
        from sklearn.cluster import KMeans

        if arm == "betula-refit":
            return vmf().fit_predict(xs)
        if arm == "sklearn-kmeans":
            return KMeans(sr.K, random_state=seed).fit_predict(xs)
        if arm == "evoc":
            import evoc

            return evoc.EVoC(random_state=seed).fit_predict(xs)
        import hdbscan
        import umap

        z = umap.UMAP(
            n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=seed
        ).fit_transform(xs)
        if arm == "umap+kmeans":
            return KMeans(sr.K, random_state=seed).fit_predict(z)
        return hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(z)

    passes = []
    for _ in range(2):
        times, est, start = [], vmf(), 0
        for cp in cuts:
            t0 = time.perf_counter()
            if arm == "betula-stream":
                est.partial_fit(x[start:cp])
                est.partial_fit()
                labels = est.predict(x[:cp])
                start = cp
            else:
                labels = refit(x[:cp])
            times.append(time.perf_counter() - t0)
        passes.append(times)
    return {
        "checkpoints": [
            {"seen": cp, "cold_s": c, "warm_s": w}
            for cp, c, w in zip(cuts, passes[0], passes[1], strict=True)
        ],
        "ari": _ari(y, labels),
    }


def child_out_of_core(seed: int) -> dict:
    """`partial_fit` in chunks over a memory-mapped file larger than RAM. Peak memory is the
    process's anonymous RSS: the mapped file's pages are page cache, which the kernel reclaims, and
    would otherwise count up to whatever RAM is free."""
    import threading

    import betula_cluster as bc

    xp, yp = synthetic(OOC_ROWS)
    x = np.load(xp, mmap_mode="r")
    peak = {"anon": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("RssAnon:"):
                    peak["anon"] = max(peak["anon"], int(line.split()[1]))
            stop.wait(0.05)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    est = bc.Betula(
        method="vmf",
        kappa="shared",
        n_clusters=SCALE_K,
        max_leaves=SCALE_LEAVES,
        threshold=0.0,
        seed=seed,
    )
    t0 = time.perf_counter()
    for a in range(0, len(x), OOC_CHUNK):
        est.partial_fit(np.ascontiguousarray(x[a : a + OOC_CHUNK]))
    t_insert = time.perf_counter() - t0
    est.partial_fit()
    t_total = time.perf_counter() - t0
    stop.set()
    sampler.join()
    probe = np.ascontiguousarray(x[:200_000])
    return {
        "cold_s": t_total,
        "insert_s": t_insert,
        "rss_mb": peak["anon"] / 1024,
        "file_gb": xp.stat().st_size / 1e9,
        "ari": _ari(np.load(yp, mmap_mode="r")[:200_000], est.predict(probe)),
    }


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
        "hdbscan",
        "evoc",
        "fast-hdbscan",
        "leidenalg",
        "pynndescent",
        "numba",
        "cuml-cu12",
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


def run_case(case: Case, seed: int) -> dict:
    if case.stage == "rust":
        return run_rust(case, seed)
    cmd = [
        sys.executable,
        str(Path(__file__)),
        "--child",
        case.stage,
        case.arm,
        case.dataset,
        str(case.n),
        str(seed),
    ]
    limit = TIMEOUT.get(case.stage, TIMEOUT_DEFAULT)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired:
        return {"dnf": f"timeout {limit:.0f} s"}
    tail = out.stdout.strip().splitlines()
    if out.returncode != 0 or not tail:
        last = (out.stderr.strip().splitlines() or ["no output"])[-1]
        return {"dnf": f"rc={out.returncode}: {last[:160]}"}
    return json.loads(tail[-1])


_RUST_FIXTURES: dict[str, Path] = {}


def run_rust(case: Case, seed: int) -> dict:
    """Two runs of the binary; each prints the seconds its fit took, CSV parsing excluded."""
    import density_rivals as dr
    import rust_rivals as rr

    if case.dataset not in _RUST_FIXTURES:
        x, y = dr.load(case.dataset)[:2]
        path = DATA / f"rust-{case.dataset}.csv"
        DATA.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, x, delimiter=",", fmt="%.17g")
        _RUST_FIXTURES[case.dataset] = path
        _RUST_FIXTURES[case.dataset + ":y"] = path.with_suffix(".labels.npy")
        np.save(_RUST_FIXTURES[case.dataset + ":y"], np.asarray(y))
    path = _RUST_FIXTURES[case.dataset]
    y = np.load(_RUST_FIXTURES[case.dataset + ":y"])
    k, mcs = len(np.unique(y)), dr.min_points(len(y))
    runs = [rr.run(case.arm, path, k, seed, mcs) for _ in range(2)]
    if "error" in runs[-1]:
        return {"dnf": runs[-1]["error"]}
    return {
        "cold_s": runs[0]["fit_s"],
        "warm_s": runs[1]["fit_s"],
        "rss_mb": runs[1]["rss_mb"],
        "ari": _ari(y, runs[1]["labels"]),
    }


def main(argv: list[str]) -> int:
    import pandas as pd

    opts = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    flags = {a for a in argv if a.startswith("--") and "=" not in a}
    stages = opts.get("stages", "").split(",")
    if stages == [""]:
        raise SystemExit(
            "name the stages: --stages="
            + ",".join(
                ["rivals", "density", "heads", "stream", "scale", "out-of-core", "rust", "gpu"]
            )
        )
    rounds = int(opts.get("rounds", 3))
    arms = set(opts["arms"].split(",")) if "arms" in opts else None
    datasets = set(opts["datasets"].split(",")) if "datasets" in opts else None

    ready = subprocess.run(["bash", str(READY)], capture_output=True, text=True)
    if ready.returncode != 0 and "--force" not in flags:
        print(ready.stdout + ready.stderr)
        print("bench-ready.sh failed: not measuring. --force runs anyway, rows marked invalid.")
        return 1
    if "out-of-core" in stages:
        free = os.statvfs(DATA if DATA.exists() else DATA.parent)
        need = OOC_ROWS * DIM * 4 + 10e9
        built = (DATA / f"synthetic-{OOC_ROWS}.npy").exists()
        if not built and free.f_bavail * free.f_frsize < need:
            raise SystemExit(f"out-of-core needs {need / 1e9:.0f} GB free under {DATA}")
    if "rust" in stages:
        import rust_rivals as rr

        rr.build()

    todo = [
        c
        for s in stages
        for c in cases(s)
        if (arms is None or c.arm in arms) and (datasets is None or c.dataset in datasets)
    ]
    if not todo:
        raise SystemExit("no case matches the filters")
    for c in todo:
        if c.stage in ("scale", "out-of-core"):
            synthetic(c.n)
    env = environment(ready.stdout)
    env["stages"], env["rounds"] = stages, rounds
    print(f"{len(todo)} cases x {rounds} rounds; bench-ready rc={ready.returncode}", flush=True)

    base = control()
    rows = []
    kept = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame()
    if len(kept):
        done = {(c.stage, c.arm, c.dataset, c.n) for c in todo}
        key = zip(kept["stage"], kept["arm"], kept["dataset"], kept["n"], strict=True)
        kept = kept[[k not in done for k in key]]
    for rnd in range(rounds):
        order = todo[:]
        random.Random(rnd).shuffle(order)
        for case in order:
            before, gpu_before = cpu_temp(), gpu_state() if case.stage == "gpu" else None
            res = run_case(case, rnd)
            after, gpu_after = cpu_temp(), gpu_state() if case.stage == "gpu" else None
            drift = control() / base - 1
            row = {
                "stage": case.stage,
                "dataset": case.dataset,
                "n": case.n,
                "arm": case.arm,
                "round": rnd,
                "seed": rnd,
            }
            common = {
                "control_drift": drift,
                "temp_before": before,
                "temp_after": after,
                "gpu_before": gpu_before,
                "gpu_after": gpu_after,
                "timing_valid": ready.returncode == 0 and abs(drift) <= DRIFT,
            }
            for cp in res.pop("checkpoints", [None]):
                rows.append(
                    row
                    | (
                        {"seen": cp["seen"], "cold_s": cp["cold_s"], "warm_s": cp["warm_s"]}
                        if cp
                        else {}
                    )
                    | res
                    | common
                )
            last = rows[-1]  # the full stream, for the stream stage
            cold, warm = (
                "     n/a" if last.get(c) is None else f"{last[c]:8.2f}"
                for c in ("cold_s", "warm_s")
            )
            shown = res.get("dnf") or f"cold {cold} s  warm {warm} s  ARI {res['ari']:.3f}"
            print(
                f"r{rnd} {case.stage:11s} {case.dataset:13s} {case.n:<8} {case.arm:40.40s} {shown}  "
                f"control {drift:+.1%}",
                flush=True,
            )
            pd.concat([kept, pd.DataFrame(rows)]).to_csv(CSV, index=False)
    env["bench_ready_after"] = subprocess.run(
        ["bash", str(READY)], capture_output=True, text=True
    ).stdout
    stored = json.loads(ENV_JSON.read_text()) if ENV_JSON.exists() else {}
    stored[",".join(stages)] = env
    ENV_JSON.write_text(json.dumps(stored, indent=2) + "\n")
    if "out-of-core" in stages and "--keep" not in flags:
        for p in DATA.glob(f"synthetic-{OOC_ROWS}*"):
            p.unlink()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, stage, arm, dataset, n, seed = sys.argv
        print(json.dumps(child(Case(stage, arm, dataset, int(n)), int(seed))))
    else:
        sys.exit(main(sys.argv[1:]))
