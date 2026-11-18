"""Where a large build spends its instructions: the profile F-19 takes at 10 M rows.

Three items of the plan are decided on it: a structure-of-arrays tree (SP-7), batched routing (P2)
and the follow-up of F-34. A wall time cannot split a fit between its phases; an instruction count
by function can. Two instruments, because each one fails where the other one works:

- `record`: `perf record` of retired user-space instructions (`instructions:u`) with frame-pointer
  call chains, sampling only inside the fit: perf starts with its counter off (`--delay=-1`), and
  the fit's own process turns it on and off around the call through perf's control FIFO, so every
  sample is the fit's, one per `PERIOD` instructions. On this CPU a sample skids into the callee
  (4.18 G instructions sampled in a kernel whose exact count is 2.81 G), so the share of a short
  function is read from the second instrument.
- `callgrind`: the same fit on a reduced input under valgrind's callgrind: an exact count per
  function. LTO inlines `descend` into `insert`, and callgrind sees no frame for inlined code, so
  its `insert` row holds the descent loop; perf's chains carry inlined frames and split the two.
  It also writes the per-line counts of `src/` (`build-<n>-<shards>.lines.txt`).

Both tables group the fit's instructions by phase. A count goes to the innermost frame that names
a phase, walking out from the function that retired it: a distance kernel, a sort, rayon's
scheduling or (under perf) a `malloc` belongs to the phase that called it. perf walks each
sample's call chain; callgrind, which keeps calls and not chains, splits an extension function's
exclusive count across its callers in the extension in proportion to their calls' counts.
Exclusive counts, not inclusive ones: an inclusive count says nothing across threads (rayon runs a
job on a worker whose stack does not hold the caller) or across recursion (it counts a nested call
twice). A phase's pattern is matched against demangled names; a kernel generic over its caller
names the caller, so `nearest_sq_euclidean_f64::<…nearest_child…>` is descent even where the call
graph has no frame between.

`build` makes the extension both read: a wheel built by maturin with frame pointers, line tables and
the inline hint the published wheels carry, unpacked under `$BENCH_DATA/profile/pkg`. The other two
steps import it from there and refuse to run on any other build of the module.

    uv run --no-sync --with maturin python bench/profile_build.py build
    PYTHONPATH=$BENCH_DATA/profile/pkg uv run --no-sync python bench/profile_build.py record \\
        [--n=10000000] [--shards=seq,8]
    PYTHONPATH=$BENCH_DATA/profile/pkg uv run --no-sync python bench/profile_build.py callgrind \\
        [--n=200000] [--shards=seq,8]

The rows are `_worker.gen("highdim", n)`, 20 dimensions and 8 clusters, the fixture of the k-means
contest of `bench/external_baselines.py`. The fit is `fit_predict(method="kmeans")` at its other
defaults: `seq` is `n_shards=None`, the sequential build every estimator fit runs, and `8` the
shard-and-merge build, the only one with a merge phase. `record` runs at the environment's thread
count (`RAYON_NUM_THREADS`), `callgrind` on one. Writes under `$BENCH_DATA/profile/`, prints the
phase tables, and changes nothing in the repository.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path(os.environ.get("BENCH_DATA", Path.home() / ".cache/betula-bench")) / "profile"
PKG = DATA / "pkg"
EXTENSION = "_core.abi3.so"
# One perf sample per this many instructions: about 2 500 samples for the 10 M-row fit, since
# resolving a sample's inlined frames is slow and a phase of 10 % needs no more to read +-1.5 %.
PERIOD = 20_000_000

# The phases a frame can name, first match wins. A frame that names none (a kernel, rayon, std,
# alloc, libc) passes its count to its caller.
PHASES = (
    ("shard merge", r"build_sharded|(^|::)(descend_cf|insert_cf)\b"),
    (
        "rebuilds",
        r"::(rebuild|reinsert|refold_node_cfs|recompute_cf|reaccumulate|fold_balanced"
        r"|collect_from|drop_merged|push_cheapest_partners)\b",
    ),
    ("descent: to the leaf", r"(^|::)(descend|nearest_(child|entry|entries|in_leaf))\b"),
    ("CF update: absorb, split", r"::(try_absorb|absorb|split_up)\b"),
    ("insert: descent loop and update, inlined", r"CFTree<.*>>::insert\b"),
    ("labelling: rows to leaves and centres", r"route_block|label_rows|::route\b|nearest_center"),
    ("head: k-means on the leaves", r"clustering::kmeans::"),
    ("the rest of the fit", r"betula_cluster::python::(build_tree|cluster|map_rows|to_rows|run_)"),
)
UNNAMED = "no phase in its call chain"


def _phase(name: str) -> str | None:
    return next((p for p, pattern in PHASES if re.search(pattern, name)), None)


def _name(raw: str) -> str:
    """A demangled name without its hash or callgrind's recursion mark."""
    return re.sub(r"::h[0-9a-f]{16}|'\d+$| \(inlined\)$", "", raw.strip())


def _options(argv: list[str]) -> dict[str, str]:
    return dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)


def build() -> None:
    """The extension with frame pointers and line tables, built apart from every other target."""
    target, wheels = DATA / "target", DATA / "wheels"
    shutil.rmtree(wheels, ignore_errors=True)
    flags = "-C force-frame-pointers=yes -C llvm-args=--inline-threshold=1000"
    if shutil.which("mold"):
        flags += " -C link-arg=-fuse-ld=mold"
    env = os.environ | {
        "RUSTFLAGS": flags,
        "CARGO_PROFILE_RELEASE_DEBUG": "line-tables-only",
        "CARGO_TARGET_DIR": str(target),
    }
    cmd = ["maturin", "build", "--release", "--out", str(wheels)]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)  # maturin reads the manifest of its cwd
    (wheel,) = wheels.glob("*.whl")
    shutil.rmtree(PKG, ignore_errors=True)
    with zipfile.ZipFile(wheel) as z:
        z.extractall(PKG)
    (PKG / "BUILD").write_text(f"{wheel.name}\nRUSTFLAGS={flags}\n")
    print(f"built {wheel.name} into {PKG}")


def _check_build() -> None:
    import betula_cluster

    where = Path(betula_cluster.__file__).resolve()
    if PKG.resolve() not in where.parents:
        raise SystemExit(f"betula_cluster comes from {where}, not {PKG}: run with PYTHONPATH={PKG}")


def fit(n: int, shards: str, ctl: str | None, ack: str | None) -> None:
    """One fit; with `ctl`, perf's counter is on for the call and off around it."""
    sys.path.insert(0, str(HERE))
    import betula_cluster as bc
    from _worker import gen

    x, k = gen("highdim", n)
    n_shards = None if shards == "seq" else int(shards)

    def control(command: str) -> None:
        if ctl is None or ack is None:
            return
        with open(ctl, "w") as c:
            c.write(command + "\n")
        with open(ack) as a:
            a.readline()

    control("enable")
    labels = bc.fit_predict(x, k, method="kmeans", threshold=0.0, seed=0, n_shards=n_shards)
    control("disable")
    print(f"fitted {n} rows, shards={shards}, {len(set(labels.tolist()))} clusters")


def _phase_table(by_phase: dict[str, float], unit: str) -> None:
    whole = sum(by_phase.values())
    print(f"{'phase':42s} {unit:>16s} {'of the fit':>10s}")
    for phase in [p for p, _ in PHASES] + [UNNAMED]:
        count = by_phase.get(phase, 0.0)
        print(f"{phase:42s} {count:16,.6g} {count / whole:10.1%}")
    print(f"{'the fit':42s} {whole:16,.6g}")


def _top(selfs: dict[str, float], unit: str) -> None:
    print(f"  {unit:>16s}  function (exclusive)")
    for name, count in sorted(selfs.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {count:16,.6g}  {name[:160]}")


def record(n: int, shards: list[str]) -> None:
    """`perf record` of each fit, then the extension's samples by phase and by function."""
    _check_build()
    DATA.mkdir(parents=True, exist_ok=True)
    for s in shards:
        out = DATA / f"build-{n}-{s}.perf.data"
        ctl, ack = DATA / "perf.ctl", DATA / "perf.ack"
        for fifo in (ctl, ack):
            fifo.unlink(missing_ok=True)
            os.mkfifo(fifo)
        cmd = ["perf", "record", "-e", "instructions:u", "-c", str(PERIOD), "--call-graph", "fp"]
        cmd += ["--delay=-1"]
        cmd += [f"--control=fifo:{ctl},{ack}", "-o", str(out), "--", sys.executable]
        cmd += [__file__, "fit", f"--n={n}", f"--shards={s}", f"--ctl={ctl}", f"--ack={ack}"]
        subprocess.run(cmd, check=True)
        by_phase, selfs = _perf_samples(out)
        print(f"\n== perf, {n} rows, shards={s}: instructions:u samples of the fit, by period")
        _phase_table(by_phase, "events")
        _top(selfs, "events")


def _perf_samples(data: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Each sample's period to the innermost frame of its chain that names a phase."""
    cmd = ["perf", "script", "-i", str(data), "-F", "period,ip,sym"]
    by_phase: dict[str, float] = {}
    selfs: dict[str, float] = {}

    def close(period: float, frames: list[str]) -> None:
        if not frames:
            return
        phase = next(filter(None, map(_phase, frames)), UNNAMED)
        by_phase[phase] = by_phase.get(phase, 0.0) + period
        # The innermost Rust frame: an inlined SIMD intrinsic names no function of the crate.
        leaf = next((f for f in frames if "::" in f), frames[0])
        selfs[leaf] = selfs.get(leaf, 0.0) + period

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        assert proc.stdout is not None
        period, frames = 0.0, []
        for line in proc.stdout:
            head = re.match(r"\s*(\d+)\s*$", line)
            if head:
                close(period, frames)
                period, frames = float(head.group(1)), []
                continue
            frame = re.match(r"\s*[0-9a-f]+\s+(.*?)(?:\+0x[0-9a-f]+)?\s*$", line)
            if frame:
                frames.append(_name(frame.group(1)))
        close(period, frames)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return by_phase, selfs


def callgrind(n: int, shards: list[str]) -> None:
    """Exact exclusive counts per function, one fit per shard setting, on one thread."""
    _check_build()
    DATA.mkdir(parents=True, exist_ok=True)
    for s in shards:
        out = DATA / f"build-{n}-{s}.callgrind"
        env = os.environ | {"RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        cmd = ["valgrind", "--tool=callgrind", f"--callgrind-out-file={out}", sys.executable]
        cmd += [__file__, "fit", f"--n={n}", f"--shards={s}"]
        subprocess.run(cmd, check=True, env=env)
        by_phase, selfs = _callgrind_counts(out)
        print(f"\n== callgrind, {n} rows, shards={s}: the fit's instructions (exact)")
        _phase_table(by_phase, "instructions")
        _top(selfs, "instructions")
        lines = DATA / f"build-{n}-{s}.lines.txt"
        annotate = ["callgrind_annotate", "--auto=yes", "--context=0", "--threshold=99", str(out)]
        with lines.open("w") as f:
            subprocess.run(annotate, cwd=ROOT, stdout=f, check=True)
        print(f"per-line counts of src/: {lines}")


def _callgrind_counts(out: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Each function's exclusive count to the phase its name gives, or else to its callers in
    proportion to their calls' inclusive counts, recursively. Only the extension's functions are
    counted: the interpreter's start-up and the data generation are not the fit."""
    cmd = ["callgrind_annotate", "--inclusive=no", "--tree=caller", "--threshold=100", str(out)]
    report = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    selfs: dict[str, float] = {}
    callers: dict[str, dict[str, float]] = {}
    pending: dict[str, float] = {}
    in_extension: set[str] = set()
    row = re.compile(
        r"\s*([\d,]+)\s+\(\s*[\d.]+%\)\s+([<*])\s+(?:[^:\s]*:)?(.*?)"
        r"(?:\s+\([\d,]+x\))?(?:\s+\[(\S+)\])?$"
    )
    for line in report.splitlines():
        m = row.match(line)
        if not m:
            pending = {}
            continue
        count, mark, name = float(m.group(1).replace(",", "")), m.group(2), _name(m.group(3))
        if mark == "<":
            pending[name] = pending.get(name, 0.0) + count
            continue
        selfs[name] = selfs.get(name, 0.0) + count
        mine = callers.setdefault(name, {})  # one function, listed once per source file it spans
        for caller, cost in pending.items():
            mine[caller] = mine.get(caller, 0.0) + cost
        # A function's code inlined from another source file is listed under that file, and
        # without its object; a Rust path (`::`) is the extension's, the interpreter's C is not.
        if (m.group(4) or "").endswith(EXTENSION) or (m.group(4) is None and "::" in name):
            in_extension.add(name)
        pending = {}

    memo: dict[str, dict[str, float]] = {}

    def phase_of(name: str) -> dict[str, float]:
        """The share of `name`'s count each phase gets; a caller outside the extension, or a call
        cycle, names none."""
        if name in memo:
            return memo[name]
        memo[name] = {UNNAMED: 1.0}  # what a cycle back to `name` reads
        own = _phase(name)
        up = {c: v for c, v in callers.get(name, {}).items() if c in in_extension}
        total = sum(up.values())
        if own or not total:
            memo[name] = {own or UNNAMED: 1.0}
            return memo[name]
        shares: dict[str, float] = {}
        for caller, count in up.items():
            for phase, part in phase_of(caller).items():
                shares[phase] = shares.get(phase, 0.0) + part * count / total
        memo[name] = shares
        return shares

    by_phase: dict[str, float] = {}
    for name in in_extension:
        for phase, part in phase_of(name).items():
            by_phase[phase] = by_phase.get(phase, 0.0) + part * selfs[name]
    return by_phase, {k: v for k, v in selfs.items() if k in in_extension}


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("build | record | callgrind | fit")
    opts = _options(argv[1:])
    shards = opts.get("shards", "seq,8").split(",")
    if argv[0] == "build":
        build()
    elif argv[0] == "record":
        record(int(opts.get("n", 10_000_000)), shards)
    elif argv[0] == "callgrind":
        callgrind(int(opts.get("n", 200_000)), shards)
    elif argv[0] == "fit":
        fit(int(opts["n"]), opts["shards"], opts.get("ctl"), opts.get("ack"))
    else:
        raise SystemExit(f"unknown step {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
