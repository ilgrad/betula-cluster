"""Win / tie / loss matrix over the committed benchmark CSVs, and a ratchet against regressions.

Nothing in this repo detected a *new* benchmark loss automatically, which is how four wrong published
claims of the 0.2.0 era survived a month. This reads the canonical `results_*.csv` (plus the
`_spread` sidecars where they exist), pairs every betula row against its rivals, and prints one
verdict per cell under the tie rule `bench/RESULTS.md` already states: **a difference smaller than
the wider of the two cells' three-seed spreads is not a result**.

Four pairings, because "did we win" has four different meanings:

- `vs-same` — betula's k-means against scikit-learn's k-means, and so on down. The like-for-like
  question, and the one the README's parity claims rest on.
- `vs-best` — betula's *best* head on a dataset against the best non-betula method there, whatever
  algorithm that is. This is the pairing that owns the two `sklearn-birch` losses; the same-algorithm
  view cannot see them, since Birch has no betula counterpart.
- `vs-external` — against `bench/results_external.csv` if it exists, i.e. the strongest specialist
  per axis rather than scikit-learn's general-purpose implementation. Produce it with
  `bench/external_baselines.py`; absent, these cells simply do not appear.
- `vs-rival` — betula's best head against **each** rival of the census harnesses
  (`bench/rivals.py`, `bench/head_references.py`, `bench/density_rivals.py`), one cell per rival,
  told-k and no-k kept apart. `--claims` renders these and the other named pairings as sentences
  for `bench/RESULTS.md`, and `--claims-check` fails when a committed block has drifted. Their
  speed cells come from the timing campaign (`bench/results_timings.csv`, written by
  `bench/timings.py`): the second fit in a process is judged, the first is printed beside it, and
  a ratio under 1.5× is a tie (`LEVEL`).

`bench/scoreboard.json` records the verdicts as of the last accepted run. `--check` re-derives them
and exits non-zero if any cell got worse (win → tie/loss, tie → loss) or vanished; `--update`
rewrites it, which is the deliberate act of accepting a new board.

A cell is identified by *what it compares* — axis, table, pairing, dataset slice — and never by who
won it. The two `vs-*-best` pairings pick a champion per side, and a champion that changes is news
about the run, not a missing comparison: with the winner's name in the key, `betula-svd` overtaking
`betula-sparse` as our fastest sparse head reported two VANISHED cells and failed the gate. The
champions are printed on the rendered line instead.

    uv run --with pandas python bench/scoreboard.py            # print the matrix
    uv run --with pandas python bench/scoreboard.py --check    # CI gate
    uv run --with pandas python bench/scoreboard.py --update   # accept the current board

`--data DIR` reads the CSVs and the board from `DIR` instead of `bench/`. The fixture in
`bench/fixtures/scoreboard/` holds one census table, a timing CSV in the format `bench/timings.py`
writes, and the claims block they must produce; it checks the timing reader end to end:

    uv run --with pandas python bench/scoreboard.py --data bench/fixtures/scoreboard \\
        --claims-check bench/fixtures/scoreboard/claims.md

Tables without a spread sidecar (speed, memory, sparse, real-scale) are single runs by construction —
`bench/_worker.py` pins `seed=0` for them — so they fall back to the per-axis tolerance in `SLACK`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

HERE = Path(__file__).resolve().parent
# Where the CSVs and the board are read from; `--data` points both at a fixture directory.
DATA = HERE
BOARD = DATA / "scoreboard.json"

# Slack for a comparison with no seed spread behind it: an absolute floor plus a relative term.
# Timing on a shared machine moves by a few percent between runs; peak RSS is far steadier; an ARI
# gets an absolute floor because a relative one would call 0.004-vs-0.056 a decisive result.
SLACK = {
    "quality": (0.01, 0.0),
    "speed": (0.0, 0.05),
    "memory": (0.0, 0.02),
}

# A census speed cell from the timing campaign is "level" below a ratio of 1.5: the slack is a
# third of the slower side. With 5 % run noise and the 28 % that DRAM placement adds on this
# machine, three rounds resolve 1.5x and no finer (1.3x needs six), so a smaller ratio is not a
# result whatever the round spread.
LEVEL = 1 / 3

# The stage of `bench/timings.py` that times each census table's arms.
TIMED_STAGE = {
    "results_rivals": "rivals",
    "results_head_refs": "heads",
    "results_density_rivals": "density",
}

# Which betula row is answered by which rival, for the like-for-like pairing.
SAME_ALGORITHM = {
    "betula-kmeans": "sklearn-kmeans",
    "betula-gmm": "sklearn-gmm",
    "betula-gmm-full": "sklearn-gmm",
    "betula-ward": "sklearn-ward",
    "betula-hdbscan": "sklearn-hdbscan",
    "betula-sparse": "sklearn-kmeans",
    "betula-svd": "sklearn-svd",
    "betula-kmeans-on-sklearn-nmf": "sklearn-nmf",
    "betula (streaming)": "sklearn-kmeans (one-shot)",
    "betula (one-shot)": "sklearn-kmeans (one-shot)",
}

QUALITY_TABLES = [
    ("results_quality", ["dataset", "method"], "ARI"),
    ("results_real", ["dataset", "n", "method"], "ARI"),
    ("results_real_hires", ["dataset", "n", "max_leaves", "method"], "ARI"),
    ("results_real_scale", ["dataset", "method"], "ari"),
    ("results_sparse", ["dataset", "method"], "ari"),
]

COST_TABLES = [
    ("results_scaling", ["method", "n"]),
    ("results_real_scale", ["dataset", "method"]),
    ("results_sparse", ["dataset", "method"]),
    ("results_memory", ["method", "n"]),
]

ORDER = {"win": 0, "tie": 1, "loss": 2}


class Cell:
    """One comparison: a betula value against a rival's, with whatever spread each side carries."""

    def __init__(
        self,
        axis,
        pairing,
        table,
        label,
        ours,
        theirs,
        spreads,
        higher_is_better,
        contenders="",
        rel=None,
        cold=None,
    ):
        self.axis, self.pairing, self.table, self.label = axis, pairing, table, label
        self.ours, self.theirs = ours, theirs
        self.higher = higher_is_better
        # The relative slack, where it is not the axis's own (`LEVEL` for the timing campaign).
        self.rel = rel
        # A timed cell judges the second fit in a process; the first, `(ours, theirs)`, is printed
        # beside it and never judged.
        self.cold = cold
        # Who won the slice, for `vs-best`. Reported, never part of `name()`: a champion that
        # changes is data about the run, and folding it into the identity made an improvement
        # ("betula-svd is now our fastest sparse head") indistinguishable from a vanished result.
        self.contenders = contenders
        self.slack = self._slack(spreads)
        # The two methods behind the numbers, for the claims block. Never part of `name()`.
        self.champion, _, self.rival = contenders.partition("-vs-") if contenders else ("", "", "")

    def _slack(self, spreads):
        """The margin below which this comparison is not a result."""
        floor, rel = SLACK.get(self.axis, (0.0, 0.0))
        rel = rel if self.rel is None else self.rel
        seen = [s for s in spreads if s is not None and not math.isnan(s)]
        return max(floor, rel * max(abs(self.ours), abs(self.theirs)), *seen)

    @property
    def verdict(self):
        diff = self.ours - self.theirs
        if abs(diff) <= self.slack:
            return "tie"
        return "win" if (diff > 0) == self.higher else "loss"

    @property
    def margin(self):
        return self.ours - self.theirs if self.higher else self.theirs - self.ours

    def name(self):
        return f"{self.table}/{self.axis}/{self.pairing}/{self.label}"


def _read(name):
    path = DATA / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def _index(df, keys, column):
    """`{key-tuple: value}` over the rows that carry a value in `column`."""
    if df is None or column not in df.columns:
        return {}
    out = {}
    for _, r in df.iterrows():
        if pd.isna(r[column]):
            continue
        out[tuple(str(r[k]) for k in keys)] = float(r[column])
    return out


def _spreads(table, keys):
    return _index(_read(f"{table}_spread"), keys, "range")


def _swap(key, at, method):
    return (*key[:at], method, *key[at + 1 :])


def _label(key, at):
    return "/".join((*key[:at], *key[at + 1 :], key[at]))


def _same_algorithm(table, keys, column, axis, higher):
    """betula-X against sklearn-X, cell by cell."""
    at = keys.index("method")
    values = _index(_read(table), keys, column)
    spread = _spreads(table, keys) if axis == "quality" else {}
    cells = []
    for key, ours in values.items():
        rival = SAME_ALGORITHM.get(key[at])
        rkey = _swap(key, at, rival) if rival else None
        if rkey is None or rkey not in values:
            continue
        cells.append(
            Cell(
                axis,
                "vs-same",
                table,
                _label(key, at),
                ours,
                values[rkey],
                (spread.get(key), spread.get(rkey)),
                higher,
                f"{key[at]}-vs-{rival}",
            )
        )
    return cells


def _best_of_breed(table, keys, column, axis, higher):
    """betula's best head on a slice against the best non-betula method on the same slice.

    This is where `sklearn-birch` lands: it is a CF-tree method with no betula counterpart, so the
    like-for-like pairing is blind to it, and it currently leads the all-methods table twice.

    Each side's champion is chosen by its **worst seed**, not by its median, wherever a spread
    sidecar exists — "which head can you rely on", not "which head got lucky once". Without it a
    head like `covtype betula-spectral`, whose three seeds span −0.015 to 0.128, would be nominated
    on a median it cannot reproduce and would then absorb a real gap into its own slack. The two
    champions are still *judged* on their medians, under the same tie rule as everything else.
    """
    at = keys.index("method")
    values = _index(_read(table), keys, column)
    spread = _spreads(table, keys) if axis == "quality" else {}
    floor = _index(_read(f"{table}_spread"), keys, "min" if higher else "max")
    slices = {}
    for key, v in values.items():
        slices.setdefault((*key[:at], *key[at + 1 :]), []).append((key, v))

    pick = max if higher else min
    cells = []
    for slice_key, rows in slices.items():
        ours = [r for r in rows if r[0][at].startswith("betula")]
        theirs = [r for r in rows if not r[0][at].startswith("betula")]
        if not ours or not theirs:
            continue
        worst = lambda r: floor.get(r[0], r[1])  # noqa: E731 — the reliable value, else the only one
        bk, bv = pick(ours, key=worst)
        rk, rv = pick(theirs, key=worst)
        cells.append(
            Cell(
                axis,
                "vs-best",
                table,
                "/".join(str(s) for s in slice_key if s != ""),
                bv,
                rv,
                (spread.get(bk), spread.get(rk)),
                higher,
                f"{bk[at]}-vs-{rk[at]}",
            )
        )
    return cells


def _external():
    """Specialist baselines from `bench/results_external.csv`, if that file has been produced."""
    df = _read("results_external")
    if df is None:
        return []
    cells = []
    for (dataset, n, contest, order), grp in df.groupby(
        ["dataset", "n", "contest", "order"], sort=True
    ):
        rows = {str(r["method"]): r for _, r in grp.iterrows()}
        # Label order keeps the key the table had before it ran both orders.
        where = f"{dataset}/{n}/{contest}" + ("" if order == "label" else f"/{order}")
        for axis, column, higher in (("quality", "ari", True), ("speed", "time_s", False)):
            # A `--quality-only` run leaves the time empty: no speed cell rather than a NaN verdict.
            timed = [m for m in rows if pd.notna(rows[m][column])]
            ours = [m for m in timed if m.startswith("betula")]
            theirs = [m for m in timed if not m.startswith("betula")]
            if not ours or not theirs:
                continue
            # The quality axis carries each row's seed range: champions are nominated by their
            # worst seed and judged on medians, the rule `_best_of_breed` states.
            seeded = axis == "quality" and "ari_min" in grp.columns
            worst = {m: float(rows[m]["ari_min" if seeded else column]) for m in timed}
            pick = max if higher else min
            bm, rm = pick(ours, key=worst.__getitem__), pick(theirs, key=worst.__getitem__)
            spread = {
                m: float(rows[m]["ari_max"] - rows[m]["ari_min"]) if seeded else None
                for m in (bm, rm)
            }
            cells.append(
                Cell(
                    axis,
                    "vs-external",
                    "results_external",
                    where,
                    float(rows[bm][column]),
                    float(rows[rm][column]),
                    (spread[bm], spread[rm]),
                    higher,
                    f"{bm}-vs-{rm}",
                )
            )
    return cells


NO_STRUCTURE = 0.01

# A reference that stands for one head is answered by that head, not by betula's best arm: FasterPAM
# says how good `kmedoids` is, and a k-means win over it would say nothing about that.
HEAD_OF_REFERENCE = {
    "kmedoids (fasterpam)": "betula-kmedoids",
    "skfuzzy-cmeans": "betula-fuzzy-cmeans",
    "sklearn-kmeans": "betula-kmeans",
    "sklearn-bisecting-kmeans": "betula-kmeans",
    "kmodes-kprototypes": "betula-kprototypes",
}


# What one contest is, per census table: the slices over which a user would pick one method. The
# density fixtures are one kind of data; MNIST, the sentence embeddings and the CLIP image
# embeddings are three more.
CONTEST = {
    "results_rivals": ("kind", "told_k"),
    "results_head_refs": ("dataset", "told_k"),
    "results_density_rivals": ("told_k",),
}

# The kind of data a dataset is, where one kind spans several datasets: 20 newsgroups under two
# sentence models is one choice for a user, and a head picked per model would need the labels. A
# dataset not listed is a kind of its own.
KIND = {"news-minilm": "sentence-embeddings", "news-mpnet": "sentence-embeddings"}


# Rows a user gets by composing public calls, not by one `method=`: printed beside the heads, never
# chosen as betula's arm, until the recipe is one library call.
RECIPES = {"betula-gmm-full-merge"}


def _contest_champions(df, by):
    """betula's arm per contest: present on every slice, best mean of its worst-seed ARI."""
    out = {}
    heads = df[df["arm"].str.startswith("betula") & ~df["arm"].isin(RECIPES)]
    for key, grp in heads.groupby(list(by), sort=True):
        worst = grp.groupby(["arm", "dataset", "n"])["ari"].min().unstack(["dataset", "n"])
        worst = worst.dropna()  # an arm missing from a slice cannot be the contest's one choice
        out[key if isinstance(key, tuple) else (key,)] = worst.mean(axis=1).idxmax()
    return out


def _rivals(table):
    """`bench/results_rivals.csv` and `bench/results_head_refs.csv`: betula's best head against
    **each** rival, one cell per rival.

    The champion is betula's best arm in the same class (or, for a head reference, the head it
    stands for: `HEAD_OF_REFERENCE`) — told the cluster count, or choosing it —
    because a method that is given `k` and one that has to find it answer different questions.
    It is chosen **once per contest** (`CONTEST`), as a user picks one method for a kind of data,
    and then meets every rival on every slice of that contest: picking the best of a dozen heads
    afresh on each slice, with the labels in hand, is a choice no user can make, and it turned
    every density slice into a betula win while hdbscan led on the mean. The choice is the arm with
    the best mean, over the contest's slices, of its worst-seed ARI. It
    is chosen on quality alone, and the memory and speed cells are that arm's: the lightest arm is
    often one whose labels no one would use, and a memory win it scored would be no claim at all.
    Values are medians over the seeds, and the seed range is the spread; a DNF arm gives no cell.

    The speed cells come from `bench/results_timings.csv` (`_timings`), never from the harness's own
    `time_s`, which is a first fit on whatever machine was at hand. A speed cell needs both arms
    timed, and judges the second fit in a process (`warm_s`, the median over the rounds, their range
    as the spread) under `LEVEL`; the first fit (`cold_s`) is printed beside it.

    A rival that recovers no structure on a slice (median ARI under `NO_STRUCTURE`) gives no cell
    there either. At its documented defaults CLASSIX, DenMune and OPTICS return one cluster on the
    standardised fixtures: that measures a default against data it was not set for, not the
    method, and a win over it would be no claim. The rule can only remove betula's wins — a slice
    where betula reads 0 and the rival does not is still scored.
    """
    df = _read(table)
    if df is None:
        return []
    if "dnf" in df.columns:
        df = df[df["dnf"].isna()]
    df = df.assign(kind=df["dataset"].map(lambda d: KIND.get(d, d)))
    champion = _contest_champions(df, CONTEST[table])
    timed = _timings(TIMED_STAGE[table])
    cells = []
    for (dataset, n, told), grp in df.groupby(["dataset", "n", "told_k"], sort=True):
        stats = grp.groupby("arm")[["ari", "rss_mb"]].agg(["median", "min", "max"])
        arms = list(stats.index)
        ours = [a for a in arms if a.startswith("betula")]
        theirs = [a for a in arms if not a.startswith("betula")]
        if not ours:
            continue
        slice_ = {"dataset": dataset, "kind": KIND.get(dataset, dataset), "told_k": told}
        key = tuple(slice_[c] for c in CONTEST[table])
        best = champion[key]
        label = f"{dataset}/{n}/{'told-k' if told else 'no-k'}"
        pairs = []
        for rm in theirs:
            if stats["ari"].loc[rm, "median"] < NO_STRUCTURE:
                continue
            paired = HEAD_OF_REFERENCE.get(rm) if table == "results_head_refs" else None
            pairs.append((paired if paired in ours else best, rm))
        for axis, column, higher in (("quality", "ari", True), ("memory", "rss_mb", False)):
            col = stats[column]
            for bm, rm in pairs:
                cells.append(
                    Cell(
                        axis,
                        "vs-rival",
                        table,
                        f"{label}/{rm}",
                        float(col.loc[bm, "median"]),
                        float(col.loc[rm, "median"]),
                        (
                            float(col.loc[bm, "max"] - col.loc[bm, "min"]),
                            float(col.loc[rm, "max"] - col.loc[rm, "min"]),
                        ),
                        higher,
                        f"{bm}-vs-{rm}",
                    )
                )
        for bm, rm in pairs:
            mine, yours = timed.get((dataset, int(n), bm)), timed.get((dataset, int(n), rm))
            if mine is None or yours is None:
                continue
            cells.append(
                Cell(
                    "speed",
                    "vs-rival",
                    table,
                    f"{label}/{rm}",
                    mine.warm,
                    yours.warm,
                    (mine.spread, yours.spread),
                    False,
                    f"{bm}-vs-{rm}",
                    rel=LEVEL,
                    cold=(mine.cold, yours.cold),
                )
            )
    return cells


class Timed(NamedTuple):
    """One arm's time on one slice of the timing campaign, in seconds."""

    warm: float  # median second fit over the rounds: the value a verdict judges
    spread: float  # max - min of the second fit over the rounds
    cold: float  # median first fit, imports and compilation included: printed, never judged


def _timings(stage):
    """`{(dataset, n, arm): Timed}` from `bench/results_timings.csv`, one stage of it.

    A row counts when the run recorded `timing_valid` (`bench-ready.sh` passed and the control
    stayed within its drift), the arm finished, and a second fit was taken. Rows of other stages
    stay out even for the same arm and slice: the `threads` stage, for one, runs betula on one
    thread, and its rows answer a different question.
    """
    df = _read("results_timings")
    if df is None or df.empty:
        return {}
    keep = (df["stage"] == stage) & (df["timing_valid"].astype(str).str.lower() == "true")
    if "dnf" in df.columns:
        keep &= df["dnf"].isna()
    df = df[keep & df["warm_s"].notna()]
    out = {}
    for (dataset, n, arm), grp in df.groupby(["dataset", "n", "arm"], sort=True):
        warm = grp["warm_s"].astype(float)
        spread = float(warm.max() - warm.min())
        out[(str(dataset), int(n), str(arm))] = Timed(
            float(warm.median()), spread, float(grp["cold_s"].astype(float).median())
        )
    return out


def collect():
    """Every comparison the committed CSVs support, ordered quality → speed → memory."""
    cells = []
    for table, keys, column in QUALITY_TABLES:
        cells += _same_algorithm(table, keys, column, "quality", True)
        cells += _best_of_breed(table, keys, column, "quality", True)
    for table, keys in COST_TABLES:
        for axis, column in (("speed", "time_s"), ("memory", "rss_mb")):
            cells += _same_algorithm(table, keys, column, axis, False)
            cells += _best_of_breed(table, keys, column, axis, False)
    cells += _external()
    for table in ("results_rivals", "results_head_refs", "results_density_rivals"):
        cells += _rivals(table)
    return cells


def render(cells):
    by_axis = {}
    for c in cells:
        by_axis.setdefault(c.axis, []).append(c)
    for axis in ("quality", "speed", "memory"):
        group = by_axis.get(axis, [])
        if not group:
            continue
        tally = {v: sum(c.verdict == v for c in group) for v in ORDER}
        print(f"\n## {axis} — {tally['win']} win · {tally['tie']} tie · {tally['loss']} loss")
        for c in sorted(group, key=lambda c: (ORDER[c.verdict], c.name())):
            mark = {"win": "WIN ", "tie": "tie ", "loss": "LOSS"}[c.verdict]
            who = f"  [{c.contenders}]" if c.contenders else ""
            cold = f"  cold {c.cold[0]:.4f} vs {c.cold[1]:.4f}" if c.cold else ""
            print(
                f"  {mark} {c.name():<72} {c.ours:>11.4f} vs {c.theirs:>11.4f}"
                f"  margin {c.margin:+.4f}  slack {c.slack:.4f}{who}{cold}"
            )


UNITS = {"quality": ("ARI", "{:.3f}"), "speed": ("s", "{:.2f}"), "memory": ("MB", "{:.0f}")}


def claims(cells):
    """The comparison as sentences, grouped by rival: every win, tie and loss, with both values.

    `bench/RESULTS.md` carries this block verbatim between `<!-- claims:begin -->` and
    `<!-- claims:end -->` (the README points to it: the block is 300 lines); `--claims-check` fails when the committed block differs from what the
    CSVs produce, so a hand-written claim cannot drift from the numbers behind it.
    """
    by_rival = {}
    for c in cells:
        if c.rival:
            by_rival.setdefault(c.rival, []).append(c)
    lines = []
    for rival in sorted(by_rival):
        group = sorted(by_rival[rival], key=lambda c: (ORDER[c.verdict], c.axis, c.name()))
        tally = {v: sum(c.verdict == v for c in group) for v in ORDER}
        lines.append(
            f"- **{rival}** — {tally['win']} win · {tally['tie']} tie · {tally['loss']} loss"
        )
        for c in group:
            unit, fmt = UNITS[c.axis]
            where = c.label.removesuffix(f"/{rival}")
            mine, yours = f"{fmt.format(c.ours)} {unit}", f"{fmt.format(c.theirs)} {unit}"
            if c.cold:  # a timed cell names the fit it judges, with the first fit beside it
                mine += f" warm, {fmt.format(c.cold[0])} {unit} cold"
                yours += f" warm, {fmt.format(c.cold[1])} {unit} cold"
            lines.append(
                f"  - {c.verdict}, {c.axis}, `{where}` ({c.table.removeprefix('results_')}): "
                f"betula {mine} (`{c.champion}`) against {yours}"
            )
    return "\n".join(lines) + "\n"


def claims_check(cells, paths):
    """Exit non-zero when a file's claims block differs from the one the CSVs produce."""
    want = claims(cells)
    bad = 0
    for path in paths:
        text = Path(path).read_text()
        begin, end = "<!-- claims:begin -->\n", "<!-- claims:end -->"
        if begin not in text or end not in text:
            print(f"{path}: no claims block")
            bad = 1
            continue
        have = text.split(begin, 1)[1].split(end, 1)[0]
        if have != want:
            print(f"{path}: the claims block is stale; regenerate it with --claims")
            bad = 1
    if not bad:
        print(f"claims: {len(paths)} file(s) match the CSVs")
    return bad


def check(cells):
    """Fail on any cell that got worse, or that the CSVs no longer produce."""
    if not BOARD.exists():
        print(f"{BOARD.name} is missing — run with --update to record the first board.")
        return 1
    prev = json.loads(BOARD.read_text())["cells"]
    now = {c.name(): c.verdict for c in cells}

    worse = [(k, v, now[k]) for k, v in prev.items() if k in now and ORDER[now[k]] > ORDER[v]]
    gone = sorted(k for k in prev if k not in now)
    if not worse and not gone:
        print(f"scoreboard: {len(now)} cells, none regressed against {BOARD.name}.")
        return 0
    for k, was, is_ in sorted(worse):
        print(f"REGRESSED {k}: {was} -> {is_}")
    for k in gone:
        print(f"VANISHED  {k}: the CSVs no longer produce this comparison")
    print(
        "\nIf the move is intended, re-measure and accept it with --update in the same commit that "
        "explains it in bench/RESULTS.md."
    )
    return 1


def main():
    ap = argparse.ArgumentParser(
        description="win/tie/loss matrix over the committed benchmark CSVs"
    )
    ap.add_argument("--check", action="store_true", help="exit non-zero on any regression")
    ap.add_argument("--update", action="store_true", help="accept the current board")
    ap.add_argument("--claims", action="store_true", help="print the claims block (markdown)")
    ap.add_argument(
        "--claims-check", nargs="+", metavar="FILE", help="fail if a file's claims block is stale"
    )
    ap.add_argument(
        "--data", type=Path, metavar="DIR", help="read the CSVs and the board from DIR (a fixture)"
    )
    args = ap.parse_args()
    if args.data:
        global DATA, BOARD
        DATA, BOARD = args.data, args.data / "scoreboard.json"

    cells = collect()
    if not cells:
        print("no comparable cells — are the results_*.csv files present?")
        return 1
    if args.check:
        return check(cells)
    if args.claims:
        print(claims(cells), end="")
        return 0
    if args.claims_check:
        return claims_check(cells, args.claims_check)
    render(cells)
    if args.update:
        BOARD.write_text(
            json.dumps(
                {"cells": {c.name(): c.verdict for c in sorted(cells, key=lambda c: c.name())}},
                indent=1,
            )
            + "\n"
        )
        print(f"\nwrote {BOARD.name} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
