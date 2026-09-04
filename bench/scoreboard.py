"""Win / tie / loss matrix over the committed benchmark CSVs, and a ratchet against regressions.

Nothing in this repo detected a *new* benchmark loss automatically, which is how four wrong published
claims of the 0.2.0 era survived a month. This reads the canonical `results_*.csv` (plus the
`_spread` sidecars where they exist), pairs every betula row against its rivals, and prints one
verdict per cell under the tie rule `bench/RESULTS.md` already states: **a difference smaller than
the wider of the two cells' three-seed spreads is not a result**.

Three pairings, because "did we win" has three different meanings:

- `vs-same` — betula's k-means against scikit-learn's k-means, and so on down. The like-for-like
  question, and the one the README's parity claims rest on.
- `vs-best` — betula's *best* head on a dataset against the best non-betula method there, whatever
  algorithm that is. This is the pairing that owns the two `sklearn-birch` losses; the same-algorithm
  view cannot see them, since Birch has no betula counterpart.
- `vs-external` — against `bench/results_external.csv` if it exists, i.e. the strongest specialist
  per axis rather than scikit-learn's general-purpose implementation. Produce it with
  `bench/external_baselines.py`; absent, these cells simply do not appear.

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

Tables without a spread sidecar (speed, memory, sparse, real-scale) are single runs by construction —
`bench/_worker.py` pins `seed=0` for them — so they fall back to the per-axis tolerance in `SLACK`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BOARD = HERE / "scoreboard.json"

# Slack for a comparison with no seed spread behind it: an absolute floor plus a relative term.
# Timing on a shared machine moves by a few percent between runs; peak RSS is far steadier; an ARI
# gets an absolute floor because a relative one would call 0.004-vs-0.056 a decisive result.
SLACK = {
    "quality": (0.01, 0.0),
    "speed": (0.0, 0.05),
    "memory": (0.0, 0.02),
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
    "betula-nmf": "sklearn-nmf",
    "betula (streaming)": "sklearn-kmeans (one-shot)",
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
        self, axis, pairing, table, label, ours, theirs, spreads, higher_is_better, contenders=""
    ):
        self.axis, self.pairing, self.table, self.label = axis, pairing, table, label
        self.ours, self.theirs = ours, theirs
        self.higher = higher_is_better
        # Who won the slice, for `vs-best`. Reported, never part of `name()`: a champion that
        # changes is data about the run, and folding it into the identity made an improvement
        # ("betula-svd is now our fastest sparse head") indistinguishable from a vanished result.
        self.contenders = contenders
        self.slack = self._slack(spreads)

    def _slack(self, spreads):
        """The margin below which this comparison is not a result."""
        floor, rel = SLACK.get(self.axis, (0.0, 0.0))
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
    path = HERE / f"{name}.csv"
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
    for (dataset, n, contest), grp in df.groupby(["dataset", "n", "contest"], sort=True):
        rows = {str(r["method"]): r for _, r in grp.iterrows()}
        ours = [m for m in rows if m.startswith("betula")]
        theirs = [m for m in rows if not m.startswith("betula")]
        if not ours or not theirs:
            continue
        for axis, column, higher in (("quality", "ari", True), ("speed", "time_s", False)):
            pick = max if higher else min
            bm = pick(ours, key=lambda m: float(rows[m][column]))
            rm = pick(theirs, key=lambda m: float(rows[m][column]))
            cells.append(
                Cell(
                    axis,
                    "vs-external",
                    "results_external",
                    f"{dataset}/{n}/{contest}",
                    float(rows[bm][column]),
                    float(rows[rm][column]),
                    (None, None),
                    higher,
                    f"{bm}-vs-{rm}",
                )
            )
    return cells


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
            print(
                f"  {mark} {c.name():<72} {c.ours:>11.4f} vs {c.theirs:>11.4f}"
                f"  margin {c.margin:+.4f}  slack {c.slack:.4f}{who}"
            )


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
    args = ap.parse_args()

    cells = collect()
    if not cells:
        print("no comparable cells — are the results_*.csv files present?")
        return 1
    if args.check:
        return check(cells)
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
