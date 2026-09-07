"""Is `mutants-baseline.txt` still describing mutants that exist?

The weekly `mutants.yml` run scores surviving mutants against that file, matching on the exact
`path:line:col: description` string cargo-mutants prints. Any edit that shifts a line invalidates
every entry below it in that file, and the workflow's only symptom is a "trim them" note buried in a
step summary of a job that takes three hours and had been red since 2026-08-19 for an unrelated
reason. By the time it was checked by hand, 147 of the 315 entries no longer matched anything.

`cargo mutants --list` builds and runs nothing -- about a second on this crate -- so the staleness
question is answerable on every push instead of every Monday. This does exactly that, and separates
the two cases that need different work:

  moved    the same mutation exists in the same file at a different line. The argument recorded for
           it is still valid; the entry needs re-anchoring. Where the same description occurs more
           than once in the file the new line is ambiguous and only a re-measure can say which is
           which, so those are counted but not proposed.
  vanished no mutation with that description exists in that file at all -- the code it described is
           gone. The entry is dead and its justification with it.

    uv run --no-sync python scripts/check_mutants_baseline.py
    uv run --no-sync python scripts/check_mutants_baseline.py --list-from mutants.out/caught.txt
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "mutants-baseline.txt"


def parse(lines: list[str]) -> list[str]:
    """Drop comments and blanks, and normalise trailing space, as `mutants.yml` does."""
    out = []
    for raw in lines:
        entry = raw.split("#", 1)[0].rstrip()
        if entry:
            out.append(entry)
    return out


def split(entry: str) -> tuple[str, str]:
    """`src/order.rs:109:15: replace || with &&` -> (`src/order.rs`, `replace || with &&`)."""
    path, _line, _col, description = entry.split(":", 3)
    return path, description.strip()


def current(list_from: pathlib.Path | None) -> list[str]:
    if list_from is not None:
        return parse(list_from.read_text().splitlines())
    done = subprocess.run(
        ["cargo", "mutants", "--list"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return parse(done.stdout.splitlines())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--list-from",
        type=pathlib.Path,
        help="read the mutant list from a file instead of running cargo mutants",
    )
    args = ap.parse_args()

    baseline = parse(BASELINE.read_text().splitlines())
    live = current(args.list_from)
    live_set = set(live)

    # Every description cargo-mutants currently emits for each file, with the lines it emits it at.
    by_file: dict[str, dict[str, list[str]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for entry in live:
        path, description = split(entry)
        by_file[path][description].append(entry)

    moved: list[tuple[str, list[str]]] = []
    vanished: list[str] = []
    for entry in baseline:
        if entry in live_set:
            continue
        path, description = split(entry)
        candidates = by_file.get(path, {}).get(description, [])
        (moved if candidates else vanished).append((entry, candidates) if candidates else entry)

    print(f"baseline {len(baseline)} entries · list {len(live)} mutants")
    if not moved and not vanished:
        print("every baseline entry matches a mutant that exists")
        return 0

    unambiguous = [(e, c[0]) for e, c in moved if len(c) == 1]
    print(f"STALE: {len(moved)} moved ({len(unambiguous)} unambiguously), {len(vanished)} vanished")

    worst = collections.Counter(split(e)[0] for e, _ in moved)
    worst.update(split(e)[0] for e in vanished)
    print("\nby file:")
    for path, n in worst.most_common():
        print(f"  {n:>4}  {path}")

    if unambiguous:
        print("\nre-anchor these -- the description occurs exactly once in the file now:")
        for old, new in unambiguous:
            print(f"  - {old}\n  + {new}")
    ambiguous = [e for e, c in moved if len(c) > 1]
    if ambiguous:
        print(
            f"\n{len(ambiguous)} moved entries are ambiguous (the same description occurs several\n"
            "times in the file); which one carried the accepted argument needs a re-measure:"
        )
        for entry in ambiguous:
            print(f"    {entry}")
    if vanished:
        print(
            f"\n{len(vanished)} entries describe a mutation that no longer exists -- delete them:"
        )
        for entry in vanished:
            print(f"    {entry}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
