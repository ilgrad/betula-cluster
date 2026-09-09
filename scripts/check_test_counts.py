"""Do the published test counts still match the suite?

`README.md` and `paper/paper.md` quote how many tests there are. Those numbers were typed by hand
and then aged: at the time this check was written the README claimed 728 Rust tests and a 457-case
Python suite against an actual 807 and 547, and the two README sentences did not agree with each
other either. A wrong count in a JOSS paper is a correctness claim about the software, so it is
cheaper to derive it than to remember it.

Neither collector runs a test. `cargo test -- --list` and `pytest --collect-only` are both about a
second once the crate is built, so this belongs in the gate rather than in a release checklist.

    uv run --no-sync python scripts/check_test_counts.py           # compare, exit 1 on a mismatch
    uv run --no-sync python scripts/check_test_counts.py --print   # just the counts

The Python collector needs the wrapper's test dependencies on the path:

    uv run --no-sync --with pytest --with scikit-learn --with scipy --with networkx \
        python scripts/check_test_counts.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Each claim is a file, a regex whose single group is the number, and the count it must equal.
#: The regex carries enough surrounding words to fail loudly if the sentence is rewritten around it.
CLAIMS = [
    (pathlib.Path("README.md"), r"\*\*(\d+)-case\*\* Python suite at \*\*100% wrapper", "python"),
    (pathlib.Path("README.md"), r"coverage\*\* \+ \*\*(\d+)\*\* Rust tests", "rust"),
    (pathlib.Path("README.md"), r"Verified: \*\*(\d+)\*\* Rust tests", "rust"),
    (pathlib.Path("README.md"), r"CLI\) \+ a \*\*(\d+)-case\*\*\nPython suite", "python"),
    (pathlib.Path("paper/paper.md"), r"a (\d+)-case Python test suite", "python"),
    (pathlib.Path("paper/paper.md"), r"coverage of the wrapper, (\d+) Rust tests", "rust"),
]


def rust_count() -> int:
    """Every test the crate declares, over every target, ignored ones included.

    `--list` prints one `N tests, M benchmarks` line per test binary; summing those lines is the
    same arithmetic the gate's five `cargo test` invocations report, without running anything.
    """
    done = subprocess.run(
        ["cargo", "test", "--all-features", "--", "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(int(m) for m in re.findall(r"^(\d+) tests, \d+ benchmarks$", done.stdout, re.M))


def python_count() -> int:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    match = re.search(r"^(\d+) tests? collected", done.stdout, re.M)
    if match is None:
        raise SystemExit(f"pytest printed no collection summary:\n{done.stdout[-2000:]}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print", action="store_true", help="print the counts and stop")
    args = parser.parse_args()

    counts = {"rust": rust_count(), "python": python_count()}
    if args.print:
        for name, n in counts.items():
            print(f"{name}: {n}")
        return 0

    stale = []
    for path, pattern, key in CLAIMS:
        text = (ROOT / path).read_text()
        found = re.findall(pattern, text)
        if not found:
            stale.append(f"{path}: no text matching /{pattern}/ — the sentence was rewritten")
            continue
        for got in found:
            if int(got) != counts[key]:
                stale.append(f"{path}: claims {got} {key} tests, the suite declares {counts[key]}")
    for line in stale:
        print(line)
    print(
        f"test counts: rust {counts['rust']}, python {counts['python']}, "
        f"{len(stale)} stale claim(s)"
    )
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
