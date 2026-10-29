"""Check that a test fails when the code it guards is broken, then restore the file.

A test that passes both before and after a fix is not a regression test. This applies one textual
mutation, runs one test command, and restores the file whatever happens:

    python scripts/revert_check.py src/clustering/kmeans.rs \\
        "if cost < best.0 {" "if cost > best.0 {" \\
        -- cargo test --release --lib clustering::kmeans::tests::after_hartigan

The old text must occur exactly once, so a mutation can neither miss its target (a silent no-op
that "passes") nor land on the wrong line. Exit 0 when the test command fails under the mutation —
the test sees the defect — and 1 when it still passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if "--" not in argv or argv.index("--") != 3:
        print(__doc__)
        return 2
    path, old, new = Path(argv[0]), argv[1], argv[2]
    cmd = argv[4:]
    original = path.read_text()
    count = original.count(old)
    if count != 1:
        print(f"refusing: the old text occurs {count} times in {path}, not once")
        return 2
    try:
        path.write_text(original.replace(old, new))
        rc = subprocess.run(cmd).returncode
    finally:
        path.write_text(original)
    if rc == 0:
        print(f"NOT CAUGHT: `{' '.join(cmd)}` still passes with the mutation in {path}")
        return 1
    print(f"caught: `{' '.join(cmd)}` fails (exit {rc}) with the mutation in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
