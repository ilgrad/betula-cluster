"""Recompute every hard-coded reference constant in the crate against independent high precision.

Provenance archaeology is not a check. `watson.rs` carried two values that were ~5 ulp wrong for a
month while their comment correctly said "from Maxima at fpprec: 40" -- the comment was true and the
numbers were still wrong, because the generator had silently dropped into double precision. The only
thing that settles it is recomputing the values from a *different* engine and diffing.

Three tables, three engines of origin, one checker (mpmath at 60 dps):

  src/clustering/watson.rs  GOLDEN         <- Maxima 5.50, fpprec 40   -> mp.hyp1f1
  src/clustering/vmf.rs     asymptotic fixture <- mpmath 50 dps        -> mp.besseli   (self-check)
  src/stats.rs              CDF_TABLE, QUANTILE_TABLE <- mpmath 80 dps -> mp.gammainc  (self-check)

Inputs are parsed as f64 and fed to mpmath as exact binary values: the Rust code is called with the
f64 the literal denotes, so the reference has to be the function's value *at that f64*, not at the
decimal it was typed as.

Run: timeout 900 uv run --no-sync --with mpmath python scripts/recheck_goldens.py
Exits 1 if any row is more than 2 ulp off.
"""

from __future__ import annotations

import re
from pathlib import Path

import mpmath as mp

mp.mp.dps = 60

ROOT = Path(__file__).resolve().parents[1]
NUM = r"[-+]?[0-9][0-9_]*\.?[0-9_]*(?:[eE][-+]?[0-9]+)?"


def f64(tok: str) -> float:
    return float(tok.replace("_", ""))


def ulps(got: float, want: float) -> float:
    """Signed error in units of the last place of `want`."""
    if got == want:
        return 0.0
    spacing = mp.mpf(math_ulp(want))
    return float((mp.mpf(got) - mp.mpf(want)) / spacing)


def math_ulp(x: float) -> float:
    import math

    return math.ulp(x)


def block(path: Path, start_pat: str, end: str = "];") -> str:
    text = path.read_text()
    i = text.index(start_pat)
    return text[i : text.index(end, i)]


def report(name: str, rows: list[tuple[str, float, float]], expect: int) -> int:
    """rows: (label, committed, reference). Returns the number of rows worse than 2 ulp.

    `expect` is the row count as of the last audit. A regex that stops matching -- because the
    table was reformatted, or a literal gained a `_f64` suffix -- otherwise reports "ok, 0 rows",
    which is the exact silent pass this script exists to catch.
    """
    if len(rows) != expect:
        raise SystemExit(
            f"{name}: parsed {len(rows)} rows, expected {expect}. Fix the pattern (or update"
            " `expect` if the table really changed) -- a partial parse checks nothing."
        )
    bad = 0
    worst = (0.0, "")
    for label, got, want in rows:
        u = ulps(got, want)
        if abs(u) > abs(worst[0]):
            worst = (u, label)
        if abs(u) > 2.0:
            bad += 1
            print(f"  BAD  {label:<34} committed {got!r}")
            print(f"       {'':34} reference {float(want)!r}  ({u:+.1f} ulp)")
    flag = "FAIL" if bad else "ok"
    print(f"{flag:>6}  {name:<28} {len(rows):>3} rows, worst {worst[0]:+.2f} ulp at {worst[1]}")
    return bad


def check_watson() -> int:
    src = block(ROOT / "src/clustering/watson.rs", "const GOLDEN:")
    pat = re.compile(rf"\(\s*({NUM}),\s*({NUM}),\s*({NUM}),\s*({NUM}),?\s*\)", re.S)
    rows = []
    for d_s, k_s, logm_s, g_s in pat.findall(src):
        d, k = int(f64(d_s)), mp.mpf(f64(k_s))
        a, b = mp.mpf(1) / 2, mp.mpf(d) / 2
        m = mp.hyp1f1(a, b, k)
        rows.append((f"d={d} k={f64(k_s)} logM", f64(logm_s), mp.log(m)))
        g = mp.hyp1f1(a + 1, b + 1, k) / m / d
        rows.append((f"d={d} k={f64(k_s)} g", f64(g_s), g))
    return report("watson.rs GOLDEN", rows, 48)


def check_vmf() -> int:
    src = block(
        ROOT / "src/clustering/vmf.rs",
        "for &(nu, kappa, want) in &[",
        "] {",
    )
    pat = re.compile(rf"\(\s*({NUM})(?:_f64)?,\s*({NUM})(?:_f64)?,\s*({NUM})(?:_f64)?,?\s*\)")
    rows = []
    for nu_s, k_s, want_s in pat.findall(src):
        nu, kappa = mp.mpf(f64(nu_s)), mp.mpf(f64(k_s))
        rows.append(
            (f"nu={f64(nu_s):g} k={f64(k_s):g}", f64(want_s), mp.log(mp.besseli(nu, kappa)))
        )
    return report("vmf.rs asymptotic fixture", rows, 11)


def check_stats_cdf() -> int:
    src = block(ROOT / "src/stats.rs", "const CDF_TABLE:")
    pat = re.compile(rf"\(\s*({NUM}),\s*({NUM}),\s*({NUM}),?\s*\)")
    rows = [
        (
            f"a={f64(a):g} x={f64(x):g}",
            f64(p),
            mp.gammainc(mp.mpf(f64(a)), 0, mp.mpf(f64(x)), regularized=True),
        )
        for a, x, p in pat.findall(src)
    ]
    return report("stats.rs CDF_TABLE", rows, 59)


def check_stats_quantile() -> int:
    src = block(ROOT / "src/stats.rs", "const QUANTILE_TABLE:")
    pat = re.compile(rf"\(\s*({NUM}),\s*({NUM}),\s*({NUM}),?\s*\)")
    rows = []
    for a_s, p_s, x_s in pat.findall(src):
        a, p = mp.mpf(f64(a_s)), mp.mpf(f64(p_s))
        # Bisect on log(x): the grid reaches 1e-151, which plain bisection on x cannot resolve.
        lo, hi = mp.mpf(-400), mp.mpf(50)
        for _ in range(400):
            mid = (lo + hi) / 2
            if mp.gammainc(a, 0, mp.e**mid, regularized=True) < p:
                lo = mid
            else:
                hi = mid
        rows.append((f"a={f64(a_s):g} p={f64(p_s):g}", f64(x_s), mp.e ** ((lo + hi) / 2)))
    return report("stats.rs QUANTILE_TABLE", rows, 234)


def main() -> None:
    print(f"mpmath {mp.__version__} at {mp.mp.dps} dps; tolerance 2 ulp\n")
    bad = check_watson() + check_vmf() + check_stats_cdf() + check_stats_quantile()
    print(f"\n{'ALL TABLES AGREE' if bad == 0 else f'{bad} ROWS DISAGREE'}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
