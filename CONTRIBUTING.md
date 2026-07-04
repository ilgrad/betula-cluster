# Contributing

Thanks for your interest in betula-cluster! It is a numerically stable BETULA clustering engine — a
Rust core (`src/`) with a thin scikit-learn-style Python wrapper (`python/betula_cluster/`) built by
[maturin](https://www.maturin.rs/).

## Development setup

```bash
# Rust toolchain (1.82+) and a Python 3.11+ venv
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release        # builds the extension into the venv (editable Python source)
```

## The gates (CI enforces all of these)

Rust:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings          # also: --no-default-features --features persistence; --features cli
cargo test                                         # also: --no-default-features; --features persistence; --features cli --bin betula
cargo llvm-cov --features persistence,cli --summary-only --fail-under-lines 95
```

Python (against the built extension):

```bash
pytest tests/test_python.py --cov=betula_cluster --cov-fail-under=100   # 100% wrapper coverage is enforced
ruff check python/ tests/ && ruff format --check python/ tests/
ty check python/                                   # or mypy / pyright
python -m mypy.stubtest betula_cluster             # the .pyi stubs must match the runtime
```

## Guidelines

- **Numerical correctness first.** New CF math must be cancellation-free and property-tested
  (`proptest` in `src/**` `mod prop_tests`), with the derivations kept in `docs/MATH.md` and checked
  against symbolic (Maxima) / `mpmath` ground truth.
- **Keep it lean.** No new **runtime** dependencies without discussion — NumPy is the only one, and the
  Rust core links no LAPACK/BLAS. Optional extras (e.g. `optuna` for tuning) go behind
  `[project.optional-dependencies]`.
- **Honest benchmarks.** Any change to the speed / quality / memory story must be reproducible from
  `bench/comprehensive.py` and reconciled in `bench/RESULTS.md` — wins *and* losses.
- **Illegal states unrepresentable.** Prefer the type system / invariants over scattered runtime guards;
  validate untrusted input once, at the boundary.
- Conventional-commit messages (`feat:` / `fix:` / `test:` / `docs:` / `chore:` …).

By contributing you agree that your contributions are licensed under the project's
[MIT license](LICENSE-MIT).
