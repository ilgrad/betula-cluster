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
pytest tests/test_python.py -q   # again with NO optional deps installed: 501 passed, 48 skipped
ruff check python/ tests/ bench/ examples/   # examples/ is linted but not formatted -- it is
ruff format --check python/ tests/ bench/    # notebook cells, and reflowing them desyncs the .ipynb
ty check python/                                   # or mypy / pyright
python -m mypy.stubtest betula_cluster             # the .pyi stubs must match the runtime
```

The second `pytest` line is not a duplicate. scikit-learn, SciPy and networkx are **optional test**
dependencies: the `python (pytest · pyX)` CI jobs install the wheel without them, so a test that
imports one without `pytest.importorskip` fails there and in no other command on this page. That is
exactly how `main` went red for three runs on 2026-09-10.

When a PR adds or changes a **parallel reduction over floats** — a rayon `fold`/`reduce`,
`par_iter().sum()`, anything whose chunking the pool size decides:

```bash
python scripts/thread_determinism.py   # every head, fitted at 1 and 8 threads, compared by digest
```

Floating-point addition does not associate, so a work-stealing fold makes the answer a function of
how the threads interleaved. It is invisible at one thread, which is why `cargo test` cannot see it,
and it shipped once in `nmf.rs`. CI runs this as the `threads` job.

Public surface, when a PR touches one of the thirteen documented modules (`tree`, `feature`,
`distance`, `bregman`, `model`, `clustering`, `types`, `sparse`, `order`, `coreset`, `stream`,
`window`, `validity` — see *Compatibility* in `README.md`):

```bash
cargo semver-checks --baseline-version <last released version>
```

It classifies each difference against the published crate rather than against a hand-kept list, so
"is this breaking?" is answered by the tool. Exit 100 means it found breaking changes; that is a
finding to justify in the PR body, not a failure to route around. The `#[doc(hidden)]` modules are
outside this contract by construction — the same command reports moving one *into* hiding as major,
which is why it is a release-boundary decision. CI runs this on every pull request; before 1.0.0 was
published it reported rather than gated, since 0.x was allowed to break.

`public-api.txt` is the same contract as a list — 1 034 items, regenerated with

```bash
cargo public-api --simplified > public-api.txt   # needs a nightly toolchain: rustdoc JSON
```

It answers a different question from `semver-checks`: not "is this breaking?" but "what exactly is
public?", as a file a reviewer can diff. `cargo public-api` omits `#[doc(hidden)]` items, so the
snapshot contains the thirteen modules and nothing else — which is also how the curation is checked
to be complete. Regenerate it in any PR that changes the surface, and before a release. It is
deliberately **not** a CI job: the tool builds rustdoc JSON, which is nightly-only and whose format
moves with the toolchain, and `cargo install cargo-public-api` costs five minutes per run to answer
a question `semver-checks` already answers semantically.

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
- **Every array-taking entry point joins the layout test.** A column-major array is contiguous, so a
  zero-copy ingest can borrow its buffer and then index it row-major — which reads the transpose and
  returns a plausible wrong answer with no error anywhere. That is not hypothetical: it shipped here
  once, and no benchmark could see it because every dataset in `bench/` arrives C-contiguous. A new
  function that takes an array is added to
  `test_the_answer_depends_on_the_values_and_not_on_how_numpy_stores_them` in `tests/test_python.py`
  in the same commit that adds the function.
- Conventional-commit messages (`feat:` / `fix:` / `test:` / `docs:` / `chore:` …).

By contributing you agree that your contributions are licensed under the project's
[MIT license](LICENSE-MIT).
