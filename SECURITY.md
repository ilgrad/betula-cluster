# Security Policy

## Supported versions

The latest published `0.6.x` release on [PyPI](https://pypi.org/project/betula-cluster/) receives
security fixes. Older versions are not maintained.

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** via GitHub's
[private security advisories](https://github.com/ilgrad/betula-cluster/security/advisories/new)
rather than opening a public issue. You can expect an initial response within a few days.

## Surface

betula-cluster has **no runtime dependencies beyond NumPy** and ships a pure-Rust core (no
LAPACK / BLAS / SciPy at runtime), which keeps its supply-chain surface small. `cargo audit` runs in CI
on every change and on a nightly schedule. All untrusted input (arrays, CSR matrices) is validated at
the Python/Rust boundary — NaN/Inf are rejected before reaching the numeric core.
