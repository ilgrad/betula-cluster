# Security policy

betula-cluster compresses data into cluster features in a Rust core and returns results through
PyO3. This document says what the library defends against, what it deliberately does not, and how to
report something that crosses the line.

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |
| 0.x | no — the model files 0.8.0 and earlier wrote still load, so the fix is to upgrade; the code itself is not maintained |

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on
[github.com/ilgrad/betula-cluster](https://github.com/ilgrad/betula-cluster/security/advisories/new)
— please do not open a public issue first. If that form is unavailable, open an issue asking for a
private channel and nothing else — no details. A report is most useful with the array shape and
dtype, the parameters, and the smallest input that still triggers it. Expect a first response within
a few days.

## The trust boundaries

**Arrays, sparse matrices and numeric parameters.** Everything is validated where Python meets Rust:
shape and dtype are checked, a column-major array is handled as itself rather than read as its
transpose, NaN and infinity are rejected before reaching the numeric core, and a parameter the
chosen head cannot use raises instead of being silently ignored.

**Model files.** `Betula.load` and `pickle.loads` read a gzip-framed, schema-versioned CBOR
document, and since 1.0.0 they **validate the tree rather than trust it**: an arena index out of
range, a node reachable twice, a cluster feature whose own arrays disagree or whose width is not the
tree's, a parameter that would leave an insert unable to terminate — each is refused with a
`ValueError` naming what disagrees. Before that, four such documents deserialized happily and
panicked several calls later, and a panic crossing the FFI boundary arrives as a `PanicException`,
which is not something a caller can reasonably be asked to catch. Decoding also streams out of the
gzip member instead of inflating to a `Vec` first, so a compression bomb has nothing to amplify
against, and a CBOR length header claiming 2⁴⁰ elements is refused without reserving for it. Over
4 000 randomly corrupted files, every failure is a `ValueError` and none is a panic.

Validation makes a malformed file an error instead of a crash. It does not make an **untrusted**
file safe to act on: what a valid file describes is still whatever its author chose. Treat a model
from someone else like any other deserialized artefact.

## What this library does not do

It opens no sockets, spawns no processes, and touches no path except the one handed to `save` /
`load`. It is not a sandbox and not a boundary against the code that calls it.

## Surface

`unsafe` exists in exactly one module, `src/kernels.rs`: AVX2/FMA and NEON distance kernels, called
only behind runtime feature detection and backed by a scalar fallback
([`docs/adr/003-avx2-kernels-unsafe.md`](https://github.com/ilgrad/betula-cluster/blob/main/docs/adr/003-avx2-kernels-unsafe.md)).
No other module in `src/` contains the word, and a patch that adds it elsewhere is a review question
rather than a detail.

There are **no runtime dependencies beyond NumPy** and the core is pure Rust — no LAPACK, BLAS or
SciPy at run time — which keeps the supply-chain surface small. `cargo audit` runs in CI on every
change and on a nightly schedule.
