# ADR 004 — the Bregman family ships as a second estimator, not a keyword on the first

**Status:** Accepted · decided **before** the wiring, because it is a type-1 door: the crate is on
PyPI and crates.io, so the shape chosen here is the shape that has to be deprecated if it is wrong.
The Rust side (`bregman::BregmanCf`, `bregman::BregmanCentroid`, `bregman::BregmanIncrease`,
`clustering::bregman::{bregman_kmeans, bregman_agglomerative, bregman_em}`) shipped in **0.7.0**
with no Python surface at all, which is what this record closes.

## Context

`Betula(feature=…)` names a **covariance model** — `"spherical"`, `"diagonal"`, `"full"`, `"fd"` —
and every one of them summarises the *same* Euclidean geometry at a different fidelity. A Bregman
feature names something else entirely: the **geometry** itself. `BregmanCf<R, KullbackLeibler>`
carries `(n, μ, S_φ)` where `S_φ` is a Bregman *information*, not a scatter, and
`BregmanCf::variance` documents that asking it for a per-dimension variance is the wrong question.

The two axes are orthogonal, and collapsing them into one keyword changes what every neighbouring
keyword means:

- `feature="kl"` with `method="gmm"` would hand a Gaussian head a Bregman information to read as a
  variance. The head would run. The numbers would be meaningless.
- `feature="kl"` with `absorb="chi2"` would apply a Normal-Inverse-Gamma variance prior to a
  quantity that is not a variance.
- `feature="kl"` with `normalize=True` would project onto the unit sphere, leaving the simplex the
  KL divergence is defined on.

Three shapes were considered.

**(a) A second keyword, `geometry="euclidean" | "kl" | "itakura-saito" | "logistic"`.** Orthogonal to
`feature=`, which is honest about the axes — but it makes the *legal* combinations a table the user
has to consult, and the illegal ones are rejected at `fit` time, one `ValueError` per pairing. Every
future head and every future feature has to declare its row. Rejected: it multiplies the validation
surface without making anything unrepresentable.

**(b) Fold it into `method=`: `"kl-kmeans"`, `"kl-ward"`, `"kl-mixture"`.** Keeps the feature axis
clean and needs no new validation, at the cost of a Cartesian product in the method enum — four
divergences × three heads is twelve names, and adding a divergence adds three more. It also puts the
geometry in the *wrong* keyword: `method=` names an algorithm everywhere else in the API. Rejected.

**(c) A separate estimator, `BregmanBetula(divergence=…)`.** The illegal combinations are not
rejected — they cannot be written. `BregmanBetula` has no `feature=`, no `normalize=`, no `absorb=`,
because none of them mean anything in a Bregman geometry, and `Betula` gains no `divergence=`.

## Decision

**(c).** `BregmanBetula` is a second scikit-learn-style estimator in the same module.

- `divergence` ∈ `{"euclidean", "kl", "itakura-saito", "logistic"}` — the geometry, and the only
  axis this estimator adds.
- `method` ∈ `{"kmeans", "ward", "mixture"}` — the three heads that exist over a Bregman feature.
- `beta` — the mixture's inverse dispersion, documented in nats and rejected for the other two heads
  rather than silently ignored.
- The tree parameters that are geometry-independent (`n_clusters`, `threshold`, `branching`,
  `leaf_cap`, `max_leaves`, `max_iter`, `n_init`, `seed`) carry over unchanged.

The cost is a second class in the public surface. That is the price of the guarantee, and it is the
cheaper side of the trade: a keyword that is legal to write and meaningless to run is a bug report
whose root cause is the API.

## Consequences

**Domain validation moves to the Python boundary and is not optional.** `BregmanCf::push` only
`debug_assert!`s its domain, so a release build fed `x ≤ 0` under KL returns `NaN` rather than
failing. `BregmanBetula.fit` validates before the data reaches Rust: `kl` and `itakura-saito` need
`x > 0` everywhere, `logistic` needs `x ∈ (0, 1)`, and `euclidean` needs nothing. The error names the
offending value and its position, not just the constraint.

**`divergence="euclidean"` is deliberately kept.** It makes `BregmanBetula` reduce to the shipped
Euclidean path exactly — `SquaredEuclidean` *is* a Bregman divergence — which is what lets the test
suite assert that the two estimators agree, and what gives a user a one-keyword A/B between
geometries without changing anything else.

**Adding a divergence is one enum arm on each side.** No new keyword, no new legality row, no change
to `Betula`. Adding a Bregman *head* is one arm of `method`.

**What this forecloses.** A future non-Euclidean feature that genuinely is a covariance model in a
Bregman geometry (a Bregman analogue of `feature="full"`) would need a `feature=` axis inside
`BregmanBetula`. That is additive and does not disturb anything decided here.

## Alternatives rejected

(a) and (b) above. Both were rejected for the same underlying reason: they place the geometry in a
keyword whose other values do not name geometries, so the API stops being readable by inspection —
and neither of them makes a single meaningless combination impossible to write.
