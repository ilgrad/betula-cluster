# Mathematical foundation & improvements

Every formula below is verified symbolically (Maxima) and/or numerically (mpmath/Julia ground
truth) — see [`DESIGN.md`](../DESIGN.md) and `research/`.

## Numerically stable cluster features `(n, μ, S)`

Classic BIRCH stores `(N, LS = Σx, SS = Σx²)` and recovers variance as `SS/N − (LS/N)²` — a
**difference of two large, nearly-equal numbers**. On real data with an offset (timestamps, money,
geo-coordinates, un-centered embeddings) this **catastrophically cancels**: in `f64` the variance
collapses to noise — and can go *negative* — around coordinate magnitude `1e7`, silently corrupting
every downstream radius, covariance, and label.

betula-cluster stores **`(n, μ, S)`** — weight, mean, and the sum of squared deviations
`S = Σ w (x − μ)(x − μ)ᵀ`. `S` is a **sum of non-negative terms**, so the variance `S/n ≥ 0` and the
covariance is **positive-semidefinite by construction** — there is nothing to cancel. The updates are
algebraically exact (not approximations):

```
add (Welford/West):  W' = W + w;   μ ← μ + (w/W')·δ;     S ← S + w·(1 − w/W')·(δ⊙δ)     [δ = x − μ_old]
merge (Chan):        W  = W_A+W_B; μ = μ_A + (W_B/W)·Δ;  S = S_A + S_B + (W_A·W_B/W)·(Δ⊙Δ) [Δ = μ_B − μ_A]
```

(full covariance: `δ⊙δ → δδᵀ`, `Δ⊙Δ → ΔΔᵀ`). Tested bit-stable at offset `1e7–1e8` against an
mpmath reference, where the classic `(N, LS, SS)` form loses all significant digits.

## GMM E-step: expected-log (variant C), not the paper's convolution

Running EM on the leaf CFs, each leaf is a mini-Gaussian `N(μ_i, Σ_i)`. The textbook / paper move
treats it by **convolution** — `N(μ_i | μ_k, Σ_k + Σ_i)` — which inflates each component by the
leaf's own spread and **washes out components on a coarse CF-tree**. betula-cluster instead uses the
**expected log-likelihood** responsibility (measured to give higher ARI — `research/RESULTS-estep.md`):

```
log r_ik = log π_k + log N(μ_i | μ_k, Σ_k) − ½·tr(Σ_k⁻¹ Σ_i)        (then log-sum-exp normalized)
```

The M-step folds the within-leaf scatter back in, so the fitted components stay exact:

```
Σ_k = Σ_i w_ik·(Σ_i + (μ_i − μ_k)(μ_i − μ_k)ᵀ) / N_k,    w_ik = n_i·r_ik
```

with NIW/MAP regularization `Σ_k = (Ψ + …)/(ν + N_k + d + 1)` so a 1-point leaf never yields a
singular covariance.

## Other verified improvements

- **Distances `D0–D4`** are the BIRCH measures re-derived on `(n, μ, S)` (Maxima-verified
  *equivalent*, computed stably). Variance-increase / Ward is `D4 = (n_A·n_B / n_AB)·‖Δμ‖²` — the `S`
  terms cancel by König–Huygens, so it is an exact centroid measure (no Lance-Williams approximation).
- **k-means on CFs** minimizes the true point objective, not the leaf-centroid proxy:
  `SSE = Σ_i [S_i + n_i·‖μ_i − c‖²]` folds each leaf's own scatter `S_i` back in, so compressing to a
  CF-tree first does not change what is being optimized.
- **Full covariance** uses a matrix Welford (PSD) with on-demand Cholesky for `logdet` /
  `mahalanobis`; the packed upper-triangular index is the tested `(j−1)·j/2` form — a reference
  implementation shipped a `(j−1)·dim/2` variant that silently corrupts `dim ≥ 4`.
- **χ² absorption gate** (`absorb="chi2"`): a mass-invariant Mahalanobis-χ² threshold with a
  Normal-Inverse-Gamma prior `var_eff = (S + κ·s₀)/(n + κ)`, finite at `n = 1`. Fixes the
  size-imbalance failure where a 12-point vs a 10⁴-point cluster decide differently (sklearn #22854).
- **Frequent-Directions sketch** (high `d`): the full-cov GMM consumes it in **low-rank** form —
  `tr(Σ_k⁻¹ Σ_i) = Σ_r ‖L_k⁻¹ f_r‖²` — so it never materializes a `d×d` matrix per leaf and keeps
  `O(ℓ·d)` memory through clustering. Identical math to the dense path.
- **Rebuild threshold** is the within-leaf mean nearest-sibling gap (ELKI/BETULA-standard,
  `O(M·capacity)`), raised monotonically — no global all-pairs scan, no over-growth collapse.
- **Robust insertion** (`huber_k = k`, optional): before a point `x` is folded into its target
  microcluster `(n, μ, S)`, each coordinate is winsorized to the cluster's own scale,

  ```
  x̃_j = clip(x_j, μ_j − k·σ_j, μ_j + k·σ_j),   σ_j = √(S_j / n)
  ```

  and the stable Welford update runs on `x̃` instead of `x`. A coordinate with `σ_j = 0` (no scale
  yet) passes through unchanged, and the clip is skipped until the target holds ≥ 5 points (so the
  scale estimate is trustworthy). This bounds any single point's pull on the centroid to `O(k·σ/n)`
  — outliers can no longer drag a centroid or inflate a radius — while leaving the CF a valid
  `(n, μ, S)` triple, so every downstream head is unchanged. The clipped value flows identically
  into the leaf entry and its ancestors, preserving the "each node = merge of its subtree" invariant.

## Relation to BIRCH and BETULA

This library is a from-scratch Rust implementation of the **BETULA** cluster feature — the
numerically stable `(n, μ, S)` summary introduced by Lang & Schubert to replace classic BIRCH's
cancellation-prone `(N, LS, SS)`:

- **BETULA: Numerically Stable CF-Trees for BIRCH Clustering** — Andreas Lang & Erich Schubert,
  *SISAP 2020* ([arXiv:2006.12881](https://arxiv.org/abs/2006.12881) ·
  [Springer](https://link.springer.com/chapter/10.1007/978-3-030-60936-8_22)); extended journal
  version, *Information Systems* 2022
  ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306437921001253)).
- **BIRCH** — Zhang, Ramakrishnan & Livny, *SIGMOD 1996*.

Reference implementations: [ELKI](https://elki-project.github.io/) (Java), and
[**betulars**](https://pypi.org/project/betulars/) ([source](https://github.com/andiwg/betula)) — a
Rust+PyO3 package by paper co-author **Andreas Lang**. betulars is a faithful, highly optimised
**Phase-1 CF-tree builder**: it builds the tree and exposes leaf cluster statistics, but (as of
v0.1.0) produces **no cluster labels** and **no global clustering** — k-means / GMM / hierarchical are
listed as planned. If you need *just* the canonical BETULA CF-tree primitive, fast, use betulars.

**betula-cluster is a different thing: an end-to-end clustering library.** It re-derives the same
stable CF from scratch and then adds everything betulars leaves to the user:

- the full Phase-2 pipeline — k-means / GMM / **full-covariance** GMM / Ward / HDBSCAN with
  automatic `k` — producing **per-point labels** and `predict`, behind a real **scikit-learn API**
  (the de-facto Python BIRCH, `sklearn.cluster.Birch`, is *classic* BIRCH: the unstable CF);
- `f32` trees, streaming `partial_fit`, a mass-invariant **χ² absorption gate**, a
  Frequent-Directions sketch for high `d`, `normalize=True` for embeddings, an inspection API
  (outliers / near-duplicates / representatives / geometry), and serde persistence;
- auto-vectorized distance kernels (tight inline reductions) and rayon-parallel build + labeling.

The concrete, reproducible quality/speed/memory comparison is against the labeled scikit-learn
clusterers practitioners actually reach for: at **matching ARI**, betula labels 1 M points **~38×
faster** than `sklearn.cluster.Birch` (8.1 s → 0.21 s) and **~14×** faster than `KMeans`, while
streaming memory stays flat at ~57 MB; see [`bench/RESULTS.md`](../bench/RESULTS.md) and the
[method-comparison notebook](../examples/04_method_comparison.ipynb). (betulars produces no labels, so
it is not in that comparison; on the raw Phase-1 *build* the two are at parity — betula-cluster builds
an **identical tree** at every `N` and, with matched `target-cpu=native` flags, matches betulars'
wall-clock to within ~2 %.)
