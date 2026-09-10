//! Numerically stable Clustering Features — the BETULA core.
//!
//! A feature summarises a (weighted) set of points by `(n, μ, S)`: weight, mean, and sum of
//! squared deviations from the mean. Three covariance models are provided:
//! [`Spherical`] (scalar `S`), [`Diagonal`] (per-dimension `S`), and [`Full`] (full scatter
//! matrix). Features are built with a weighted Welford update and merged with the Chan parallel
//! update — both are sums of non-negative contributions, so variances never suffer catastrophic
//! cancellation and the covariance is positive semi-definite by construction.
//!
//! Merge is homogeneous (same model) — trees use a single feature type — which avoids the
//! lossy cross-model fallbacks and the upper-triangular index-bug class of earlier impls.

use crate::linalg;
use crate::types::Real;

/// A feature's covariance `Σ = M / n` in the form the full-covariance GMM E-step needs: `Dense` for
/// the dense models, or `LowRank` rows `{f_r}` plus an isotropic floor with
/// `Σ = Σ_r f_r f_rᵀ + iso·I` for the Frequent-Directions sketch. The low-rank form keeps the GMM at
/// `O(ℓ·d)` per leaf instead of materialising `d×d` (which would undo FD's whole memory advantage).
/// Both encode the same matrix, so every accessor here — `trace_under`, `trace`, `apply_rows`,
/// `add_scaled` — returns identical values for either variant.
pub enum SecondMoment<R: Real> {
    /// Dense `d×d` covariance.
    Dense(Vec<Vec<R>>),
    /// Rows `f_r` and an isotropic term with `Σ = Σ_r f_r f_rᵀ + iso·I`. `iso` carries the scatter
    /// the sketch shrink discarded, whose directions are no longer known — see [`FdSketch`].
    LowRank {
        /// Rank-1 factors of the retained directions.
        rows: Vec<Vec<R>>,
        /// Per-dimension variance of the discarded mass (`0` for an exact sketch).
        iso: R,
        /// Ambient dimension — `rows` may be empty, and `iso·I` still has a trace.
        dim: usize,
    },
}

#[allow(clippy::needless_range_loop)] // dense matrix arithmetic reads clearest with (i, j) indices
impl<R: Real> SecondMoment<R> {
    /// `tr(A⁻¹ Σ)` where `chol` is the Cholesky factor `L` of `A` (`A = L Lᵀ`) and `inv = A⁻¹`.
    /// For `LowRank`, `tr(A⁻¹ Σ_r f_r f_rᵀ) = Σ_r ‖L⁻¹ f_r‖²` — no `d×d` matrix is formed.
    pub fn trace_under(&self, chol: &[Vec<R>], inv: &[Vec<R>]) -> R {
        match self {
            SecondMoment::Dense(cov) => {
                let d = cov.len();
                let mut s = R::zero();
                for i in 0..d {
                    for j in 0..d {
                        s = s + inv[i][j] * cov[i][j];
                    }
                }
                s
            }
            SecondMoment::LowRank { rows, iso, .. } => {
                let low = rows
                    .iter()
                    .map(|f| linalg::mahalanobis_sq_from_chol(chol, f))
                    .fold(R::zero(), |a, b| a + b);
                // tr(A⁻¹·iso·I) = iso·tr(A⁻¹).
                let t = (0..inv.len())
                    .map(|i| inv[i][i])
                    .fold(R::zero(), |a, b| a + b);
                low + *iso * t
            }
        }
    }

    /// `tr(Σ)`. For `LowRank` this is `Σ_r ‖f_r‖² + iso·d`, so the diagonal is never materialised.
    pub fn trace(&self) -> R {
        match self {
            SecondMoment::Dense(cov) => cov
                .iter()
                .enumerate()
                .map(|(d, row)| row[d])
                .fold(R::zero(), |a, b| a + b),
            SecondMoment::LowRank { rows, iso, dim } => {
                rows.iter()
                    .map(|f| f.iter().map(|&v| v * v).fold(R::zero(), |a, b| a + b))
                    .fold(R::zero(), |a, b| a + b)
                    + *iso * R::from_usize(*dim).unwrap()
            }
        }
    }

    /// `out[r] += w · Σ v_r` for each row `v_r` of `v` — the only product a subspace head needs of a
    /// leaf's scatter. For `LowRank`, `Σ v = Σ_j f_j (f_j·v)` costs `O(ℓ·d)` per row against the
    /// dense `O(d²)`, which is what keeps an `ℓ`-row sketch cheaper than the covariance it stands for.
    pub fn apply_rows(&self, v: &[Vec<R>], out: &mut [Vec<R>], w: R) {
        match self {
            SecondMoment::Dense(cov) => {
                for (vr, o) in v.iter().zip(out.iter_mut()) {
                    for (a, row) in cov.iter().enumerate() {
                        let s = row
                            .iter()
                            .zip(vr)
                            .map(|(&c, &x)| c * x)
                            .fold(R::zero(), |p, q| p + q);
                        o[a] = o[a] + w * s;
                    }
                }
            }
            SecondMoment::LowRank { rows, iso, .. } => {
                for (vr, o) in v.iter().zip(out.iter_mut()) {
                    for f in rows {
                        let dot = f
                            .iter()
                            .zip(vr)
                            .map(|(&a, &b)| a * b)
                            .fold(R::zero(), |p, q| p + q);
                        let c = w * dot;
                        if c != R::zero() {
                            for (ov, &fv) in o.iter_mut().zip(f) {
                                *ov = *ov + c * fv;
                            }
                        }
                    }
                    let c = w * *iso; // (iso·I)v = iso·v
                    if c != R::zero() {
                        for (ov, &vv) in o.iter_mut().zip(vr) {
                            *ov = *ov + c * vv;
                        }
                    }
                }
            }
        }
    }

    /// `target += w · Σ`.
    pub fn add_scaled(&self, target: &mut [Vec<R>], w: R) {
        match self {
            SecondMoment::Dense(cov) => {
                for (tr, cr) in target.iter_mut().zip(cov) {
                    for (t, &c) in tr.iter_mut().zip(cr) {
                        *t = *t + w * c;
                    }
                }
            }
            SecondMoment::LowRank { rows, iso, .. } => {
                let d = target.len();
                for f in rows {
                    for a in 0..d {
                        let wfa = w * f[a];
                        if wfa != R::zero() {
                            for b in 0..d {
                                target[a][b] = target[a][b] + wfa * f[b];
                            }
                        }
                    }
                }
                let c = w * *iso;
                if c != R::zero() {
                    for a in 0..d {
                        target[a][a] = target[a][a] + c;
                    }
                }
            }
        }
    }

    /// `out[j] += w · Σ[j][j]` — the diagonal alone, which a per-dimension noise model needs and
    /// neither [`trace`](Self::trace) (which has already summed it) nor
    /// [`apply_rows`](Self::apply_rows) can answer. For `LowRank` it is `Σ_r f_r[j]² + iso`, so the
    /// `d×d` matrix `add_scaled` would build to read `d` numbers off its diagonal never exists.
    pub fn add_diagonal_scaled(&self, out: &mut [R], w: R) {
        match self {
            SecondMoment::Dense(cov) => {
                for (j, o) in out.iter_mut().enumerate() {
                    *o = *o + w * cov[j][j];
                }
            }
            SecondMoment::LowRank { rows, iso, .. } => {
                for f in rows {
                    for (o, &fv) in out.iter_mut().zip(f) {
                        *o = *o + w * fv * fv;
                    }
                }
                let c = w * *iso;
                if c != R::zero() {
                    out.iter_mut().for_each(|o| *o = *o + c);
                }
            }
        }
    }
}

/// A weighted point summary: weight `n`, mean `μ`, and second-moment information `S`.
///
/// `Send + Sync` lets features be shared across rayon worker threads (e.g. parallel rebuilds).
pub trait ClusterFeature<R: Real>: Clone + Send + Sync {
    /// Empty feature for `dim`-dimensional points.
    fn new(dim: usize) -> Self;
    /// Point dimensionality.
    fn dim(&self) -> usize;
    /// Aggregated weight `n` (point count, or total weight for weighted/decayed data).
    fn weight(&self) -> R;
    /// Mean vector `μ`.
    fn mean(&self) -> &[R];
    /// Total sum of squared deviations `S = Σ w ‖x-μ‖²` (the trace of the scatter matrix).
    fn ssd(&self) -> R;
    /// Population variance of dimension `d` (`S_d / n`); `0` for an empty feature.
    fn variance(&self, d: usize) -> R;
    /// Add point `x` with weight `w` (weighted Welford; no cancellation).
    fn push(&mut self, x: &[R], w: R);
    /// Merge another feature of the same model (Chan parallel update; exact and stable).
    fn merge(&mut self, other: &Self);
    /// Exponentially decay the feature's mass by `factor ∈ (0, 1]`: weight and second moments
    /// scale together, so the mean and variance are unchanged — only the influence (weight) of the
    /// accumulated points shrinks. Used for time-decayed / concept-drift streaming.
    fn decay(&mut self, factor: R);
    /// Dense `d×d` covariance estimate `Σ = M/n`. Default is diagonal (`diag(variance(i))`); the
    /// full model overrides it with the off-diagonal cross-covariances.
    fn cov_dense(&self) -> Vec<Vec<R>> {
        let d = self.dim();
        let mut m = vec![vec![R::zero(); d]; d];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = self.variance(i);
        }
        m
    }
    /// Covariance for the full-cov GMM E-step. Default wraps [`ClusterFeature::cov_dense`] as a dense
    /// matrix; the FD sketch overrides it with low-rank factors so the GMM avoids materialising a
    /// `d×d` matrix per leaf (preserving its `O(ℓ·d)` memory).
    fn second_moment(&self) -> SecondMoment<R> {
        SecondMoment::Dense(self.cov_dense())
    }
    /// Are this feature's own arrays mutually consistent?
    ///
    /// Dead weight on every path but one. A feature built by [`ClusterFeature::new`],
    /// [`ClusterFeature::push`] and [`ClusterFeature::merge`] cannot be malformed — but one
    /// *deserialized* came from bytes, and bytes are input. A mean of length 1 inside a
    /// 4-dimensional tree used to reach an `index out of bounds` in `cluster_centers_`, from a file
    /// rather than from a bug. [`crate::tree::CFTree::validate`] is what calls this.
    ///
    /// Defaults to `true` so an out-of-crate feature is not forced to implement it; the shipped
    /// models override it wherever they hold two arrays that have to agree. [`Spherical`] does not:
    /// its dimension *is* its mean's length and its scatter is a scalar, so it has nothing to
    /// disagree with.
    #[doc(hidden)]
    fn is_well_formed(&self) -> bool {
        true
    }
}

/// `R -> f64`. Infallible for both types `Real` admits.
#[inline]
fn wide<R: Real>(x: R) -> f64 {
    x.to_f64().unwrap()
}

/// `f64 -> R`. Infallible for both types `Real` admits; saturates to `±inf` outside `f32`'s range.
#[inline]
fn narrow<R: Real>(x: f64) -> R {
    R::from_f64(x).unwrap()
}

// ───────────────────────── Spherical (scalar SSD) ─────────────────────────

/// Isotropic feature: a single scalar `S`. Covariance is `(S / (n·d)) · I`.
///
/// The weight and the scalar scatter are `f64` whatever `R` is, and that is not a detail: they are
/// *running totals*, so their precision is set by how many rows a leaf absorbs rather than by how
/// precisely each row is stored. In binary32 `w + 1 == w` from `2^24` upward, so an `f32` leaf used
/// to stop counting after 16.8 M rows — the mean kept creeping and the scatter kept growing against
/// a weight that did not, so the reported variance inflated. Both accumulate in `f64` now; the mean,
/// which is bounded by the data rather than by the row count, stays in `R`, so an `f32` tree still
/// costs `4·d` bytes per leaf plus sixteen.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Spherical<R: Real> {
    w: f64,
    mean: Vec<R>,
    ssd: f64,
}

impl<R: Real> Spherical<R> {
    /// Build a spherical feature directly from its moments `(n, μ, S)`. Used by the sparse-native
    /// path, which accumulates the mean and scatter in a scaled sparse form and materialises the
    /// dense feature once, at the end.
    pub fn from_moments(weight: R, mean: Vec<R>, ssd: R) -> Self {
        Self {
            w: wide(weight),
            mean,
            ssd: wide(ssd),
        }
    }
}

impl<R: Real> ClusterFeature<R> for Spherical<R> {
    fn new(dim: usize) -> Self {
        Self {
            w: 0.0,
            mean: vec![R::zero(); dim],
            ssd: 0.0,
        }
    }
    fn dim(&self) -> usize {
        self.mean.len()
    }
    fn weight(&self) -> R {
        narrow(self.w)
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        narrow(self.ssd)
    }
    fn variance(&self, _d: usize) -> R {
        if self.w <= 0.0 {
            R::zero()
        } else {
            narrow(self.ssd / self.w / self.mean.len() as f64)
        }
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let wi = wide(w);
        let w_new = self.w + wi;
        let factor: R = narrow(wi / w_new);
        let coef: R = narrow(wi * (1.0 - wi / w_new));
        let mut normsq = R::zero();
        for (m, &xi) in self.mean.iter_mut().zip(x) {
            let d = xi - *m;
            normsq = normsq + d * d;
            *m = *m + factor * d;
        }
        self.ssd += wide(coef * normsq);
        self.w = w_new;
    }
    fn merge(&mut self, other: &Self) {
        if other.w <= 0.0 {
            return;
        }
        if self.w <= 0.0 {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor: R = narrow(other.w / w_new);
        let c: R = narrow(self.w * other.w / w_new);
        let mut normsq = R::zero();
        for (m, &om) in self.mean.iter_mut().zip(&other.mean) {
            let d = om - *m;
            normsq = normsq + d * d;
            *m = *m + factor * d;
        }
        self.ssd = self.ssd + other.ssd + wide(c * normsq);
        self.w = w_new;
    }
    fn decay(&mut self, factor: R) {
        let f = wide(factor);
        self.w *= f;
        self.ssd *= f;
    }
}

// ───────────────────────── Diagonal (per-dimension SSD) ─────────────────────────

/// Axis-aligned feature: per-dimension `S`. Covariance is `diag(S_d / n)`.
///
/// The weight is `f64` for the reason given on [`Spherical`]. The per-axis scatter is **not**, and
/// that is a measured trade rather than an oversight: it is `d` values per leaf touched once per
/// (row, dimension), so widening it cost **+11 %** on the insert path (200 000 × 32,
/// `max_leaves = 2000`, median of five, A-B-A) and took an `f32` tree from half the size of the
/// `f64` one to three quarters — paid by every `f32` user, to correct a **0.89 %** low variance that
/// only appears once a single leaf is past `2^24` rows. `an_f32_leaf_past_the_ceiling_pins_what_a`
/// `_narrow_scatter_still_costs` pins the residual so it cannot drift unrecorded.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Diagonal<R: Real> {
    w: f64,
    mean: Vec<R>,
    ssd: Vec<R>,
}

impl<R: Real> ClusterFeature<R> for Diagonal<R> {
    fn new(dim: usize) -> Self {
        Self {
            w: 0.0,
            mean: vec![R::zero(); dim],
            ssd: vec![R::zero(); dim],
        }
    }
    fn is_well_formed(&self) -> bool {
        self.mean.len() == self.ssd.len()
    }
    fn dim(&self) -> usize {
        self.mean.len()
    }
    fn weight(&self) -> R {
        narrow(self.w)
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        self.ssd.iter().copied().sum()
    }
    fn variance(&self, d: usize) -> R {
        if self.w <= 0.0 {
            R::zero()
        } else {
            self.ssd[d] / narrow(self.w)
        }
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let wi = wide(w);
        let w_new = self.w + wi;
        let factor: R = narrow(wi / w_new);
        let coef: R = narrow(wi * (1.0 - wi / w_new));
        for ((m, s), &xi) in self.mean.iter_mut().zip(self.ssd.iter_mut()).zip(x) {
            let d = xi - *m;
            *s = *s + coef * d * d;
            *m = *m + factor * d;
        }
        self.w = w_new;
    }
    fn merge(&mut self, other: &Self) {
        if other.w <= 0.0 {
            return;
        }
        if self.w <= 0.0 {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor: R = narrow(other.w / w_new);
        let c: R = narrow(self.w * other.w / w_new);
        for (((m, s), &om), &os) in self
            .mean
            .iter_mut()
            .zip(self.ssd.iter_mut())
            .zip(&other.mean)
            .zip(&other.ssd)
        {
            let d = om - *m;
            *s = *s + os + c * d * d;
            *m = *m + factor * d;
        }
        self.w = w_new;
    }
    fn decay(&mut self, factor: R) {
        self.w *= wide(factor);
        for s in &mut self.ssd {
            *s = *s * factor;
        }
    }
}

// ───────────────────────── Full (scatter matrix) ─────────────────────────

/// Full-covariance feature: the scatter matrix `M = Σ w (x-μ)(x-μ)ᵀ`, stored as the flat
/// upper triangle (row-major, including the diagonal). Covariance is `M / n`.
///
/// The weight is `f64` for the reason given on [`Spherical`]. The scatter stays in `R`: it is
/// `d(d+1)/2` values per leaf, and widening those would undo the memory an `f32` tree is chosen for.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Full<R: Real> {
    w: f64,
    mean: Vec<R>,
    scatter: Vec<R>,
    dim: usize,
}

impl<R: Real> Full<R> {
    /// Flat index of upper-triangular element `(i, j)` with `i <= j`.
    ///
    /// Row `i` starts at `i·dim − T(i)` with the triangular number `T(i) = i(i-1)/2`; the
    /// element is `+ (j - i)` further. This is the single source of truth for indexing — the
    /// reference implementation duplicated it and used `dim` instead of `i` in `T`, silently
    /// corrupting cross-products for `dim >= 4` (see `math_improove/09-vvv-bug`, local-only notes).
    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.dim - i * i.saturating_sub(1) / 2 + (j - i)
    }
    /// Symmetric scatter access.
    fn at(&self, i: usize, j: usize) -> R {
        let (a, b) = if i <= j { (i, j) } else { (j, i) };
        self.scatter[self.idx(a, b)]
    }
    /// Dense covariance matrix `Σ = M / n` (`d×d`, symmetric). Zeros for an empty feature.
    pub fn covariance(&self) -> Vec<Vec<R>> {
        let d = self.dim;
        if self.w <= 0.0 {
            return vec![vec![R::zero(); d]; d];
        }
        let inv: R = narrow(1.0 / self.w);
        (0..d)
            .map(|i| (0..d).map(|j| self.at(i, j) * inv).collect())
            .collect()
    }
    /// Lower-triangular Cholesky factor of the covariance; `None` if not positive-definite.
    pub fn cholesky(&self) -> Option<Vec<Vec<R>>> {
        linalg::cholesky_lower(&self.covariance())
    }
    /// `log|Σ|`; `None` if the covariance is not positive-definite.
    pub fn logdet(&self) -> Option<R> {
        self.cholesky().map(|l| linalg::logdet_from_chol(&l))
    }
    /// Squared Mahalanobis distance `(x-μ)ᵀ Σ⁻¹ (x-μ)`; `None` if covariance is not PD.
    pub fn mahalanobis_sq(&self, x: &[R]) -> Option<R> {
        let l = self.cholesky()?;
        let delta: Vec<R> = (0..self.dim).map(|i| x[i] - self.mean[i]).collect();
        Some(linalg::mahalanobis_sq_from_chol(&l, &delta))
    }
}

// Triangular scatter access reads clearest with explicit `(i, j)` indices.
#[allow(clippy::needless_range_loop)]
impl<R: Real> ClusterFeature<R> for Full<R> {
    fn new(dim: usize) -> Self {
        Self {
            w: 0.0,
            mean: vec![R::zero(); dim],
            scatter: vec![R::zero(); dim * (dim + 1) / 2],
            dim,
        }
    }
    fn is_well_formed(&self) -> bool {
        self.mean.len() == self.dim && self.scatter.len() == self.dim * (self.dim + 1) / 2
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn weight(&self) -> R {
        narrow(self.w)
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        (0..self.dim).map(|i| self.at(i, i)).sum()
    }
    fn variance(&self, d: usize) -> R {
        if self.w <= 0.0 {
            R::zero()
        } else {
            self.at(d, d) / narrow(self.w)
        }
    }
    fn cov_dense(&self) -> Vec<Vec<R>> {
        self.covariance()
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let wi = wide(w);
        let w_new = self.w + wi;
        let factor: R = narrow(wi / w_new);
        let coef: R = narrow(wi * (1.0 - wi / w_new));
        let d: Vec<R> = (0..self.dim).map(|i| x[i] - self.mean[i]).collect();
        for i in 0..self.dim {
            for j in 0..=i {
                let k = self.idx(j, i);
                self.scatter[k] = self.scatter[k] + coef * d[i] * d[j];
            }
        }
        for i in 0..self.dim {
            self.mean[i] = self.mean[i] + factor * d[i];
        }
        self.w = w_new;
    }
    fn merge(&mut self, other: &Self) {
        if other.w <= 0.0 {
            return;
        }
        if self.w <= 0.0 {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor: R = narrow(other.w / w_new);
        let c: R = narrow(self.w * other.w / w_new);
        let d: Vec<R> = (0..self.dim)
            .map(|i| other.mean[i] - self.mean[i])
            .collect();
        for i in 0..self.dim {
            for j in 0..=i {
                let k = self.idx(j, i);
                self.scatter[k] = self.scatter[k] + other.scatter[k] + c * d[i] * d[j];
            }
        }
        for i in 0..self.dim {
            self.mean[i] = self.mean[i] + factor * d[i];
        }
        self.w = w_new;
    }
    fn decay(&mut self, factor: R) {
        self.w *= wide(factor);
        for s in &mut self.scatter {
            *s = *s * factor;
        }
    }
}

// ──────────────────── Frequent-Directions sketch (high-dimensional) ────────────────────

/// Default sketch size for [`FdSketch::new`]: `ℓ = min(dim, FD_DEFAULT_ELL)` rows.
pub const FD_DEFAULT_ELL: usize = 32;

/// High-dimensional feature whose scatter matrix is approximated by a Frequent-Directions sketch
/// `B` (an `ℓ × d` matrix, `ℓ ≪ d`): `M ≈ BᵀB` in `O(ℓ·d)` memory instead of `O(d²)`. Mean and
/// weight are exact; each Welford rank-1 scatter update inserts one weighted, centred row, and the
/// sketch is periodically shrunk (Liberty 2013) keeping the dominant directions. For large `d`
/// (embeddings) where a full `d×d` covariance per leaf is too big to store.
///
/// The shrink discards real mass — measured at 10–34 % of a leaf's scatter for `ℓ` between 16 and 64
/// — and used to drop it on the floor, so the radius, the absorption gate, Ward's cost and the GMM
/// all saw a leaf up to a third tighter than it is. `lost` now records the discarded trace exactly,
/// and [`ssd`](ClusterFeature::ssd) reports `tr(BᵀB) + lost`, which is the leaf's true total scatter.
///
/// Giving that trace back to the *shape* is a modelling choice, since the shrink destroyed the
/// directions along with the magnitudes, and both candidates were measured on the `gmm` head:
/// scaling the retained directions up to carry it (chosen) against spreading it isotropically over
/// all `d` (refuted — it fills `d − ℓ` directions that hold no data, and costs up to 0.32 ARI). See
/// `bench/RESULTS.md`. When a shrink leaves no direction at all, isotropic is the only choice left.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct FdSketch<R: Real> {
    w: f64,
    mean: Vec<R>,
    sketch: Vec<Vec<R>>, // `ell` rows × `dim`; rows `[0, self.rows)` are live
    rows: usize,
    ell: usize,
    dim: usize,
    /// Trace of the scatter the shrinks discarded, in the same units as [`ClusterFeature::ssd`].
    #[cfg_attr(feature = "persistence", serde(default))]
    lost: f64,
}

#[allow(clippy::needless_range_loop)] // sketch/gram math reads clearest with explicit indices
impl<R: Real> FdSketch<R> {
    /// Empty sketch with an explicit row budget `ell` (clamped to `[1, dim]`).
    pub fn with_ell(dim: usize, ell: usize) -> Self {
        let ell = ell.max(1).min(dim.max(1));
        Self {
            w: 0.0,
            mean: vec![R::zero(); dim],
            sketch: vec![vec![R::zero(); dim]; ell],
            rows: 0,
            ell,
            dim,
            lost: 0.0,
        }
    }

    /// Trace of the scatter the sketch still holds explicitly, `tr(BᵀB)`.
    fn kept(&self) -> R {
        self.sketch
            .iter()
            .take(self.rows)
            .flatten()
            .map(|&x| x * x)
            .fold(R::zero(), |a, b| a + b)
    }

    /// How to give the discarded trace back: a factor on the retained directions, or — only when the
    /// shrink left no direction at all — an isotropic per-dimension variance.
    fn completion(&self) -> (R, R) {
        if self.lost <= 0.0 || self.w <= 0.0 || self.dim == 0 {
            return (R::one(), R::zero());
        }
        let kept = self.kept();
        if kept > R::zero() {
            ((kept + narrow(self.lost)) / kept, R::zero())
        } else {
            (R::one(), narrow(self.lost / (self.w * self.dim as f64)))
        }
    }

    /// Frequent-Directions shrink of the (full) sketch: subtract the smallest squared singular
    /// value from all of them, zeroing ≥1 row. Uses the eigendecomposition of the small `ℓ×ℓ`
    /// Gram matrix `BBᵀ = U Σ² Uᵀ`; the shrunk sketch is `diag(σ'/σ) Uᵀ B`.
    fn reduce(&mut self) {
        let (ell, dim) = (self.ell, self.dim);
        let mut g = vec![vec![R::zero(); ell]; ell];
        for i in 0..ell {
            for j in 0..=i {
                let dot: R = (0..dim)
                    .map(|d| self.sketch[i][d] * self.sketch[j][d])
                    .fold(R::zero(), |a, b| a + b);
                g[i][j] = dot;
                g[j][i] = dot;
            }
        }
        let (eig, u) = linalg::jacobi_eigen(&g);
        // Shrink by the lower-median squared singular value (Ghashami et al.): this zeroes ~half the
        // rows per reduce — so the sketch is rebuilt every ~ℓ/2 inserts instead of every insert —
        // while the dominant directions (and any exact low-rank structure) survive.
        let mut sorted = eig.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let delta = sorted[(ell - 1) / 2].max(R::zero());
        let tiny = R::from_f64(1e-300).unwrap();
        let mut next = vec![vec![R::zero(); dim]; ell];
        let mut new_rows = 0;
        for i in 0..ell {
            let sigma2 = eig[i].max(R::zero());
            let sigma = sigma2.sqrt();
            let sigma_p = (sigma2 - delta).max(R::zero()).sqrt();
            // Whatever this direction gives up — `delta`, or all of `sigma2` for a freed row — is
            // real scatter of the leaf. Bank its trace instead of dropping it on the floor.
            self.lost = self.lost + wide(sigma2) - wide(sigma_p * sigma_p);
            if sigma <= tiny || sigma_p <= tiny {
                continue; // null or shrunk-to-zero direction → row freed
            }
            let scale = sigma_p / sigma;
            let dst = &mut next[new_rows];
            for j in 0..ell {
                let coef = scale * u[j][i];
                if coef != R::zero() {
                    for d in 0..dim {
                        dst[d] = dst[d] + coef * self.sketch[j][d];
                    }
                }
            }
            new_rows += 1;
        }
        self.sketch = next;
        self.rows = new_rows;
    }

    /// Insert one row, shrinking first if the sketch is full.
    fn insert_row(&mut self, row: Vec<R>) {
        if self.rows == self.ell {
            self.reduce();
        }
        self.sketch[self.rows] = row;
        self.rows += 1;
    }
}

#[allow(clippy::needless_range_loop)] // Welford / sketch updates read clearest with explicit indices
impl<R: Real> ClusterFeature<R> for FdSketch<R> {
    fn new(dim: usize) -> Self {
        Self::with_ell(dim, FD_DEFAULT_ELL)
    }
    fn is_well_formed(&self) -> bool {
        self.mean.len() == self.dim
            && self.ell >= 1
            && self.sketch.len() == self.ell
            && self.rows <= self.ell
            && self.sketch.iter().all(|r| r.len() == self.dim)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn weight(&self) -> R {
        narrow(self.w)
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        self.kept() + narrow(self.lost)
    }
    fn variance(&self, d: usize) -> R {
        if self.w <= 0.0 {
            return R::zero();
        }
        let (scale, iso) = self.completion();
        let s: R = (0..self.rows)
            .map(|i| self.sketch[i][d] * self.sketch[i][d])
            .fold(R::zero(), |a, b| a + b);
        scale * s / narrow(self.w) + iso
    }
    fn cov_dense(&self) -> Vec<Vec<R>> {
        let dim = self.dim;
        let mut m = vec![vec![R::zero(); dim]; dim];
        for i in 0..self.rows {
            for a in 0..dim {
                let sia = self.sketch[i][a];
                if sia != R::zero() {
                    for b in 0..dim {
                        m[a][b] = m[a][b] + sia * self.sketch[i][b];
                    }
                }
            }
        }
        let (scale, iso) = self.completion();
        if self.w > 0.0 {
            let f = scale / narrow(self.w);
            for row in m.iter_mut() {
                for x in row.iter_mut() {
                    *x = *x * f;
                }
            }
        }
        if iso != R::zero() {
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = row[i] + iso;
            }
        }
        m
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let wi = wide(w);
        let w_new = self.w + wi;
        let factor: R = narrow(wi / w_new);
        let coef: R = narrow(wi * (1.0 - wi / w_new));
        let delta: Vec<R> = (0..self.dim).map(|i| x[i] - self.mean[i]).collect();
        let sq = coef.sqrt();
        let row: Vec<R> = delta.iter().map(|&di| sq * di).collect();
        self.insert_row(row);
        for i in 0..self.dim {
            self.mean[i] = self.mean[i] + factor * delta[i];
        }
        self.w = w_new;
    }
    fn merge(&mut self, other: &Self) {
        if other.w <= 0.0 {
            return;
        }
        if self.w <= 0.0 {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor: R = narrow(other.w / w_new);
        let c: R = narrow(self.w * other.w / w_new);
        let delta: Vec<R> = (0..self.dim)
            .map(|i| other.mean[i] - self.mean[i])
            .collect();
        self.lost += other.lost;
        for i in 0..other.rows {
            self.insert_row(other.sketch[i].clone());
        }
        let sq = c.sqrt();
        let corr: Vec<R> = delta.iter().map(|&di| sq * di).collect();
        self.insert_row(corr);
        for i in 0..self.dim {
            self.mean[i] = self.mean[i] + factor * delta[i];
        }
        self.w = w_new;
    }
    fn decay(&mut self, factor: R) {
        let f = wide(factor);
        self.w *= f;
        self.lost *= f; // rows scale by √factor, so their scatter scales by factor
        let s = factor.sqrt();
        for row in self.sketch.iter_mut().take(self.rows) {
            for x in row.iter_mut() {
                *x = *x * s;
            }
        }
    }
    fn second_moment(&self) -> SecondMoment<R> {
        // Σ = BᵀB / n + iso·I = Σ_r (b_r/√n)(b_r/√n)ᵀ + iso·I — the sketch rows scaled by 1/√n are
        // the low-rank factors, so the GMM never reconstructs the `d×d` covariance for this leaf.
        if self.w <= 0.0 {
            return SecondMoment::LowRank {
                rows: Vec::new(),
                iso: R::zero(),
                dim: self.dim,
            };
        }
        let (fill, iso) = self.completion();
        let scale = (fill / narrow(self.w)).sqrt();
        let rows = self
            .sketch
            .iter()
            .take(self.rows)
            .map(|row| row.iter().map(|&x| x * scale).collect())
            .collect();
        SecondMoment::LowRank {
            rows,
            iso,
            dim: self.dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    fn push_all<C: ClusterFeature<f64>>(dim: usize, pts: &[&[f64]]) -> C {
        let mut c = C::new(dim);
        for p in pts {
            c.push(p, 1.0);
        }
        c
    }

    /// `is_well_formed` earns its place only where a feature holds two arrays that have to agree
    /// *and* `dim()` cannot see the disagreement — which is the case for all three of these, and
    /// for none of `Spherical`, whose dimension is its mean's length.
    #[test]
    fn a_feature_whose_own_arrays_disagree_is_not_well_formed() {
        let mut diag: Diagonal<f64> = push_all(3, &[&[1.0, 2.0, 3.0], &[2.0, 0.0, 1.0]]);
        assert!(diag.is_well_formed());
        diag.ssd.pop();
        assert_eq!(
            diag.dim(),
            3,
            "the mean still says 3, so only this check can see it"
        );
        assert!(!diag.is_well_formed());

        let mut full: Full<f64> = push_all(3, &[&[1.0, 2.0, 3.0], &[2.0, 0.0, 1.0]]);
        assert!(full.is_well_formed());
        full.scatter.pop();
        assert_eq!(full.dim(), 3);
        assert!(!full.is_well_formed());

        let mut fd: FdSketch<f64> = push_all(4, &[&[1.0, 2.0, 3.0, 4.0], &[2.0, 0.0, 1.0, 5.0]]);
        assert!(fd.is_well_formed());
        fd.sketch[0].pop();
        assert!(!fd.is_well_formed(), "a sketch row narrower than the mean");
        let mut fd: FdSketch<f64> = push_all(4, &[&[1.0, 2.0, 3.0, 4.0]]);
        fd.rows = fd.ell + 1;
        assert!(!fd.is_well_formed(), "more live rows than the sketch has");

        let sph: Spherical<f64> = push_all(3, &[&[1.0, 2.0, 3.0]]);
        assert!(sph.is_well_formed(), "nothing in a Spherical can disagree");
    }

    #[test]
    fn spherical_basic() {
        let c: Spherical<f64> = push_all(3, &[&[1., 2., 3.], &[3., 4., 5.], &[5., 6., 7.]]);
        assert!(close(c.weight(), 3.0));
        assert!(close(c.mean()[0], 3.0) && close(c.mean()[1], 4.0) && close(c.mean()[2], 5.0));
        assert!(close(c.ssd(), 24.0)); // each dim [1,3,5] mean 3 -> 8; total 24
        assert!(close(c.variance(0), 24.0 / 3.0 / 3.0)); // ssd / n / dim
    }

    #[test]
    fn diagonal_basic() {
        let c: Diagonal<f64> = push_all(2, &[&[0., 0.], &[10., 1.], &[20., 2.]]);
        assert!(close(c.variance(0), 200.0 / 3.0));
        assert!(close(c.variance(1), 2.0 / 3.0));
        assert!(close(c.ssd(), 202.0));
    }

    #[test]
    fn full_covariance_known() {
        // y = 2x: (0,0),(1,2),(2,4); mean (1,2)
        let c: Full<f64> = push_all(2, &[&[0., 0.], &[1., 2.], &[2., 4.]]);
        let cov = c.covariance();
        assert!(close(cov[0][0], 2.0 / 3.0));
        assert!(close(cov[1][1], 8.0 / 3.0));
        assert!(close(cov[0][1], 4.0 / 3.0));
        assert!(close(cov[0][1], cov[1][0]));
    }

    #[test]
    fn merge_equals_sequential_all_models() {
        let pts: Vec<Vec<f64>> = (0..12)
            .map(|i| {
                let f = i as f64;
                vec![
                    f.sin(),
                    (f * 0.7).cos(),
                    (i % 5) as f64,
                    f.sqrt(),
                    ((i * i) % 7) as f64,
                ]
            })
            .collect();
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        for d in [2usize, 5] {
            macro_rules! check {
                ($t:ty) => {{
                    let mut full = <$t>::new(d);
                    for p in &refs {
                        full.push(&p[..d], 1.0);
                    }
                    let mut a = <$t>::new(d);
                    for p in &refs[..7] {
                        a.push(&p[..d], 1.0);
                    }
                    let mut b = <$t>::new(d);
                    for p in &refs[7..] {
                        b.push(&p[..d], 1.0);
                    }
                    a.merge(&b);
                    assert!(close(a.weight(), full.weight()));
                    for i in 0..d {
                        assert!(close(a.mean()[i], full.mean()[i]), "mean d={} i={}", d, i);
                    }
                    assert!(close(a.ssd(), full.ssd()), "ssd d={}", d);
                    for i in 0..d {
                        assert!(
                            close(a.variance(i), full.variance(i)),
                            "var d={} i={}",
                            d,
                            i
                        );
                    }
                }};
            }
            check!(Spherical<f64>);
            check!(Diagonal<f64>);
            check!(Full<f64>);
        }
    }

    #[test]
    fn full_merge_preserves_cross_products_dim5() {
        // dim>=4 is exactly where the reference upper-triangular index bug corrupted cross terms.
        let pts: Vec<Vec<f64>> = (0..8)
            .map(|i| (0..5).map(|j| ((i * 7 + j * 3) % 11) as f64).collect())
            .collect();
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let mut full: Full<f64> = Full::new(5);
        for p in &refs {
            full.push(p, 1.0);
        }
        let mut a: Full<f64> = Full::new(5);
        for p in &refs[..3] {
            a.push(p, 1.0);
        }
        let mut b: Full<f64> = Full::new(5);
        for p in &refs[3..] {
            b.push(p, 1.0);
        }
        a.merge(&b);
        let (ca, cf) = (a.covariance(), full.covariance());
        for i in 0..5 {
            for j in 0..5 {
                assert!(close(ca[i][j], cf[i][j]), "cov mismatch ({}, {})", i, j);
            }
        }
    }

    /// 8 points confined to a 2-D subspace of R⁴ (dims 2,3 are 0): rank ≤ 2 < ℓ, so the FD sketch
    /// is lossless and must match the exact full covariance.
    fn low_rank_pts() -> Vec<Vec<f64>> {
        (0..8)
            .map(|i| {
                let f = i as f64;
                vec![f.sin(), (f * 0.7).cos(), 0.0, 0.0]
            })
            .collect()
    }

    #[test]
    fn fd_sketch_matches_full_on_low_rank() {
        let pts = low_rank_pts();
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let full: Full<f64> = push_all(4, &refs);
        let fd: FdSketch<f64> = push_all(4, &refs);
        let tol = |a: f64, b: f64| (a - b).abs() < 1e-6;
        assert!(tol(fd.weight(), full.weight()));
        for i in 0..4 {
            assert!(tol(fd.mean()[i], full.mean()[i]), "mean {i}");
        }
        assert!(
            tol(fd.ssd(), full.ssd()),
            "ssd {} vs {}",
            fd.ssd(),
            full.ssd()
        );
        let (cfd, cfull) = (fd.cov_dense(), full.cov_dense());
        for i in 0..4 {
            for j in 0..4 {
                assert!(tol(cfd[i][j], cfull[i][j]), "cov ({i},{j})");
            }
        }
    }

    #[test]
    fn fd_sketch_merge_matches_full_on_low_rank() {
        let pts = low_rank_pts();
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let full: Full<f64> = push_all(4, &refs);
        let mut a: FdSketch<f64> = push_all(4, &refs[..3]);
        let b: FdSketch<f64> = push_all(4, &refs[3..]);
        a.merge(&b);
        let tol = |x: f64, y: f64| (x - y).abs() < 1e-6;
        assert!(tol(a.weight(), full.weight()));
        let (ca, cf) = (a.cov_dense(), full.cov_dense());
        for i in 0..4 {
            for j in 0..4 {
                assert!(tol(ca[i][j], cf[i][j]), "cov ({i},{j})");
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // symmetric-matrix check reads clearest with (i, j)
    fn fd_sketch_reports_the_exact_scatter_and_stays_symmetric() {
        // Full-rank data with ℓ < d: the retained sketch underestimates the scatter (FD's
        // BᵀB ⪯ AᵀA), but the shrink banks what it drops, so the total this leaf reports is exact —
        // the radius, the absorption gate and Ward all read `ssd()`, and a third of it going
        // missing is not a rounding question.
        let pts: Vec<Vec<f64>> = (0..40)
            .map(|i| {
                (0..10)
                    .map(|j| (((i * 7 + j * 3) % 11) as f64).sin())
                    .collect()
            })
            .collect();
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let full: Full<f64> = push_all(10, &refs);
        let mut fd: FdSketch<f64> = FdSketch::with_ell(10, 4); // ℓ = 4 < d = 10
        for p in &refs {
            fd.push(p, 1.0);
        }
        let retained: f64 = fd
            .cov_dense()
            .iter()
            .enumerate()
            .map(|(i, r)| r[i])
            .sum::<f64>()
            * fd.weight()
            - fd.ssd();
        assert!(retained.abs() < 1e-9, "cov_dense trace must equal ssd");
        assert!(
            (fd.ssd() - full.ssd()).abs() < 1e-9,
            "fd ssd {} != full {}",
            fd.ssd(),
            full.ssd()
        );
        let cov = fd.cov_dense();
        for i in 0..10 {
            for j in 0..10 {
                assert!((cov[i][j] - cov[j][i]).abs() < 1e-9, "asymmetric ({i},{j})");
            }
        }
    }

    /// A deterministic full-rank cloud: no circulant or trigonometric structure that could make
    /// the sketch lossless behind the test's back.
    fn full_rank_cloud(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
        };
        (0..n).map(|_| (0..d).map(|_| next()).collect()).collect()
    }

    /// The sketch on data it **cannot** hold exactly — the regime every other FD test here avoids.
    ///
    /// `fd_sketch_reports_the_exact_scatter_and_stays_symmetric` and
    /// `the_low_rank_factors_reconstruct_the_dense_covariance` both feed the sketch data of rank
    /// `<= ell`, so `lost` stays 0, `completion` returns from its first line, and everything after
    /// it is unmeasured. A whole-file mutation run of `src/feature.rs` (2026-09-10) left 20 mutants
    /// alive in `completion`, `variance`, `cov_dense`, `merge` and `reduce` for exactly that reason.
    /// The `assert!(lost > 0)` is load-bearing: without it a future fixture could drift back into
    /// the lossless regime and this test would keep passing while measuring nothing.
    #[test]
    fn a_sketch_that_really_loses_a_direction_still_reports_the_exact_scatter() {
        let (n, d, ell) = (30, 6, 3);
        let pts = full_rank_cloud(n, d, 20260910);
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let full: Full<f64> = push_all(d, &refs);
        let mut fd: FdSketch<f64> = FdSketch::with_ell(d, ell);
        for p in &refs {
            fd.push(p, 1.0);
        }
        assert!(
            fd.lost > 0.0,
            "the fixture is not lossy, so it measures nothing"
        );
        assert!(fd.kept() > 0.0, "and it must still retain a direction");

        // FD's promise: the *total* scatter survives the shrink even though the directions do not.
        assert!(
            (fd.ssd() - full.ssd()).abs() < 1e-9,
            "fd ssd {} != full {}",
            fd.ssd(),
            full.ssd()
        );
        // Both readings of that total have to agree with it: `variance` summed over dimensions and
        // the trace of `cov_dense` are the same number by construction, and both go through
        // `completion`.
        let by_variance: f64 = (0..d).map(|j| fd.variance(j)).sum::<f64>() * fd.weight();
        let by_trace: f64 = fd
            .cov_dense()
            .iter()
            .enumerate()
            .map(|(i, r)| r[i])
            .sum::<f64>()
            * fd.weight();
        assert!(
            (by_variance - fd.ssd()).abs() < 1e-9,
            "Σ variance = {by_variance}"
        );
        assert!(
            (by_trace - fd.ssd()).abs() < 1e-9,
            "tr cov_dense = {by_trace}"
        );

        let cov = fd.cov_dense();
        for (i, row) in cov.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                assert!((x - cov[j][i]).abs() < 1e-12, "asymmetric ({i},{j})");
            }
        }
        // With a direction still retained the completion is a *scale* on it, never an isotropic
        // floor — which is the branch `completion` picks, not a detail of it.
        let SecondMoment::LowRank { rows, iso, dim } = fd.second_moment() else {
            panic!("FdSketch must expose low-rank factors");
        };
        assert_eq!(dim, d);
        assert_eq!(
            iso, 0.0,
            "a retained direction must not be completed isotropically"
        );
        for a in 0..d {
            for b in 0..d {
                let got: f64 = rows.iter().map(|r| r[a] * r[b]).sum();
                assert!((got - cov[a][b]).abs() < 1e-9, "factors ≠ cov at ({a},{b})");
            }
        }

        // Merging two *lossy* sketches: `lost` is additive, and the Chan correction row keeps the
        // combined total exact even though neither side holds its own directions any more.
        let (l, r) = pts.split_at(n / 2);
        let mut a: FdSketch<f64> = FdSketch::with_ell(d, ell);
        let mut b: FdSketch<f64> = FdSketch::with_ell(d, ell);
        for p in l {
            a.push(p, 1.0);
        }
        for p in r {
            b.push(p, 1.0);
        }
        assert!(
            a.lost > 0.0 && b.lost > 0.0,
            "both sides must have lost something"
        );
        a.merge(&b);
        assert!(
            (a.ssd() - full.ssd()).abs() < 1e-9,
            "merged ssd {} != full {}",
            a.ssd(),
            full.ssd()
        );
    }

    /// The other branch of `completion`: a shrink that leaves **no** direction at all, where the
    /// only honest way to give the banked trace back is to spread it evenly.
    ///
    /// Reachable, not hypothetical: with `ell = 1` a point that lands exactly on the running mean
    /// contributes a zero row, so the sketch holds a row and retains nothing.
    #[test]
    fn a_sketch_shrunk_to_no_direction_reports_an_isotropic_variance() {
        let d = 3;
        let mut fd: FdSketch<f64> = FdSketch::with_ell(d, 1);
        let pts: Vec<Vec<f64>> = vec![
            vec![-2.0, 1.0, 0.5],
            vec![2.0, -1.0, -0.5],
            vec![1.0, 3.0, -2.0],
            vec![-1.0, -3.0, 2.0],
        ];
        for p in &pts {
            fd.push(p, 1.0);
        }
        let mean = fd.mean().to_vec();
        fd.push(&mean, 1.0); // a zero row: the sketch now holds a direction of length nothing

        assert_eq!(fd.kept(), 0.0, "the fixture must retain nothing");
        assert!(fd.lost > 0.0);
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let mut full: Full<f64> = push_all(d, &refs);
        full.push(&mean, 1.0);
        assert!(
            (fd.ssd() - full.ssd()).abs() < 1e-9,
            "fd ssd {} != full {}",
            fd.ssd(),
            full.ssd()
        );

        let v: Vec<f64> = (0..d).map(|j| fd.variance(j)).collect();
        assert!(
            v.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-12),
            "with no direction left the variance can only be isotropic: {v:?}"
        );
        assert!((v.iter().sum::<f64>() * fd.weight() - fd.ssd()).abs() < 1e-9);
        let SecondMoment::LowRank { rows, iso, .. } = fd.second_moment() else {
            panic!("FdSketch must expose low-rank factors");
        };
        assert!(
            rows.iter().flatten().all(|&x| x == 0.0),
            "the retained factors carry no direction: {rows:?}"
        );
        assert!(
            iso > 0.0,
            "the banked trace has to come back as an isotropic floor"
        );
        assert!((iso * (d as f64) * fd.weight() - fd.ssd()).abs() < 1e-9);
    }

    /// The shrink is by the **lower median** squared singular value, not by the smallest: that is
    /// what frees about half the rows per reduce and so rebuilds the sketch every ~ℓ/2 inserts
    /// instead of every insert. It is a cost policy, so a test has to pin it — the exactness
    /// assertions above cannot see it, since `lost` banks whatever the shrink gives up either way.
    #[test]
    fn a_reduce_frees_about_half_the_rows_not_one() {
        let (d, ell) = (8, 6);
        let pts = full_rank_cloud(ell + 1, d, 7);
        let mut fd: FdSketch<f64> = FdSketch::with_ell(d, ell);
        for p in &pts {
            fd.push(p, 1.0);
        }
        assert!(fd.lost > 0.0, "the fixture must have triggered a reduce");
        // `ell + 1` inserts fire exactly one reduce, so `rows` reads (ell − freed) + 1. The lower
        // median frees ⌈ell/2⌉ = 3 of 6 and leaves 4; the smallest singular value would free one
        // and leave 6 — full again, and reducing on every insert from then on.
        assert!(
            fd.rows <= ell / 2 + 1,
            "a reduce left {} of {ell} rows; the lower-median rule leaves {}",
            fd.rows,
            ell / 2 + 1
        );
    }

    #[test]
    fn decay_scales_mass_but_preserves_shape() {
        let pts: &[&[f64]] = &[&[0.0, 0.0], &[2.0, 4.0], &[4.0, 2.0], &[1.0, 3.0]];
        // Full: weight scales; mean, variance, covariance unchanged.
        let mut f: Full<f64> = push_all(2, pts);
        let (w0, m0, c01) = (f.weight(), f.mean().to_vec(), f.covariance()[0][1]);
        let (v0, v1) = (f.variance(0), f.variance(1));
        f.decay(0.5);
        assert!(close(f.weight(), 0.5 * w0));
        assert!(close(f.mean()[0], m0[0]) && close(f.mean()[1], m0[1]));
        assert!(close(f.variance(0), v0) && close(f.variance(1), v1));
        assert!(close(f.covariance()[0][1], c01));
        // FdSketch: rows scale by √factor, so weight scales and variance is preserved.
        let mut s: FdSketch<f64> = push_all(2, pts);
        let (sw, sv) = (s.weight(), s.variance(0));
        s.decay(0.25);
        assert!(close(s.weight(), 0.25 * sw));
        assert!(close(s.variance(0), sv));
    }

    /// What an `f32` tree can and cannot still promise about a weight, now that the running totals
    /// are `f64`.
    ///
    /// The *accumulator* counts every row: `w` is `f64`, so `2^24 + 1` is held exactly. The
    /// *report* is not, because [`ClusterFeature::weight`] hands back an `R` — so a caller reading
    /// an `f32` tree sees the nearest binary32, one part in `2^24`. That is a rounding of a correct
    /// total, not the old defect, which lost the rows outright: every subsequent row used to add
    /// nothing at all, and the mean and scatter went on moving against a weight that had stopped.
    #[test]
    fn an_f32_weight_is_counted_in_f64_and_reported_rounded() {
        let mut c: Spherical<f32> = Spherical::new(1);
        c.push(&[0.0], 16_777_216.0); // 2^24, reached in one weighted push
        c.push(&[1.0], 1.0);
        assert_eq!(
            c.weight(),
            16_777_216.0,
            "binary32 cannot name 2^24 + 1, so the report rounds"
        );
        // But the row is in the total: 4095 more of them cross into the next representable value,
        // which the old f32 accumulator could never reach one unit-weight row at a time.
        for _ in 0..4095 {
            c.push(&[1.0], 1.0);
        }
        assert_eq!(c.weight(), 16_781_312.0, "4096 rows, all of them counted");
        assert!(c.mean()[0] > 0.0);
        assert!(c.ssd() > 0.0);
    }

    /// The acceptance test for the `f64`-accumulator fix, by the route a user actually reaches the
    /// ceiling by: one leaf of an `f32` tree absorbing more than `2^24` unit-weight rows.
    ///
    /// It failed until 2026-09-10, reporting 16 777 216 against the 16 781 312 rows it was given —
    /// the measurement behind a limit `docs/USAGE.md` used to document and the estimator used to
    /// warn about. It costs 2 s in a debug build, which is why it runs in the suite rather than
    /// behind `#[ignore]`: a 33 M-row loop is the only way to reach the boundary honestly, and the
    /// promise is worth two seconds. The `f64` control in the same body was exact before and after.
    #[test]
    fn an_f32_leaf_counts_every_row_it_absorbs() {
        const ROWS: usize = 16_781_312; // 2^24 + 4096
        let mut narrow: Spherical<f32> = Spherical::new(1);
        let mut wide: Spherical<f64> = Spherical::new(1);
        for _ in 0..ROWS {
            narrow.push(&[1.0], 1.0);
            wide.push(&[1.0], 1.0);
        }
        assert_eq!(wide.weight(), ROWS as f64, "the f64 control is exact");
        assert_eq!(narrow.weight(), ROWS as f32);
    }

    /// The consequence the weight ceiling actually had on an answer, and the reason the *scalar*
    /// scatter is `f64` too.
    ///
    /// A frozen weight did not stop the leaf: the scatter kept accumulating against it, so
    /// `variance = ssd / w` drifted by roughly the fraction of rows the weight had dropped — on this
    /// fixture, 200 000 unit-variance rows past `2^24`, it read **1.0058134**. It now reads 1.0 to a
    /// part in a million; the bound is loose only so that a different rounding mode cannot make the
    /// test brittle about a defect three orders of magnitude larger than it.
    ///
    /// Widening the weight alone would have moved the ceiling rather than removed it, because `ssd`
    /// reaches `2^24` on this fixture at the same time the weight does. That is free here — one
    /// value per leaf — and is not free for the per-axis models, which is the subject of
    /// [`an_f32_diagonal_leaf_past_the_ceiling_still_loses_scatter`].
    #[test]
    fn an_f32_leaf_past_the_ceiling_reports_the_variance_it_holds() {
        const ROWS: usize = 16_977_216; // 2^24 + 200 000
        let mut sph: Spherical<f32> = Spherical::new(1);
        for i in 0..ROWS {
            sph.push(&[if i % 2 == 0 { 1.0 } else { -1.0 }], 1.0);
        }
        let v = sph.variance(0);
        assert!(
            (v - 1.0).abs() < 1e-6,
            "spherical reports {v} on unit-variance data past 2^24; the defect read 1.0058134"
        );
    }

    /// What an `f32` [`Diagonal`] still gets wrong past `2^24`, stated as a test so the figure in
    /// `docs/USAGE.md` cannot drift away from the arithmetic.
    ///
    /// The per-axis scatter is `d` values per leaf, touched once per (row, dimension). Widening it
    /// was implemented and measured: **+11 %** on the insert path and an `f32` tree three quarters
    /// the size of the `f64` one instead of half, charged to every `f32` user — to correct a leaf
    /// that has to be past `2^24` rows before it is wrong at all. So it stays `f32`, and this is
    /// what that costs: with the weight now exact the scatter is the term that saturates, and the
    /// variance comes back **0.89 % low** where the original defect had it 0.58 % high.
    ///
    /// `spherical` (the default) and `f64` are exact; `full` and `fd` share this residual and reach
    /// it in the same regime.
    #[test]
    fn an_f32_diagonal_leaf_past_the_ceiling_still_loses_scatter() {
        const ROWS: usize = 16_977_216; // 2^24 + 200 000
        let mut diag: Diagonal<f32> = Diagonal::new(1);
        let mut control: Diagonal<f64> = Diagonal::new(1);
        for i in 0..ROWS {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            diag.push(&[x], 1.0);
            control.push(&[f64::from(x)], 1.0);
        }
        assert_eq!(diag.weight(), ROWS as f32, "the weight itself is exact");
        assert!(
            (control.variance(0) - 1.0).abs() < 1e-12,
            "the f64 control is exact: {}",
            control.variance(0)
        );
        let v = diag.variance(0);
        assert!(
            (0.985..0.995).contains(&v),
            "the f32 per-axis scatter loses about 0.9 % here; it read {v}"
        );
    }

    #[test]
    fn weighted_push_equals_repeats() {
        let mut a: Diagonal<f64> = Diagonal::new(2);
        a.push(&[1.0, 2.0], 3.0);
        a.push(&[4.0, 0.0], 1.0);
        let mut b: Diagonal<f64> = Diagonal::new(2);
        for _ in 0..3 {
            b.push(&[1.0, 2.0], 1.0);
        }
        b.push(&[4.0, 0.0], 1.0);
        assert!(close(a.weight(), b.weight()));
        assert!(close(a.mean()[0], b.mean()[0]) && close(a.mean()[1], b.mean()[1]));
        assert!(close(a.ssd(), b.ssd()));
    }

    #[test]
    fn full_cholesky_and_mahalanobis() {
        // 4 points -> mean (0,0), covariance diag(1,4)
        let c: Full<f64> = push_all(2, &[&[-1., -2.], &[1., 2.], &[-1., 2.], &[1., -2.]]);
        assert!(close(c.mahalanobis_sq(&[0., 0.]).unwrap(), 0.0));
        assert!(close(c.mahalanobis_sq(&[2., 2.]).unwrap(), 5.0)); // 4/1 + 4/4
        assert!(close(c.logdet().unwrap(), 4.0_f64.ln())); // det = 1*4
    }

    #[test]
    fn full_cholesky_none_when_rank_deficient() {
        let c: Full<f64> = push_all(3, &[&[1., 2., 3.], &[4., 1., 2.]]); // 2 points in 3D
        assert!(c.cholesky().is_none());
    }

    #[test]
    fn decay_spherical_and_diagonal() {
        let pts: &[&[f64]] = &[&[0.0, 0.0], &[2.0, 4.0], &[4.0, 2.0]];
        let mut s: Spherical<f64> = push_all(2, pts);
        let (sw, sv, sm) = (s.weight(), s.variance(0), s.mean().to_vec());
        s.decay(0.5);
        assert!(close(s.weight(), 0.5 * sw) && close(s.variance(0), sv));
        assert!(close(s.mean()[0], sm[0]) && close(s.mean()[1], sm[1]));
        let mut d: Diagonal<f64> = push_all(2, pts);
        let (dw, dv0, dv1) = (d.weight(), d.variance(0), d.variance(1));
        d.decay(0.5);
        assert!(close(d.weight(), 0.5 * dw));
        assert!(close(d.variance(0), dv0) && close(d.variance(1), dv1));
    }

    #[test]
    fn edge_guards_all_models() {
        // Empty features, non-positive-weight pushes, and merges with an empty feature must be safe
        // no-ops / clones across every covariance model — exercises the boundary guards directly.
        macro_rules! check {
            ($t:ty) => {{
                let mut e: $t = <$t>::new(3);
                assert_eq!(e.dim(), 3);
                assert!(close(e.weight(), 0.0));
                assert!(close(e.variance(0), 0.0)); // empty → zero variance
                assert_eq!(e.cov_dense().len(), 3); // empty covariance is well-formed
                e.push(&[1.0, 2.0, 3.0], 0.0); // non-positive weight is a no-op
                assert!(close(e.weight(), 0.0) && close(e.mean()[0], 0.0)); // no NaN
                let full: $t = push_all(3, &[&[1., 2., 3.], &[4., 5., 6.]]);
                let mut from_empty: $t = <$t>::new(3);
                from_empty.merge(&full); // empty.merge(full) → clones
                assert!(close(from_empty.weight(), full.weight()));
                let mut keep = full.clone();
                keep.merge(&<$t>::new(3)); // full.merge(empty) → no-op
                assert!(close(keep.weight(), full.weight()));
                assert_eq!(full.cov_dense().len(), 3);
            }};
        }
        check!(Spherical<f64>);
        check!(Diagonal<f64>);
        check!(Full<f64>);
        check!(FdSketch<f64>);
    }

    #[test]
    fn fd_sketch_empty_second_moment_is_empty_low_rank() {
        let e: FdSketch<f64> = FdSketch::new(3);
        match e.second_moment() {
            SecondMoment::LowRank { rows, iso, dim } => {
                assert!(rows.is_empty());
                assert_eq!(iso, 0.0);
                assert_eq!(dim, 3);
            }
            SecondMoment::Dense(_) => panic!("FD sketch must yield a low-rank second moment"),
        }
    }

    /// `FdSketch::second_moment` hands the GMM the low-rank factors `b_r/√n` instead of a dense
    /// `d×d` block, so `Σ_r (b_r/√n)(b_r/√n)ᵀ` must reproduce `cov_dense` exactly. Nothing asserted
    /// that identity, which left the `1/√n` scaling and the merge's cross-term free to drift.
    #[test]
    fn the_low_rank_factors_reconstruct_the_dense_covariance() {
        let d = 4;
        let mut fd: FdSketch<f64> = FdSketch::with_ell(d, d);
        for p in [
            [1.0, 0.0, 2.0, -1.0],
            [0.0, 3.0, 1.0, 0.5],
            [2.0, 1.0, 0.0, 1.5],
            [-1.0, 2.0, 3.0, 0.0],
        ] {
            fd.push(&p, 1.0);
        }
        let dense = fd.cov_dense();
        let SecondMoment::LowRank { rows, iso, .. } = fd.second_moment() else {
            panic!("FdSketch must expose low-rank factors");
        };
        for a in 0..d {
            for b in 0..d {
                let got: f64 =
                    rows.iter().map(|r| r[a] * r[b]).sum::<f64>() + if a == b { iso } else { 0.0 };
                assert!(
                    (got - dense[a][b]).abs() < 1e-9,
                    "Σ_r r_a r_b at ({a},{b}) = {got}, cov_dense = {}",
                    dense[a][b]
                );
            }
        }

        // An empty sketch has no covariance and therefore no factors.
        let e: FdSketch<f64> = FdSketch::with_ell(d, d);
        assert!(matches!(e.second_moment(), SecondMoment::LowRank { rows, .. } if rows.is_empty()));
        for row in e.cov_dense() {
            assert!(
                row.iter().all(|&x| x == 0.0),
                "empty sketch has a covariance"
            );
        }
    }

    #[test]
    fn merging_sketches_matches_folding_the_same_points_into_one() {
        // The Chan parallel update is exact in weight and mean, and the cross-term row
        // √(n₁n₂/n)·(μ₂−μ₁) is what keeps the *scatter* exact too. `ℓ` is chosen larger than the
        // number of rows either side ever holds (merge inserts `other.rows` plus one correction),
        // so no Frequent-Directions shrink fires and the comparison can be exact rather than
        // approximate.
        let d = 8;
        let left = [
            [1.0, 0.0, 2.0, -1.0, 0.5, 1.0, 0.0, 3.0],
            [0.0, 1.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0],
        ];
        let right = [
            [4.0, 2.0, 0.0, 1.0, 0.0, 3.0, 2.0, 1.0],
            [3.0, 5.0, 1.0, 2.0, 1.0, 0.0, 0.5, 2.0],
            [2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 1.5, 0.0],
        ];

        let mut a: FdSketch<f64> = FdSketch::with_ell(d, d);
        for p in &left {
            a.push(p, 1.0);
        }
        let mut b: FdSketch<f64> = FdSketch::with_ell(d, d);
        for p in &right {
            b.push(p, 1.0);
        }
        a.merge(&b);

        let mut all: FdSketch<f64> = FdSketch::with_ell(d, d);
        for p in left.iter().chain(&right) {
            all.push(p, 1.0);
        }

        assert!((a.weight() - all.weight()).abs() < 1e-12, "weight");
        for i in 0..d {
            assert!(
                (a.mean()[i] - all.mean()[i]).abs() < 1e-12,
                "mean {i}: {} vs {}",
                a.mean()[i],
                all.mean()[i]
            );
        }
        let (ca, cb) = (a.cov_dense(), all.cov_dense());
        for i in 0..d {
            for j in 0..d {
                assert!(
                    (ca[i][j] - cb[i][j]).abs() < 1e-9,
                    "covariance ({i},{j}): {} vs {}",
                    ca[i][j],
                    cb[i][j]
                );
            }
        }

        // Merging an empty sketch is a no-op in both directions.
        let empty: FdSketch<f64> = FdSketch::with_ell(d, d);
        let before = a.mean().to_vec();
        a.merge(&empty);
        assert!(
            (a.weight() - all.weight()).abs() < 1e-12,
            "empty merge changed the weight"
        );
        assert_eq!(a.mean().to_vec(), before);
        let mut fresh: FdSketch<f64> = FdSketch::with_ell(d, d);
        fresh.merge(&all);
        assert!((fresh.weight() - all.weight()).abs() < 1e-12);
        for i in 0..d {
            assert!((fresh.mean()[i] - all.mean()[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn mahalanobis_uses_the_signed_deviation_from_the_mean() {
        // Away from the origin, so `x − μ` and `x + μ` differ. Two points at 3 and 5 in dim 0 and
        // at 10 and 10 in dim 1 give μ = (4, 10) and a singular covariance, so a ridge point is
        // needed: add a third point to make Σ positive definite.
        let mut f: Full<f64> = Full::new(2);
        f.push(&[3.0, 9.0], 1.0);
        f.push(&[5.0, 11.0], 1.0);
        f.push(&[4.0, 13.0], 1.0);
        let at_mean = f
            .mahalanobis_sq(f.mean().to_vec().as_slice())
            .expect("PD covariance");
        assert!(
            at_mean.abs() < 1e-12,
            "the mean is not at distance 0: {at_mean}"
        );

        let mu = f.mean().to_vec();
        let probe = [mu[0] + 1.0, mu[1] + 1.0];
        let mirrored = [mu[0] - 1.0, mu[1] - 1.0];
        let d1 = f.mahalanobis_sq(&probe).unwrap();
        let d2 = f.mahalanobis_sq(&mirrored).unwrap();
        assert!(d1 > 0.0, "a point off the mean scored {d1}");
        assert!(
            (d1 - d2).abs() < 1e-9,
            "the metric is not symmetric about the mean: {d1} vs {d2}"
        );
        // Doubling the deviation quadruples the squared distance.
        let far = [mu[0] + 2.0, mu[1] + 2.0];
        assert!(
            (f.mahalanobis_sq(&far).unwrap() - 4.0 * d1).abs() < 1e-9,
            "not quadratic in the deviation"
        );
    }

    /// `trace` and `apply_rows` are the two products the MPPCA head takes of a leaf's scatter, and
    /// both have a `LowRank` path that never forms `Σ`. Check each against `Σ` built densely, on a
    /// sketch whose rows are neither orthogonal nor axis-aligned.
    #[test]
    fn the_low_rank_scatter_products_agree_with_the_dense_ones() {
        let rows = vec![
            vec![1.0_f64, -2.0, 0.5, 3.0],
            vec![0.0, 1.0, 2.0, -1.0],
            vec![-1.5, 0.25, 1.0, 0.75],
        ];
        let mut dense = vec![vec![0.0_f64; 4]; 4];
        for r in &rows {
            for i in 0..4 {
                for j in 0..4 {
                    dense[i][j] += r[i] * r[j];
                }
            }
        }
        let low = SecondMoment::LowRank {
            rows,
            iso: 0.0,
            dim: 4,
        };
        let full = SecondMoment::Dense(dense.clone());
        assert!(close(low.trace(), full.trace()));
        assert!(close(low.trace(), (0..4).map(|i| dense[i][i]).sum()));

        let v = vec![vec![2.0_f64, 0.0, -1.0, 0.5], vec![0.3, 1.7, 0.0, -2.0]];
        let mut a = vec![vec![0.0_f64; 4]; 2];
        let mut b = vec![vec![0.0_f64; 4]; 2];
        low.apply_rows(&v, &mut a, 1.5);
        full.apply_rows(&v, &mut b, 1.5);
        for (ra, rb) in a.iter().zip(&b) {
            for (&x, &y) in ra.iter().zip(rb) {
                assert!(close(x, y), "{x} vs {y}");
            }
        }
        // …and against the longhand `1.5 · Σ v` rather than only against each other.
        for (r, vr) in v.iter().enumerate() {
            for i in 0..4 {
                let want: f64 = 1.5 * (0..4).map(|j| dense[i][j] * vr[j]).sum::<f64>();
                assert!(
                    close(a[r][i], want),
                    "row {r} dim {i}: {} vs {want}",
                    a[r][i]
                );
            }
        }

        // The diagonal is the third product, and the one the diagonal-noise head needs: `trace`
        // has already summed it away and `apply_rows` cannot isolate it.
        let (mut da, mut db) = (vec![0.0_f64; 4], vec![0.0_f64; 4]);
        low.add_diagonal_scaled(&mut da, 1.5);
        full.add_diagonal_scaled(&mut db, 1.5);
        for i in 0..4 {
            assert!(close(da[i], db[i]), "dim {i}: {} vs {}", da[i], db[i]);
            assert!(close(da[i], 1.5 * dense[i][i]));
        }
        assert!(close(da.iter().sum::<f64>(), 1.5 * low.trace()));

        // The shrink mass a sketch could not place has to reach the diagonal too, or a
        // diagonal-noise head would read a floor of zero where the sketch says otherwise.
        let iso = SecondMoment::LowRank {
            rows: Vec::new(),
            iso: 0.25,
            dim: 4,
        };
        let mut di = vec![0.0_f64; 4];
        iso.add_diagonal_scaled(&mut di, 2.0);
        assert!(di.iter().all(|&v| close(v, 0.5)), "{di:?}");
    }

    /// The same accessors with a **non-zero isotropic floor**, which is what the sketch returns
    /// once a shrink has discarded a direction — and with `trace_under` and `add_scaled`, which the
    /// `iso = 0` test above does not reach at all.
    ///
    /// Every `iso` term in `SecondMoment`, and the `!= 0` guards that skip it, were unmeasured: a
    /// whole-file mutation run of `src/feature.rs` (2026-09-10) left 24 arithmetic mutants alive
    /// across `trace`, `trace_under`, `apply_rows` and `add_scaled` for exactly that reason. Every
    /// number here is checked against longhand as well as across the two variants, so a mutation in
    /// either arm is a failure rather than a disagreement nobody reads.
    #[test]
    fn the_low_rank_products_carry_the_isotropic_floor() {
        let d = 3;
        let iso = 0.25_f64;
        let rows = vec![vec![1.0_f64, 2.0, -1.0], vec![0.5, -1.0, 2.0]];
        let mut dense = vec![vec![0.0_f64; d]; d];
        for r in &rows {
            for a in 0..d {
                for b in 0..d {
                    dense[a][b] += r[a] * r[b];
                }
            }
        }
        for (a, row) in dense.iter_mut().enumerate() {
            row[a] += iso;
        }
        let low = SecondMoment::LowRank {
            rows: rows.clone(),
            iso,
            dim: d,
        };
        let full = SecondMoment::Dense(dense.clone());

        // tr(Σ) = Σ_r ‖f_r‖² + iso·d = 6 + 5.25 + 0.75.
        assert!(close(low.trace(), 12.0), "{}", low.trace());
        assert!(close(full.trace(), 12.0));

        // tr(A⁻¹ Σ) through the Cholesky, against the dense double sum. `a` is not diagonal, so
        // the off-diagonal terms of both formulas are load-bearing.
        let a = vec![
            vec![4.0_f64, 1.0, 0.5],
            vec![1.0, 3.0, -0.5],
            vec![0.5, -0.5, 2.0],
        ];
        let chol = linalg::cholesky_lower(&a).expect("the fixture must be positive definite");
        let inv = linalg::inv_from_chol(&chol);
        let want: f64 = (0..d)
            .map(|i| (0..d).map(|j| inv[i][j] * dense[i][j]).sum::<f64>())
            .sum();
        assert!(close(low.trace_under(&chol, &inv), want), "low: {want}");
        assert!(close(full.trace_under(&chol, &inv), want), "dense: {want}");

        let w = 1.5;
        let v = vec![vec![2.0_f64, 0.0, -1.0], vec![0.3, 1.7, -2.0]];
        let mut la = vec![vec![0.0_f64; d]; 2];
        let mut fa = vec![vec![0.0_f64; d]; 2];
        low.apply_rows(&v, &mut la, w);
        full.apply_rows(&v, &mut fa, w);
        for (r, vr) in v.iter().enumerate() {
            for i in 0..d {
                let want: f64 = w * (0..d).map(|j| dense[i][j] * vr[j]).sum::<f64>();
                assert!(
                    close(la[r][i], want),
                    "low ({r},{i}): {} vs {want}",
                    la[r][i]
                );
                assert!(close(fa[r][i], want), "dense ({r},{i})");
            }
        }

        // `add_scaled` accumulates into a target that is already non-zero, so a `+` that became a
        // `*` cannot hide behind a zero.
        let base = vec![
            vec![0.5_f64, -0.25, 1.0],
            vec![-0.25, 2.0, 0.75],
            vec![1.0, 0.75, -0.5],
        ];
        let (mut lt, mut ft) = (base.clone(), base.clone());
        low.add_scaled(&mut lt, w);
        full.add_scaled(&mut ft, w);
        for i in 0..d {
            for j in 0..d {
                let want = base[i][j] + w * dense[i][j];
                assert!(
                    close(lt[i][j], want),
                    "low ({i},{j}): {} vs {want}",
                    lt[i][j]
                );
                assert!(close(ft[i][j], want), "dense ({i},{j})");
            }
        }

        let (mut ld, mut fd_) = (vec![0.0_f64; d], vec![0.0_f64; d]);
        low.add_diagonal_scaled(&mut ld, w);
        full.add_diagonal_scaled(&mut fd_, w);
        for i in 0..d {
            assert!(close(ld[i], w * dense[i][i]), "low diag {i}");
            assert!(close(fd_[i], w * dense[i][i]), "dense diag {i}");
        }

        // The degenerate case the floor exists for: a shrink that left no direction at all, so the
        // whole covariance *is* the floor.
        let bare: SecondMoment<f64> = SecondMoment::LowRank {
            rows: Vec::new(),
            iso,
            dim: d,
        };
        assert!(close(bare.trace(), iso * d as f64));
        assert!(close(
            bare.trace_under(&chol, &inv),
            iso * (0..d).map(|i| inv[i][i]).sum::<f64>()
        ));
        let mut bt = vec![vec![0.0_f64; d]; d];
        bare.add_scaled(&mut bt, w);
        for (i, row) in bt.iter().enumerate() {
            for (j, &x) in row.iter().enumerate() {
                let want = if i == j { w * iso } else { 0.0 };
                assert!(close(x, want), "bare ({i},{j}): {x}");
            }
        }
    }

    /// `apply_rows` accumulates: two half-weight calls must equal one full-weight call, which is
    /// what lets the M-step sum leaf contributions into one buffer.
    #[test]
    fn apply_rows_accumulates_into_its_target() {
        let low = SecondMoment::LowRank {
            rows: vec![vec![1.0_f64, 2.0], vec![-0.5, 0.25]],
            iso: 0.0,
            dim: 2,
        };
        let v = vec![vec![1.0_f64, -1.0]];
        let mut once = vec![vec![0.0_f64; 2]; 1];
        let mut twice = vec![vec![0.0_f64; 2]; 1];
        low.apply_rows(&v, &mut once, 1.0);
        low.apply_rows(&v, &mut twice, 0.4);
        low.apply_rows(&v, &mut twice, 0.6);
        for (&x, &y) in once[0].iter().zip(&twice[0]) {
            assert!(close(x, y), "{x} vs {y}");
        }
    }

    /// The boundary of what a cluster feature can express, pinned by an example rather than argued.
    ///
    /// A feature is a sum-decomposition `f(X) = φ(Σ ψ(x))` with `ψ(x) = (1, x, xxᵀ)`, so it carries
    /// the permutation-invariant polynomials of degree ≤ 2 and nothing else. These two sets have the
    /// same weight, mean and scatter **exactly** — integer coordinates on purpose, so nothing rests
    /// on a tolerance — yet their pairwise distance sets differ, and with them any head built on a
    /// `min` or `max` over pairs: single linkage joins at `2` in one and at `3` in the other, and
    /// DBSCAN at `eps = 2`, `minPts = 2` calls one two clusters and the other one cluster with two
    /// noise points.
    ///
    /// This test asserts an *impossibility*, which is why it is worth its own name: no richer
    /// feature in the family rescues it, because the full model already fails.
    #[test]
    fn two_point_sets_can_share_a_feature_and_not_a_geometry() {
        let a: [&[f64]; 4] = [&[-3.0, 0.0], &[-1.0, 0.0], &[2.0, 0.0], &[2.0, 0.0]];
        let b: [&[f64]; 4] = [&[-3.0, 0.0], &[0.0, 0.0], &[0.0, 0.0], &[3.0, 0.0]];

        fn moments<C: ClusterFeature<f64>>(pts: &[&[f64]]) -> (f64, Vec<f64>, f64) {
            let c: C = push_all(2, pts);
            (c.weight(), c.mean().to_vec(), c.ssd())
        }
        let agree = |name: &str,
                     (wa, ma, ssa): (f64, Vec<f64>, f64),
                     (wb, mb, ssb): (f64, Vec<f64>, f64)| {
            assert!(close(wa, wb), "{name}: weights differ");
            assert!(
                ma.iter().zip(&mb).all(|(&x, &y)| close(x, y)),
                "{name}: means differ"
            );
            assert!(close(ssa, ssb), "{name}: scatter differs, {ssa} vs {ssb}");
        };
        agree(
            "spherical",
            moments::<Spherical<f64>>(&a),
            moments::<Spherical<f64>>(&b),
        );
        agree(
            "diagonal",
            moments::<Diagonal<f64>>(&a),
            moments::<Diagonal<f64>>(&b),
        );
        agree("full", moments::<Full<f64>>(&a), moments::<Full<f64>>(&b));
        agree(
            "fd sketch",
            moments::<FdSketch<f64>>(&a),
            moments::<FdSketch<f64>>(&b),
        );

        let (fa, fb): (Full<f64>, Full<f64>) = (push_all(2, &a), push_all(2, &b));
        let (ca, cb) = (fa.cov_dense(), fb.cov_dense());
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    close(ca[i][j], cb[i][j]),
                    "full covariance differs at ({i}, {j})"
                );
            }
        }

        let spread = |pts: &[&[f64]]| -> Vec<f64> {
            let mut d: Vec<f64> = (0..pts.len())
                .flat_map(|i| (i + 1..pts.len()).map(move |j| (i, j)))
                .map(|(i, j)| (pts[i][0] - pts[j][0]).abs())
                .collect();
            d.sort_by(|x, y| x.partial_cmp(y).expect("no NaN in a fixed fixture"));
            d.dedup_by(|x, y| close(*x, *y));
            d
        };
        assert_eq!(spread(&a), vec![0.0, 2.0, 3.0, 5.0]);
        assert_eq!(spread(&b), vec![0.0, 3.0, 6.0]);
    }

    /// What summarising costs, as an identity rather than a bound.
    ///
    /// With every leaf inside one cluster, `WCSS_points = WCSS_summary + Σ_leaf ssd`, because
    /// `Σ_{x ∈ leaf} ‖x − c‖² = ssd + w‖μ − c‖²` for any `c`. The total leaf scatter — the quantity
    /// `threshold` and `max_leaves` control — is therefore exactly the error of reading `k`-means,
    /// Ward and Calinski–Harabasz off the summary, which is what makes the leaf budget a budget.
    #[test]
    fn the_summary_costs_exactly_the_total_leaf_scatter() {
        let pts: Vec<Vec<f64>> = (0..24)
            .map(|i| {
                let t = f64::from(i);
                vec![(t * 0.7).sin() * 3.0, (t * 1.3).cos() * 2.0 + t * 0.1]
            })
            .collect();
        let leaf_of = |i: usize| i % 4; // four leaves, two per cluster
        let cluster_of_leaf = [0usize, 0, 1, 1];

        let mut exact = 0.0;
        for c in 0..2 {
            let members: Vec<&Vec<f64>> = pts
                .iter()
                .enumerate()
                .filter(|(i, _)| cluster_of_leaf[leaf_of(*i)] == c)
                .map(|(_, p)| p)
                .collect();
            let n = members.len() as f64;
            let mean: Vec<f64> = (0..2)
                .map(|d| members.iter().map(|p| p[d]).sum::<f64>() / n)
                .collect();
            exact += members
                .iter()
                .map(|p| (0..2).map(|d| (p[d] - mean[d]).powi(2)).sum::<f64>())
                .sum::<f64>();
        }

        let leaves: Vec<Full<f64>> = (0..4)
            .map(|l| {
                let mut f = Full::new(2);
                for (i, p) in pts.iter().enumerate() {
                    if leaf_of(i) == l {
                        f.push(p, 1.0);
                    }
                }
                f
            })
            .collect();
        let leaf_scatter: f64 = leaves.iter().map(ClusterFeature::ssd).sum();
        let mut summary = 0.0;
        for c in 0..2 {
            let sel: Vec<&Full<f64>> = leaves
                .iter()
                .enumerate()
                .filter(|(l, _)| cluster_of_leaf[*l] == c)
                .map(|(_, f)| f)
                .collect();
            let w: f64 = sel.iter().map(|f| f.weight()).sum();
            let centre: Vec<f64> = (0..2)
                .map(|d| sel.iter().map(|f| f.weight() * f.mean()[d]).sum::<f64>() / w)
                .collect();
            summary += sel
                .iter()
                .map(|f| {
                    f.weight()
                        * (0..2)
                            .map(|d| (f.mean()[d] - centre[d]).powi(2))
                            .sum::<f64>()
                })
                .sum::<f64>();
        }
        assert!(
            (exact - summary - leaf_scatter).abs() < 1e-9,
            "exact {exact} != summary {summary} + leaf scatter {leaf_scatter}"
        );
        assert!(
            leaf_scatter > 1.0,
            "a fixture with no leaf spread proves nothing"
        );
    }
}

#[cfg(test)]
mod prop_tests {
    //! Property-based checks of the invariants DESIGN.md advertises: the CF is a commutative monoid
    //! (merge = Σ points, order-independent), the full-covariance upper-triangular index is a
    //! bijection (the `dim >= 4` cross-product regime), and the FD sketch is lossless on low-rank
    //! data and never overshoots the exact scatter.
    #![allow(clippy::needless_range_loop)] // symmetric-matrix checks read clearest with (i, j)
    use super::*;
    use proptest::prelude::*;

    /// Relative closeness — merge reorders the stable Welford/Chan updates, so results agree up to
    /// rounding rather than bit-for-bit.
    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-7 * a.abs().max(b.abs()).max(1.0)
    }

    /// `(dim, points)` with every point sharing `dim`, `dim ∈ [min_dim, max_dim]`.
    fn dim_points(
        min_dim: usize,
        max_dim: usize,
        min_pts: usize,
        max_pts: usize,
    ) -> impl Strategy<Value = (usize, Vec<Vec<f64>>)> {
        (min_dim..=max_dim).prop_flat_map(move |d| {
            prop::collection::vec(
                prop::collection::vec(-100.0f64..100.0, d..=d),
                min_pts..=max_pts,
            )
            .prop_map(move |pts| (d, pts))
        })
    }

    fn build<C: ClusterFeature<f64>>(d: usize, pts: &[Vec<f64>]) -> C {
        let mut c = C::new(d);
        for p in pts {
            c.push(p, 1.0);
        }
        c
    }

    /// Weight, mean, ssd and per-dimension variance all agree (diagonal moments).
    fn same<C: ClusterFeature<f64>>(a: &C, b: &C) -> bool {
        close(a.weight(), b.weight())
            && close(a.ssd(), b.ssd())
            && (0..a.dim())
                .all(|i| close(a.mean()[i], b.mean()[i]) && close(a.variance(i), b.variance(i)))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Merging two arbitrary partitions reconstructs the whole-dataset feature — for every
        /// model. The CF-level "merge = Σ points" / commutative-monoid property.
        #[test]
        fn merge_equals_sequential((d, pts) in dim_points(2, 8, 2, 40), cut in 0usize..=40) {
            let k = cut.min(pts.len());
            macro_rules! check {
                ($t:ty) => {{
                    let full: $t = build(d, &pts);
                    let mut a: $t = build(d, &pts[..k]);
                    let b: $t = build(d, &pts[k..]);
                    a.merge(&b);
                    prop_assert!(same(&a, &full));
                }};
            }
            check!(Spherical<f64>);
            check!(Diagonal<f64>);
            check!(Full<f64>);
        }

        /// `A ⊕ B == B ⊕ A`.
        #[test]
        fn merge_is_commutative((d, pts) in dim_points(2, 6, 2, 30), cut in 0usize..=30) {
            let k = cut.min(pts.len());
            let mut ab: Full<f64> = build(d, &pts[..k]);
            ab.merge(&build::<Full<f64>>(d, &pts[k..]));
            let mut ba: Full<f64> = build(d, &pts[k..]);
            ba.merge(&build::<Full<f64>>(d, &pts[..k]));
            prop_assert!(same(&ab, &ba));
        }

        /// `(A ⊕ B) ⊕ C == A ⊕ (B ⊕ C)`.
        #[test]
        fn merge_is_associative(
            (d, pts) in dim_points(2, 6, 3, 30),
            c1 in 0usize..=30,
            c2 in 0usize..=30,
        ) {
            let n = pts.len();
            let mut cuts = [c1.min(n), c2.min(n)];
            cuts.sort_unstable();
            let (p, q) = (cuts[0], cuts[1]);
            let mut left: Full<f64> = build(d, &pts[..p]);
            left.merge(&build::<Full<f64>>(d, &pts[p..q]));
            left.merge(&build::<Full<f64>>(d, &pts[q..]));
            let mut bc: Full<f64> = build(d, &pts[p..q]);
            bc.merge(&build::<Full<f64>>(d, &pts[q..]));
            let mut right: Full<f64> = build(d, &pts[..p]);
            right.merge(&bc);
            prop_assert!(same(&left, &right));
        }

        /// Full covariance (off-diagonals included) survives an arbitrary merge at `dim >= 4` — the
        /// regime where the reference upper-triangular index bug corrupted cross-products.
        #[test]
        fn full_merge_preserves_covariance_dim_ge_4(
            (d, pts) in dim_points(4, 8, 3, 30),
            cut in 0usize..=30,
        ) {
            let k = cut.min(pts.len());
            let full: Full<f64> = build(d, &pts);
            let mut a: Full<f64> = build(d, &pts[..k]);
            a.merge(&build::<Full<f64>>(d, &pts[k..]));
            let (ca, cf) = (a.covariance(), full.covariance());
            for i in 0..d {
                for j in 0..d {
                    prop_assert!(close(ca[i][j], cf[i][j]), "cov ({i},{j})");
                }
            }
        }

        /// The upper-triangular packing index is a bijection onto `0..d(d+1)/2` — no `(i, j)`
        /// collides and every slot is used (round-trip), including `d >= 4`.
        #[test]
        fn upper_tri_index_is_a_bijection(d in 1usize..=12) {
            let f: Full<f64> = Full::new(d);
            let mut seen = vec![false; d * (d + 1) / 2];
            for i in 0..d {
                for j in i..d {
                    let k = f.idx(i, j);
                    prop_assert!(k < seen.len(), "idx out of range at ({i},{j})");
                    prop_assert!(!seen[k], "collision at ({i},{j}) -> {k}");
                    seen[k] = true;
                }
            }
            prop_assert!(seen.into_iter().all(|b| b), "some slot never indexed");
        }

        /// FD sketch is lossless on genuinely low-rank data: rank-≤2 points in `R^d` (`d >= 4`, so
        /// `ell = d` clears FD's lower-median shrink threshold `ceil((ell-1)/2) >= 2`) reproduce the
        /// exact full covariance. `with_ell` clamps `ell` to `[1, dim]`, so `ell > d` is impossible.
        #[test]
        fn fd_lossless_on_low_rank(
            (d, u, v, coeffs) in (4usize..=8).prop_flat_map(|d| {
                (
                    Just(d),
                    prop::collection::vec(-10.0f64..10.0, d),
                    prop::collection::vec(-10.0f64..10.0, d),
                    prop::collection::vec((-5.0f64..5.0, -5.0f64..5.0), 3..=30),
                )
            })
        ) {
            let pts: Vec<Vec<f64>> = coeffs
                .iter()
                .map(|&(a, b)| (0..d).map(|k| a * u[k] + b * v[k]).collect())
                .collect();
            let full: Full<f64> = build(d, &pts);
            let mut fd = FdSketch::<f64>::with_ell(d, d);
            for p in &pts {
                fd.push(p, 1.0);
            }
            prop_assert!(close(fd.weight(), full.weight()));
            let (cfd, cfull) = (fd.cov_dense(), full.cov_dense());
            for i in 0..d {
                for j in 0..d {
                    prop_assert!(close(cfd[i][j], cfull[i][j]), "cov ({i},{j})");
                }
            }
        }

        /// FD sketch with `ell < d` loses directions, not mass: `BᵀB ⪯ AᵀA` strictly, but the shrink's
        /// discarded trace is banked, so the reported total scatter is exact. Covariance stays
        /// symmetric.
        #[test]
        fn fd_reports_the_exact_scatter_and_stays_symmetric((d, pts) in dim_points(3, 8, 6, 40), e in 0usize..100) {
            let ell = 1 + e % (d - 1); // 1..=d-1
            let full: Full<f64> = build(d, &pts);
            let mut fd = FdSketch::<f64>::with_ell(d, ell);
            for p in &pts {
                fd.push(p, 1.0);
            }
            prop_assert!(
                (fd.ssd() - full.ssd()).abs() <= full.ssd() * 1e-9 + 1e-9,
                "scatter {} != exact {}",
                fd.ssd(),
                full.ssd()
            );
            let cov = fd.cov_dense();
            for i in 0..d {
                for j in 0..d {
                    prop_assert!(close(cov[i][j], cov[j][i]), "asymmetry ({i},{j})");
                }
            }
        }
    }
}
