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
/// the dense models, or `LowRank` rows `{f_r}` with `Σ = Σ_r f_r f_rᵀ` for the Frequent-Directions
/// sketch. The low-rank form keeps the GMM at `O(ℓ·d)` per leaf instead of materialising `d×d`
/// (which would undo FD's whole memory advantage). Both encode the same matrix, so `trace_under` and
/// `add_scaled` return identical values for either variant.
pub enum SecondMoment<R: Real> {
    /// Dense `d×d` covariance.
    Dense(Vec<Vec<R>>),
    /// Rows `f_r` such that `Σ = Σ_r f_r f_rᵀ`.
    LowRank(Vec<Vec<R>>),
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
            SecondMoment::LowRank(rows) => rows
                .iter()
                .map(|f| linalg::mahalanobis_sq_from_chol(chol, f))
                .fold(R::zero(), |a, b| a + b),
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
            SecondMoment::LowRank(rows) => {
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
}

// ───────────────────────── Spherical (scalar SSD) ─────────────────────────

/// Isotropic feature: a single scalar `S`. Covariance is `(S / (n·d)) · I`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Spherical<R: Real> {
    w: R,
    mean: Vec<R>,
    ssd: R,
}

impl<R: Real> Spherical<R> {
    /// Build a spherical feature directly from its moments `(n, μ, S)`. Used by the sparse-native
    /// path, which accumulates the mean and scatter in a scaled sparse form and materialises the
    /// dense feature once, at the end.
    pub fn from_moments(weight: R, mean: Vec<R>, ssd: R) -> Self {
        Self {
            w: weight,
            mean,
            ssd,
        }
    }
}

impl<R: Real> ClusterFeature<R> for Spherical<R> {
    fn new(dim: usize) -> Self {
        Self {
            w: R::zero(),
            mean: vec![R::zero(); dim],
            ssd: R::zero(),
        }
    }
    fn dim(&self) -> usize {
        self.mean.len()
    }
    fn weight(&self) -> R {
        self.w
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        self.ssd
    }
    fn variance(&self, _d: usize) -> R {
        if self.w <= R::zero() {
            R::zero()
        } else {
            self.ssd / self.w / R::from_usize(self.mean.len()).unwrap()
        }
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let w_new = self.w + w;
        let factor = w / w_new;
        let coef = w * (R::one() - factor);
        let mut normsq = R::zero();
        for (m, &xi) in self.mean.iter_mut().zip(x) {
            let d = xi - *m;
            normsq = normsq + d * d;
            *m = *m + factor * d;
        }
        self.ssd = self.ssd + coef * normsq;
        self.w = w_new;
    }
    fn merge(&mut self, other: &Self) {
        if other.w <= R::zero() {
            return;
        }
        if self.w <= R::zero() {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor = other.w / w_new;
        let c = self.w * other.w / w_new;
        let mut normsq = R::zero();
        for (m, &om) in self.mean.iter_mut().zip(&other.mean) {
            let d = om - *m;
            normsq = normsq + d * d;
            *m = *m + factor * d;
        }
        self.ssd = self.ssd + other.ssd + c * normsq;
        self.w = w_new;
    }
    fn decay(&mut self, factor: R) {
        self.w = self.w * factor;
        self.ssd = self.ssd * factor;
    }
}

// ───────────────────────── Diagonal (per-dimension SSD) ─────────────────────────

/// Axis-aligned feature: per-dimension `S`. Covariance is `diag(S_d / n)`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Diagonal<R: Real> {
    w: R,
    mean: Vec<R>,
    ssd: Vec<R>,
}

impl<R: Real> ClusterFeature<R> for Diagonal<R> {
    fn new(dim: usize) -> Self {
        Self {
            w: R::zero(),
            mean: vec![R::zero(); dim],
            ssd: vec![R::zero(); dim],
        }
    }
    fn dim(&self) -> usize {
        self.mean.len()
    }
    fn weight(&self) -> R {
        self.w
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        self.ssd.iter().copied().sum()
    }
    fn variance(&self, d: usize) -> R {
        if self.w <= R::zero() {
            R::zero()
        } else {
            self.ssd[d] / self.w
        }
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let w_new = self.w + w;
        let factor = w / w_new;
        let coef = w * (R::one() - factor);
        for ((m, s), &xi) in self.mean.iter_mut().zip(self.ssd.iter_mut()).zip(x) {
            let d = xi - *m;
            *s = *s + coef * d * d;
            *m = *m + factor * d;
        }
        self.w = w_new;
    }
    fn merge(&mut self, other: &Self) {
        if other.w <= R::zero() {
            return;
        }
        if self.w <= R::zero() {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor = other.w / w_new;
        let c = self.w * other.w / w_new;
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
        self.w = self.w * factor;
        for s in &mut self.ssd {
            *s = *s * factor;
        }
    }
}

// ───────────────────────── Full (scatter matrix) ─────────────────────────

/// Full-covariance feature: the scatter matrix `M = Σ w (x-μ)(x-μ)ᵀ`, stored as the flat
/// upper triangle (row-major, including the diagonal). Covariance is `M / n`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct Full<R: Real> {
    w: R,
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
        if self.w <= R::zero() {
            return vec![vec![R::zero(); d]; d];
        }
        let inv = R::one() / self.w;
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
            w: R::zero(),
            mean: vec![R::zero(); dim],
            scatter: vec![R::zero(); dim * (dim + 1) / 2],
            dim,
        }
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn weight(&self) -> R {
        self.w
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        (0..self.dim).map(|i| self.at(i, i)).sum()
    }
    fn variance(&self, d: usize) -> R {
        if self.w <= R::zero() {
            R::zero()
        } else {
            self.at(d, d) / self.w
        }
    }
    fn cov_dense(&self) -> Vec<Vec<R>> {
        self.covariance()
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let w_new = self.w + w;
        let factor = w / w_new;
        let coef = w * (R::one() - factor);
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
        if other.w <= R::zero() {
            return;
        }
        if self.w <= R::zero() {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor = other.w / w_new;
        let c = self.w * other.w / w_new;
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
        self.w = self.w * factor;
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
#[derive(Clone, Debug)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct FdSketch<R: Real> {
    w: R,
    mean: Vec<R>,
    sketch: Vec<Vec<R>>, // `ell` rows × `dim`; rows `[0, self.rows)` are live
    rows: usize,
    ell: usize,
    dim: usize,
}

#[allow(clippy::needless_range_loop)] // sketch/gram math reads clearest with explicit indices
impl<R: Real> FdSketch<R> {
    /// Empty sketch with an explicit row budget `ell` (clamped to `[1, dim]`).
    pub fn with_ell(dim: usize, ell: usize) -> Self {
        let ell = ell.max(1).min(dim.max(1));
        Self {
            w: R::zero(),
            mean: vec![R::zero(); dim],
            sketch: vec![vec![R::zero(); dim]; ell],
            rows: 0,
            ell,
            dim,
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
    fn dim(&self) -> usize {
        self.dim
    }
    fn weight(&self) -> R {
        self.w
    }
    fn mean(&self) -> &[R] {
        &self.mean
    }
    fn ssd(&self) -> R {
        self.sketch
            .iter()
            .take(self.rows)
            .flatten()
            .map(|&x| x * x)
            .fold(R::zero(), |a, b| a + b)
    }
    fn variance(&self, d: usize) -> R {
        if self.w <= R::zero() {
            return R::zero();
        }
        let s: R = (0..self.rows)
            .map(|i| self.sketch[i][d] * self.sketch[i][d])
            .fold(R::zero(), |a, b| a + b);
        s / self.w
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
        if self.w > R::zero() {
            for row in m.iter_mut() {
                for x in row.iter_mut() {
                    *x = *x / self.w;
                }
            }
        }
        m
    }
    fn push(&mut self, x: &[R], w: R) {
        if w <= R::zero() {
            return;
        }
        let w_new = self.w + w;
        let factor = w / w_new;
        let coef = w * (R::one() - factor);
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
        if other.w <= R::zero() {
            return;
        }
        if self.w <= R::zero() {
            *self = other.clone();
            return;
        }
        let w_new = self.w + other.w;
        let factor = other.w / w_new;
        let c = self.w * other.w / w_new;
        let delta: Vec<R> = (0..self.dim)
            .map(|i| other.mean[i] - self.mean[i])
            .collect();
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
        self.w = self.w * factor;
        let s = factor.sqrt();
        for row in self.sketch.iter_mut().take(self.rows) {
            for x in row.iter_mut() {
                *x = *x * s;
            }
        }
    }
    fn second_moment(&self) -> SecondMoment<R> {
        // Σ = BᵀB / n = Σ_r (b_r/√n)(b_r/√n)ᵀ — the sketch rows scaled by 1/√n are the low-rank
        // factors, so the GMM never reconstructs the `d×d` covariance for this leaf.
        if self.w <= R::zero() {
            return SecondMoment::LowRank(Vec::new());
        }
        let scale = R::one() / self.w.sqrt();
        let rows = self
            .sketch
            .iter()
            .take(self.rows)
            .map(|row| row.iter().map(|&x| x * scale).collect())
            .collect();
        SecondMoment::LowRank(rows)
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
    fn fd_sketch_undershoots_and_stays_symmetric() {
        // Full-rank data with ℓ < d: the sketch covariance is symmetric and underestimates the
        // scatter (FD's BᵀB ⪯ AᵀA), never blowing up.
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
        assert!(
            fd.ssd() <= full.ssd() + 1e-9,
            "fd ssd {} > full {}",
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
            SecondMoment::LowRank(rows) => assert!(rows.is_empty()),
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
        let SecondMoment::LowRank(rows) = fd.second_moment() else {
            panic!("FdSketch must expose low-rank factors");
        };
        for a in 0..d {
            for b in 0..d {
                let got: f64 = rows.iter().map(|r| r[a] * r[b]).sum();
                assert!(
                    (got - dense[a][b]).abs() < 1e-9,
                    "Σ_r r_a r_b at ({a},{b}) = {got}, cov_dense = {}",
                    dense[a][b]
                );
            }
        }

        // An empty sketch has no covariance and therefore no factors.
        let e: FdSketch<f64> = FdSketch::with_ell(d, d);
        assert!(matches!(e.second_moment(), SecondMoment::LowRank(r) if r.is_empty()));
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

        /// FD sketch with `ell < d` never overshoots (`BᵀB ⪯ AᵀA`, so total scatter <= exact) and
        /// stays symmetric.
        #[test]
        fn fd_undershoots_and_symmetric((d, pts) in dim_points(3, 8, 6, 40), e in 0usize..100) {
            let ell = 1 + e % (d - 1); // 1..=d-1
            let full: Full<f64> = build(d, &pts);
            let mut fd = FdSketch::<f64>::with_ell(d, ell);
            for p in &pts {
                fd.push(p, 1.0);
            }
            prop_assert!(
                fd.ssd() <= full.ssd() * (1.0 + 1e-9) + 1e-9,
                "overshoot: {} > {}",
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
