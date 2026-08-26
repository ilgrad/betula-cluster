//! Mixture-Wasserstein `MW₂`: a label-free distance between two *fitted* Gaussian mixtures.
//!
//! Delon & Desolneux (*A Wasserstein-type distance in the space of Gaussian mixture models*, SIAM
//! J. Imaging Sci. 13(2), 2020) restrict the coupling in the Wasserstein problem to be itself a
//! Gaussian mixture. The restriction buys a closed form for every pair cost — `W₂` between two
//! Gaussians is the Bures metric, no sampling — and leaves a discrete optimal-transport problem
//! over the `k_A × k_B` component grid, which at head scale (`k ≤ 64`) is a few thousand cells.
//!
//! `MW₂` is a genuine metric on the space of Gaussian mixtures, and it upper-bounds the true `W₂`
//! between the same two densities: the restricted coupling is feasible for the unrestricted problem.
//!
//! Two uses here, both diagnostic rather than a head:
//!
//! - **Drift.** The distance between the model fitted at `t₁` and the model fitted at `t₂` needs no
//!   labels and no shared point set, which is what makes it usable where ARI-over-time is not — the
//!   two windows do not have to contain the same points, or the same number of them.
//! - **Cross-implementation agreement.** "The two fitted mixtures are `MW₂ = x` apart" is a sharper
//!   statement than an ARI between their labellings, which conflates a parameter difference with a
//!   tie broken the other way on points near a boundary.
//!
//! ## What is exact and what is not
//!
//! The pair cost is exact. The transport is solved exactly, by the transportation simplex with
//! Bland's rule — no entropic regularisation, so `MW₂(m, m)` is exactly zero rather than a small
//! positive number, which is the property that makes the diagnostic readable at all.
//!
//! The `MW₂` value is *not* the `W₂` between the two densities. It is the `W₂` under a coupling
//! restricted to Gaussian mixtures, which is an upper bound and coincides with `W₂` only when the
//! optimal unrestricted coupling happens to be of that form (for instance, two mixtures that differ
//! by a common translation).

use crate::linalg::jacobi_eigen;
use crate::types::Real;

/// The covariance of one Gaussian component, in whichever form its head already holds.
///
/// The split is not decoration: `gmm` at `d = 784` holds one variance *vector* per component, and
/// densifying it into a `784 × 784` matrix just to take a distance would cost more memory than the
/// fit did. [`Spread::Diagonal`] against [`Spread::Diagonal`] never builds a matrix at all.
#[derive(Debug, Clone, Copy)]
pub enum Spread<'a, R: Real> {
    /// One variance per coordinate: `Σ = diag(σ²₀, …, σ²_{d-1})`.
    Diagonal(&'a [R]),
    /// A dense symmetric positive-semidefinite `d × d`, row by row.
    Full(&'a [Vec<R>]),
}

impl<R: Real> Spread<'_, R> {
    fn dim(&self) -> usize {
        match self {
            Spread::Diagonal(v) => v.len(),
            Spread::Full(m) => m.len(),
        }
    }

    fn trace(&self) -> R {
        match self {
            Spread::Diagonal(v) => v.iter().copied().fold(R::zero(), |a, b| a + b),
            Spread::Full(m) => (0..m.len()).fold(R::zero(), |a, i| a + m[i][i]),
        }
    }

    fn dense(&self) -> Vec<Vec<R>> {
        match self {
            Spread::Diagonal(v) => {
                let mut m = vec![vec![R::zero(); v.len()]; v.len()];
                for (i, &s) in v.iter().enumerate() {
                    m[i][i] = s;
                }
                m
            }
            Spread::Full(m) => m.to_vec(),
        }
    }
}

/// A fitted Gaussian mixture, in the form the transport reads it: one weight, mean and covariance
/// per component. Borrowed rather than owned — every caller already holds these arrays.
#[derive(Debug, Clone, Copy)]
pub struct GaussianMixture<'a, R: Real> {
    /// Mixing weights. Need not sum to one; they are normalized internally.
    pub weights: &'a [R],
    /// One mean vector per component.
    pub means: &'a [Vec<R>],
    /// One covariance per component.
    pub covs: &'a [Spread<'a, R>],
}

impl<R: Real> GaussianMixture<'_, R> {
    /// `Some(dim)` when the three arrays agree on component count and every component agrees on
    /// dimension, `None` otherwise. A mixture that fails this is a caller bug, not a degenerate
    /// input, so the entry points return `None` rather than guessing a dimension.
    fn checked_dim(&self) -> Option<usize> {
        let k = self.weights.len();
        if k == 0 || self.means.len() != k || self.covs.len() != k {
            return None;
        }
        let dim = self.means[0].len();
        let ok = (0..k).all(|c| self.means[c].len() == dim && self.covs[c].dim() == dim);
        ok.then_some(dim)
    }
}

/// `tr((A^½ B A^½)^½)`, the cross term of the Bures metric, via two symmetric eigendecompositions.
///
/// `A^½ B A^½` is similar to `AB`, so its eigenvalues are those of `AB` and are real and
/// non-negative for PSD `A`, `B` — but forming it symmetrically is what lets the in-house Jacobi
/// routine be used at all, and it keeps the rounding symmetric too. Negative eigenvalues can only
/// come out of rounding on a near-singular pair, so they are clamped rather than propagated as NaN.
fn bures_cross<R: Real>(a: &[Vec<R>], b: &[Vec<R>]) -> R {
    let n = a.len();
    let (eig, vecs) = jacobi_eigen(a);
    // A^½ = V diag(√λ) Vᵀ.
    let root: Vec<R> = eig.iter().map(|&l| l.max(R::zero()).sqrt()).collect();
    let mut a_half = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = R::zero();
            for k in 0..n {
                s = s + vecs[i][k] * root[k] * vecs[j][k];
            }
            a_half[i][j] = s;
            a_half[j][i] = s;
        }
    }
    // M = A^½ B A^½, formed as (A^½ B) A^½ and written symmetrically: only the lower triangle
    // is computed and mirrored, so rounding cannot make the two halves disagree.
    let mut ab = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = R::zero();
            for k in 0..n {
                s = s + a_half[i][k] * b[k][j];
            }
            ab[i][j] = s;
        }
    }
    let mut m = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = R::zero();
            for k in 0..n {
                s = s + ab[i][k] * a_half[k][j];
            }
            m[i][j] = s;
            m[j][i] = s;
        }
    }
    let (eig_m, _) = jacobi_eigen(&m);
    eig_m
        .iter()
        .fold(R::zero(), |acc, &l| acc + l.max(R::zero()).sqrt())
}

/// Squared `W₂` between two Gaussians: `‖m₁ − m₂‖² + tr(Σ₁ + Σ₂ − 2(Σ₁^½ Σ₂ Σ₁^½)^½)`.
///
/// Returns `None` unless all four arguments agree on dimension.
///
/// Two diagonal covariances take the closed form `Σ_d (σ₁ − σ₂)²` — commuting matrices make the
/// Bures term collapse — and never build a `d × d` matrix. Any other combination densifies, which
/// is what a full covariance already costs.
pub fn gaussian_w2_sq<R: Real>(
    mean_a: &[R],
    cov_a: Spread<'_, R>,
    mean_b: &[R],
    cov_b: Spread<'_, R>,
) -> Option<R> {
    let dim = mean_a.len();
    if mean_b.len() != dim || cov_a.dim() != dim || cov_b.dim() != dim {
        return None;
    }
    let gap = mean_a
        .iter()
        .zip(mean_b)
        .fold(R::zero(), |acc, (&x, &y)| acc + (x - y) * (x - y));
    let two = R::one() + R::one();
    let spread = match (cov_a, cov_b) {
        (Spread::Diagonal(u), Spread::Diagonal(v)) => {
            u.iter().zip(v).fold(R::zero(), |acc, (&s, &t)| {
                let d = s.max(R::zero()).sqrt() - t.max(R::zero()).sqrt();
                acc + d * d
            })
        }
        _ => {
            let cross = bures_cross(&cov_a.dense(), &cov_b.dense());
            cov_a.trace() + cov_b.trace() - two * cross
        }
    };
    Some(gap + spread.max(R::zero()))
}

/// The mixture-Wasserstein distance `MW₂(a, b)`, in the same units as the data.
///
/// Returns `None` when either mixture is empty, internally inconsistent, carries no positive weight,
/// or disagrees with the other on dimension.
pub fn mixture_w2<R: Real>(a: GaussianMixture<'_, R>, b: GaussianMixture<'_, R>) -> Option<R> {
    let dim_a = a.checked_dim()?;
    if dim_a != b.checked_dim()? {
        return None;
    }
    let wa = normalized(a.weights)?;
    let wb = normalized(b.weights)?;
    let mut cost = vec![vec![0.0f64; wb.len()]; wa.len()];
    for (i, row) in cost.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let pair = gaussian_w2_sq(&a.means[i], a.covs[i], &b.means[j], b.covs[j])?;
            *cell = pair.to_f64().unwrap_or(f64::MAX).max(0.0);
        }
    }
    let (total, _) = transport(&wa, &wb, &cost);
    R::from_f64(total.max(0.0).sqrt())
}

/// Weights as a probability vector, or `None` if none of them is positive. Negative entries are
/// clamped: a mixing weight below zero is not a small mass, it is a broken fit.
fn normalized<R: Real>(w: &[R]) -> Option<Vec<f64>> {
    let raw: Vec<f64> = w
        .iter()
        .map(|&x| x.to_f64().unwrap_or(0.0).max(0.0))
        .collect();
    let sum: f64 = raw.iter().sum();
    if sum <= 0.0 {
        return None;
    }
    Some(raw.iter().map(|&x| x / sum).collect())
}

/// Anything below this is a rounding artefact of the dual update rather than a real improvement.
const SIMPLEX_TOL: f64 = 1e-12;

/// The pivot budget. A backstop for a broken invariant, not a convergence parameter: Bland's rule
/// visits no basis twice and the number of bases is finite, so a solve that reaches this has a bug
/// rather than a hard instance — which is why the tests assert the pivot count stays under it.
fn iteration_cap(na: usize, nb: usize) -> usize {
    64 * (na * nb + na + nb)
}

/// The optimal value of the transportation problem `min Σ π_ij c_ij` over couplings of `a` and `b`,
/// and the number of pivots it took.
///
/// The transportation simplex with **Bland's rule** on both the entering and the leaving cell, which
/// is what makes it terminate on a degenerate basis instead of cycling between two bases of the same
/// cost — and degeneracy is the common case here, not the exotic one: two components of equal
/// mixing weight produce it immediately.
///
/// Both marginals must be non-negative and sum to one.
fn transport(a: &[f64], b: &[f64], cost: &[Vec<f64>]) -> (f64, usize) {
    let (na, nb) = (a.len(), b.len());
    let mut flow = vec![vec![0.0f64; nb]; na];
    let mut basic = vec![vec![false; nb]; na];

    // North-west corner. Each step fixes one cell and advances exactly one index, so it marks
    // na + nb - 1 cells and ends at the last one — a spanning tree, degenerate zero cells included.
    let (mut ra, mut rb) = (a.to_vec(), b.to_vec());
    let (mut i, mut j) = (0usize, 0usize);
    loop {
        let t = ra[i].min(rb[j]);
        flow[i][j] = t;
        basic[i][j] = true;
        ra[i] -= t;
        rb[j] -= t;
        if i + 1 == na && j + 1 == nb {
            break;
        }
        if (ra[i] <= rb[j] && i + 1 < na) || j + 1 == nb {
            i += 1;
        } else {
            j += 1;
        }
    }

    let mut pivots = 0usize;
    for _ in 0..iteration_cap(na, nb) {
        let Some((u, v)) = duals(&basic, cost, na, nb) else {
            break;
        };
        // Bland: the first entering cell in row-major order, not the most negative one.
        let entering = (0..na)
            .flat_map(|r| (0..nb).map(move |c| (r, c)))
            .find(|&(r, c)| !basic[r][c] && cost[r][c] - u[r] - v[c] < -SIMPLEX_TOL);
        let Some((er, ec)) = entering else {
            break;
        };
        let Some(cycle) = tree_path(&basic, na, nb, er, ec) else {
            break;
        };
        // The cycle alternates: the entering cell gains, its neighbours lose, and so on.
        let mut theta = f64::INFINITY;
        let mut leaving = None;
        for (step, &(r, c)) in cycle.iter().enumerate() {
            if step % 2 == 1 {
                // Bland again on the exit: strict `<` keeps the earliest cell of a tie.
                if flow[r][c] < theta {
                    theta = flow[r][c];
                    leaving = Some((r, c));
                }
            }
        }
        let Some((lr, lc)) = leaving else {
            break;
        };
        for (step, &(r, c)) in cycle.iter().enumerate() {
            if step % 2 == 0 {
                flow[r][c] += theta;
            } else {
                flow[r][c] -= theta;
            }
        }
        basic[er][ec] = true;
        basic[lr][lc] = false;
        flow[lr][lc] = 0.0;
        pivots += 1;
    }

    let total = (0..na)
        .flat_map(|r| (0..nb).map(move |c| (r, c)))
        .map(|(r, c)| flow[r][c] * cost[r][c])
        .sum();
    (total, pivots)
}

/// Potentials `u`, `v` with `u_i + v_j = c_ij` on every basic cell, fixing `u₀ = 0`.
///
/// `None` if the basis is not connected — which cannot happen from a north-west-corner start
/// followed by simplex pivots, and is reported rather than papered over if it ever does.
fn duals(
    basic: &[Vec<bool>],
    cost: &[Vec<f64>],
    na: usize,
    nb: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut u = vec![0.0; na];
    let mut v = vec![0.0; nb];
    // Explicit flags rather than a `NaN` sentinel for "not yet settled". A `NaN` anywhere in `cost`
    // makes `is_nan()` stay true for a potential that *has* been written, so its node is pushed on
    // every later visit and the stack grows without bound. With a flag a node is pushed exactly
    // when its flag flips, which bounds the traversal at `na + nb` pushes by construction and puts
    // the arithmetic in `cost` outside the control flow entirely.
    let mut u_set = vec![false; na];
    let mut v_set = vec![false; nb];
    u_set[0] = true;
    let mut stack = vec![0usize]; // row indices; columns are pushed as `na + j`
    while let Some(node) = stack.pop() {
        if node < na {
            let r = node;
            for c in 0..nb {
                if basic[r][c] && !v_set[c] {
                    v[c] = cost[r][c] - u[r];
                    v_set[c] = true;
                    stack.push(na + c);
                }
            }
        } else {
            let c = node - na;
            for (r, (urow, uset)) in u.iter_mut().zip(u_set.iter_mut()).enumerate() {
                if basic[r][c] && !*uset {
                    *urow = cost[r][c] - v[c];
                    *uset = true;
                    stack.push(r);
                }
            }
        }
    }
    // The advertised contract, stated directly: a spanning tree settles every potential.
    (u_set.iter().all(|&s| s) && v_set.iter().all(|&s| s)).then_some((u, v))
}

/// The unique cycle that adding `(er, ec)` creates in the basis tree, starting at the entering cell.
///
/// Returned as the cell sequence around the cycle, so index parity is the `+ / −` alternation the
/// pivot needs. The walk alternates row-step and column-step by construction — a basic cell is an
/// edge between a row node and a column node, so any path through the tree alternates sides.
fn tree_path(
    basic: &[Vec<bool>],
    na: usize,
    nb: usize,
    er: usize,
    ec: usize,
) -> Option<Vec<(usize, usize)>> {
    // Depth-first search from the entering column back to the entering row, over basic cells only.
    let mut parent = vec![usize::MAX; na + nb];
    let mut seen = vec![false; na + nb];
    let start = na + ec;
    seen[start] = true;
    let mut stack = vec![start];
    let mut found = false;
    while let Some(node) = stack.pop() {
        if node == er {
            found = true;
            break;
        }
        if node < na {
            for c in 0..nb {
                if basic[node][c] && !seen[na + c] {
                    seen[na + c] = true;
                    parent[na + c] = node;
                    stack.push(na + c);
                }
            }
        } else {
            let c = node - na;
            for r in 0..na {
                if basic[r][c] && !seen[r] {
                    seen[r] = true;
                    parent[r] = c + na;
                    stack.push(r);
                }
            }
        }
    }
    if !found {
        return None;
    }
    // Walk back from the entering row to the entering column, turning node pairs into cells.
    let mut nodes = vec![er];
    let mut node = er;
    while node != start {
        node = parent[node];
        nodes.push(node);
    }
    let mut cells = vec![(er, ec)];
    for pair in nodes.windows(2) {
        let (x, y) = (pair[0], pair[1]);
        cells.push(if x < na { (x, y - na) } else { (y, x - na) });
    }
    Some(cells)
}

/// A reproducible stream without a dependency, shared by the randomized tests and the cross-check
/// dump so that both can be replayed from the seed alone.
#[cfg(test)]
struct Lcg(u64);

#[cfg(test)]
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag_mixture<'a>(
        weights: &'a [f64],
        means: &'a [Vec<f64>],
        covs: &'a [Spread<'a, f64>],
    ) -> GaussianMixture<'a, f64> {
        GaussianMixture {
            weights,
            means,
            covs,
        }
    }

    /// Three well-separated unit-variance components in the plane.
    fn base() -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        (
            vec![0.5, 0.3, 0.2],
            vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![0.0, 7.0]],
            vec![vec![1.0, 1.0], vec![2.0, 0.5], vec![0.25, 3.0]],
        )
    }

    fn spreads(vars: &[Vec<f64>]) -> Vec<Spread<'_, f64>> {
        vars.iter()
            .map(|v| Spread::Diagonal(v.as_slice()))
            .collect()
    }

    #[test]
    fn a_mixture_is_at_zero_distance_from_itself() {
        let (w, m, v) = base();
        let s = spreads(&v);
        let mix = diag_mixture(&w, &m, &s);
        let d = mixture_w2(mix, mix).expect("a well-formed mixture has a distance to itself");
        assert!(d < 1e-9, "MW2(m, m) = {d}, expected 0");
    }

    #[test]
    fn relabelling_the_components_does_not_move_the_distance() {
        let (w, m, v) = base();
        let s = spreads(&v);
        let order = [2usize, 0, 1];
        let wp: Vec<f64> = order.iter().map(|&i| w[i]).collect();
        let mp: Vec<Vec<f64>> = order.iter().map(|&i| m[i].clone()).collect();
        let vp: Vec<Vec<f64>> = order.iter().map(|&i| v[i].clone()).collect();
        let sp = spreads(&vp);
        let d = mixture_w2(diag_mixture(&w, &m, &s), diag_mixture(&wp, &mp, &sp))
            .expect("a permuted copy is still a mixture");
        assert!(
            d < 1e-9,
            "a permuted copy is the same mixture, got MW2 = {d}"
        );
    }

    #[test]
    fn a_common_translation_costs_exactly_its_own_length() {
        // Every component moves by the same vector, so the identity coupling is optimal and MW2
        // collapses to the translation length. This is also one of the cases where MW2 equals the
        // unrestricted W2, which is why it can be checked against a number known in advance.
        let (w, m, v) = base();
        let s = spreads(&v);
        let shift = [1.5, -2.0];
        let mt: Vec<Vec<f64>> = m
            .iter()
            .map(|r| r.iter().zip(shift).map(|(&x, t)| x + t).collect())
            .collect();
        let d = mixture_w2(diag_mixture(&w, &m, &s), diag_mixture(&w, &mt, &s))
            .expect("a translated copy is still a mixture");
        let want = (shift[0] * shift[0] + shift[1] * shift[1]).sqrt();
        assert!(
            (d - want).abs() < 1e-9,
            "a common translation costs {want}, got {d}"
        );
    }

    #[test]
    fn one_component_each_is_the_plain_bures_distance() {
        let (wa, wb) = (vec![1.0], vec![1.0]);
        let (ma, mb) = (vec![vec![0.0]], vec![vec![3.0]]);
        let (va, vb) = (vec![vec![4.0]], vec![vec![9.0]]);
        let (sa, sb) = (spreads(&va), spreads(&vb));
        let d = mixture_w2(diag_mixture(&wa, &ma, &sa), diag_mixture(&wb, &mb, &sb)).unwrap();
        // In one dimension W2 is |Δμ| and |Δσ| in quadrature: √(9 + 1).
        assert!((d - 10.0f64.sqrt()).abs() < 1e-9, "got {d}");
    }

    #[test]
    fn a_diagonal_covariance_scores_the_same_through_the_dense_path() {
        // The two branches of `gaussian_w2_sq` must agree where they overlap, or the cheap path is
        // a second implementation rather than an optimization.
        let (mean_a, mean_b) = (vec![0.0f64, 1.0], vec![2.0f64, -1.0]);
        let (va, vb) = (vec![3.0f64, 0.5], vec![1.0f64, 2.0]);
        let dense_a = vec![vec![3.0, 0.0], vec![0.0, 0.5]];
        let dense_b = vec![vec![1.0, 0.0], vec![0.0, 2.0]];
        let cheap = gaussian_w2_sq(
            &mean_a,
            Spread::Diagonal(&va),
            &mean_b,
            Spread::Diagonal(&vb),
        )
        .unwrap();
        let dense = gaussian_w2_sq(
            &mean_a,
            Spread::Full(&dense_a),
            &mean_b,
            Spread::Full(&dense_b),
        )
        .unwrap();
        assert!(
            (cheap - dense).abs() < 1e-9,
            "diagonal path {cheap} vs dense path {dense}"
        );
    }

    #[test]
    fn a_rotated_pair_costs_less_than_the_axis_aligned_one_it_came_from() {
        // Two elongated Gaussians with the same shape but rotated relative to each other: the Bures
        // term must see the rotation. A diagonal-only reading would report zero spread cost.
        let flat = vec![vec![9.0, 0.0], vec![0.0, 1.0]];
        let tall = vec![vec![1.0, 0.0], vec![0.0, 9.0]];
        let zero = vec![0.0, 0.0];
        let rotated = gaussian_w2_sq(&zero, Spread::Full(&flat), &zero, Spread::Full(&tall))
            .expect("two 2x2 covariances have a distance");
        assert!(
            rotated > 1.0,
            "a 90-degree rotation of an elongated Gaussian must cost something, got {rotated}"
        );
        let same = gaussian_w2_sq(&zero, Spread::Full(&flat), &zero, Spread::Full(&flat)).unwrap();
        assert!(same < 1e-9, "a covariance against itself costs {same}");
    }

    #[test]
    fn the_transport_beats_the_coupling_that_keeps_the_component_order() {
        // Components deliberately listed in crossed order: the identity coupling pairs 0-0 and 1-1
        // at distance 10 each, the optimal one pairs 0-1 and 1-0 at distance 0. A solver that
        // silently returned the diagonal coupling would score 10 here.
        let w = vec![0.5, 0.5];
        let ma = vec![vec![0.0], vec![10.0]];
        let mb = vec![vec![10.0], vec![0.0]];
        let v = vec![vec![1.0], vec![1.0]];
        let s = spreads(&v);
        let d = mixture_w2(diag_mixture(&w, &ma, &s), diag_mixture(&w, &mb, &s)).unwrap();
        assert!(d < 1e-9, "the crossed coupling is free, got MW2 = {d}");
    }

    #[test]
    fn splitting_one_component_into_two_at_the_same_place_changes_nothing() {
        // A mixture and its own refinement describe the same density, so MW2 must be zero even
        // though the component counts differ. This is the property that lets two fits with
        // different k be compared at all.
        let w1 = vec![1.0];
        let m1 = vec![vec![0.0, 0.0]];
        let v1 = vec![vec![1.0, 1.0]];
        let s1 = spreads(&v1);
        let w2 = vec![0.4, 0.6];
        let m2 = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let v2 = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let s2 = spreads(&v2);
        let d = mixture_w2(diag_mixture(&w1, &m1, &s1), diag_mixture(&w2, &m2, &s2)).unwrap();
        assert!(d < 1e-9, "a refinement of the same density scores {d}");
    }

    #[test]
    fn moving_mass_between_two_far_components_moves_the_distance_monotonically() {
        let ma = vec![vec![0.0], vec![20.0]];
        let v = vec![vec![1.0], vec![1.0]];
        let s = spreads(&v);
        let base_w = vec![0.5, 0.5];
        let mut previous = 0.0;
        for step in 1..=4 {
            let p = 0.5 + 0.1 * f64::from(step);
            let w = vec![p, 1.0 - p];
            let d = mixture_w2(diag_mixture(&base_w, &ma, &s), diag_mixture(&w, &ma, &s)).unwrap();
            assert!(
                d > previous,
                "shifting more mass must cost more: {d} after {previous}"
            );
            previous = d;
        }
    }

    #[test]
    fn the_distance_is_symmetric_and_obeys_the_triangle_inequality() {
        // MW2 is a metric, and a transport solver that stopped early on one of the three legs would
        // usually break the triangle inequality before it broke anything else.
        let w = vec![0.6, 0.4];
        let s_a = vec![vec![1.0, 2.0], vec![0.5, 0.5]];
        let s_b = vec![vec![2.0, 1.0], vec![1.5, 0.25]];
        let s_c = vec![vec![0.75, 0.75], vec![3.0, 1.0]];
        let (sa, sb, sc) = (spreads(&s_a), spreads(&s_b), spreads(&s_c));
        let ma = vec![vec![0.0, 0.0], vec![4.0, 1.0]];
        let mb = vec![vec![1.0, -3.0], vec![6.0, 2.0]];
        let mc = vec![vec![-2.0, 5.0], vec![0.5, 0.5]];
        let a = diag_mixture(&w, &ma, &sa);
        let b = diag_mixture(&w, &mb, &sb);
        let c = diag_mixture(&w, &mc, &sc);
        let ab = mixture_w2(a, b).unwrap();
        let ba = mixture_w2(b, a).unwrap();
        let bc = mixture_w2(b, c).unwrap();
        let ac = mixture_w2(a, c).unwrap();
        assert!((ab - ba).abs() < 1e-9, "asymmetric: {ab} vs {ba}");
        assert!(
            ac <= ab + bc + 1e-9,
            "triangle inequality violated: {ac} > {ab} + {bc}"
        );
    }

    #[test]
    fn the_degenerate_inputs_answer_rather_than_panic() {
        let w = vec![1.0];
        let m = vec![vec![0.0, 0.0]];
        let v = vec![vec![1.0, 1.0]];
        let s = spreads(&v);
        let good = diag_mixture(&w, &m, &s);

        let empty_w: Vec<f64> = vec![];
        let empty_m: Vec<Vec<f64>> = vec![];
        let empty_s: Vec<Spread<'_, f64>> = vec![];
        let empty = GaussianMixture {
            weights: &empty_w,
            means: &empty_m,
            covs: &empty_s,
        };
        assert!(mixture_w2(good, empty).is_none(), "an empty mixture");

        let wide_m = vec![vec![0.0, 0.0, 0.0]];
        let wide_v = vec![vec![1.0, 1.0, 1.0]];
        let wide_s = spreads(&wide_v);
        let wide = diag_mixture(&w, &wide_m, &wide_s);
        assert!(mixture_w2(good, wide).is_none(), "a dimension mismatch");

        let zero_w = vec![0.0];
        let dead = diag_mixture(&zero_w, &m, &s);
        assert!(mixture_w2(good, dead).is_none(), "no positive mass");

        let ragged_m = vec![vec![0.0, 0.0], vec![1.0]];
        let ragged_v = vec![vec![1.0, 1.0], vec![1.0]];
        let ragged_s = spreads(&ragged_v);
        let ragged_w = vec![0.5, 0.5];
        let ragged = diag_mixture(&ragged_w, &ragged_m, &ragged_s);
        assert!(mixture_w2(good, ragged).is_none(), "a ragged component");

        assert!(
            gaussian_w2_sq(
                &[0.0],
                Spread::Diagonal(&[1.0]),
                &[0.0, 0.0],
                Spread::Diagonal(&[1.0, 1.0])
            )
            .is_none(),
            "a pair of different widths"
        );
    }

    #[test]
    fn one_wide_argument_is_enough_to_refuse() {
        // Three one-at-a-time mismatches. The test above disagrees on the mean *and* the covariance
        // at once, which any one of the three guards alone would still reject.
        let two = vec![0.0f64, 0.0];
        let three = vec![0.0f64, 0.0, 0.0];
        let v2 = vec![1.0f64, 1.0];
        let v3 = vec![1.0f64, 1.0, 1.0];
        for (label, mean_b, cov_a, cov_b) in [
            ("only the second mean", &three, &v2, &v2),
            ("only the first covariance", &two, &v3, &v2),
            ("only the second covariance", &two, &v2, &v3),
        ] {
            assert!(
                gaussian_w2_sq(
                    &two,
                    Spread::Diagonal(cov_a),
                    mean_b,
                    Spread::Diagonal(cov_b)
                )
                .is_none(),
                "{label} is wide, and that alone must refuse"
            );
        }
    }

    #[test]
    fn checked_dim_rejects_what_nothing_downstream_can_see() {
        let m = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let v = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let s = spreads(&v);
        let w = vec![0.5, 0.5];
        assert_eq!(diag_mixture(&w, &m, &s).checked_dim(), Some(2));

        // Two weights against one mean, with the covariance count *agreeing* with the weights: the
        // one malformed shape that reaches the cost loop, where it would index past `means`.
        let one_m = vec![vec![0.0, 0.0]];
        assert_eq!(
            diag_mixture(&w, &one_m, &s).checked_dim(),
            None,
            "the weights outnumber the means"
        );

        let one_s = spreads(&v[..1]);
        assert_eq!(
            diag_mixture(&w, &m, &one_s).checked_dim(),
            None,
            "the weights outnumber the covariances"
        );

        // A per-component disagreement: the means agree with each other, one covariance does not.
        // `gaussian_w2_sq` would also refuse this, so only a direct call can tell the two apart.
        let wide_v = vec![vec![1.0, 1.0], vec![1.0, 1.0, 1.0]];
        let wide_s = spreads(&wide_v);
        assert_eq!(
            diag_mixture(&w, &m, &wide_s).checked_dim(),
            None,
            "one covariance of another width"
        );

        let ragged_m = vec![vec![0.0, 0.0], vec![1.0]];
        assert_eq!(
            diag_mixture(&w, &ragged_m, &s).checked_dim(),
            None,
            "one mean of another width"
        );

        let empty_w: Vec<f64> = vec![];
        let empty_m: Vec<Vec<f64>> = vec![];
        let empty_s: Vec<Spread<'_, f64>> = vec![];
        let empty = GaussianMixture {
            weights: &empty_w,
            means: &empty_m,
            covs: &empty_s,
        };
        assert_eq!(empty.checked_dim(), None, "no components at all");
    }

    #[test]
    fn a_diagonal_against_a_dense_scores_the_same_as_two_dense_ones() {
        // `Spread::trace` is reached on a `Diagonal` only through this mixed arm: two diagonals take
        // the closed form and never call it, two dense ones take the `Full` branch of it. The
        // variances are distinct and none is one, so a trace that combined them wrongly cannot
        // land on the same number by accident.
        let mean_a = vec![0.0f64, 1.0, -2.0];
        let mean_b = vec![2.0f64, -1.0, 0.5];
        let va = vec![3.0f64, 0.5, 2.0];
        let dense_a = vec![
            vec![3.0, 0.0, 0.0],
            vec![0.0, 0.5, 0.0],
            vec![0.0, 0.0, 2.0],
        ];
        let dense_b = vec![
            vec![1.0, 0.2, 0.0],
            vec![0.2, 2.0, 0.0],
            vec![0.0, 0.0, 0.75],
        ];
        let mixed = gaussian_w2_sq(
            &mean_a,
            Spread::Diagonal(&va),
            &mean_b,
            Spread::Full(&dense_b),
        )
        .expect("a diagonal and a dense covariance of the same width");
        let both_dense = gaussian_w2_sq(
            &mean_a,
            Spread::Full(&dense_a),
            &mean_b,
            Spread::Full(&dense_b),
        )
        .expect("two dense covariances");
        assert!(
            (mixed - both_dense).abs() < 1e-9,
            "mixed path {mixed} vs dense path {both_dense}"
        );
    }

    #[test]
    fn the_weights_need_not_sum_to_one() {
        // The documented contract of `GaussianMixture::weights`, and the only test that exercises
        // the division in `normalized`: every other fixture already sums to one, where dividing by
        // the total and multiplying by it agree.
        let (w, m, v) = base();
        let s = spreads(&v);
        let shift = [1.5, -2.0];
        let mt: Vec<Vec<f64>> = m
            .iter()
            .map(|r| r.iter().zip(shift).map(|(&x, t)| x + t).collect())
            .collect();
        let scaled: Vec<f64> = w.iter().map(|x| x * 7.0).collect();
        let plain = mixture_w2(diag_mixture(&w, &m, &s), diag_mixture(&w, &mt, &s)).unwrap();
        let heavy = mixture_w2(
            diag_mixture(&scaled, &m, &s),
            diag_mixture(&scaled, &mt, &s),
        )
        .unwrap();
        assert!(
            (plain - heavy).abs() < 1e-9,
            "scaling both weight vectors by 7 moved the distance: {plain} vs {heavy}"
        );
    }

    #[test]
    fn a_non_finite_cost_terminates_rather_than_growing_a_stack() {
        // `mixture_w2` scrubs non-finite pair costs before they reach the solver, so this states an
        // invariant rather than a reachable input. It is worth stating because the dual traversal
        // keys off `is_nan`: a potential that comes out `NaN` never stops being unset, so the node
        // is pushed again on every visit and the stack grows without limit. Completing at all is
        // the assertion; the pivot count is the part that can be written down.
        let half = vec![0.5, 0.5];
        for cost in [
            vec![vec![f64::NAN, 1.0], vec![1.0, 0.0]],
            vec![vec![f64::INFINITY, 1.0], vec![1.0, 0.0]],
            vec![vec![0.0, f64::NEG_INFINITY], vec![2.0, 1.0]],
        ] {
            let (_, pivots) = transport(&half, &half, &cost);
            assert!(
                pivots < iteration_cap(2, 2),
                "a non-finite cost ran the solver to its backstop after {pivots} pivots"
            );
        }
    }

    #[test]
    fn the_iteration_cap_is_the_documented_multiple_of_the_grid() {
        // The cap is a backstop no solve reaches, so its arithmetic is unobservable through
        // `transport` and is asserted directly instead of through a behaviour no instance produces.
        assert_eq!(iteration_cap(1, 1), 64 * 3);
        assert_eq!(iteration_cap(3, 4), 64 * (12 + 7));
        assert_eq!(iteration_cap(64, 64), 64 * (4096 + 128));
    }

    /// A random composition of `q` unit atoms into `n` parts.
    ///
    /// `allow_empty` decides whether a part may end up with no mass. Both cases are real: a fitted
    /// mixture can carry a zero-weight component, and `normalized` passes it through rather than
    /// dropping it, so the north-west corner has to walk past an exhausted marginal with columns
    /// still to come. That is the only state in which its guards differ from each other.
    fn composition(rng: &mut Lcg, n: usize, q: usize, allow_empty: bool) -> Vec<usize> {
        let seed = usize::from(!allow_empty);
        let mut parts = vec![seed; n];
        for _ in 0..q - seed * n {
            let i = ((rng.next() * n as f64) as usize).min(n - 1);
            parts[i] += 1;
        }
        parts
    }

    /// The lexicographically next permutation, or `false` at the last one.
    fn next_permutation(p: &mut [usize]) -> bool {
        let n = p.len();
        let Some(i) = (0..n.saturating_sub(1)).rev().find(|&i| p[i] < p[i + 1]) else {
            return false;
        };
        let j = (i + 1..n)
            .rev()
            .find(|&j| p[j] > p[i])
            .expect("p[i] < p[i + 1] guarantees a larger element to the right");
        p.swap(i, j);
        p[i + 1..].reverse();
        true
    }

    /// The transportation optimum computed a second way, sharing no code with the simplex.
    ///
    /// With every marginal a multiple of `1/q`, the problem is an assignment problem over `q` unit
    /// atoms per side, so its optimum is the cheapest of the `q!` matchings. Exponential, and that
    /// is the point: a different algorithm rather than a rearrangement of the one under test.
    fn assignment_optimum(a: &[usize], b: &[usize], cost: &[Vec<f64>], q: usize) -> f64 {
        let atoms = |counts: &[usize]| -> Vec<usize> {
            counts
                .iter()
                .enumerate()
                .flat_map(|(i, &n)| std::iter::repeat_n(i, n))
                .collect()
        };
        let (rows, cols) = (atoms(a), atoms(b));
        let mut perm: Vec<usize> = (0..q).collect();
        let mut best = f64::INFINITY;
        loop {
            let total: f64 = (0..q).map(|t| cost[rows[t]][cols[perm[t]]]).sum();
            best = best.min(total);
            if !next_permutation(&mut perm) {
                break;
            }
        }
        best / q as f64
    }

    #[test]
    fn the_simplex_matches_a_brute_force_optimum_and_stops_short_of_its_backstop() {
        const Q: usize = 6;
        let mut rng = Lcg(20_260_826);
        let mut worked = 0usize;
        for (na, nb) in [(2usize, 2usize), (2, 5), (5, 2), (3, 3), (3, 4), (4, 3)] {
            for trial in 0..8 {
                // Half the instances get integer costs. Ties there are the common case, so reduced
                // costs of exactly zero and pivots of zero step size are too -- which is the state
                // Bland's rule exists for and the one a cycling solver never leaves.
                let ties = trial % 2 == 1;
                let cost: Vec<Vec<f64>> = (0..na)
                    .map(|_| {
                        (0..nb)
                            .map(|_| {
                                let x = 10.0 * rng.next();
                                if ties { x.round() } else { x }
                            })
                            .collect()
                    })
                    .collect();
                let empty = trial >= 4; // a quarter of the marginals carry a zero-weight component
                let ca = composition(&mut rng, na, Q, empty);
                let cb = composition(&mut rng, nb, Q, empty);
                let a: Vec<f64> = ca.iter().map(|&n| n as f64 / Q as f64).collect();
                let b: Vec<f64> = cb.iter().map(|&n| n as f64 / Q as f64).collect();
                let (got, pivots) = transport(&a, &b, &cost);
                let want = assignment_optimum(&ca, &cb, &cost, Q);
                assert!(
                    (got - want).abs() < 1e-12,
                    "{na}x{nb} trial {trial}: simplex {got}, brute force {want}"
                );
                assert!(
                    pivots < iteration_cap(na, nb),
                    "{na}x{nb} trial {trial}: ran to the backstop after {pivots} pivots"
                );
                worked += pivots;
            }
        }
        // Without this the pivot assertion above is satisfied by a solver that never pivots at all,
        // which the north-west corner alone is not entitled to be: it produces a feasible basis, not
        // an optimal one, and these instances are not optimal where it leaves them.
        assert!(worked > 0, "no instance in the sweep needed a single pivot");
    }
}

/// Emits random instances and this crate's answer for them, so an independent solver can be handed
/// the same problem. Not a test: it asserts nothing, and the comparison lives in
/// `local/scratch/mw2_crosscheck.py`.
#[cfg(test)]
mod measure {
    use super::*;

    #[test]
    #[ignore = "prints instances for an external cross-check"]
    fn instances_for_an_independent_transport_solver() {
        let mut rng = Lcg(20_260_825);
        for case in 0..12 {
            let dim = 1 + case % 3;
            let ka = 2 + case % 4;
            let kb = 2 + (case * 3) % 5;
            let mut emit = |k: usize| {
                let w: Vec<f64> = (0..k).map(|_| 0.05 + rng.next()).collect();
                let m: Vec<Vec<f64>> = (0..k)
                    .map(|_| (0..dim).map(|_| 10.0 * rng.next() - 5.0).collect())
                    .collect();
                let v: Vec<Vec<f64>> = (0..k)
                    .map(|_| (0..dim).map(|_| 0.05 + 4.0 * rng.next()).collect())
                    .collect();
                (w, m, v)
            };
            let (wa, ma, va) = emit(ka);
            let (wb, mb, vb) = emit(kb);
            let sa: Vec<Spread<'_, f64>> =
                va.iter().map(|x| Spread::Diagonal(x.as_slice())).collect();
            let sb: Vec<Spread<'_, f64>> =
                vb.iter().map(|x| Spread::Diagonal(x.as_slice())).collect();
            let a = GaussianMixture {
                weights: &wa,
                means: &ma,
                covs: &sa,
            };
            let b = GaussianMixture {
                weights: &wb,
                means: &mb,
                covs: &sb,
            };
            let d = mixture_w2(a, b).expect("a well-formed pair");
            println!(
                "{{\"wa\":{wa:?},\"ma\":{ma:?},\"va\":{va:?},\"wb\":{wb:?},\"mb\":{mb:?},\"vb\":{vb:?},\"mw2\":{d}}}"
            );
        }
    }
}
