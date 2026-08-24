//! Sensitivity-sampling coresets on the leaf summary, with the summarization error made explicit.
//!
//! A coreset is a small weighted set on which *every* candidate solution scores within `(1 ± ε)` of
//! its score on the full data. That is a much stronger promise than "a summary that clusters well",
//! and it is the one this module makes — in two independent pieces, because the summary and the
//! sample fail in different ways and hiding either inside the other would make both unfalsifiable.
//!
//! ## Piece one: what the leaf summary itself costs
//!
//! Write `Δ = Σ_i S_i` for the total within-leaf scatter and, for a candidate centre set `C`,
//!
//! ```text
//! cost(C)  = Σ_p  d²(p, C)                      what the points actually cost
//! ĉost(C)  = Σ_i (S_i + n_i·d²(μ_i, C))          what the summary reports
//! ```
//!
//! `ĉost` is exactly the cost of assigning every point of a leaf to the centre nearest that leaf's
//! *centroid* — the CF identity `Σ_{p∈i} ‖p − c‖² = S_i + n_i‖μ_i − c‖²` makes it exact, not an
//! estimate. A constrained assignment can only cost more, so `ĉost ≥ cost` always, and the gap is
//! bounded. For a point `p` in leaf `i` at radius `r_p = ‖p − μ_i‖`, with `c_i` the centre nearest
//! `μ_i` and `c*` the centre nearest `p`:
//!
//! ```text
//! ‖p − c_i‖ ≤ r_p + ‖μ_i − c_i‖ ≤ r_p + ‖μ_i − c*‖ ≤ 2 r_p + ‖p − c*‖
//! ```
//!
//! — the middle step is the only place `c_i` is used, and it holds because `c_i` minimises the
//! distance to `μ_i`. Squaring, summing over all points, and applying Cauchy-Schwarz to the cross
//! term gives the whole bound in one line:
//!
//! ```text
//! 0 ≤ ĉost(C) − cost(C) ≤ 4·√(Δ · cost(C)) + 4·Δ           for every C, every k
//! ```
//!
//! So the relative error is `4√ρ + 4ρ` with `ρ = Δ / cost(C)`, and since `cost(C) ≥ OPT_k` for
//! every `C`, `ρ ≤ Δ / OPT_k` bounds it uniformly. `Δ` is known exactly from the tree;
//! `OPT_k` is not, which is why [`Coreset::summary_epsilon`] makes the caller name the
//! approximation factor it is willing to assume rather than quietly picking one.
//!
//! ## Piece two: sampling the leaves down to `Õ(k/ε²)`
//!
//! `ĉost(C) = Δ + Σ_i n_i·d²(μ_i, C)`, and **`Δ` does not depend on `C`**. So a coreset of the
//! weighted point set `{(μ_i, n_i)}` is a coreset of `ĉost` up to that additive constant, which the
//! export carries as [`Coreset::offset`] instead of folding it in and losing it.
//!
//! The construction is sensitivity sampling (Feldman & Langberg, STOC 2011; Braverman et al.).
//! From an α-approximate solution `A` on the leaves, with `B_j` its clusters, `c_i` the cost leaf
//! `i` contributes, `C_j = Σ_{i∈B_j} c_i` and `W_j = Σ_{i∈B_j} n_i`:
//!
//! ```text
//! s_i = 8·c_i/Ctot  +  2·(n_i/W_j)·(C_j/Ctot)  +  4·(n_i/W_j),      S = Σ_i s_i = 10 + 4k
//! ```
//!
//! Sample `T` leaves with probability `s_i/S` and weight each `n_i·S/(T·s_i)`; the estimator is
//! unbiased for any positive `s_i`, and it is this particular `s_i` — an upper bound on the true
//! sensitivity — that bounds the variance and hence the sample size. Sensitivity sampling is now
//! known to attain the *optimal* worst-case coreset size `Õ(k·ε⁻²·min(√k, ε⁻²))`, matching the
//! STOC 2022 lower bound, and `Õ(k/ε²)` on `Ω(1)`-cost-stable instances (arXiv 2405.01339).
//!
//! The α-approximate solution comes from the crate's own weighted k-means, whose seeding samples on
//! the exact CF potential `S_i + n_i·D²_i`, so the guarantee is stated over the weighted instance
//! the leaves define rather than over the points — which is the correct object here.

use crate::clustering::kmeans;
use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// A weighted sample of leaves that scores every candidate solution like the full leaf set does.
pub struct Coreset<R> {
    /// Index into the input features, one per retained leaf, ascending and without repeats.
    pub indices: Vec<usize>,
    /// The retained leaves' means.
    pub points: Vec<Vec<R>>,
    /// Sampling weights. `Σ_j w_j` estimates `Σ_i n_i` but is not equal to it — an unbiased
    /// estimator of a sum is not a partition of it.
    pub weights: Vec<R>,
    /// `Δ = Σ_i S_i` over **all** leaves, not just the retained ones. Add it to a weighted cost
    /// over `points` to recover `ĉost`; it is a constant in the candidate solution, so leaving it
    /// out changes every cost by the same amount and none of the comparisons between them.
    pub offset: R,
    /// `ĉost(A)` of the α-approximate solution the sensitivities were derived from. Reported so
    /// [`Coreset::summary_epsilon`] has a scale to work against; it upper-bounds `OPT_k`.
    pub reference_cost: R,
    /// `S = Σ_i s_i`. Grows like `10 + 4k`; a value far from that means the reference solution
    /// left a cluster empty.
    pub total_sensitivity: f64,
    /// Leaves the tree held before sampling.
    pub n_leaves: usize,
}

impl<R: Real> Coreset<R> {
    /// Relative summarization error `4√ρ + 4ρ` from piece one, evaluated at
    /// `ρ = alpha · offset / reference_cost`.
    ///
    /// `alpha` is the approximation factor the caller is willing to assume for the reference
    /// solution, and it is a required argument for a reason: `reference_cost ≥ OPT_k` always, so
    /// `offset / reference_cost` *under*-states `Δ/OPT_k` and `summary_epsilon(1.0)` is an
    /// optimistic reading, not a certificate. The shipped seeding is k-means++ with greedy trials,
    /// whose guarantee is `O(log k)` in expectation — a distribution, not a bound on the run in
    /// hand. Pass the factor you can defend.
    ///
    /// The result covers the summary only. Sampling error is on top of it, and is what the coreset
    /// size buys down.
    pub fn summary_epsilon(&self, alpha: f64) -> f64 {
        let cost = self.reference_cost.to_f64().unwrap_or(0.0);
        if !(cost.is_finite() && cost > 0.0) {
            return 0.0;
        }
        let rho = alpha * self.offset.to_f64().unwrap_or(0.0) / cost;
        4.0 * rho.max(0.0).sqrt() + 4.0 * rho.max(0.0)
    }

    /// `Σ_j w_j·d²(x_j, C) + offset` — the coreset's estimate of `ĉost(C)`.
    pub fn cost(&self, centers: &[Vec<R>]) -> R {
        let mut total = self.offset;
        for (x, &w) in self.points.iter().zip(&self.weights) {
            let mut best = R::infinity();
            for c in centers {
                let d = sq_euclidean(x, c);
                if d < best {
                    best = d;
                }
            }
            if best.is_finite() {
                total = total + w * best;
            }
        }
        total
    }
}

/// `ĉost(C) = Σ_i (S_i + n_i·d²(μ_i, C))` over the full leaf set — the quantity a coreset
/// approximates, and the reference any empirical `(k, ε)` check has to be taken against.
pub fn summary_cost<R: Real, C: ClusterFeature<R>>(features: &[C], centers: &[Vec<R>]) -> R {
    let mut total = R::zero();
    for f in features {
        total = total + f.ssd();
        let mut best = R::infinity();
        for c in centers {
            let d = sq_euclidean(f.mean(), c);
            if d < best {
                best = d;
            }
        }
        if best.is_finite() {
            total = total + f.weight() * best;
        }
    }
    total
}

/// Sample `size` leaves into a `(k, ε)`-coreset by sensitivity sampling.
///
/// `size ≥ features.len()` returns every leaf with its own weight — the exact summary, no sampling
/// error — rather than drawing `size` times from `m < size` leaves and reporting a noisy version of
/// something it already holds exactly.
pub fn sensitivity_coreset<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    size: usize,
    seed: u64,
) -> Coreset<R> {
    let m = features.len();
    if m == 0 {
        return Coreset {
            indices: Vec::new(),
            points: Vec::new(),
            weights: Vec::new(),
            offset: R::zero(),
            reference_cost: R::zero(),
            total_sensitivity: 0.0,
            n_leaves: 0,
        };
    }
    let offset = features.iter().fold(R::zero(), |a, f| a + f.ssd());
    let k = k.max(1).min(m);

    let reference = kmeans(features, k, 100, 3, seed);
    // `c_i` is what leaf `i` contributes to the reference solution's cost, and it is deliberately
    // the *centroid* term only: the `S_i` half of the leaf's cost is the same under every candidate
    // solution, so charging it to a leaf's sensitivity would rank leaves by how internally spread
    // they are rather than by how much any solution can disagree about them.
    let mut ci = vec![0.0f64; m];
    let mut cj = vec![0.0f64; k];
    let mut wj = vec![0.0f64; k];
    for (i, f) in features.iter().enumerate() {
        let j = reference.labels[i];
        let n = f.weight().to_f64().unwrap_or(0.0);
        let d = sq_euclidean(f.mean(), &reference.centers[j])
            .to_f64()
            .unwrap_or(0.0);
        ci[i] = n * d;
        cj[j] += ci[i];
        wj[j] += n;
    }
    let ctot: f64 = ci.iter().sum();

    let mut sens = vec![0.0f64; m];
    for (i, f) in features.iter().enumerate() {
        let j = reference.labels[i];
        let n = f.weight().to_f64().unwrap_or(0.0);
        // A cluster with no mass contributes no membership term; without the guard an empty
        // reference cluster turns the whole sensitivity vector into NaN and the sampler silently
        // degenerates to "leaf 0, repeated".
        let share = if wj[j] > 0.0 { n / wj[j] } else { 0.0 };
        let cost_term = if ctot > 0.0 { ci[i] / ctot } else { 0.0 };
        let group_term = if ctot > 0.0 {
            share * cj[j] / ctot
        } else {
            0.0
        };
        sens[i] = 8.0 * cost_term + 2.0 * group_term + 4.0 * share;
    }
    let total: f64 = sens.iter().sum();

    // Degenerate summaries — one leaf, or every leaf a single point already at a centre — leave
    // every sensitivity at zero. There is nothing to prefer, so keep everything.
    if size >= m || !(total.is_finite() && total > 0.0) {
        return exact(features, offset, reference.inertia, total);
    }

    let mut cumulative = Vec::with_capacity(m);
    let mut running = 0.0;
    for &s in &sens {
        running += s;
        cumulative.push(running);
    }
    let mut hits = vec![0usize; m];
    let mut rng = SplitMix64::new(seed ^ 0x0C0F_FEE1_5BAD_u64);
    for _ in 0..size {
        let u = rng.next_f64() * total;
        let i = cumulative.partition_point(|&c| c <= u).min(m - 1);
        hits[i] += 1;
    }

    let t = size as f64;
    let mut indices = Vec::new();
    let mut points = Vec::new();
    let mut weights = Vec::new();
    for (i, &h) in hits.iter().enumerate() {
        if h == 0 {
            continue;
        }
        let n = features[i].weight().to_f64().unwrap_or(0.0);
        let w = h as f64 * n * total / (t * sens[i]);
        indices.push(i);
        points.push(features[i].mean().to_vec());
        weights.push(R::from_f64(w).unwrap_or_else(R::zero));
    }
    Coreset {
        indices,
        points,
        weights,
        offset,
        reference_cost: reference.inertia,
        total_sensitivity: total,
        n_leaves: m,
    }
}

/// Every leaf, at its own weight: the summary itself, viewed as a coreset with no sampling error.
fn exact<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    offset: R,
    reference_cost: R,
    total_sensitivity: f64,
) -> Coreset<R> {
    Coreset {
        indices: (0..features.len()).collect(),
        points: features.iter().map(|f| f.mean().to_vec()).collect(),
        weights: features.iter().map(|f| f.weight()).collect(),
        offset,
        reference_cost,
        total_sensitivity,
        n_leaves: features.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::Spherical;

    fn leaf(points: &[[f64; 2]]) -> Spherical<f64> {
        let mut cf = Spherical::new(2);
        for p in points {
            cf.push(p, 1.0);
        }
        cf
    }

    /// Leaves with real internal spread and unequal masses, over three groups.
    fn summary(
        seed: u64,
        leaves_per_group: usize,
        per_leaf: usize,
    ) -> (Vec<Spherical<f64>>, Vec<[f64; 2]>) {
        let mut rng = SplitMix64::new(seed);
        let mut feats = Vec::new();
        let mut pts = Vec::new();
        for (cx, cy) in [(0.0, 0.0), (14.0, 1.0), (3.0, 12.0)] {
            for _ in 0..leaves_per_group {
                let (lx, ly) = (cx + 2.0 * rng.gauss(), cy + 2.0 * rng.gauss());
                let group: Vec<[f64; 2]> = (0..per_leaf)
                    .map(|_| [lx + 0.4 * rng.gauss(), ly + 0.4 * rng.gauss()])
                    .collect();
                pts.extend_from_slice(&group);
                feats.push(leaf(&group));
            }
        }
        (feats, pts)
    }

    fn point_cost(points: &[[f64; 2]], centers: &[Vec<f64>]) -> f64 {
        points
            .iter()
            .map(|p| {
                centers
                    .iter()
                    .map(|c| sq_euclidean(p.as_slice(), c))
                    .fold(f64::INFINITY, f64::min)
            })
            .sum()
    }

    fn candidates(seed: u64, k: usize, n: usize) -> Vec<Vec<Vec<f64>>> {
        let mut rng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                (0..k)
                    .map(|_| vec![14.0 * rng.next_f64() - 2.0, 14.0 * rng.next_f64() - 2.0])
                    .collect()
            })
            .collect()
    }

    #[test]
    fn the_summary_never_underestimates_and_stays_inside_the_derived_bound() {
        // Piece one, checked against the points themselves rather than against another summary.
        let (feats, pts) = summary(1, 12, 40);
        let delta: f64 = feats.iter().map(|f| f.ssd()).sum();
        for k in [1usize, 3, 6] {
            for c in candidates(k as u64 * 31, k, 40) {
                let hat = summary_cost(&feats, &c);
                let real = point_cost(&pts, &c);
                assert!(hat >= real - 1e-9, "summary underestimated: {hat} < {real}");
                let bound = 4.0 * (delta * real).sqrt() + 4.0 * delta;
                assert!(
                    hat - real <= bound + 1e-9,
                    "gap {} exceeds {bound}",
                    hat - real
                );
            }
        }
    }

    #[test]
    fn the_bound_is_not_vacuous_on_the_fixture_it_is_checked_on() {
        // A bound larger than the cost it bounds proves nothing, so pin that this fixture is in the
        // regime where the bound has content -- otherwise the test above passes on any code at all.
        let (feats, pts) = summary(1, 12, 40);
        let delta: f64 = feats.iter().map(|f| f.ssd()).sum();
        let c = candidates(7, 3, 1).pop().unwrap();
        let real = point_cost(&pts, &c);
        let bound = 4.0 * (delta * real).sqrt() + 4.0 * delta;
        assert!(
            bound < real,
            "bound {bound} is looser than the cost {real} it bounds"
        );
    }

    #[test]
    fn the_estimator_is_unbiased_over_seeds() {
        let (feats, _) = summary(2, 15, 30);
        let c = candidates(5, 4, 1).pop().unwrap();
        let want = summary_cost(&feats, &c);
        let mean: f64 = (0..80)
            .map(|s| sensitivity_coreset(&feats, 4, 40, s).cost(&c))
            .sum::<f64>()
            / 80.0;
        assert!(
            (mean - want).abs() < 0.02 * want,
            "mean {mean} vs {want} over 80 seeds"
        );
    }

    #[test]
    fn every_candidate_solution_scores_within_epsilon_on_the_coreset() {
        // The acceptance criterion, as a test: sweep candidate solutions, not just the one the
        // sensitivities were derived from -- a coreset that only works for its own reference
        // solution is not a coreset.
        let (feats, _) = summary(3, 20, 25);
        for k in [2usize, 4, 8] {
            let cs = sensitivity_coreset(&feats, k, 200, 11);
            let mut worst: f64 = 0.0;
            for c in candidates(k as u64 * 17 + 1, k, 60) {
                let want = summary_cost(&feats, &c);
                let got = cs.cost(&c);
                worst = worst.max((got - want).abs() / want);
            }
            assert!(worst < 0.10, "k = {k}: worst relative error {worst}");
        }
    }

    #[test]
    fn a_larger_sample_is_a_better_sample() {
        let (feats, _) = summary(4, 20, 25);
        let cands = candidates(9, 5, 40);
        let err = |size: usize| -> f64 {
            let mut acc = 0.0;
            for s in 0..12u64 {
                let cs = sensitivity_coreset(&feats, 5, size, s);
                for c in &cands {
                    let want = summary_cost(&feats, c);
                    acc += ((cs.cost(c) - want) / want).powi(2);
                }
            }
            (acc / (12.0 * cands.len() as f64)).sqrt()
        };
        let (small, large) = (err(30), err(300));
        assert!(
            large < small,
            "300 samples ({large}) not better than 30 ({small})"
        );
    }

    #[test]
    fn a_budget_at_or_above_the_leaf_count_returns_the_summary_exactly() {
        let (feats, _) = summary(5, 4, 20);
        let cs = sensitivity_coreset(&feats, 3, feats.len(), 0);
        assert_eq!(cs.indices.len(), feats.len());
        let c = candidates(3, 3, 1).pop().unwrap();
        assert!((cs.cost(&c) - summary_cost(&feats, &c)).abs() < 1e-9);
    }

    #[test]
    fn total_sensitivity_tracks_the_ten_plus_four_k_the_derivation_predicts() {
        let (feats, _) = summary(6, 20, 20);
        for k in [1usize, 3, 5, 9] {
            let cs = sensitivity_coreset(&feats, k, 50, 2);
            let want = 10.0 + 4.0 * k as f64;
            assert!(
                (cs.total_sensitivity - want).abs() < 1e-6,
                "k = {k}: {} vs {want}",
                cs.total_sensitivity
            );
        }
    }

    #[test]
    fn summary_epsilon_scales_with_the_approximation_factor_it_is_given() {
        let (feats, _) = summary(7, 20, 20);
        let cs = sensitivity_coreset(&feats, 3, 100, 0);
        let (one, four) = (cs.summary_epsilon(1.0), cs.summary_epsilon(4.0));
        assert!(one > 0.0 && four > one, "{one} then {four}");
        // 4 rho + 4 sqrt(rho): doubling alpha at most doubles the linear half and multiplies the
        // square-root half by sqrt(2), so the ratio is bounded by the factor itself.
        assert!(four <= 4.0 * one, "{four} vs {one}");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let empty: Vec<Spherical<f64>> = Vec::new();
        let cs = sensitivity_coreset(&empty, 3, 10, 0);
        assert!(cs.indices.is_empty() && cs.summary_epsilon(1.0) == 0.0);

        let one = vec![leaf(&[[1.0, 1.0]])];
        let cs = sensitivity_coreset(&one, 5, 10, 0);
        assert_eq!(cs.indices, vec![0]);

        // Coincident leaves: the reference solution costs nothing, so both cost terms vanish and
        // only the membership term `4·n_i/W_j` survives. That is still a positive, well-defined
        // distribution -- the fallback is for a sensitivity vector that is *entirely* zero, which
        // this is not, so sampling proceeds and must not divide by any of the vanished terms.
        let same = vec![leaf(&[[2.0, 2.0]]); 6];
        let cs = sensitivity_coreset(&same, 2, 3, 0);
        assert!(!cs.indices.is_empty() && cs.indices.len() <= 3);
        assert!(cs.weights.iter().all(|w| w.is_finite() && *w > 0.0));
        assert!(cs.offset == 0.0 && cs.reference_cost == 0.0);
    }
}
