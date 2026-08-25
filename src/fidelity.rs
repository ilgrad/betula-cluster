//! How faithful is the leaf summary to the data it replaced? A label-free, head-independent number.
//!
//! Every other quality number in the crate needs something the summary itself cannot supply: ground
//! truth labels, or a choice of head, or a value of `k`. The sum-of-squares identity
//! `Σ‖x − c‖² = S_i + n_i‖μ_i − c‖²` is exact but answers only the k-means question — a summary can
//! preserve every within-cluster sum of squares and still misplace the shape of the distribution.
//!
//! Maximum mean discrepancy asks the distributional question instead. With a characteristic kernel
//! `k`, `MMD(P, Q) = ‖E_P k(x, ·) − E_Q k(x, ·)‖_H` is zero exactly when `P = Q`, so it measures the
//! summary against the sample without reference to any downstream task.
//!
//! # The closed form
//!
//! The crate models a leaf as an isotropic cloud `N(μ_i, s_i I)` with `s_i = S_i / (n_i·d)` — the
//! same surrogate the scale-space head mollifies with — so the surrogate measure is the mixture
//! `P = Σ_i (n_i / N) · N(μ_i, s_i I)` and no sampling is needed to integrate against it. For the
//! Gaussian kernel `k(x, y) = exp(−‖x − y‖² / 2h²)` and independent `X ~ N(a, A)`, `Y ~ N(b, B)`,
//! the difference `X − Y` is `N(a − b, A + B)`, so
//!
//! ```text
//! E k(X, Y) = E_{Z ~ N(m, C)} exp(−Z'Z / 2h²) = |I + C/h²|^(−1/2) · exp(−m'(h²I + C)⁻¹ m / 2)
//! ```
//!
//! which for `C = s·I` collapses to
//!
//! ```text
//! g(‖m‖², s) = (1 + s/h²)^(−d/2) · exp(−‖m‖² / (2(h² + s)))
//! ```
//!
//! and `g(·, 0)` is the plain kernel, so raw sample points are the `s = 0` case of the same formula
//! rather than a second code path. Verified against Monte Carlo before use: worst |z| = 3.02 over 24
//! stochastic cells at 4·10⁶ draws, with the 8 deterministic cells exact to 10⁻¹².
//!
//! # What it costs and what it does not say
//!
//! `O(m² + m·M + M²)` kernel evaluations for `m` leaves and `M` sample rows, each `O(d)` — this is a
//! diagnostic to run at a few budgets, not something to put inside a fit. It is a **V**-statistic:
//! the empirical measure of the sample is taken as `Q` itself, diagonal terms included, so the value
//! is the distance to that sample rather than an unbiased estimate of the distance to the population
//! it came from.
//!
//! And it inherits the isotropic leaf surrogate. A `Full` leaf carrying a strongly anisotropic
//! scatter is still integrated as a ball of the same trace, so the number scores the summary the
//! *crate's own leaf model* describes, not the tightest one the stored moments could support.

use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// At most this many sample rows feed the median-heuristic bandwidth. The median is a scale
/// estimate, not a statistic anyone reports, and the pairwise distances it needs would otherwise be
/// the one part of this that allocates `O(M²)`.
const MEDIAN_ROWS: usize = 1024;

/// Isotropic radius of the leaf's surrogate cloud, `s_i = S_i / (n_i·d)`.
fn leaf_scale<R: Real, C: ClusterFeature<R>>(f: &C) -> R {
    let n = f.weight();
    let d = R::from_usize(f.dim()).unwrap_or_else(R::one);
    if n <= R::zero() || d <= R::zero() {
        R::zero()
    } else {
        f.ssd() / (n * d)
    }
}

/// `g(‖m‖², s)` — the expected Gaussian kernel between two isotropic clouds whose centres are
/// `‖m‖²` apart and whose variances sum to `s`. With `s = 0` this is the plain kernel.
fn expected_kernel<R: Real>(sq_dist: R, s: R, h2: R, dim: usize) -> R {
    let half_d = R::from_usize(dim).unwrap_or_else(R::one) / (R::one() + R::one());
    let inflation = (R::one() + s / h2).powf(-half_d);
    inflation * (-sq_dist / ((R::one() + R::one()) * (h2 + s))).exp()
}

/// Median heuristic: `h² = median‖x − y‖² / 2`, so the kernel is `exp(−1)` at the median pair.
fn median_bandwidth_sq<R: Real>(rows: &[&[R]]) -> R {
    let take = rows.len().min(MEDIAN_ROWS);
    let mut d2: Vec<R> = Vec::with_capacity(take * take / 2);
    for i in 0..take {
        for j in (i + 1)..take {
            d2.push(sq_euclidean(rows[i], rows[j]));
        }
    }
    if d2.is_empty() {
        return R::one();
    }
    let mid = d2.len() / 2;
    d2.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let m = d2[mid];
    if m > R::zero() {
        m / (R::one() + R::one())
    } else {
        R::one()
    }
}

/// Maximum mean discrepancy between the leaf surrogate and a raw sample, in the Gaussian-kernel RKHS.
/// Lower is better; `0` means the surrogate reproduces the sample's kernel mean embedding exactly.
///
/// `sample` is row-major `M × d`, with `d` taken from the leaves — a trailing partial row is
/// ignored. `bandwidth` is the kernel's `h`; `None` picks it by the median heuristic on the sample,
/// which makes the value comparable across leaf budgets of the *same* data and meaningless across
/// different data, exactly like the heuristic it comes from.
///
/// Returns `0` when either side is empty, which is the only answer available and not a claim that
/// the summary is good.
pub fn summary_mmd<R: Real, C: ClusterFeature<R>>(
    leaves: &[C],
    sample: &[R],
    bandwidth: Option<R>,
) -> R {
    let Some(dim) = leaves.first().map(ClusterFeature::dim) else {
        return R::zero();
    };
    if dim == 0 {
        return R::zero();
    }
    let rows: Vec<&[R]> = sample.chunks_exact(dim).collect();
    if rows.is_empty() {
        return R::zero();
    }
    let total: R = leaves.iter().map(ClusterFeature::weight).sum();
    if total <= R::zero() {
        return R::zero();
    }

    let h2 = match bandwidth {
        Some(h) if h > R::zero() => h * h,
        _ => median_bandwidth_sq(&rows),
    };
    let scale: Vec<R> = leaves.iter().map(leaf_scale).collect();
    let m_inv = R::one() / R::from_usize(rows.len()).unwrap_or_else(R::one);

    let mut e_pp = R::zero();
    for (i, fi) in leaves.iter().enumerate() {
        for (j, fj) in leaves.iter().enumerate() {
            let d2 = sq_euclidean(fi.mean(), fj.mean());
            let k = expected_kernel(d2, scale[i] + scale[j], h2, dim);
            e_pp = e_pp + (fi.weight() / total) * (fj.weight() / total) * k;
        }
    }

    let mut e_pq = R::zero();
    for (i, fi) in leaves.iter().enumerate() {
        let w = fi.weight() / total;
        for row in &rows {
            let d2 = sq_euclidean(fi.mean(), row);
            e_pq = e_pq + w * m_inv * expected_kernel(d2, scale[i], h2, dim);
        }
    }

    let mut e_qq = R::zero();
    for a in &rows {
        for b in &rows {
            e_qq = e_qq + m_inv * m_inv * expected_kernel(sq_euclidean(a, b), R::zero(), h2, dim);
        }
    }

    let two = R::one() + R::one();
    // Nonnegative in exact arithmetic — it is a squared norm in the RKHS — so a negative value is
    // cancellation between three sums of the same magnitude, not a distance.
    (e_pp - two * e_pq + e_qq).max(R::zero()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::feature::{Full, Spherical};

    /// `per` points around each centre, one blob after another — consecutive, so that chunking
    /// consecutive points into a leaf produces a leaf that lies inside one blob. Interleaving the
    /// centres instead makes every leaf straddle every blob and the coarsening tests measure the
    /// fixture rather than the summary.
    fn blob_points(seed: u64, per: usize, centres: &[[f64; 2]], spread: f64) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        let mut flat = Vec::with_capacity(per * centres.len() * 2);
        for c in centres {
            for _ in 0..per {
                flat.push(c[0] + spread * rng.gauss());
                flat.push(c[1] + spread * rng.gauss());
            }
        }
        flat
    }

    /// One leaf per point: a summary that discarded nothing.
    fn exact_leaves(flat: &[f64]) -> Vec<Spherical<f64>> {
        flat.chunks_exact(2)
            .map(|p| {
                let mut f = Spherical::new(2);
                f.push(p, 1.0);
                f
            })
            .collect()
    }

    /// `chunk` consecutive points per leaf: a summary that discarded something.
    fn coarse_leaves(flat: &[f64], chunk: usize) -> Vec<Spherical<f64>> {
        flat.chunks_exact(2)
            .collect::<Vec<_>>()
            .chunks(chunk)
            .map(|group| {
                let mut f = Spherical::new(2);
                for p in group {
                    f.push(p, 1.0);
                }
                f
            })
            .collect()
    }

    /// The median heuristic written out independently: `median‖x − y‖² / 2` over all row pairs.
    fn median_h2(flat: &[f64], dim: usize) -> f64 {
        let rows: Vec<&[f64]> = flat.chunks_exact(dim).collect();
        let mut d2: Vec<f64> = Vec::new();
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                d2.push(
                    rows[i]
                        .iter()
                        .zip(rows[j])
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum(),
                );
            }
        }
        d2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        d2[d2.len() / 2] / 2.0
    }

    #[test]
    fn the_default_bandwidth_is_the_median_heuristic_and_nothing_near_it() {
        // Pins `median_bandwidth_sq` exactly: the pair loop, the rank it selects and the halving.
        // Every neighbouring rank and every arithmetic slip lands on a different `h`, so the
        // equality is a much narrower claim than "the default is some sensible bandwidth".
        let flat = blob_points(5, 12, &[[0.0, 0.0], [4.0, 3.0]], 0.7);
        let leaves = coarse_leaves(&flat, 4);
        let h2 = median_h2(&flat, 2);
        let auto = summary_mmd(&leaves, &flat, None);
        assert!(
            (auto - summary_mmd(&leaves, &flat, Some(h2.sqrt()))).abs() < 1e-12,
            "the default is not the median heuristic"
        );
        for wrong in [h2 * 0.5, h2 * 2.0, h2 / 1.05] {
            assert!(
                (auto - summary_mmd(&leaves, &flat, Some(wrong.sqrt()))).abs() > 1e-9,
                "a bandwidth of {wrong} scores the same as the median {h2}"
            );
        }
    }

    #[test]
    fn a_sample_with_no_spread_falls_back_to_a_unit_bandwidth() {
        // Every row identical ⇒ the median squared distance is exactly zero, and `h² = 0` would
        // divide by zero inside the kernel. The fallback is `h² = 1`, and the test pins that
        // choice rather than only checking the result is finite.
        let flat = vec![2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0];
        let leaves = coarse_leaves(&[0.0, 0.0, 5.0, 5.0], 1);
        let auto = summary_mmd(&leaves, &flat, None);
        assert!(auto.is_finite(), "a zero-spread sample produced {auto}");
        assert!(
            (auto - summary_mmd(&leaves, &flat, Some(1.0))).abs() < 1e-12,
            "the zero-median fallback is not h = 1"
        );
    }

    #[test]
    fn a_non_positive_bandwidth_falls_back_instead_of_dividing_by_zero() {
        let flat = blob_points(6, 10, &[[0.0, 0.0], [3.0, 0.0]], 0.5);
        let leaves = coarse_leaves(&flat, 5);
        let auto = summary_mmd(&leaves, &flat, None);
        for bad in [0.0, -1.0] {
            let got = summary_mmd(&leaves, &flat, Some(bad));
            assert!(got.is_finite(), "bandwidth {bad} produced {got}");
            assert!(
                (got - auto).abs() < 1e-12,
                "bandwidth {bad} must fall back to the median heuristic"
            );
        }
    }

    #[test]
    fn an_explicit_bandwidth_enters_the_kernel_as_its_own_square() {
        // The whole statistic recomputed from the closed form by hand at h = 3, where h·h, h + h
        // and h / h are three different numbers. A tiny fixture, so the triple sum is checkable.
        let leaves = coarse_leaves(&[0.0, 0.0, 1.0, 0.0, 4.0, 2.0, 4.0, 3.0], 2);
        let flat = [0.0, 0.0, 4.0, 2.5, 1.0, 1.0];
        let (h, dim) = (3.0_f64, 2usize);
        let h2 = h * h;
        let kernel = |m2: f64, s: f64| {
            (1.0 + s / h2).powf(-(dim as f64) / 2.0) * (-m2 / (2.0 * (h2 + s))).exp()
        };
        let w: Vec<f64> = leaves.iter().map(|f| f.weight()).collect();
        let total: f64 = w.iter().sum();
        let mu: Vec<Vec<f64>> = leaves.iter().map(|f| f.mean().to_vec()).collect();
        let s: Vec<f64> = leaves
            .iter()
            .map(|f| f.ssd() / (f.weight() * dim as f64))
            .collect();
        let rows: Vec<&[f64]> = flat.chunks_exact(dim).collect();
        let sq =
            |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
        let mut e_pp = 0.0;
        for i in 0..leaves.len() {
            for j in 0..leaves.len() {
                e_pp += (w[i] / total) * (w[j] / total) * kernel(sq(&mu[i], &mu[j]), s[i] + s[j]);
            }
        }
        let m_inv = 1.0 / rows.len() as f64;
        let mut e_pq = 0.0;
        for i in 0..leaves.len() {
            for r in &rows {
                e_pq += (w[i] / total) * m_inv * kernel(sq(&mu[i], r), s[i]);
            }
        }
        let mut e_qq = 0.0;
        for a in &rows {
            for b in &rows {
                e_qq += m_inv * m_inv * kernel(sq(a, b), 0.0);
            }
        }
        let want = (e_pp - 2.0 * e_pq + e_qq).max(0.0).sqrt();
        let got = summary_mmd(&leaves, &flat, Some(h));
        assert!(
            (got - want).abs() < 1e-12,
            "got {got}, the closed form says {want}"
        );
    }

    #[test]
    fn a_leaf_carrying_no_mass_has_no_scale_rather_than_a_nan() {
        // `S/(n·d)` on an empty leaf is `0/0`. It contributes zero weight to every term, so the
        // answer must be exactly the one the same summary gives without it.
        let flat = blob_points(7, 10, &[[0.0, 0.0], [5.0, 5.0]], 0.6);
        let leaves = coarse_leaves(&flat, 4);
        let mut padded = leaves.clone();
        padded.push(Spherical::new(2));
        let got = summary_mmd(&padded, &flat, Some(1.5));
        assert!(got.is_finite(), "an empty leaf produced {got}");
        assert!(
            (got - summary_mmd(&leaves, &flat, Some(1.5))).abs() < 1e-12,
            "an empty leaf changed the score"
        );
    }

    #[test]
    fn a_summary_that_kept_every_point_is_at_zero_distance_from_it() {
        let flat = blob_points(7, 40, &[[0.0, 0.0], [6.0, 5.0], [-4.0, 3.0]], 0.6);
        let mmd = summary_mmd(&exact_leaves(&flat), &flat, None);
        assert!(
            mmd < 1e-12,
            "P and Q are the same measure, so the MMD must vanish, got {mmd}"
        );
    }

    #[test]
    fn the_closed_form_matches_a_monte_carlo_average_of_the_kernel_it_integrates() {
        // `expected_kernel` is the whole claim; the double sums are bookkeeping over it. Draw pairs
        // from two isotropic Gaussians, average the plain kernel, and require the closed form to
        // land within four standard errors. Cells are chosen where the kernel is not so far into
        // its tail that the Monte Carlo estimator is itself dominated by rare draws.
        let mut rng = SplitMix64::new(23);
        let draws = 200_000;
        for &(dim, sa, sb, h) in &[
            (1usize, 0.5f64, 0.0f64, 0.7f64),
            (3, 1.0, 2.5, 0.7),
            (8, 0.05, 0.05, 2.0),
            (8, 1.0, 2.5, 2.0),
            (30, 0.5, 0.0, 2.0),
        ] {
            let a: Vec<f64> = (0..dim).map(|i| 0.3 * i as f64).collect();
            let b: Vec<f64> = (0..dim).map(|i| 0.9 - 0.2 * i as f64).collect();
            let closed = expected_kernel(sq_euclidean(&a, &b), sa + sb, h * h, dim);
            let (mut sum, mut sum_sq) = (0.0, 0.0);
            for _ in 0..draws {
                let mut d2 = 0.0;
                for t in 0..dim {
                    let x = a[t] + sa.sqrt() * rng.gauss();
                    let y = b[t] + sb.sqrt() * rng.gauss();
                    d2 += (x - y) * (x - y);
                }
                let k = (-d2 / (2.0 * h * h)).exp();
                sum += k;
                sum_sq += k * k;
            }
            let n = draws as f64;
            let mean = sum / n;
            let se = ((sum_sq / n - mean * mean) / n).sqrt();
            assert!(
                (closed - mean).abs() < 4.0 * se.max(1e-15),
                "dim {dim}, s = ({sa}, {sb}), h = {h}: closed {closed}, monte carlo {mean} +- {se}"
            );
        }
    }

    #[test]
    fn coarsening_costs_fidelity_but_not_monotonically_in_the_leaf_count() {
        // Two errors compete, and the shipped documentation says so because this test measures it.
        // Coarsening loses the ability to place mass where the data is, which grows the distance;
        // but it also lets each leaf's isotropic surrogate be a *better* model of a genuinely
        // Gaussian blob than a two-point leaf, whose ball is fitted to a single gap. The second
        // effect wins past a point, so the curve rises and then falls back.
        //
        // Asserting the non-monotonicity rather than skipping it is the point: a future change that
        // makes this monotone has changed what the number means, and should have to say so.
        let flat = blob_points(5, 240, &[[0.0, 0.0]], 1.0);
        let curve: Vec<f64> = [1usize, 2, 4, 8, 16, 80]
            .iter()
            .map(|&c| summary_mmd(&coarse_leaves(&flat, c), &flat, Some(2.5)))
            .collect();
        assert_eq!(curve[0], 0.0, "the exact summary is the sample itself");
        assert!(
            curve[1..].iter().all(|&v| v > 0.0),
            "every coarsening must cost something: {curve:?}"
        );
        assert!(
            curve.windows(2).any(|w| w[1] < w[0]),
            "the curve is monotone here, which contradicts the documented shape: {curve:?}"
        );
    }

    #[test]
    fn the_diagnostic_moves_with_the_data_and_not_with_the_frame() {
        // A distance between measures cannot depend on where the origin is, provided the bandwidth
        // travels with the data — which is what the median heuristic does.
        let flat = blob_points(9, 75, &[[0.0, 0.0], [4.0, 4.0]], 0.8);
        let shifted: Vec<f64> = flat
            .chunks_exact(2)
            .flat_map(|p| [p[0] - 137.5, p[1] + 62.25])
            .collect();
        let here = summary_mmd(&coarse_leaves(&flat, 8), &flat, None);
        let there = summary_mmd(&coarse_leaves(&shifted, 8), &shifted, None);
        assert!(
            (here - there).abs() < 1e-9,
            "translation moved the fidelity number: {here} against {there}"
        );
    }

    #[test]
    fn a_summary_of_the_wrong_data_scores_worse_than_a_summary_of_the_right_data() {
        let flat = blob_points(13, 100, &[[0.0, 0.0], [6.0, 0.0]], 0.5);
        let elsewhere = blob_points(13, 100, &[[0.0, 0.0], [6.0, 9.0]], 0.5);
        let right = summary_mmd(&coarse_leaves(&flat, 10), &flat, Some(2.0));
        let wrong = summary_mmd(&coarse_leaves(&elsewhere, 10), &flat, Some(2.0));
        assert!(
            wrong > 10.0 * right,
            "a summary of a different distribution scored {wrong} against {right}"
        );
    }

    #[test]
    fn the_full_feature_is_read_through_the_same_isotropic_surrogate() {
        // Documented behaviour, not an accident: only the trace of the scatter reaches the formula,
        // so a Full leaf and a Spherical leaf over the same points score identically.
        let flat = blob_points(17, 48, &[[0.0, 0.0], [3.0, 3.0]], 1.1);
        let full: Vec<Full<f64>> = flat
            .chunks_exact(2)
            .collect::<Vec<_>>()
            .chunks(8)
            .map(|g| {
                let mut f = Full::new(2);
                for p in g {
                    f.push(p, 1.0);
                }
                f
            })
            .collect();
        let a = summary_mmd(&full, &flat, Some(1.5));
        let b = summary_mmd(&coarse_leaves(&flat, 8), &flat, Some(1.5));
        assert!((a - b).abs() < 1e-12, "{a} against {b}");
    }

    #[test]
    fn the_degenerate_inputs_answer_rather_than_panic() {
        let flat = blob_points(2, 20, &[[0.0, 0.0]], 1.0);
        let leaves = coarse_leaves(&flat, 4);
        assert_eq!(summary_mmd::<f64, Spherical<f64>>(&[], &flat, None), 0.0);
        assert_eq!(summary_mmd(&leaves, &[], None), 0.0);
        // One row shorter than `dim` is no row at all.
        assert_eq!(summary_mmd(&leaves, &flat[..1], None), 0.0);
        // A zero-dimensional feature has no points to compare.
        assert_eq!(summary_mmd(&[Spherical::<f64>::new(0)], &flat, None), 0.0);
        // A single sample row leaves the median heuristic no pair to take a median of.
        assert!(summary_mmd(&leaves, &flat[..2], None).is_finite());
        // An empty leaf contributes no mass and must not divide by it.
        assert_eq!(summary_mmd(&[Spherical::<f64>::new(2)], &flat, None), 0.0);
    }
}

#[cfg(test)]
mod measure {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::distance::CentroidEuclidean;
    use crate::feature::Spherical;
    use crate::tree::CFTree;

    type EuclidTree = CFTree<f64, Spherical<f64>, CentroidEuclidean, CentroidEuclidean>;

    /// Does the MMD say anything the quantization error does not, on the crate's own tree?
    ///
    /// `cargo test --lib fidelity::measure -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn summary_mmd_against_the_leaf_budget() {
        for (name, dim, per, sep, spread) in [
            ("d=2 separated", 2usize, 800usize, 12.0, 1.0),
            ("d=16 separated", 16, 800, 12.0, 1.0),
            ("d=16 overlapping", 16, 800, 4.0, 3.0),
        ] {
            println!("\n{name}  dim = {dim}, 6 components, spread {spread}, seeds 31/32/33");
            println!(
                "{:>10} {:>8} {:>14} {:>12} {:>8}",
                "budget", "leaves", "mean_sq_radius", "mmd", "ari"
            );
            for budget in [8usize, 16, 32, 64, 128, 256, 512, 1024] {
                let mut cells = Vec::new();
                for seed in [31u64, 32, 33] {
                    let flat = gaussian_mixture(seed, dim, per, 6, sep, spread);
                    let n = flat.len() / dim;
                    let mut t = EuclidTree::new(
                        dim,
                        16,
                        16,
                        0.0,
                        budget,
                        CentroidEuclidean,
                        CentroidEuclidean,
                    );
                    for row in flat.chunks_exact(dim) {
                        t.insert(row);
                    }
                    let lv = t.leaf_features();
                    let mass: f64 = lv.iter().map(|f| f.weight()).sum();
                    let msr = lv.iter().map(|f| f.ssd()).sum::<f64>() / mass.max(1.0);
                    let mmd = summary_mmd(lv, &flat, None);
                    // The question the flat high-dimensional column raises: does quality stay flat
                    // too? If it does, the MMD agrees with the head where `mean_sq_radius` does not.
                    let km = crate::clustering::kmeans(lv, 6, 100, 4, 7);
                    let assign: Vec<usize> = flat
                        .chunks_exact(dim)
                        .map(|row| km.labels[t.nearest_entry(row)])
                        .collect();
                    let truth: Vec<usize> = (0..n).map(|i| i / per).collect();
                    let ari = crate::clustering::testutil::ari(&assign, &truth);
                    cells.push((lv.len() as f64, msr, mmd, ari));
                }
                let med = |f: fn(&(f64, f64, f64, f64)) -> f64| {
                    let mut v: Vec<f64> = cells.iter().map(f).collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    v[1]
                };
                println!(
                    "{budget:>10} {:>8.0} {:>14.4} {:>12.6} {:>8.4}",
                    med(|c| c.0),
                    med(|c| c.1),
                    med(|c| c.2),
                    med(|c| c.3)
                );
            }
        }
    }

    /// Is the flat `d = 16` column above a property of MMD or of the median bandwidth? Sweep `h`.
    ///
    /// `cargo test --lib fidelity::measure::the_bandwidth -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn the_bandwidth_sweep_behind_the_flat_high_dimensional_column() {
        let dim = 16;
        let flat = gaussian_mixture(31, dim, 800, 6, 12.0, 1.0);
        let rows: Vec<&[f64]> = flat.chunks_exact(dim).collect();
        let h_med = median_bandwidth_sq(&rows).sqrt();
        println!("\nd = {dim}, median-heuristic h = {h_med:.4}");
        print!("{:>10}", "h/h_med");
        for budget in [8usize, 64, 512] {
            print!("{:>14}", format!("{budget} leaves"));
        }
        println!();
        for f in [0.05f64, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0] {
            print!("{f:>10.2}");
            for budget in [8usize, 64, 512] {
                let mut t = EuclidTree::new(
                    dim,
                    16,
                    16,
                    0.0,
                    budget,
                    CentroidEuclidean,
                    CentroidEuclidean,
                );
                for row in flat.chunks_exact(dim) {
                    t.insert(row);
                }
                print!(
                    "{:>14.6}",
                    summary_mmd(t.leaf_features(), &flat, Some(f * h_med))
                );
            }
            println!();
        }
    }

    fn gaussian_mixture(
        seed: u64,
        dim: usize,
        per: usize,
        k: usize,
        sep: f64,
        spread: f64,
    ) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        let centres: Vec<Vec<f64>> = (0..k)
            .map(|_| (0..dim).map(|_| sep * rng.gauss()).collect())
            .collect();
        let mut flat = Vec::with_capacity(k * per * dim);
        for c in &centres {
            for _ in 0..per {
                for &mu in c {
                    flat.push(mu + spread * rng.gauss());
                }
            }
        }
        flat
    }
}
