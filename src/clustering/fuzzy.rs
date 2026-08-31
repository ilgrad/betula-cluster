//! Weighted fuzzy c-means over CF microclusters (Dunn 1973; Bezdek 1981).
//!
//! Every other soft head in the crate is a *generative mixture*: it fits a density and its
//! responsibilities are a posterior. Fuzzy c-means fits no density. It minimises
//!
//! ```text
//! J_m(U, C) = Σ_x Σ_j u_xj^m ‖x − c_j‖²      subject to   Σ_j u_xj = 1,  u ≥ 0,
//! ```
//!
//! and the memberships that fall out are a *partition of unity*, not a probability. The distinction
//! is not pedantic: `m` is a free parameter with no likelihood to fix it, `u` has no calibration, and
//! at `m → 1⁺` the head degenerates to hard k-means while at `m → ∞` every membership goes to `1/k`.
//! What it buys is a soft assignment with no distributional assumption at all — the one place in the
//! crate where "this point is half in each cluster" is not also a claim that the clusters are Gaussian.
//!
//! **Exact on the summary**, under the whole-leaf restriction the mixture heads already accept. Tie
//! the membership within a leaf and the leaf's contribution is closed-form in the cluster feature:
//!
//! ```text
//! Σ_{x ∈ leaf i} u_ij^m ‖x − c_j‖² = u_ij^m (S_i + n_i‖μ_i − c_j‖²) = u_ij^m · n_i · d_ij
//! ```
//!
//! with `d_ij = ‖μ_i − c_j‖² + S_i/n_i` the leaf's mean squared distance to `c_j`. The constrained
//! minimiser of that over `u_i·` is `u_ij ∝ d_ij^{−1/(m−1)}` — the `n_i` cancels in the normalisation,
//! so the classical membership rule survives the weighting unchanged — and the centre update becomes
//! `c_j = Σ_i u_ij^m n_i μ_i / Σ_i u_ij^m n_i`. Both are the point-level updates evaluated exactly;
//! the tying is the only approximation, and it is the same one `crate::clustering::gmm`'s E-step makes.
//!
//! `k` is chosen by the Xie–Beni index when asked, which is this head's own family's validity
//! measure rather than a criterion borrowed from a likelihood the head does not have.

use crate::clustering::kmeans::kmeans_plus_plus;
use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// k-means++ restarts, matching the hard head's default so the two differ in objective and not in
/// how hard they look for a start.
const N_INIT: usize = 4;

/// Result of a [`fuzzy_cmeans`] run.
pub struct FuzzyCMeans<R: Real> {
    /// Hard label per input feature — the membership's argmax, which is also the nearest centre.
    pub labels: Vec<usize>,
    /// `[leaf][cluster]` memberships, each row summing to one. Not a posterior; see the module docs.
    pub memberships: Vec<Vec<R>>,
    /// The fuzzy-weighted centre of each realised cluster, in label order.
    pub centers: Vec<Vec<R>>,
    /// `J_m` at convergence, over the **points** the leaves stand for.
    pub loss: f64,
    /// The fuzzifier the run used, after clamping.
    pub fuzzifier: f64,
}

/// Everything the iteration reads, in the working precision of the objective.
struct Leaves {
    mu: Vec<Vec<f64>>,
    w: Vec<f64>,
    /// `S_i / n_i`, the leaf's own mean squared spread about its mean.
    spread: Vec<f64>,
}

impl Leaves {
    fn of<R: Real, C: ClusterFeature<R>>(features: &[C]) -> Self {
        let to_f = |r: R| r.to_f64().unwrap_or(0.0);
        let mu = features
            .iter()
            .map(|f| f.mean().iter().map(|&v| to_f(v)).collect())
            .collect();
        let w: Vec<f64> = features.iter().map(|f| to_f(f.weight())).collect();
        // A massless leaf has no scatter to report and `S/n` on it is `0/0`; a NaN loose here reaches
        // the membership normalisation and silently wins every comparison it takes part in.
        let spread = features
            .iter()
            .zip(&w)
            .map(|(f, &n)| if n > 0.0 { to_f(f.ssd()) / n } else { 0.0 })
            .collect();
        Self { mu, w, spread }
    }

    fn len(&self) -> usize {
        self.mu.len()
    }

    /// `d_ij` — leaf `i`'s mean squared distance to the point `c`, scatter included.
    fn dist(&self, i: usize, c: &[f64]) -> f64 {
        sq_euclidean(&self.mu[i], c) + self.spread[i]
    }
}

/// The membership row of one leaf, written into `u`, and its `Σ_j u^m d` contribution returned.
///
/// The exponent is applied to the *ratio* `d_min / d_j ∈ (0, 1]` rather than to `1/d_j`: at
/// `m = 1.1` the exponent is 10, and a leaf sitting a rounding error from a centre raises `1/d` to
/// the tenth power, which overflows to infinity and turns the normalisation into `inf/inf`.
fn membership_row(lv: &Leaves, i: usize, centers: &[Vec<f64>], m: f64, u: &mut [f64]) -> f64 {
    let mut dmin = f64::INFINITY;
    for (j, c) in centers.iter().enumerate() {
        u[j] = lv.dist(i, c);
        dmin = dmin.min(u[j]);
    }
    if dmin <= 0.0 {
        // A leaf coincident with one or more centres: the constrained minimum is to split its
        // membership equally over exactly those and give the rest nothing, which is the standard
        // singleton rule and the only finite answer here.
        let hits = u.iter().filter(|&&d| d <= 0.0).count() as f64;
        for v in u.iter_mut() {
            *v = if *v <= 0.0 { 1.0 / hits } else { 0.0 };
        }
        return 0.0;
    }
    let mut sum = 0.0;
    for v in u.iter_mut() {
        *v = (dmin / *v).powf(1.0 / (m - 1.0));
        sum += *v;
    }
    let mut contrib = 0.0;
    for (j, v) in u.iter_mut().enumerate() {
        *v /= sum;
        contrib += v.powf(m) * lv.dist(i, &centers[j]);
    }
    contrib
}

/// What one alternating-minimisation run produced: memberships, centres and `J_m`.
type Run = (Vec<Vec<f64>>, Vec<Vec<f64>>, f64);

/// One fuzzy c-means run from the given centres.
fn fcm_from(lv: &Leaves, mut centers: Vec<Vec<f64>>, m: f64, max_iter: usize) -> Run {
    let (n, k, dim) = (lv.len(), centers.len(), lv.mu[0].len());
    let mut u = vec![vec![0.0; k]; n];
    let mut loss = f64::INFINITY;
    for _ in 0..max_iter.max(1) {
        let mut next = 0.0;
        for (i, row) in u.iter_mut().enumerate() {
            next += lv.w[i] * membership_row(lv, i, &centers, m, row);
        }
        let mut num = vec![vec![0.0; dim]; k];
        let mut den = vec![0.0; k];
        for (i, row) in u.iter().enumerate() {
            for (j, &uij) in row.iter().enumerate() {
                let g = uij.powf(m) * lv.w[i];
                den[j] += g;
                for (s, &mu) in num[j].iter_mut().zip(&lv.mu[i]) {
                    *s += g * mu;
                }
            }
        }
        for (j, c) in centers.iter_mut().enumerate() {
            // A centre no leaf gives any weight has no update defined; leaving it where it is keeps
            // it a live candidate for a later sweep rather than moving it to the origin.
            if den[j] > 0.0 {
                for (v, &s) in c.iter_mut().zip(&num[j]) {
                    *v = s / den[j];
                }
            }
        }
        let done = (loss - next).abs() <= 1e-12 * loss.abs().max(1.0);
        loss = next;
        if done {
            break;
        }
    }
    // One last E-step so the returned memberships are the ones the returned centres imply, rather
    // than the ones that produced them.
    let mut final_loss = 0.0;
    for (i, row) in u.iter_mut().enumerate() {
        final_loss += lv.w[i] * membership_row(lv, i, &centers, m, row);
    }
    (u, centers, final_loss)
}

/// Drop clusters no leaf's argmax claims, renumbering to stay contiguous.
///
/// An empty cluster is a centre the fit gave no point, and keeping it would let `predict` return a
/// label `fit_predict` never produced — the same rule the centroid heads apply.
fn compact(u: &[Vec<f64>], centers: Vec<Vec<f64>>) -> (Vec<usize>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let argmax = |row: &Vec<f64>| {
        row.iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (j, &v)| {
                if v > bv { (j, v) } else { (bi, bv) }
            })
            .0
    };
    let hard: Vec<usize> = u.iter().map(argmax).collect();
    let mut keep = vec![usize::MAX; centers.len()];
    let mut order = Vec::new();
    for &h in &hard {
        if keep[h] == usize::MAX {
            keep[h] = order.len();
            order.push(h);
        }
    }
    let labels = hard.iter().map(|&h| keep[h]).collect();
    let kept_centers = order.iter().map(|&j| centers[j].clone()).collect();
    // The memberships are renumbered, not renormalised: dropping a column would change what the
    // remaining ones mean, and every dropped column is one no leaf preferred, not one that is zero.
    let memberships = u
        .iter()
        .map(|row| order.iter().map(|&j| row[j]).collect())
        .collect();
    (labels, kept_centers, memberships)
}

fn to_r<R: Real>(v: &[Vec<f64>]) -> Vec<Vec<R>> {
    v.iter()
        .map(|row| row.iter().map(|&x| R::from(x).unwrap()).collect())
        .collect()
}

/// Weighted fuzzy c-means over the leaf centroids, each leaf weighted by its mass.
///
/// `fuzzifier` is `m`, clamped to `> 1`; `2.0` is Bezdek's default and the one the Python layer
/// passes. `m → 1⁺` degenerates to hard k-means and `m → ∞` sends every membership to `1/k`, so it
/// is the knob that decides how soft "soft" is — and it is a modelling choice, not something the
/// objective can select, because `J_m` is not comparable across `m`.
///
/// `k` is clamped to the leaf count; `max_iter` bounds the alternating updates and `0` means the
/// default. Cost is `O(n_init · iter · m · k · d)` on the `m ≪ N` microclusters.
pub fn fuzzy_cmeans<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    fuzzifier: f64,
    max_iter: usize,
    seed: u64,
) -> FuzzyCMeans<R> {
    let lv = Leaves::of(features);
    let m = if fuzzifier.is_finite() {
        fuzzifier.max(1.0 + 1e-6)
    } else {
        2.0
    };
    if lv.len() == 0 {
        return FuzzyCMeans {
            labels: Vec::new(),
            memberships: Vec::new(),
            centers: Vec::new(),
            loss: 0.0,
            fuzzifier: m,
        };
    }
    let k = k.clamp(1, lv.len());
    let max_iter = if max_iter == 0 { 100 } else { max_iter };
    let mut rng = SplitMix64::new(seed);
    let ssd: Vec<f64> = lv.w.iter().zip(&lv.spread).map(|(&n, &s)| n * s).collect();

    let mut best: Option<Run> = None;
    for _ in 0..N_INIT {
        let start = kmeans_plus_plus(&lv.mu, &lv.w, &ssd, k, &mut rng);
        let got = fcm_from(&lv, start, m, max_iter);
        if best.as_ref().is_none_or(|b| got.2 < b.2) {
            best = Some(got);
        }
    }
    let (u, centers, loss) = best.expect("N_INIT >= 1");
    let (labels, centers, memberships) = compact(&u, centers);
    FuzzyCMeans {
        labels,
        memberships: to_r(&memberships),
        centers: to_r(&centers),
        loss,
        fuzzifier: m,
    }
}

/// Xie–Beni index of a fit: `J_m / (W · min_{i≠j} ‖c_i − c_j‖²)`. Lower is better.
///
/// `J_m` falls monotonically as `k` grows, so it cannot choose `k` on its own; the separation
/// denominator is what makes the ratio turn. `f64::INFINITY` when there is no pair of centres to
/// separate, which is why the sweep starts at `k = 2`.
fn xie_beni(loss: f64, total: f64, centers: &[Vec<f64>]) -> f64 {
    let mut sep = f64::INFINITY;
    for (i, a) in centers.iter().enumerate() {
        for b in &centers[i + 1..] {
            sep = sep.min(sq_euclidean(a, b));
        }
    }
    // `sep` is still infinite when there was no pair to compare, and dividing by that would score a
    // collapsed fit at zero — the best possible index — and let it win the sweep outright.
    if sep.is_finite() && sep > 0.0 && total > 0.0 {
        loss / (total * sep)
    } else {
        f64::INFINITY
    }
}

/// [`fuzzy_cmeans`] with `k` chosen by the lowest Xie–Beni index over `2..=k_max`.
///
/// The index is this head's own family's validity measure (Xie & Beni 1991): the fuzzy objective per
/// unit of mass, divided by the tightest gap between two centres. A BIC would be the wrong tool —
/// there is no likelihood here to penalise — and the partition coefficient, the other classical
/// choice, is monotone in `k` on most data and would answer `2` almost always.
///
/// Falls back to `k = 1` only when the leaf set cannot support two clusters.
pub fn fuzzy_cmeans_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_max: usize,
    fuzzifier: f64,
    max_iter: usize,
    seed: u64,
) -> FuzzyCMeans<R> {
    let hi = k_max.min(features.len());
    if hi < 2 {
        return fuzzy_cmeans(features, 1, fuzzifier, max_iter, seed);
    }
    let total: f64 = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap_or(0.0))
        .sum();
    let mut best: Option<(f64, FuzzyCMeans<R>)> = None;
    for k in 2..=hi {
        let fit = fuzzy_cmeans::<R, C>(features, k, fuzzifier, max_iter, seed);
        let centers: Vec<Vec<f64>> = fit
            .centers
            .iter()
            .map(|c| c.iter().map(|&v| v.to_f64().unwrap_or(0.0)).collect())
            .collect();
        let score = xie_beni(fit.loss, total, &centers);
        if best.as_ref().is_none_or(|(s, _)| score < *s) {
            best = Some((score, fit));
        }
    }
    match best {
        Some((s, fit)) if s.is_finite() => fit,
        _ => fuzzy_cmeans(features, 1, fuzzifier, max_iter, seed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::testutil::{ari, blob_leaves, blobs, grid_micros};
    use crate::feature::Spherical;

    fn fixture(seed: u64, spread: f64) -> (Vec<Spherical<f64>>, Vec<usize>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let centres = [[0.0, 0.0], [12.0, 0.0], [6.0, 11.0], [18.0, 11.0]];
        let (pts, truth) = blobs(&mut rng, 150, &centres, spread);
        let (micros, assign) = grid_micros(&pts, 0.6);
        (micros, assign, truth)
    }

    #[test]
    fn every_membership_row_is_a_partition_of_unity() {
        let (micros, _, _) = fixture(1, 0.8);
        let fit = fuzzy_cmeans::<f64, _>(&micros, 4, 2.0, 100, 1);
        for row in &fit.memberships {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row sums to {sum}");
            assert!(row.iter().all(|&v| (0.0..=1.0).contains(&v)), "{row:?}");
        }
    }

    #[test]
    fn the_reported_loss_is_the_objective_over_the_points_not_over_the_leaves() {
        // `Σ_{x ∈ leaf i} u_ij^m ‖x − c_j‖² = u_ij^m (S_i + n_i‖μ_i − c_j‖²)`. Recomputed here from
        // the raw points, which the head never sees.
        let mut rng = SplitMix64::new(3);
        let centres = [[0.0, 0.0], [12.0, 0.0], [6.0, 11.0], [18.0, 11.0]];
        let (pts, _) = blobs(&mut rng, 150, &centres, 0.8);
        let (micros, assign) = grid_micros(&pts, 0.6);
        let fit = fuzzy_cmeans::<f64, _>(&micros, 4, 2.0, 100, 3);
        let direct: f64 = pts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let u = &fit.memberships[assign[i]];
                u.iter()
                    .zip(&fit.centers)
                    .map(|(&uij, c)| uij.powf(2.0) * sq_euclidean(p, c))
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (fit.loss - direct).abs() <= 1e-9 * direct,
            "loss {} against the point-level objective {direct}",
            fit.loss
        );
    }

    #[test]
    fn the_hard_label_is_the_membership_argmax_and_also_the_nearest_centre() {
        // `u_ij ∝ d_ij^{−1/(m−1)}` is strictly decreasing in `d_ij`, so the two orderings are the
        // same one — which is what lets the head publish a plain centroid rule for `predict`.
        let (micros, _, _) = fixture(5, 0.8);
        let fit = fuzzy_cmeans::<f64, _>(&micros, 5, 2.0, 100, 5);
        let lv = Leaves::of(&micros);
        let centers: Vec<Vec<f64>> = fit.centers.clone();
        for (i, (row, &l)) in fit.memberships.iter().zip(&fit.labels).enumerate() {
            let by_u = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            let by_d = centers
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    lv.dist(i, a.1)
                        .partial_cmp(&lv.dist(i, b.1))
                        .expect("finite distances")
                })
                .unwrap()
                .0;
            assert_eq!(l, by_u, "leaf {i}");
            assert_eq!(l, by_d, "leaf {i}");
        }
    }

    #[test]
    fn a_fuzzifier_at_the_hard_limit_reproduces_k_means() {
        // `m → 1⁺` sends every membership to an indicator, so `J_m` becomes the k-means SSE and the
        // partition becomes k-means'. The clamp is what keeps `1/(m−1)` finite; this checks the
        // limit is approached rather than merely not crashing.
        let (micros, _, _) = fixture(2, 0.8);
        let hard = crate::clustering::kmeans::kmeans::<f64, _>(&micros, 4, 100, 4, 2);
        let fit = fuzzy_cmeans::<f64, _>(&micros, 4, 1.001, 200, 2);
        for row in &fit.memberships {
            let top = row.iter().cloned().fold(f64::MIN, f64::max);
            assert!(top > 0.999, "membership {top} is not an indicator at m → 1");
        }
        assert!(
            (fit.loss - hard.inertia).abs() <= 1e-3 * hard.inertia,
            "J_m {} against the k-means SSE {}",
            fit.loss,
            hard.inertia
        );
    }

    #[test]
    fn a_large_fuzzifier_sends_every_membership_to_one_over_k() {
        let (micros, _, _) = fixture(4, 0.8);
        let fit = fuzzy_cmeans::<f64, _>(&micros, 4, 40.0, 200, 4);
        for row in &fit.memberships {
            for &v in row {
                assert!((v - 0.25).abs() < 0.05, "{row:?}");
            }
        }
    }

    #[test]
    fn a_leaf_sitting_on_a_centre_takes_that_cluster_whole_rather_than_dividing_by_zero() {
        // Two coincident zero-scatter leaves at the two centres: `d = 0` makes `d^{−p}` infinite and
        // the normalisation `inf/inf`. The singleton rule is the only finite constrained minimum.
        let lv = Leaves {
            mu: vec![vec![0.0, 0.0], vec![10.0, 0.0]],
            w: vec![1.0, 1.0],
            spread: vec![0.0, 0.0],
        };
        let centers = vec![vec![0.0, 0.0], vec![10.0, 0.0]];
        let mut u = vec![0.0; 2];
        assert_eq!(membership_row(&lv, 0, &centers, 2.0, &mut u), 0.0);
        assert_eq!(u, [1.0, 0.0]);
        assert_eq!(membership_row(&lv, 1, &centers, 2.0, &mut u), 0.0);
        assert_eq!(u, [0.0, 1.0]);
    }

    #[test]
    fn a_leaf_equidistant_from_two_coincident_centres_splits_between_them() {
        let lv = Leaves {
            mu: vec![vec![0.0, 0.0]],
            w: vec![1.0],
            spread: vec![0.0],
        };
        let centers = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![5.0, 0.0]];
        let mut u = vec![0.0; 3];
        membership_row(&lv, 0, &centers, 2.0, &mut u);
        assert_eq!(u, [0.5, 0.5, 0.0]);
    }

    #[test]
    fn the_head_recovers_the_blobs_it_was_given() {
        for seed in [0u64, 1, 2] {
            let (micros, assign, truth) = fixture(seed, 0.8);
            let fit = fuzzy_cmeans::<f64, _>(&micros, 4, 2.0, 100, seed);
            let labels: Vec<usize> = assign.iter().map(|&i| fit.labels[i]).collect();
            assert!(
                ari(&labels, &truth) > 0.99,
                "seed {seed}: ARI = {}",
                ari(&labels, &truth)
            );
        }
    }

    #[test]
    fn xie_beni_chooses_the_number_of_blobs_without_being_told_it() {
        let (feats, truth) = blob_leaves(5, 6, 40, 3);
        let fit = fuzzy_cmeans_auto::<f64, _>(&feats, 12, 2.0, 100, 3);
        assert_eq!(fit.centers.len(), 5, "chose {}", fit.centers.len());
        assert!(ari(&fit.labels, &truth) > 0.99, "{:?}", fit.labels);
    }

    #[test]
    fn the_index_is_the_objective_over_the_tightest_gap_and_is_undefined_below_two_centres() {
        assert_eq!(xie_beni(1.0, 10.0, &[vec![0.0]]), f64::INFINITY);
        assert_eq!(
            xie_beni(1.0, 10.0, &[vec![0.0], vec![0.0]]),
            f64::INFINITY,
            "coincident centres have no separation to divide by"
        );
        // Three centres, tightest gap 4: the index reads the *minimum*, not the first pair.
        let got = xie_beni(20.0, 10.0, &[vec![0.0], vec![10.0], vec![12.0]]);
        assert!((got - 20.0 / (10.0 * 4.0)).abs() < 1e-12, "{got}");
    }

    #[test]
    fn the_degenerate_inputs_answer_rather_than_panic() {
        let empty: Vec<Spherical<f64>> = Vec::new();
        let fit = fuzzy_cmeans::<f64, _>(&empty, 4, 2.0, 100, 0);
        assert!(fit.labels.is_empty());
        assert!(fit.centers.is_empty());
        assert_eq!(fit.loss, 0.0);
        assert!(
            fuzzy_cmeans_auto::<f64, _>(&empty, 8, 2.0, 100, 0)
                .labels
                .is_empty()
        );

        let (micros, _, _) = fixture(6, 0.8);
        // More clusters asked for than leaves: `k` is clamped, not honoured.
        let two = &micros[..2];
        assert!(fuzzy_cmeans::<f64, _>(two, 50, 2.0, 100, 0).centers.len() <= 2);
        // A fuzzifier at or below the hard limit is clamped rather than dividing by zero, and a
        // non-finite one falls back to the documented default.
        for bad in [1.0, 0.5, -3.0, f64::NAN] {
            let fit = fuzzy_cmeans::<f64, _>(&micros, 3, bad, 50, 0);
            assert!(fit.fuzzifier > 1.0, "{bad} left m at {}", fit.fuzzifier);
            assert_eq!(fit.memberships.len(), micros.len());
        }
    }
}
