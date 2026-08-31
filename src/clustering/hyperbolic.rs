//! Hyperbolic clustering on the Lorentz (hyperboloid) model of `H^d`.
//!
//! Hierarchies embed into hyperbolic space with vanishing distortion where Euclidean space needs
//! exponentially many dimensions (Sarkar 2011; Nickel & Kiela, NeurIPS 2017), so taxonomies, trees
//! and scale-free graphs increasingly arrive as points of `H^d` rather than of `R^d`. Clustering
//! them with a Euclidean head throws away the geometry the embedding was built to carry.
//!
//! The model is the upper sheet
//!
//! ```text
//! H^d = { x ∈ R^(d+1) : ⟨x,x⟩_L = −1, x_0 > 0 },   ⟨x,y⟩_L = −x_0 y_0 + Σ_{i≥1} x_i y_i
//! ```
//!
//! whose geodesic distance is `d_H(x,y) = arccosh(−⟨x,y⟩_L)`.
//!
//! **The head does not use `d_H`, and that is the whole design.** `arccosh` has no closed-form
//! Fréchet mean, so a Riemannian k-means there needs an inner gradient loop per centre and — worse
//! for this crate — a quantity that is not a sum, which a cluster feature cannot carry. The
//! **squared Lorentzian distance** of Law, Liao, Snavely & Dhillon (ICML 2019),
//!
//! ```text
//! d_L²(x,y) = ‖x − y‖²_L = −2 − 2⟨x,y⟩_L
//! ```
//!
//! does have one. It is a strictly increasing function of `d_H` on the sheet, so it orders pairs
//! identically and its k-means partition is a hyperbolic partition; and its centroid is the
//! *normalised sum*
//!
//! ```text
//! μ = R / |R|_L,   R = Σ_i n_i x_i,   |R|_L = √(−⟨R,R⟩_L)
//! ```
//!
//! ## Why this is exact on a summary, and exact in a stronger sense than Euclidean k-means
//!
//! `d_L²` is **affine in each argument**. So the cost of assigning a whole leaf to a centre,
//!
//! ```text
//! Σ_{x ∈ leaf i} d_L²(x, c) = −2 n_i − 2 ⟨R_i, c⟩_L
//! ```
//!
//! depends on the leaf only through `(n_i, R_i)` — and `R_i = n_i · μ_i` is exactly what every
//! cluster feature already stores. Euclidean k-means needs the leaf's scatter `S_i` as well, by
//! König–Huygens; here there is **no scatter term at all**, because a linear function has no second
//! order. A `Spherical` leaf therefore loses nothing relative to a `Full` one, which is not true of
//! any other head in the crate.
//!
//! The consequence is that both steps of Lloyd are closed-form and exact on the summary:
//!
//! | step | on the summary | approximation |
//! |---|---|---|
//! | assign | `argmax_c ⟨R_i, c⟩_L` | only that a leaf's points share a label |
//! | update | `c = (Σ_{i∈c} R_i)/‖·‖_L` | none |
//! | cost | `Σ_c 2(\|R_c\|_L − W_c)` | none |
//!
//! and the merge increase, the hyperbolic analogue of Ward's, is
//! `ΔS = 2(|R_a + R_b|_L − |R_a|_L − |R_b|_L) ≥ 0`.
//!
//! All of it was verified in Maxima before any of this was written —
//! `local/scratch/lorentz_identities.mac`: the lift lands on the sheet, the Lagrange stationarity
//! `R − |R|_L μ = 0` holds componentwise, `S = 2(|R|_L − W)` has residual exactly zero, and
//! `S ≥ 0`, `ΔS ≥ 0` survive 400 random weighted sums.
//!
//! ## The working radius, which is finite and is not a detail
//!
//! `⟨x,y⟩_L` is a difference of two nearly equal positive numbers: at hyperbolic radius `r` both
//! `x_0 y_0` and `Σ x_i y_i` are `Θ(cosh² r)` while the result is `Θ(1)`. The absolute error is
//! therefore `≈ ε cosh² r` against an answer of order one, and the model has a **working radius**
//! `r_max ≈ ½ ln(2/ε) ≈ 18.4` for `f64` — where "working" already means "the sign is right".
//! Measured against 60-digit mpmath (`local/scratch/lorentz_precision.py`), on two points a unit
//! apart:
//!
//! | radius | 0 | 5 | 10 | 15 | 17 | 18 |
//! |---|---|---|---|---|---|---|
//! | relative error of `d_L²` | 1e−16 | 4e−12 | 1e−8 | 2e−3 | 8e−2 | **sign flips** |
//!
//! and `|R|_L` is 10 % low at `r = 18` and collapses to zero at `r = 20`. So: `r ≲ 10` is exact for
//! any purpose here, `r ≈ 15` is the last radius whose answer is still recognisable, and past 18
//! nothing is. That is a property of the representation, not of this code — a Poincaré-ball
//! implementation saturates *earlier* — and it is why [`hyperbolic_kmeans`] reports the largest
//! radius it saw rather than leaving the caller to discover it.
//!
//! ## What it buys, measured
//!
//! A 4-ary tree of depth 4 laid out in geodesic polar coordinates (`τ = 1.6` per level), 15 360
//! points, 16 true groups, `max_leaves = 2000`, median of seeds 0/1/2, ARI.
//!
//! On the **Lorentz** coordinates it reads **0.731** against `kmeans` 0.407 and `gmm-full` 0.663.
//! But the same points converted to the **Poincaré ball** and handed to `gmm-full` read **0.817** —
//! the ball keeps angle and compresses radius by `tanh(r/2)`, which is what a Euclidean head wants.
//! **On a centred embedding the chart change beats this head.** What separates them is a Lorentz
//! boost, an isometry of `H^d`: at rapidity 3 the ball route falls to 0.311 and this holds 0.596.
//! The deliverable is invariance, not ARI.
//!
//! The residual drift is the *tree's* — routing and absorption are Euclidean. Over a fixed leaf set
//! the partition is exactly boost-invariant, and at one leaf per point the measured ARI is 0.772 at
//! every rapidity tried. Tables in `bench/RESULTS.md`.
//!
//! ## Input convention
//!
//! Rows are `(d+1)`-dimensional **Lorentz** coordinates and coordinate 0 is the time-like one. It is
//! recomputed from the spatial part as `x_0 = √(1 + ‖s‖²)` before anything else happens, exactly as
//! the directional heads L2-normalise theirs: the projection is idempotent on a valid point, total
//! on an invalid one, and it makes a row that is off the sheet unrepresentable further in. A
//! Poincaré-ball embedding `p` converts with `x = (1 + ‖p‖², 2p) / (1 − ‖p‖²)`.

use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::types::Real;

/// k-means restarts kept for the best cost, matching the Euclidean head.
const HYPERBOLIC_N_INIT: usize = 4;

/// Result of a hyperbolic k-means run over cluster features.
pub struct HyperbolicKMeans<R: Real> {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
    /// Cluster centres, on the sheet.
    pub centers: Vec<Vec<R>>,
    /// Total squared Lorentzian cost `Σ_c 2(|R_c|_L − W_c)`. Exact for the underlying points, not
    /// only for the leaf centroids — `d_L²` is affine, so there is no within-leaf term to drop.
    pub cost: R,
    /// The largest hyperbolic radius `arccosh(x_0)` seen among the leaf centroids.
    ///
    /// Reported because the answer silently stops being meaningful past ≈ 18 in `f64` and the
    /// caller cannot see that from the labels. Compare against [`f64_working_radius`].
    pub max_radius: R,
}

/// The hyperbolic radius past which `f64` cannot represent the Lorentz form at all:
/// `½ ln(2/ε) ≈ 18.4`. Full relative accuracy needs `cosh² r ≪ 1/ε`, i.e. roughly `r ≤ 10`.
#[must_use]
pub fn f64_working_radius() -> f64 {
    0.5 * (2.0 / f64::EPSILON).ln()
}

/// Minkowski inner product `−x_0 y_0 + Σ_{i≥1} x_i y_i`.
#[inline]
pub(crate) fn lorentz_dot<R: Real>(a: &[R], b: &[R]) -> R {
    let spatial = a[1..]
        .iter()
        .zip(&b[1..])
        .map(|(&x, &y)| x * y)
        .fold(R::zero(), |p, q| p + q);
    spatial - a[0] * b[0]
}

/// `|v|_L = √(−⟨v,v⟩_L)` for a future-directed timelike `v`, floored at zero.
#[inline]
fn lorentz_norm<R: Real>(v: &[R]) -> R {
    (-lorentz_dot(v, v)).max(R::zero()).sqrt()
}

/// Project a raw row onto the sheet by recomputing its time-like coordinate.
///
/// Idempotent where the row is already a point of `H^d`, and total where it is not — which is the
/// property that keeps an off-sheet row from reaching the algebra, where `|R|_L` would go imaginary
/// and the cost would stop being a cost.
pub fn project_to_sheet<R: Real>(row: &[R]) -> Vec<R> {
    let mut out = row.to_vec();
    let sq = row[1..]
        .iter()
        .map(|&x| x * x)
        .fold(R::zero(), |p, q| p + q);
    out[0] = (R::one() + sq).sqrt();
    out
}

/// `R / |R|_L`, the Lorentzian centroid of the weighted sum `R`. Degenerate sums fall back to the
/// origin of the model, `(1, 0, …, 0)`, which is the only choice that is still on the sheet.
fn normalize_to_sheet<R: Real>(r: &[R]) -> Vec<R> {
    let n = lorentz_norm(r);
    if n <= R::from_f64(1e-12).unwrap() {
        let mut origin = vec![R::zero(); r.len()];
        origin[0] = R::one();
        return origin;
    }
    r.iter().map(|&v| v / n).collect()
}

/// Squared Lorentzian distance between two points of the sheet, floored at zero.
#[inline]
fn dl2<R: Real>(a: &[R], b: &[R]) -> R {
    let two = R::one() + R::one();
    (-two - two * lorentz_dot(a, b)).max(R::zero())
}

/// Cluster `features` into `k` groups under the squared Lorentzian distance.
///
/// `k == 0` is rejected by the caller; `max_iter` bounds the Lloyd sweeps and `seed` fixes the
/// k-means++ draw.
///
/// **Rows must already be on the sheet when they enter the tree.** [`project_to_sheet`] is applied
/// once at the boundary, per row, exactly as the directional heads L2-normalize theirs — it cannot
/// be applied here, because projection does not commute with averaging and the leaf carries only the
/// average.
pub fn hyperbolic_kmeans<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> HyperbolicKMeans<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(!features.is_empty(), "hyperbolic k-means needs a feature");
    let m = features.len();
    let dim = features[0].dim();
    assert!(dim >= 2, "the Lorentz model needs at least two coordinates");

    // `R_i = n_i · μ_i` is the leaf's exact Lorentz sum of its own points. Reprojecting the mean
    // onto the sheet here would look like a safety net and is a defect: the mean of on-sheet points
    // is off the sheet, and `project_to_sheet` keeps the spatial part while rewriting the time
    // coordinate, so it returns a *different* vector than the sum — which is the one quantity the
    // exactness claim rests on.
    let w: Vec<R> = features.iter().map(ClusterFeature::weight).collect();
    let sums: Vec<Vec<R>> = features
        .iter()
        .zip(&w)
        .map(|(f, &n)| f.mean().iter().map(|&v| v * n).collect())
        .collect();
    // The leaf's own Lorentzian centroid, which *is* on the sheet. Only the seeding metric and the
    // reported radius read it; the objective reads `sums`.
    let pts: Vec<Vec<R>> = sums.iter().map(|r| normalize_to_sheet(r)).collect();
    let max_radius = pts
        .iter()
        .map(|p| acosh_clamped(p[0]))
        .fold(R::zero(), R::max);

    let k = k.min(m);
    let mut best: Option<(R, Vec<usize>, Vec<Vec<R>>)> = None;
    for restart in 0..HYPERBOLIC_N_INIT {
        let mut centers = seed_centers(&pts, &w, k, seed ^ (restart as u64).wrapping_mul(0x9E37));
        let mut labels = vec![0usize; m];
        let mut cost = R::infinity();
        for _ in 0..max_iter.max(1) {
            let mut moved = false;
            for (i, s) in sums.iter().enumerate() {
                // `argmin_c −2n_i − 2⟨R_i, c⟩_L` is `argmax_c ⟨R_i, c⟩_L`, and since
                // `R_i = |R_i|_L x_i` with a positive norm that is the same argmax as on `x_i`.
                let mut bestc = 0;
                let mut bestv = R::neg_infinity();
                for (c, ctr) in centers.iter().enumerate() {
                    let v = lorentz_dot(s, ctr);
                    if v > bestv {
                        bestv = v;
                        bestc = c;
                    }
                }
                if labels[i] != bestc {
                    moved = true;
                }
                labels[i] = bestc;
            }
            let (new_centers, new_cost) = update(&sums, &w, &labels, k, dim, &centers);
            centers = new_centers;
            cost = new_cost;
            if !moved {
                break;
            }
        }
        if best.as_ref().is_none_or(|(b, _, _)| cost < *b) {
            best = Some((cost, labels, centers));
        }
    }
    let (cost, labels, centers) = best.expect("at least one restart");
    HyperbolicKMeans {
        labels,
        centers,
        cost,
        max_radius,
    }
}

/// The merge increase of the hyperbolic cost, `ΔS = 2(|R_a + R_b|_L − |R_a|_L − |R_b|_L) ≥ 0`.
///
/// The hyperbolic analogue of Ward's variance increase, and mergeable on the same terms: it reads
/// only the two weighted Lorentz sums. Non-negativity is the reverse triangle inequality for
/// future-directed timelike vectors, checked numerically alongside the identities.
#[must_use]
pub fn merge_increase<R: Real>(ra: &[R], rb: &[R]) -> R {
    let two = R::one() + R::one();
    let joint: Vec<R> = ra.iter().zip(rb).map(|(&x, &y)| x + y).collect();
    (two * (lorentz_norm(&joint) - lorentz_norm(ra) - lorentz_norm(rb))).max(R::zero())
}

/// `arccosh` of a time-like coordinate, clamped at the sheet's floor of 1.
fn acosh_clamped<R: Real>(x0: R) -> R {
    let one = R::one();
    let x = x0.max(one);
    (x + (x * x - one).max(R::zero()).sqrt()).ln()
}

/// k-means++ in the `d_L²` metric, weighted by leaf mass.
fn seed_centers<R: Real>(pts: &[Vec<R>], w: &[R], k: usize, seed: u64) -> Vec<Vec<R>> {
    let m = pts.len();
    let mut rng = SplitMix64::new(seed);
    let mut centers = Vec::with_capacity(k);
    let first = (rng.next_u64() % m as u64) as usize;
    centers.push(pts[first].clone());
    let mut d2: Vec<R> = pts.iter().map(|p| dl2(p, &centers[0])).collect();
    while centers.len() < k {
        let total = d2
            .iter()
            .zip(w)
            .map(|(&d, &n)| d * n)
            .fold(R::zero(), |a, b| a + b);
        let pick = if total > R::zero() {
            let target = R::from_f64(rng.next_f64()).unwrap_or_else(R::zero) * total;
            let mut acc = R::zero();
            let mut chosen = m - 1;
            for (i, (&d, &n)) in d2.iter().zip(w).enumerate() {
                acc = acc + d * n;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            (rng.next_u64() % m as u64) as usize
        };
        centers.push(pts[pick].clone());
        let last = centers.last().expect("just pushed");
        for (d, p) in d2.iter_mut().zip(pts) {
            *d = (*d).min(dl2(p, last));
        }
    }
    centers
}

/// Closed-form update: each centre is the normalised Lorentz sum of its members, and the cost is
/// read off the same sums. An emptied cluster keeps its previous centre.
fn update<R: Real>(
    sums: &[Vec<R>],
    w: &[R],
    labels: &[usize],
    k: usize,
    dim: usize,
    prev: &[Vec<R>],
) -> (Vec<Vec<R>>, R) {
    let two = R::one() + R::one();
    let mut acc = vec![vec![R::zero(); dim]; k];
    let mut mass = vec![R::zero(); k];
    for ((s, &n), &c) in sums.iter().zip(w).zip(labels) {
        mass[c] = mass[c] + n;
        for (a, &v) in acc[c].iter_mut().zip(s) {
            *a = *a + v;
        }
    }
    let mut cost = R::zero();
    let mut centers = Vec::with_capacity(k);
    for c in 0..k {
        if mass[c] <= R::zero() {
            centers.push(prev[c].clone());
            continue;
        }
        cost = cost + two * (lorentz_norm(&acc[c]) - mass[c]);
        centers.push(normalize_to_sheet(&acc[c]));
    }
    (centers, cost.max(R::zero()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::kmeans::kmeans;
    use crate::clustering::testutil::ari;
    use crate::feature::{Full, Spherical};

    /// `k` arms of `H^2`, each a narrow angular sector spanning hyperbolic radii `[r_lo, r_hi]`.
    ///
    /// This is what a hyperbolic embedding of a tree looks like — branches radiating from the root,
    /// a branch spanning depths rather than sitting at one — and it is the shape on which the two
    /// geometries genuinely disagree. `k` blobs at a single radius do **not**: they separate under
    /// both metrics, so a test built on them would pass with the Lorentz form deleted.
    ///
    /// The disagreement is a ranking flip. Hyperbolically, two points on one arm are at most
    /// `r_hi − r_lo` apart while two on different arms are `≈ r_a + r_b + 2 ln sin(Δθ/2)`; in the
    /// ambient Euclidean coordinates the radial extent is `sinh r_hi − sinh r_lo`, which at
    /// `[4, 8]` is 538 against 17 for the angular gap at the inner end. So a Euclidean head cuts
    /// the arms into radial bands across all of them.
    fn hyperbolic_arms(
        seed: u64,
        k: usize,
        per: usize,
        r_lo: f64,
        r_hi: f64,
        ang: f64,
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for c in 0..k {
            let theta = std::f64::consts::TAU * (c as f64) / (k as f64);
            for _ in 0..per {
                let r = r_lo + (r_hi - r_lo) * rng.next_f64();
                let a = theta + ang * rng.gauss();
                let s = r.sinh();
                pts.push(vec![r.cosh(), s * a.cos(), s * a.sin()]);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    fn leaves(pts: &[Vec<f64>]) -> Vec<Full<f64>> {
        pts.iter()
            .map(|p| {
                let mut f = Full::new(p.len());
                f.push(p, 1.0);
                f
            })
            .collect()
    }

    /// The identity the head exists for: at a large radius the Euclidean geometry of the ambient
    /// coordinates has nothing to do with the hyperbolic geometry of the points.
    #[test]
    fn hyperbolic_kmeans_beats_euclidean_where_the_radius_is_large() {
        let (pts, truth) = hyperbolic_arms(4, 3, 150, 4.0, 8.0, 0.10);
        let lv = leaves(&pts);
        let hyp = hyperbolic_kmeans(&lv, 3, 100, 7);
        let euc = kmeans(&lv, 3, 100, 4, 7);
        let a_hyp = ari(&hyp.labels, &truth);
        let a_euc = ari(&euc.labels, &truth);
        assert!(
            a_hyp > 0.9,
            "hyperbolic head lost its own fixture: {a_hyp:.3}"
        );
        assert!(
            a_hyp > a_euc + 0.05,
            "no separation: hyperbolic {a_hyp:.3} vs euclidean {a_euc:.3}"
        );
    }

    /// Every centre must be a point of the model. An update that returned the raw sum would still
    /// produce plausible labels and a monotone cost, and would be wrong.
    #[test]
    fn every_centre_lands_on_the_sheet() {
        let (pts, _) = hyperbolic_arms(11, 3, 80, 3.6, 4.4, 0.10);
        let lv = leaves(&pts);
        let out = hyperbolic_kmeans(&lv, 3, 100, 3);
        for c in &out.centers {
            assert!(
                (lorentz_dot(c, c) + 1.0).abs() < 1e-9,
                "centre off the sheet: <c,c>_L = {}",
                lorentz_dot(c, c)
            );
            assert!(c[0] > 0.0, "centre on the lower sheet");
        }
    }

    /// The exactness claim, stated as a test rather than as prose: because `d_L²` is affine, the
    /// summary carries the objective with no residual, so a `Spherical` leaf — which keeps only the
    /// trace of the scatter — reaches the identical partition and the identical cost as a `Full`
    /// one. No other head in this crate can say that.
    #[test]
    fn a_spherical_leaf_loses_nothing_a_full_leaf_keeps() {
        let (pts, _) = hyperbolic_arms(21, 3, 200, 2.5, 3.5, 0.15);
        let cell = 0.25;
        let mut keys: Vec<(i64, i64)> = Vec::new();
        let mut full: Vec<Full<f64>> = Vec::new();
        let mut sph: Vec<Spherical<f64>> = Vec::new();
        for p in &pts {
            let key = ((p[1] / cell) as i64, (p[2] / cell) as i64);
            let idx = keys.iter().position(|&k| k == key).unwrap_or_else(|| {
                keys.push(key);
                full.push(Full::new(3));
                sph.push(Spherical::new(3));
                keys.len() - 1
            });
            full[idx].push(p, 1.0);
            sph[idx].push(p, 1.0);
        }
        let a = hyperbolic_kmeans(&full, 3, 100, 5);
        let b = hyperbolic_kmeans(&sph, 3, 100, 5);
        assert_eq!(a.labels, b.labels);
        assert!((a.cost - b.cost).abs() < 1e-9, "{} vs {}", a.cost, b.cost);
    }

    /// The Maxima identity, re-checked on the shipped code path: the closed-form centroid is the
    /// minimiser, so no perturbation of it lowers the cost.
    #[test]
    fn the_closed_form_centroid_is_the_minimiser() {
        let (pts, _) = hyperbolic_arms(31, 1, 60, 1.4, 2.6, 0.6);
        let mut sum = vec![0.0; 3];
        for p in &pts {
            for (s, &v) in sum.iter_mut().zip(p) {
                *s += v;
            }
        }
        let mu = normalize_to_sheet(&sum);
        let cost = |c: &[f64]| pts.iter().map(|p| dl2(p, c)).sum::<f64>();
        let base = cost(&mu);
        let claim = 2.0 * (lorentz_norm(&sum) - pts.len() as f64);
        assert!((base - claim).abs() < 1e-9, "{base} vs {claim}");
        let mut rng = SplitMix64::new(99);
        for _ in 0..200 {
            let perturbed =
                project_to_sheet(&[0.0, mu[1] + 0.05 * rng.gauss(), mu[2] + 0.05 * rng.gauss()]);
            assert!(
                cost(&perturbed) >= base - 1e-9,
                "a perturbation beat the closed form"
            );
        }
    }

    /// The boundary projection: idempotent where the row is already a point of the model, total
    /// where it is not, and it never trusts the time-like coordinate it was handed.
    #[test]
    fn the_boundary_projection_is_idempotent_on_the_sheet_and_total_off_it() {
        let (pts, _) = hyperbolic_arms(41, 3, 30, 2.6, 3.4, 0.15);
        for p in &pts {
            let once = project_to_sheet(p);
            for (a, b) in once.iter().zip(p) {
                assert!((a - b).abs() < 1e-9, "moved a point already on the sheet");
            }
            let corrupted = vec![p[0] * 3.0 + 7.0, p[1], p[2]];
            let fixed = project_to_sheet(&corrupted);
            assert!((lorentz_dot(&fixed, &fixed) + 1.0).abs() < 1e-9);
            assert!(
                (fixed[0] - p[0]).abs() < 1e-9,
                "the corrupted time survived"
            );
        }
    }

    /// The exactness claim, at the granularity where it can fail: a leaf of many points costs
    /// exactly what its points cost, `2(|Σ_j x_j|_L − n)`.
    ///
    /// The tempting alternative — project the leaf mean onto the sheet, then weight it — is wrong
    /// and this is the test that says so. The mean of on-sheet points sits *inside* the hyperboloid,
    /// and the projection keeps its spatial part while rewriting its time coordinate, so `n · proj(μ)`
    /// is not `Σ_j x_j`. Reinstating that line moves this cost.
    #[test]
    fn the_cost_is_the_sum_over_the_points_not_over_a_reprojected_centroid() {
        let (pts, _) = hyperbolic_arms(61, 1, 40, 2.0, 3.0, 0.5);
        let mut leaf = Full::new(3);
        for p in &pts {
            leaf.push(p, 1.0);
        }
        let out = hyperbolic_kmeans(std::slice::from_ref(&leaf), 1, 50, 3);
        let mut sum = vec![0.0; 3];
        for p in &pts {
            for (s, &v) in sum.iter_mut().zip(p) {
                *s += v;
            }
        }
        let mu = normalize_to_sheet(&sum);
        let direct: f64 = pts.iter().map(|p| dl2(p, &mu)).sum();
        assert!(
            (out.cost - direct).abs() < 1e-9,
            "leaf cost {} vs the points' own {direct}",
            out.cost
        );
    }

    /// `ΔS ≥ 0` and the merge algebra: merging then measuring equals measuring the merged sum.
    #[test]
    fn the_merge_increase_is_non_negative_and_reads_only_the_sums() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..300 {
            let mk = |rng: &mut SplitMix64| -> Vec<f64> {
                let n = 1 + (rng.next_u64() % 9) as usize;
                let mut acc = vec![0.0; 3];
                for _ in 0..n {
                    let p = project_to_sheet(&[0.0, 2.0 * rng.gauss(), 2.0 * rng.gauss()]);
                    for (a, v) in acc.iter_mut().zip(&p) {
                        *a += v;
                    }
                }
                acc
            };
            let (ra, rb) = (mk(&mut rng), mk(&mut rng));
            let joint: Vec<f64> = ra.iter().zip(&rb).map(|(a, b)| a + b).collect();
            let ds = merge_increase(&ra, &rb);
            let direct = 2.0 * (lorentz_norm(&joint) - lorentz_norm(&ra) - lorentz_norm(&rb));
            assert!(ds >= 0.0);
            assert!((ds - direct.max(0.0)).abs() < 1e-9);
        }
    }

    /// The reported radius is the one the caller has to compare against `f64_working_radius`.
    #[test]
    fn the_head_reports_the_radius_it_actually_saw() {
        let (pts, _) = hyperbolic_arms(51, 2, 50, 6.8, 7.0, 0.05);
        let out = hyperbolic_kmeans(&leaves(&pts), 2, 100, 1);
        assert!(
            (out.max_radius - 7.0).abs() < 0.5,
            "radius {} not near 7",
            out.max_radius
        );
        assert!(f64_working_radius() > 18.0 && f64_working_radius() < 19.0);
    }
}
