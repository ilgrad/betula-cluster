//! Scale-space mode clustering over CF microclusters (Morse persistence).
//!
//! Treats the leaf microclusters as a weighted set of **Gaussians, not points**, and clusters the
//! modes of the kernel density. A leaf is a cloud, and convolving that cloud with the kernel is what
//! the kernel density of the underlying data actually is: `N(μ_j, Σ_j) * N(0, h²I) = N(μ_j, Σ_j+h²I)`.
//! The `(n, μ, S)` summary carries the leaf's own scatter, so with `σ²_j = S_j/(n_j·d)` the isotropic
//! part of `Σ_j` is known and every leaf gets its own width `s²_j = h² + σ²_j`:
//!
//! ```text
//! ρ_h(x) = Σ_j n_j · s_j^(−d) · exp(−‖x − μ_j‖² / 2s²_j)
//! ```
//!
//! The `s_j^(−d)` factor is not decoration: once widths differ per leaf it is what stops a fat leaf
//! from having the same peak density as a tight one. On unit-mass leaves every `σ²_j` is zero, `s_j`
//! collapses to `h`, the factor becomes a constant that cancels in every ratio taken below, and the
//! head reduces exactly to the point-kernel version — which is what the tests assert.
//!
//! At a bandwidth `h`, mean-shift moves every
//! microcluster uphill to a density mode; microclusters reaching the same mode form a cluster. As `h`
//! grows, modes merge — the classic scale-space / Morse picture. Rather than ask the user for `h`
//! (or `k`), the head sweeps `h` and keeps the labelling at the **most persistent** mode count: the
//! widest plateau of the "number of modes vs `log h`" curve, i.e. the structure that survives the
//! longest range of scales. This makes it parameter-free (no `k`, no bandwidth) and non-convex-aware.
//!
//! It runs on the `M ≪ N` microclusters, so cost is `O(sweeps · iters · M² · d)` — bounded by the
//! leaf budget, not `N`.

use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Two raw modes are the same cluster when the density valley between them stays above this fraction
/// of the lower peak (a shallow saddle) — this collapses the spurious sub-peaks a single cluster
/// produces at fine bandwidths while keeping genuinely separated clusters apart.
const VALLEY_RATIO: f64 = 0.8;

/// Result of a scale-space run: one cluster label per input microcluster, plus the selected scale.
pub struct ScaleSpace {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
    /// Bandwidth `h` at the selected (most persistent) scale.
    pub bandwidth: f64,
    /// Number of density modes (clusters) at the selected scale.
    pub n_modes: usize,
}

/// Cluster `features` by KDE modes, auto-selecting the scale by mode persistence. `n_bandwidths` is
/// the number of log-spaced scales swept (`0` ⇒ a sensible default); `max_iter` bounds the mean-shift
/// iterations per scale.
pub fn scale_space<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    n_bandwidths: usize,
    max_iter: usize,
) -> ScaleSpace {
    let mu: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|&x| x.to_f64().unwrap()).collect())
        .collect();
    let n: Vec<f64> = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap())
        .collect();
    let m = mu.len();
    // Per-dimension variance of the leaf's own points: `ssd` is the total scatter `Σ‖x − μ‖²`.
    let var: Vec<f64> = features
        .iter()
        .zip(&n)
        .map(|(f, &w)| {
            let dim = f.mean().len().max(1) as f64;
            if w > 0.0 {
                (f.ssd().to_f64().unwrap() / (w * dim)).max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    if m <= 1 {
        return ScaleSpace {
            labels: vec![0; m],
            bandwidth: 0.0,
            n_modes: m,
        };
    }

    let (h_min, h_max) = bandwidth_range(&mu);
    let steps = if n_bandwidths == 0 { 15 } else { n_bandwidths }.max(2);
    // Log-spaced bandwidths h_min .. h_max.
    let ln_lo = h_min.ln();
    let ln_step = (h_max.ln() - ln_lo) / (steps as f64 - 1.0);
    let bandwidths: Vec<f64> = (0..steps)
        .map(|s| (ln_lo + ln_step * s as f64).exp())
        .collect();

    let runs: Vec<(Vec<usize>, usize)> = bandwidths
        .iter()
        .map(|&h| mean_shift(&mu, &n, &var, h, max_iter))
        .collect();

    // Most persistent scale = the widest plateau of equal mode counts. A multi-mode (`≥ 2`) plateau
    // wins only if it is at least as persistent as the merged single-mode tail — otherwise the data
    // is genuinely one cluster.
    let sel = select_scale(&runs.iter().map(|r| r.1).collect::<Vec<_>>());
    let (labels, n_modes) = runs[sel].clone();
    ScaleSpace {
        labels,
        bandwidth: bandwidths[sel],
        n_modes,
    }
}

/// Bandwidth sweep bounds from the microcluster geometry: `h_min` ≈ half the median nearest-neighbour
/// gap (resolves individual peaks), `h_max` ≈ half the point-set diameter (merges everything).
fn bandwidth_range(mu: &[Vec<f64>]) -> (f64, f64) {
    let m = mu.len();
    let mut nn = vec![f64::INFINITY; m];
    let mut diam2 = 0.0_f64;
    for i in 0..m {
        for j in (i + 1)..m {
            let d2 = sq_euclidean::<f64>(&mu[i], &mu[j]);
            if d2 < nn[i] {
                nn[i] = d2;
            }
            if d2 < nn[j] {
                nn[j] = d2;
            }
            if d2 > diam2 {
                diam2 = d2;
            }
        }
    }
    let mut nn_d: Vec<f64> = nn.iter().map(|&d2| d2.sqrt()).collect();
    nn_d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_nn = nn_d[m / 2].max(1e-12);
    let diameter = diam2.sqrt().max(median_nn * 2.0);
    (0.5 * median_nn, (0.5 * diameter).max(median_nn))
}

/// Per-leaf kernel constants at bandwidth `h`: `(1/2s²_j, n_j·s_j^(−d))`, with `s²_j = h² + σ²_j` the
/// mollified width of leaf `j` and the second entry its already-scaled mass.
fn widths(n: &[f64], var: &[f64], h: f64, d: usize) -> Vec<(f64, f64)> {
    n.iter()
        .zip(var)
        .map(|(&nj, &vj)| {
            let s2 = h * h + vj;
            (0.5 / s2, nj * s2.powf(-0.5 * d as f64))
        })
        .collect()
}

/// Mean-shift every microcluster mean uphill on `ρ_h` (the leaves fixed at `μ`, each a Gaussian of
/// width `s²_j = h² + σ²_j` carrying mass `n_j`), then group the converged positions into modes by
/// [`prominence_modes`]. Returns `(mode label per point, #modes)`.
///
/// The stationary point of a variable-width kernel sum is `x = Σ_j (w_j/s²_j)·μ_j / Σ_j (w_j/s²_j)`,
/// not `Σ w_j μ_j / Σ w_j`: the `1/s²_j` comes straight out of `∇ρ = Σ_j w_j (μ_j − x)/s²_j`
/// (Comaniciu, Ramesh & Meer 2001). With equal widths it is a constant and the two agree.
#[allow(clippy::needless_range_loop)] // mean-shift mutates pts[i] in place while reading μ/pts by index
fn mean_shift(
    mu: &[Vec<f64>],
    n: &[f64],
    var: &[f64],
    h: f64,
    max_iter: usize,
) -> (Vec<usize>, usize) {
    let m = mu.len();
    let d = mu[0].len();
    let ker = widths(n, var, h, d);
    let tol = 1e-4 * h;
    let mut pts = mu.to_vec();
    for _ in 0..max_iter.max(1) {
        let mut moved = false;
        for i in 0..m {
            let mut num = vec![0.0; d];
            let mut den = 0.0;
            for j in 0..m {
                let (inv2s2, mass) = ker[j];
                let g =
                    mass * (-sq_euclidean::<f64>(&pts[i], &mu[j]) * inv2s2).exp() * 2.0 * inv2s2;
                den += g;
                for k in 0..d {
                    num[k] += g * mu[j][k];
                }
            }
            if den > 0.0 {
                for k in 0..d {
                    let nv = num[k] / den;
                    if (nv - pts[i][k]).abs() > tol {
                        moved = true;
                    }
                    pts[i][k] = nv;
                }
            }
        }
        if !moved {
            break;
        }
    }
    prominence_modes(&pts, mu, n, var, h)
}

/// Group converged mean-shift points into modes by **prominence**: tight-unique the endpoints into
/// raw modes, then union two nearby raw modes when the density valley between them stays above
/// `VALLEY_RATIO` of the lower peak (a shallow saddle). This collapses the spurious sub-peaks a
/// single cluster produces at fine bandwidths while keeping separated clusters apart, so the
/// mode-count-vs-scale curve is clean. Returns `(mode label per point, #modes)`.
#[allow(clippy::needless_range_loop)] // pairwise valley checks read clearest with (a, b, t, k) indices
fn prominence_modes(
    pts: &[Vec<f64>],
    mu: &[Vec<f64>],
    n: &[f64],
    var: &[f64],
    h: f64,
) -> (Vec<usize>, usize) {
    let ker = widths(n, var, h, mu[0].len());
    // Tight-unique the converged points into raw modes.
    let tol2 = (0.1 * h).powi(2);
    let mut reps: Vec<Vec<f64>> = Vec::new();
    let mut raw = vec![0usize; pts.len()];
    for (i, p) in pts.iter().enumerate() {
        match reps.iter().position(|r| sq_euclidean::<f64>(p, r) <= tol2) {
            Some(c) => raw[i] = c,
            None => {
                raw[i] = reps.len();
                reps.push(p.clone());
            }
        }
    }
    let m = reps.len();
    let rho = |x: &[f64]| -> f64 {
        mu.iter()
            .zip(&ker)
            .map(|(muj, &(inv2s2, mass))| mass * (-sq_euclidean::<f64>(x, muj) * inv2s2).exp())
            .sum()
    };
    let peak: Vec<f64> = reps.iter().map(|r| rho(r)).collect();

    // Union raw modes joined by a shallow valley. Only nearby pairs can qualify — modes farther than
    // four times the *widest* leaf apart always have a deep valley, so they are skipped (keeps this
    // out of `O(m²·dim)`). The cut-off has to follow the widest `s_j`, not `h`: a fat leaf spreads
    // density further than the bandwidth alone would.
    let mut parent: Vec<usize> = (0..m).collect();
    let s_max = ker
        .iter()
        .map(|&(inv2s2, _)| (0.5 / inv2s2).sqrt())
        .fold(h, f64::max);
    let cutoff2 = (4.0 * s_max).powi(2);
    let dim = mu[0].len();
    for a in 0..m {
        for b in (a + 1)..m {
            if sq_euclidean::<f64>(&reps[a], &reps[b]) > cutoff2 {
                continue;
            }
            let mut valley = f64::INFINITY;
            for t in 1..12 {
                let f = t as f64 / 12.0; // interior points of the a→b segment
                let seg: Vec<f64> = (0..dim)
                    .map(|k| reps[a][k] * (1.0 - f) + reps[b][k] * f)
                    .collect();
                valley = valley.min(rho(&seg));
            }
            if valley >= VALLEY_RATIO * peak[a].min(peak[b]) {
                let (ra, rb) = (uf_find(&mut parent, a), uf_find(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
    }

    // Relabel components to dense ids and map every point through its raw mode.
    let mut comp = vec![usize::MAX; m];
    let mut n_modes = 0;
    for a in 0..m {
        let r = uf_find(&mut parent, a);
        if comp[r] == usize::MAX {
            comp[r] = n_modes;
            n_modes += 1;
        }
    }
    let labels = raw.iter().map(|&r| comp[uf_find(&mut parent, r)]).collect();
    (labels, n_modes)
}

/// Union-find root with path halving.
fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    let mut c = x;
    while parent[c] != r {
        let nx = parent[c];
        parent[c] = r;
        c = nx;
    }
    r
}

/// Select the scale index by mode persistence: the widest `≥ 2`-mode plateau wins if it spans at
/// least two scales (prominence merging makes spurious multi-mode runs width-1, so a wider run is
/// real structure); otherwise the data is one cluster and the widest single-mode run is chosen.
/// Returns the middle of the winning run.
fn select_scale(counts: &[usize]) -> usize {
    let (mut best2_start, mut best2_len) = (0usize, 0usize); // widest run with count ≥ 2
    let (mut best1_start, mut best1_len) = (0usize, 0usize); // widest run with count == 1
    let mut i = 0;
    while i < counts.len() {
        let mut j = i;
        while j < counts.len() && counts[j] == counts[i] {
            j += 1;
        }
        let len = j - i;
        if counts[i] >= 2 && len > best2_len {
            best2_len = len;
            best2_start = i;
        }
        if counts[i] == 1 && len > best1_len {
            best1_len = len;
            best1_start = i;
        }
        i = j;
    }
    // With prominence merging, spurious multi-mode runs are width-1, so a ≥2-wide multi-mode plateau
    // is real structure and wins outright; otherwise the data is one cluster (widest single-mode run).
    if best2_len >= 2 {
        best2_start + best2_len / 2
    } else if best1_len > 0 {
        best1_start + best1_len / 2
    } else {
        // No stable multi-mode or single-mode plateau — take the coarsest scale (fewest, most-merged).
        counts.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros};
    use crate::feature::Spherical;

    /// Per-dimension leaf variance, re-derived rather than shared with the source.
    fn leaf_variances(micros: &[Spherical<f64>]) -> Vec<f64> {
        micros
            .iter()
            .map(|f| f.ssd() / (f.weight() * f.mean().len() as f64))
            .collect()
    }

    #[test]
    fn scale_space_recovers_well_separated_blobs() {
        let mut rng = SplitMix64::new(7);
        let centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.5);
        let (micros, assign) = grid_micros(&pts, 0.6);
        let res = scale_space(&micros, 15, 100);
        assert_eq!(res.n_modes, 3, "should find 3 density modes");
        let labels: Vec<usize> = assign.iter().map(|&mi| res.labels[mi]).collect();
        assert!(
            ari(&labels, &truth) > 0.95,
            "ARI = {}",
            ari(&labels, &truth)
        );
    }

    #[test]
    fn scale_space_recovers_many_clusters() {
        // Six well-separated blobs — before prominence merging this collapsed to a single mode.
        let mut rng = SplitMix64::new(9);
        let centers = [
            [0.0, 0.0],
            [12.0, 0.0],
            [0.0, 12.0],
            [12.0, 12.0],
            [6.0, 21.0],
            [21.0, 6.0],
        ];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.6);
        let (micros, assign) = grid_micros(&pts, 0.6);
        let res = scale_space(&micros, 15, 100);
        assert!(res.n_modes >= 5, "expected ≥5 modes, got {}", res.n_modes);
        let labels: Vec<usize> = assign.iter().map(|&mi| res.labels[mi]).collect();
        assert!(
            ari(&labels, &truth) > 0.85,
            "ARI = {}",
            ari(&labels, &truth)
        );
    }

    #[test]
    fn scale_space_single_blob_is_one_mode() {
        let mut rng = SplitMix64::new(3);
        let centers = [[0.0, 0.0]];
        let (pts, _t) = blobs(&mut rng, 200, &centers, 0.7);
        let (micros, _a) = grid_micros(&pts, 0.6);
        let res = scale_space(&micros, 12, 100);
        assert_eq!(res.n_modes, 1);
    }

    #[test]
    fn scale_space_handles_tiny_input() {
        let (micros, _a) = grid_micros(&[vec![0.0, 0.0]], 1.0);
        let res = scale_space(&micros, 10, 50);
        assert_eq!(res.labels, vec![0]);
    }

    #[test]
    fn bandwidth_range_is_half_the_median_gap_and_half_the_diameter() {
        // Four collinear points at 0, 1, 3, 7. Squared nearest-neighbour gaps are 1, 1, 4, 16, so the
        // sorted distances are 1, 1, 2, 4 and the median (index m/2 = 2) is 2. The diameter is 7,
        // already above 2·median, so it is not clamped. Expect (0.5·2, 0.5·7) = (1.0, 3.5).
        let mu = vec![vec![0.0], vec![1.0], vec![3.0], vec![7.0]];
        let (lo, hi) = bandwidth_range(&mu);
        assert!((lo - 1.0).abs() < 1e-12, "h_min = {lo}");
        assert!((hi - 3.5).abs() < 1e-12, "h_max = {hi}");

        // Coincident points: every gap is 0, so the median is floored to 1e-12 and the diameter is
        // clamped up to 2·median rather than staying at 0.
        let same = vec![vec![2.0, 2.0]; 3];
        let (lo, hi) = bandwidth_range(&same);
        assert!((lo - 0.5e-12).abs() < 1e-24, "degenerate h_min = {lo}");
        assert!((hi - 1e-12).abs() < 1e-24, "degenerate h_max = {hi}");
    }

    /// Independent re-derivation of the mean-shift ascent in [`mean_shift`]: Gaussian kernel, data
    /// fixed at `μ` with weights `n`, iterate until no coordinate moves by more than `1e-4·h`.
    /// Returns the converged positions, which the source never exposes.
    #[allow(clippy::needless_range_loop)] // mutates pts[i] while reading mu by index, as the source does
    fn reference_shift_endpoints(
        mu: &[Vec<f64>],
        n: &[f64],
        var: &[f64],
        h: f64,
        max_iter: usize,
    ) -> Vec<Vec<f64>> {
        let (m, d) = (mu.len(), mu[0].len());
        let tol = 1e-4 * h;
        let mut pts = mu.to_vec();
        for _ in 0..max_iter.max(1) {
            let mut moved = false;
            for i in 0..m {
                let mut num = vec![0.0; d];
                let mut den = 0.0;
                for j in 0..m {
                    let s2 = h * h + var[j];
                    let d2: f64 = (0..d).map(|k| (pts[i][k] - mu[j][k]).powi(2)).sum();
                    let w = n[j] / s2.sqrt().powi(d as i32) * (-d2 / (2.0 * s2)).exp() / s2;
                    den += w;
                    for k in 0..d {
                        num[k] += w * mu[j][k];
                    }
                }
                if den > 0.0 {
                    for k in 0..d {
                        let nv = num[k] / den;
                        if (nv - pts[i][k]).abs() > tol {
                            moved = true;
                        }
                        pts[i][k] = nv;
                    }
                }
            }
            if !moved {
                break;
            }
        }
        pts
    }

    #[test]
    fn mean_shift_ascent_matches_an_independent_reference_across_the_sweep() {
        // Sweeping h walks the whole merge cascade, from one mode per group down to a single mode.
        // The bandwidth at which each merge happens is what the arithmetic decides, so a corrupted
        // kernel, numerator or convergence test moves a merge and changes the labelling.
        let mut rng = SplitMix64::new(31);
        let centers = [[0.0, 0.0], [3.0, 0.4], [1.4, 3.0]];
        let (pts, _) = blobs(&mut rng, 60, &centers, 0.7);
        let (micros, _) = grid_micros(&pts, 0.5);
        let mu: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let n: Vec<f64> = micros.iter().map(|f| f.weight()).collect();
        let var: Vec<f64> = leaf_variances(&micros);

        let mut counts = Vec::new();
        for step in 0..8 {
            let h = 0.25 * 1.45_f64.powi(step);
            let (labels, modes) = mean_shift(&mu, &n, &var, h, 100);
            let endpoints = reference_shift_endpoints(&mu, &n, &var, h, 100);
            let (rlabels, rmodes) = prominence_modes(&endpoints, &mu, &n, &var, h);
            assert_eq!(modes, rmodes, "h = {h}: mode count");
            assert_eq!(labels, rlabels, "h = {h}: labelling");
            counts.push(modes);
        }
        assert!(
            counts.first() > counts.last(),
            "sweep did not traverse a merge cascade: {counts:?}"
        );
    }

    /// Leaves wide enough to overlap, whose centroids are not. The point kernel sees the centroid
    /// lattice and reports one mode per leaf; the mollified kernel sees the clouds and reports the two
    /// groups. This is the whole content of the fix, so it is stated as `var` against `var = 0` on the
    /// same fixture at the same bandwidth — reverting the mollification collapses it to the second arm.
    #[test]
    fn wide_leaves_merge_under_mollification_and_not_under_the_point_kernel() {
        let mu: Vec<Vec<f64>> = [0.0, 4.0, 8.0, 30.0, 34.0, 38.0]
            .iter()
            .map(|&x| vec![x])
            .collect();
        let n = vec![30.0; 6];
        let var = vec![4.0; 6]; // sigma = 2, so neighbouring leaves overlap heavily
        let zero = vec![0.0; 6];
        let h = 1.0;

        let (labels, modes) = mean_shift(&mu, &n, &var, h, 200);
        assert_eq!(modes, 2, "mollified: the two groups");
        assert_eq!(labels[0], labels[2], "left group is one cluster");
        assert_eq!(labels[3], labels[5], "right group is one cluster");
        assert_ne!(labels[0], labels[3], "the groups stay apart");

        let (_, point_modes) = mean_shift(&mu, &n, &zero, h, 200);
        assert!(
            point_modes > modes,
            "point kernel should over-split: {point_modes} vs {modes}"
        );
    }

    /// The reduction the module docs claim: with no leaf scatter the per-leaf width collapses to `h`
    /// and the `s^-d` amplitude becomes a constant that cancels in every ratio, so the head is
    /// bit-identical to the point-kernel version it replaced.
    #[test]
    fn zero_scatter_leaves_reduce_to_the_point_kernel_exactly() {
        let mut rng = SplitMix64::new(17);
        let centers = [[0.0, 0.0], [5.0, 0.5], [2.0, 5.0]];
        let (pts, _) = blobs(&mut rng, 90, &centers, 0.5);
        // One leaf per point: weight 1, ssd 0, so every `sigma^2` is exactly zero.
        let singles: Vec<Spherical<f64>> = pts
            .iter()
            .map(|p| {
                let mut cf = Spherical::new(p.len());
                cf.push(p, 1.0);
                cf
            })
            .collect();
        assert!(leaf_variances(&singles).iter().all(|&v| v == 0.0));

        let mu: Vec<Vec<f64>> = singles.iter().map(|f| f.mean().to_vec()).collect();
        let n = vec![1.0; mu.len()];
        let zero = vec![0.0; mu.len()];
        for step in 0..6 {
            let h = 0.3 * 1.5_f64.powi(step);
            let got = mean_shift(&mu, &n, &leaf_variances(&singles), h, 100);
            let want = mean_shift(&mu, &n, &zero, h, 100);
            assert_eq!(got, want, "h = {h}");
        }
    }

    #[test]
    fn the_bandwidth_grid_is_log_spaced_and_zero_means_the_default() {
        let mut rng = SplitMix64::new(5);
        let centers = [[0.0, 0.0], [6.0, 0.0]];
        let (pts, _) = blobs(&mut rng, 120, &centers, 0.6);
        let (micros, _) = grid_micros(&pts, 0.6);
        let mu: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();

        let (h_min, h_max) = bandwidth_range(&mu);
        let steps = 9usize;
        let ln_lo = h_min.ln();
        let ln_step = (h_max.ln() - ln_lo) / (steps as f64 - 1.0);
        let grid: Vec<f64> = (0..steps)
            .map(|s| (ln_lo + ln_step * s as f64).exp())
            .collect();

        let got = scale_space(&micros, steps, 100).bandwidth;
        assert!(
            grid.iter().any(|&g| (g - got).abs() <= 1e-12 * g.max(1.0)),
            "selected bandwidth {got} is not on the log grid {grid:?}"
        );

        // `n_bandwidths = 0` selects the documented default of 15, not a two-point sweep.
        let zero = scale_space(&micros, 0, 100);
        let fifteen = scale_space(&micros, 15, 100);
        assert_eq!(zero.n_modes, fifteen.n_modes);
        assert!((zero.bandwidth - fifteen.bandwidth).abs() < 1e-12);
        assert_eq!(zero.labels, fifteen.labels);
    }

    /// The prominence merge re-derived: converged positions closer than `0.1h` are the same raw
    /// mode (first-seen order), and two raw modes within `4h` join when the lowest density on the
    /// eleven interior points of the segment between them reaches `VALLEY_RATIO` of the smaller
    /// peak, under `ρ(x) = Σ_j n_j exp(−‖x − μ_j‖² / 2h²)`.
    fn reference_prominence(
        pts: &[Vec<f64>],
        mu: &[Vec<f64>],
        n: &[f64],
        var: &[f64],
        h: f64,
    ) -> (Vec<usize>, usize) {
        let dim = mu[0].len();
        let d2 = |a: &[f64], b: &[f64]| -> f64 { (0..dim).map(|k| (a[k] - b[k]).powi(2)).sum() };
        let s2: Vec<f64> = var.iter().map(|v| h * h + v).collect();
        let rho = |x: &[f64]| -> f64 {
            mu.iter()
                .zip(n)
                .zip(&s2)
                .map(|((m, &nj), &sj)| {
                    nj / sj.sqrt().powi(dim as i32) * (-d2(x, m) / (2.0 * sj)).exp()
                })
                .sum()
        };
        let tol2 = (0.1 * h) * (0.1 * h);
        let mut reps: Vec<Vec<f64>> = Vec::new();
        let raw: Vec<usize> = pts
            .iter()
            .map(|p| match reps.iter().position(|r| d2(p, r) <= tol2) {
                Some(c) => c,
                None => {
                    reps.push(p.clone());
                    reps.len() - 1
                }
            })
            .collect();
        let m = reps.len();
        let peak: Vec<f64> = reps.iter().map(|r| rho(r)).collect();

        let mut parent: Vec<usize> = (0..m).collect();
        let root = |p: &mut Vec<usize>, x: usize| -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            r
        };
        let s_max = s2.iter().fold(h * h, |acc, &v| acc.max(v)).sqrt();
        let cutoff2 = (4.0 * s_max) * (4.0 * s_max);
        for a in 0..m {
            for b in (a + 1)..m {
                if d2(&reps[a], &reps[b]) > cutoff2 {
                    continue;
                }
                let mut valley = f64::INFINITY;
                for t in 1..12 {
                    let f = t as f64 / 12.0;
                    let seg: Vec<f64> = (0..dim)
                        .map(|k| reps[a][k] * (1.0 - f) + reps[b][k] * f)
                        .collect();
                    valley = valley.min(rho(&seg));
                }
                if valley >= VALLEY_RATIO * peak[a].min(peak[b]) {
                    let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }
        let mut comp = vec![usize::MAX; m];
        let mut n_modes = 0;
        for a in 0..m {
            let r = root(&mut parent, a);
            if comp[r] == usize::MAX {
                comp[r] = n_modes;
                n_modes += 1;
            }
        }
        let labels = raw
            .iter()
            .map(|&r| comp[root(&mut parent, r)])
            .collect::<Vec<usize>>();
        (labels, n_modes)
    }

    #[test]
    fn the_prominence_merge_matches_an_independent_reference_across_the_sweep() {
        // The density behind the merge is never returned, and the mode count on a well-separated
        // fixture survives almost any corruption of it. Sweeping the bandwidth walks the merge
        // through every count from "one mode per blob" to "everything is one", and the whole
        // sequence of counts -- not just its endpoints -- is compared.
        let mut rng = SplitMix64::new(21);
        let centers = [[0.0, 0.0], [2.2, 0.3], [1.0, 2.4], [6.5, 5.0]];
        let (pts, _truth) = blobs(&mut rng, 45, &centers, 0.45);
        let (micros, _assign) = grid_micros(&pts, 0.4);
        let mu: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let n: Vec<f64> = micros.iter().map(|f| f.weight()).collect();
        let var: Vec<f64> = leaf_variances(&micros);

        let mut counts = Vec::new();
        for step in 0..14 {
            let h = 0.18 * 1.28f64.powi(step);
            let (labels, modes) = mean_shift(&mu, &n, &var, h, 200);
            let ends = reference_shift_endpoints(&mu, &n, &var, h, 200);
            let (rlabels, rmodes) = reference_prominence(&ends, &mu, &n, &var, h);
            assert_eq!(modes, rmodes, "h = {h}: mode count");
            assert_eq!(labels, rlabels, "h = {h}: labelling");
            counts.push(modes);
        }
        assert!(
            counts.first() > counts.last()
                && counts.windows(2).filter(|w| w[0] != w[1]).count() > 2,
            "the sweep does not walk the merge: {counts:?}"
        );
    }
    /// The valley test, in closed form, on two equal Gaussian bumps `s` apart: `rho` is a plain sum
    /// of kernels here, not the implementation's cached reciprocal bandwidth, so a corrupted
    /// exponent or a corrupted `1/(2h²)` shows up as a different merge decision.
    fn expected_two_bump_modes(s: f64, h: f64) -> usize {
        let rho = |x: f64| {
            (-(x * x) / (2.0 * h * h)).exp() + (-((x - s) * (x - s)) / (2.0 * h * h)).exp()
        };
        if s <= 0.1 * h {
            return 1; // the two converged points tight-unique into one raw mode
        }
        if s * s > (4.0 * h) * (4.0 * h) {
            return 2; // farther apart than the pair cutoff: never even tested
        }
        let valley = (1..12)
            .map(|t| rho(s * t as f64 / 12.0))
            .fold(f64::INFINITY, f64::min);
        if valley >= VALLEY_RATIO * rho(0.0).min(rho(s)) {
            1
        } else {
            2
        }
    }

    /// A separation sweep over `prominence_modes` alone, with the converged points placed exactly on
    /// the two peaks so the merge decision is the only thing left to get wrong. The bandwidth sweep
    /// in the test above walks the *mean-shift* endpoints; nothing walked the valley test itself, and
    /// on a well-separated fixture it agrees with almost any corruption of the density.
    #[test]
    fn the_valley_test_flips_where_the_density_says_it_does() {
        for &h in &[0.5_f64, 1.0, 2.0] {
            let mut seen = Vec::new();
            for step in 0..40 {
                let s = 0.05 * h + 0.09 * h * step as f64;
                let mu = vec![vec![0.0, 0.0], vec![s, 0.0]];
                let n = vec![1.0, 1.0];
                let (labels, modes) = prominence_modes(&mu.clone(), &mu, &n, &[0.0, 0.0], h);
                let want = expected_two_bump_modes(s, h);
                assert_eq!(modes, want, "h = {h}, s = {s}: mode count");
                assert_eq!(labels.len(), 2, "h = {h}, s = {s}: one label per point");
                assert_eq!(
                    labels[0] == labels[1],
                    want == 1,
                    "h = {h}, s = {s}: labelling disagrees with the count"
                );
                seen.push(modes);
            }
            seen.dedup();
            assert_eq!(
                seen,
                vec![1, 2],
                "h = {h}: the sweep never crossed the valley threshold: {seen:?}"
            );
        }
    }
}
