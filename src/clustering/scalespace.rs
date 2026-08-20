//! Scale-space mode clustering over CF microclusters (Morse persistence).
//!
//! Treats the leaf microclusters as a weighted point set and clusters the **modes** of the kernel
//! density `ρ_h(x) = Σ_j n_j exp(−‖x − μ_j‖² / 2h²)`. At a bandwidth `h`, mean-shift moves every
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
        .map(|&h| mean_shift(&mu, &n, h, max_iter))
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

/// Mean-shift every microcluster mean uphill on `ρ_h` (data points fixed at `μ`, weighted by `n`),
/// then group the converged positions into modes by [`prominence_modes`]. Returns
/// `(mode label per point, #modes)`.
#[allow(clippy::needless_range_loop)] // mean-shift mutates pts[i] in place while reading μ/pts by index
fn mean_shift(mu: &[Vec<f64>], n: &[f64], h: f64, max_iter: usize) -> (Vec<usize>, usize) {
    let m = mu.len();
    let d = mu[0].len();
    let inv2h2 = 1.0 / (2.0 * h * h);
    let tol = 1e-4 * h;
    let mut pts = mu.to_vec();
    for _ in 0..max_iter.max(1) {
        let mut moved = false;
        for i in 0..m {
            let mut num = vec![0.0; d];
            let mut den = 0.0;
            for j in 0..m {
                let w = n[j] * (-sq_euclidean::<f64>(&pts[i], &mu[j]) * inv2h2).exp();
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
    prominence_modes(&pts, mu, n, h)
}

/// Group converged mean-shift points into modes by **prominence**: tight-unique the endpoints into
/// raw modes, then union two nearby raw modes when the density valley between them stays above
/// `VALLEY_RATIO` of the lower peak (a shallow saddle). This collapses the spurious sub-peaks a
/// single cluster produces at fine bandwidths while keeping separated clusters apart, so the
/// mode-count-vs-scale curve is clean. Returns `(mode label per point, #modes)`.
#[allow(clippy::needless_range_loop)] // pairwise valley checks read clearest with (a, b, t, k) indices
fn prominence_modes(pts: &[Vec<f64>], mu: &[Vec<f64>], n: &[f64], h: f64) -> (Vec<usize>, usize) {
    let inv2h2 = 1.0 / (2.0 * h * h);
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
            .zip(n)
            .map(|(muj, &nj)| nj * (-sq_euclidean::<f64>(x, muj) * inv2h2).exp())
            .sum()
    };
    let peak: Vec<f64> = reps.iter().map(|r| rho(r)).collect();

    // Union raw modes joined by a shallow valley. Only nearby pairs can qualify — modes farther than
    // `4h` apart always have a deep valley, so they are skipped (keeps this out of `O(m²·dim)`).
    let mut parent: Vec<usize> = (0..m).collect();
    let cutoff2 = (4.0 * h).powi(2);
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
                    let d2: f64 = (0..d).map(|k| (pts[i][k] - mu[j][k]).powi(2)).sum();
                    let w = n[j] * (-d2 / (2.0 * h * h)).exp();
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

        let mut counts = Vec::new();
        for step in 0..8 {
            let h = 0.25 * 1.45_f64.powi(step);
            let (labels, modes) = mean_shift(&mu, &n, h, 100);
            let endpoints = reference_shift_endpoints(&mu, &n, h, 100);
            let (rlabels, rmodes) = prominence_modes(&endpoints, &mu, &n, h);
            assert_eq!(modes, rmodes, "h = {h}: mode count");
            assert_eq!(labels, rlabels, "h = {h}: labelling");
            counts.push(modes);
        }
        assert!(
            counts.first() > counts.last(),
            "sweep did not traverse a merge cascade: {counts:?}"
        );
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
        h: f64,
    ) -> (Vec<usize>, usize) {
        let dim = mu[0].len();
        let d2 = |a: &[f64], b: &[f64]| -> f64 { (0..dim).map(|k| (a[k] - b[k]).powi(2)).sum() };
        let rho = |x: &[f64]| -> f64 {
            mu.iter()
                .zip(n)
                .map(|(m, &nj)| nj * (-d2(x, m) / (2.0 * h * h)).exp())
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
        let cutoff2 = (4.0 * h) * (4.0 * h);
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

        let mut counts = Vec::new();
        for step in 0..14 {
            let h = 0.18 * 1.28f64.powi(step);
            let (labels, modes) = mean_shift(&mu, &n, h, 200);
            let ends = reference_shift_endpoints(&mu, &n, h, 200);
            let (rlabels, rmodes) = reference_prominence(&ends, &mu, &n, h);
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
                let (labels, modes) = prominence_modes(&mu.clone(), &mu, &n, h);
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
