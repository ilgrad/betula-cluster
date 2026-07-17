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
}
