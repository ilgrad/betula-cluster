//! Spectral clustering on the CF-tree leaf microclusters.
//!
//! Builds a self-tuning RBF affinity over the leaf means (local scaling of Zelnik-Manor & Perona,
//! NIPS 2004), forms the symmetric normalized affinity `P = D^{-1/2} A D^{-1/2}` whose top
//! eigenvectors are the bottom eigenvectors of the normalized Laplacian `L_sym = I − P`
//! (Ng-Jordan-Weiss, NIPS 2001), embeds each microcluster in the space of the `k` such
//! eigenvectors, row-normalizes, and k-means-clusters the embedding. This separates non-convex /
//! manifold clusters (rings, moons, spirals) that the centroid heads cannot.
//!
//! The eigensolver is the in-house cyclic-Jacobi routine (`O(M³)`; no LAPACK/ARPACK — the crate
//! stays LEAN), so the microcluster count handed to it is capped: above [`SPECTRAL_MAX_NODES`] the
//! leaves are first reduced to that many weighted k-means landmarks, spectral-clustered, and each
//! leaf inherits its landmark's label. Spectral has no built-in cluster-count selection (the
//! eigengap is unreliable on k-NN graphs), so `k == 0` (auto) falls back to [`SPECTRAL_DEFAULT_K`].

use crate::clustering::graph::{knn_affinity, knn_affinity_with_degree};
use crate::clustering::kmeans::kmeans;
use crate::feature::{ClusterFeature, Spherical};
use crate::linalg::jacobi_eigen;
use crate::types::Real;

/// Microclusters solved directly by the eigensolver; above this, reduce to this many k-means
/// landmarks first. Keeps the `O(M³)` Jacobi eigendecomposition well under a second.
pub const SPECTRAL_MAX_NODES: usize = 256;
/// Cluster count used when `k == 0` (auto) is requested — spectral has no reliable built-in
/// selection, and two clusters is the canonical non-convex case (moons / two rings).
pub const SPECTRAL_DEFAULT_K: usize = 2;
/// k-means restarts for the embedding and landmark reductions (mirrors the k-means head).
const N_INIT: usize = 4;

/// Result of a spectral run: one cluster label per input microcluster.
pub struct Spectral {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
}

/// Spectral-cluster `features` into `k` groups (`k == 0` ⇒ [`SPECTRAL_DEFAULT_K`]). Above
/// [`SPECTRAL_MAX_NODES`] features the graph is reduced to that many k-means landmarks first.
pub fn spectral<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Spectral {
    assert!(!features.is_empty(), "spectral needs at least one feature");
    let k = if k == 0 { SPECTRAL_DEFAULT_K } else { k };
    let m = features.len();
    if m > SPECTRAL_MAX_NODES {
        // Landmark reduction: weighted k-means to SPECTRAL_MAX_NODES centers, spectral on those,
        // then every leaf inherits the label of the landmark it was assigned to.
        let land = kmeans(features, SPECTRAL_MAX_NODES, max_iter, N_INIT, seed);
        let sub = spectral_core::<R>(&land.centers, k, max_iter, seed);
        let labels = land.labels.iter().map(|&a| sub[a]).collect();
        return Spectral { labels };
    }
    let centers: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    Spectral {
        labels: spectral_core::<R>(&centers, k, max_iter, seed),
    }
}

/// Diffusion-maps density normalisation exponent `α` (Coifman & Lafon, *Diffusion maps*, ACHA 21(1),
/// 2006, §3). Replacing `A_ij` by `A_ij / (q_i^α q_j^α)` with `q_i = Σ_j A_ij` before the usual
/// symmetric normalisation interpolates between two different operators: `α = 0` is the
/// density-biased normalized Laplacian, `α = 1` divides the sampling density out entirely and the
/// limit operator is the Laplace–Beltrami of the underlying manifold, `α = 1/2` is the
/// Fokker–Planck point between them.
///
/// It matters more here than on raw points, because a leaf is a *mass-weighted* microcluster: where
/// the data is dense the tree puts more leaves **and** each leaf is tighter, so sampling density
/// enters the affinity twice.
///
/// `0.0` is the shipped value, and it is a measurement rather than an inheritance — see the
/// `measure_alpha_normalization` harness and `bench/RESULTS.md`.
const DIFFUSION_ALPHA: f64 = 0.0;

fn spectral_core<R: Real>(centers: &[Vec<R>], k: usize, max_iter: usize, seed: u64) -> Vec<usize> {
    spectral_core_alpha(centers, k, max_iter, seed, DIFFUSION_ALPHA, None)
}

fn spectral_core_alpha<R: Real>(
    centers: &[Vec<R>],
    k: usize,
    max_iter: usize,
    seed: u64,
    alpha: f64,
    degree: Option<usize>,
) -> Vec<usize> {
    let n = centers.len();
    if n == 1 {
        return vec![0];
    }
    if k <= 1 {
        return vec![0; n]; // single cluster
    }
    if k >= n {
        return (0..n).collect(); // more clusters requested than nodes ⇒ each its own
    }
    let tiny = R::from_f64(1e-12).unwrap();

    // Symmetric self-tuning k-NN affinity graph, then its degrees and the normalized affinity
    // `P = D^{-1/2} A D^{-1/2}` (`A` is exactly symmetric by construction, so `P` is too).
    let adj = match degree {
        Some(d) => knn_affinity_with_degree(centers, d),
        None => knn_affinity(centers),
    };
    let mut a = vec![vec![R::zero(); n]; n];
    for (i, row) in adj.iter().enumerate() {
        for &(j, w) in row {
            a[i][j] = w;
        }
    }
    if alpha != 0.0 {
        // `q` is the kernel density estimate the affinity itself induces; dividing it out is what
        // separates the geometry from the sampling. It must be taken *before* any of `a` is
        // rewritten, or later rows would be normalised against already-normalised ones.
        let exponent = R::from_f64(alpha).unwrap_or_else(R::zero);
        let q: Vec<R> = a
            .iter()
            .map(|row| row.iter().copied().sum::<R>().max(tiny).powf(exponent))
            .collect();
        for (i, row) in a.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                *w = *w / (q[i] * q[j]);
            }
        }
    }
    let mut deg = vec![R::zero(); n];
    for (i, row) in a.iter().enumerate() {
        deg[i] = row.iter().copied().sum::<R>();
    }
    let dinv: Vec<R> = deg.iter().map(|&d| R::one() / d.max(tiny).sqrt()).collect();
    let mut p = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            p[i][j] = a[i][j] * dinv[i] * dinv[j];
        }
    }

    // Top-`k` eigenvectors of `P` are the bottom-`k` of `L_sym = I − P` (here `2 ≤ k < n`).
    let (eigvals, vecs) = jacobi_eigen(&p);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| eigvals[j].partial_cmp(&eigvals[i]).unwrap()); // P eigenvalues descending

    // Row-normalized eigenvector embedding (Ng-Jordan-Weiss), then k-means on the rows.
    let sel = &order[0..k];
    let embed: Vec<Spherical<R>> = (0..n)
        .map(|i| {
            let mut row: Vec<R> = sel.iter().map(|&c| vecs[i][c]).collect();
            let norm = row.iter().map(|&x| x * x).sum::<R>().sqrt().max(tiny);
            for x in row.iter_mut() {
                *x = *x / norm;
            }
            let mut f = Spherical::new(k);
            f.push(&row, R::one());
            f
        })
        .collect();
    kmeans(&embed, k, max_iter, N_INIT, seed).labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros, two_moons};
    use std::collections::HashSet;

    fn n_distinct(labels: &[usize]) -> usize {
        labels.iter().copied().collect::<HashSet<_>>().len()
    }

    /// Two concentric rings — the canonical case where the graph, not the centroid, is the model.
    fn circles(rng: &mut SplitMix64, per: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, r) in [(0usize, 1.0f64), (1, 2.6)] {
            for i in 0..per {
                let t = std::f64::consts::TAU * (i as f64) / (per as f64);
                pts.push(vec![
                    r * t.cos() + noise * rng.gauss(),
                    r * t.sin() + noise * rng.gauss(),
                ]);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    /// Blobs sheared into long thin ribbons: the Euclidean heads' worst shape, and a graph whose
    /// local scales differ wildly along and across the ribbon.
    fn aniso(rng: &mut SplitMix64, per: usize) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, ctr) in [(0usize, [0.0f64, 0.0]), (1, [3.0, 3.0]), (2, [-3.0, 4.0])] {
            for _ in 0..per {
                let (u, v) = (2.5 * rng.gauss(), 0.25 * rng.gauss());
                pts.push(vec![ctr[0] + u + 0.6 * v, ctr[1] + 0.35 * u + v]);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    /// Two moons sampled at different rates. This is the case `α` exists for: the affinity's own
    /// density estimate differs by a factor of four between the two clusters, and `α = 1` is
    /// supposed to divide exactly that out.
    fn lopsided_moons(rng: &mut SplitMix64, per: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
        thinned_moons(rng, per, noise, 4)
    }

    fn thinned_moons(
        rng: &mut SplitMix64,
        per: usize,
        noise: f64,
        thin: usize,
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let (pts, truth) = two_moons(rng, per, noise);
        let mut kept_pts = Vec::new();
        let mut kept_truth = Vec::new();
        // Count within the class, not over the interleaved list: `two_moons` alternates the two
        // moons, so thinning on the global index would drop one class entirely.
        let mut seen = 0usize;
        for (p, &t) in pts.iter().zip(&truth) {
            let keep = if t == 0 {
                true
            } else {
                seen += 1;
                seen % thin == 0
            };
            if keep {
                kept_pts.push(p.clone());
                kept_truth.push(t);
            }
        }
        (kept_pts, kept_truth)
    }

    #[test]
    fn the_shipped_head_is_the_alpha_the_constant_names() {
        let mut rng = SplitMix64::new(3);
        let (pts, _) = circles(&mut rng, 220, 0.08);
        let (micros, _) = grid_micros(&pts, 0.16);
        let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        assert_eq!(
            spectral(&micros, 2, 100, 1).labels,
            spectral_core_alpha(&centers, 2, 100, 1, DIFFUSION_ALPHA, None),
            "the head and the constant have drifted apart"
        );
    }

    #[test]
    fn alpha_normalisation_changes_the_operator_it_is_supposed_to_change() {
        // Without this the `alpha != 0.0` branch could be dead and every measurement below would be
        // measuring the same operator three times.
        let mut rng = SplitMix64::new(5);
        let (pts, _) = lopsided_moons(&mut rng, 400, 0.06);
        let (micros, _) = grid_micros(&pts, 0.1);
        let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        assert_ne!(
            spectral_core_alpha(&centers, 2, 100, 1, 0.0, None),
            spectral_core_alpha(&centers, 2, 100, 1, 1.0, None),
            "alpha = 1 produced the same labelling as alpha = 0"
        );
    }

    /// Is the shipped neighbour count load-bearing, and does the Γ-convergence floor bite? Median
    /// ARI of seeds 0/1/2 per (fixture, degree). `knn_degree` returns 10 at every node count the
    /// spectral head can reach, so the comparison is against fixed alternatives around it.
    ///
    /// `cargo test --lib clustering::spectral::tests::measure_graph_degree -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn measure_graph_degree() {
        type Fixture = fn(&mut SplitMix64, usize, f64) -> (Vec<Vec<f64>>, Vec<usize>);
        let cases: [(&str, Fixture, usize, f64, f64, usize); 4] = [
            ("two-moons", two_moons, 400, 0.06, 0.10, 2),
            ("lopsided-moons", lopsided_moons, 400, 0.06, 0.10, 2),
            ("circles", circles, 400, 0.08, 0.16, 2),
            ("aniso", |r, per, _| aniso(r, per), 400, 0.0, 0.30, 3),
        ];
        let degrees = [3usize, 5, 7, 10, 16, 24];
        print!("\n{:>16} {:>7}", "fixture", "nodes");
        for d in degrees {
            print!("{:>9}", format!("k={d}"));
        }
        println!("   (shipped k = 10)");
        for (name, build, per, noise, cell, k) in cases {
            let mut row = Vec::new();
            let mut nodes = 0usize;
            for degree in degrees {
                let mut scores = Vec::new();
                for seed in 0u64..3 {
                    let mut rng = SplitMix64::new(seed);
                    let (pts, truth) = build(&mut rng, per, noise);
                    let (micros, assign) = grid_micros(&pts, cell);
                    let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
                    nodes = centers.len().min(SPECTRAL_MAX_NODES);
                    let lab = spectral_core_alpha(&centers, k, 100, seed, 0.0, Some(degree));
                    let per_point: Vec<usize> = assign.iter().map(|&m| lab[m]).collect();
                    scores.push(ari(&per_point, &truth));
                }
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                row.push(scores[1]);
            }
            print!("{name:>16} {nodes:>7}");
            for v in &row {
                print!("{v:>9.4}");
            }
            println!();
        }
    }

    /// Does dividing the sampling density out of the affinity help this crate's mass-weighted
    /// leaves? Median ARI of seeds 0/1/2 per (fixture, alpha).
    ///
    /// `cargo test --lib clustering::spectral::tests::measure_alpha -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn measure_alpha_normalization() {
        type Fixture = fn(&mut SplitMix64, usize, f64) -> (Vec<Vec<f64>>, Vec<usize>);
        let cases: [(&str, Fixture, usize, f64, f64, usize); 4] = [
            ("two-moons", two_moons, 400, 0.06, 0.10, 2),
            ("lopsided-moons", lopsided_moons, 400, 0.06, 0.10, 2),
            ("circles", circles, 400, 0.08, 0.16, 2),
            ("aniso", |r, per, _| aniso(r, per), 400, 0.0, 0.30, 3),
        ];
        println!(
            "\n{:>16} {:>8} {:>10} {:>10} {:>10}",
            "fixture", "leaves", "a=0", "a=0.5", "a=1"
        );
        for (name, build, per, noise, cell, k) in cases {
            let mut row = Vec::new();
            let mut leaves = 0usize;
            for alpha in [0.0f64, 0.5, 1.0] {
                let mut scores = Vec::new();
                for seed in 0u64..3 {
                    let mut rng = SplitMix64::new(seed);
                    let (pts, truth) = build(&mut rng, per, noise);
                    let (micros, assign) = grid_micros(&pts, cell);
                    leaves = micros.len();
                    let centers: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
                    let lab = spectral_core_alpha(&centers, k, 100, seed, alpha, None);
                    let per_point: Vec<usize> = assign.iter().map(|&m| lab[m]).collect();
                    scores.push(ari(&per_point, &truth));
                }
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                row.push(scores[1]);
            }
            println!(
                "{name:>16} {leaves:>8} {:>10.4} {:>10.4} {:>10.4}",
                row[0], row[1], row[2]
            );
        }

        // The regime `α` exists for: one moon sampled `thin` times more sparsely than the other.
        println!(
            "\n{:>16} {:>8} {:>10} {:>10} {:>10}",
            "moons 1:thin", "noise", "a=0", "a=0.5", "a=1"
        );
        for thin in [4usize, 10, 25] {
            for noise in [0.06f64, 0.10, 0.14] {
                let mut row = Vec::new();
                for alpha in [0.0f64, 0.5, 1.0] {
                    let mut scores = Vec::new();
                    for seed in 0u64..3 {
                        let mut rng = SplitMix64::new(seed);
                        let (pts, truth) = thinned_moons(&mut rng, 400, noise, thin);
                        let (micros, assign) = grid_micros(&pts, 0.10);
                        let centers: Vec<Vec<f64>> =
                            micros.iter().map(|f| f.mean().to_vec()).collect();
                        let lab = spectral_core_alpha(&centers, 2, 100, seed, alpha, None);
                        let per_point: Vec<usize> = assign.iter().map(|&m| lab[m]).collect();
                        scores.push(ari(&per_point, &truth));
                    }
                    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    row.push(scores[1]);
                }
                println!(
                    "{:>16} {noise:>8.2} {:>10.4} {:>10.4} {:>10.4}",
                    format!("1:{thin}"),
                    row[0],
                    row[1],
                    row[2]
                );
            }
        }
    }

    #[test]
    fn spectral_separates_two_moons_where_kmeans_cannot() {
        let mut rng = SplitMix64::new(7);
        let (pts, truth) = two_moons(&mut rng, 250, 0.05);
        let (micros, point_to_micro) = grid_micros(&pts, 0.12);
        let micro_labels = spectral(&micros, 2, 100, 1).labels;
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| micro_labels[m]).collect();
        assert_eq!(n_distinct(&micro_labels), 2);
        // The non-convex moons are recovered — a centroid head (k-means) scores ~0 here.
        assert!(ari(&pred, &truth) > 0.85, "moons ARI too low");
    }

    #[test]
    fn spectral_auto_k_defaults_to_two() {
        // k == 0 (auto) ⇒ SPECTRAL_DEFAULT_K = 2, which is exactly right for the two moons.
        let mut rng = SplitMix64::new(5);
        let (pts, truth) = two_moons(&mut rng, 250, 0.05);
        let (micros, point_to_micro) = grid_micros(&pts, 0.12);
        let micro_labels = spectral(&micros, 0, 100, 1).labels;
        assert_eq!(n_distinct(&micro_labels), SPECTRAL_DEFAULT_K);
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| micro_labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.85, "auto-k moons ARI too low");
    }

    #[test]
    fn spectral_more_clusters_than_nodes_is_identity() {
        let (micros, _) = grid_micros(&[vec![0.0, 0.0], vec![9.0, 0.0], vec![0.0, 9.0]], 0.5);
        let labels = spectral(&micros, 5, 100, 1).labels;
        assert_eq!(labels, vec![0, 1, 2]);
    }

    #[test]
    fn spectral_single_feature_is_one_cluster() {
        let (micros, _) = grid_micros(&[vec![1.0, 2.0]], 1.0);
        assert_eq!(spectral(&micros, 1, 100, 1).labels, vec![0]);
    }

    #[test]
    fn spectral_k_one_collapses_to_a_single_cluster() {
        let (micros, _) = grid_micros(&[vec![0.0, 0.0], vec![9.0, 0.0], vec![0.0, 9.0]], 0.5);
        let labels = spectral(&micros, 1, 100, 1).labels;
        assert_eq!(labels, vec![0, 0, 0]);
    }

    #[test]
    fn spectral_landmark_reduction_above_the_cap() {
        // > SPECTRAL_MAX_NODES microclusters force the k-means-landmark reduction path.
        let mut rng = SplitMix64::new(11);
        let centers = [[0.0, 0.0], [15.0, 0.0], [0.0, 15.0]];
        let (pts, truth) = blobs(&mut rng, 700, &centers, 0.7);
        let (micros, point_to_micro) = grid_micros(&pts, 0.12);
        assert!(
            micros.len() > SPECTRAL_MAX_NODES,
            "need > cap microclusters"
        );
        let micro_labels = spectral(&micros, 3, 100, 1).labels;
        assert_eq!(micro_labels.len(), micros.len());
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| micro_labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.9, "landmark spectral lost the blobs");
    }

    /// Three groups of very different size. Eigenvector entries scale like `1/sqrt(size)`, so the
    /// raw embedding rows of the small groups are several times longer than the large group's --
    /// which is exactly the magnitude the Ng-Jordan-Weiss row normalization exists to remove.
    fn lopsided_groups() -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(404);
        let spec = [
            ([0.0f64, 0.0], 80usize),
            ([14.0, 0.0], 10),
            ([7.0, 13.0], 5),
        ];
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, (ctr, count)) in spec.iter().enumerate() {
            for _ in 0..*count {
                pts.push(vec![
                    ctr[0] + 0.35 * rng.gauss(),
                    ctr[1] + 0.35 * rng.gauss(),
                ]);
                truth.push(c);
            }
        }
        (pts, truth)
    }

    #[test]
    fn the_embedding_is_row_normalized_before_the_final_kmeans() {
        // Without the normalization the k-means at the end separates rows by *length* -- the large
        // group's short rows cluster together with whichever small group is nearest the origin --
        // so an exact partition, not an ARI threshold, is what makes the step visible.
        let (pts, truth) = lopsided_groups();
        let labels = spectral_core(&pts, 3, 100, 5);
        assert_eq!(n_distinct(&labels), 3, "{labels:?}");
        assert!(
            (ari(&labels, &truth) - 1.0).abs() < 1e-12,
            "ARI = {}",
            ari(&labels, &truth)
        );
    }
}
