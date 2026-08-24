//! Global clustering (BIRCH Phase 3) on the leaf clustering features of a CF-tree.
//!
//! Parametric heads: weighted k-means ([`kmeans`]) and diagonal GMM-EM ([`gmm_diagonal`]).
//! A density head (HDBSCAN-on-CF, [`hdbscan`]) clusters the leaf microclusters directly.

pub mod agglomerative;
pub mod community;
pub mod gmm;
pub mod gmm_toeplitz;
pub(crate) mod graph;
pub mod hdbscan;
pub mod kmeans;
pub(crate) mod knn;
pub mod kprototypes;
pub mod mppca;
pub mod nmf;
pub(crate) mod pca;
pub(crate) mod rng;
pub mod scalespace;
pub mod spectral;
pub mod vmf;
pub mod ward;

pub use agglomerative::{Agglomerative, Linkage, agglomerative, agglomerative_auto};
pub use community::{Community, Objective, leiden};
pub use gmm::{Gmm, GmmFull, gmm_diagonal, gmm_diagonal_auto, gmm_full, gmm_full_auto};
pub use gmm_toeplitz::{
    GmmToeplitz, gmm_toeplitz, gmm_toeplitz_auto, gmm_toeplitz_full, gmm_toeplitz_full_auto,
    gmm_toeplitz_gs, gmm_toeplitz_gs_auto,
};
pub use hdbscan::{Hdbscan, hdbscan};
pub use kmeans::{ConstraintError, KMeans, cop_kmeans, kmeans, xmeans};
pub use kprototypes::{MixedCf, kprototypes, nearest_micro, summarize_mixed};
pub use mppca::{Mppca, mppca, mppca_auto};
pub use scalespace::{ScaleSpace, scale_space};
pub use spectral::{Spectral, spectral};
pub use vmf::{Movmf, SphericalKMeans, movmf, movmf_auto, spherical_kmeans};
pub use ward::{WardHac, ward_hac, ward_hac_auto};

#[cfg(test)]
pub(crate) mod testutil {
    use crate::clustering::rng::SplitMix64;
    use crate::feature::{ClusterFeature, Spherical};
    use std::collections::HashMap;

    /// Adjusted Rand Index between two labelings.
    pub fn ari(a: &[usize], b: &[usize]) -> f64 {
        let mut cont: HashMap<(usize, usize), i64> = HashMap::new();
        let mut ra: HashMap<usize, i64> = HashMap::new();
        let mut rb: HashMap<usize, i64> = HashMap::new();
        for (&x, &y) in a.iter().zip(b) {
            *cont.entry((x, y)).or_insert(0) += 1;
            *ra.entry(x).or_insert(0) += 1;
            *rb.entry(y).or_insert(0) += 1;
        }
        let c2 = |x: i64| x * (x - 1) / 2;
        let s: i64 = cont.values().map(|&v| c2(v)).sum();
        let sa: i64 = ra.values().map(|&v| c2(v)).sum();
        let sb: i64 = rb.values().map(|&v| c2(v)).sum();
        let tot = c2(a.len() as i64) as f64;
        let exp = sa as f64 * sb as f64 / tot;
        let mx = 0.5 * (sa as f64 + sb as f64);
        if (mx - exp).abs() < 1e-12 {
            1.0
        } else {
            (s as f64 - exp) / (mx - exp)
        }
    }

    /// 2D Gaussian blobs; returns (points, true labels).
    pub fn blobs(
        rng: &mut SplitMix64,
        per: usize,
        centers: &[[f64; 2]],
        spread: f64,
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for (c, ctr) in centers.iter().enumerate() {
            for _ in 0..per {
                xs.push(vec![
                    ctr[0] + spread * rng.gauss(),
                    ctr[1] + spread * rng.gauss(),
                ]);
                ys.push(c);
            }
        }
        (xs, ys)
    }

    /// Two interleaving half-moons; returns (points, true labels). k-means cannot separate them.
    pub fn two_moons(rng: &mut SplitMix64, per: usize, noise: f64) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..per {
            let t = std::f64::consts::PI * (i as f64) / (per as f64);
            xs.push(vec![
                t.cos() + noise * rng.gauss(),
                t.sin() + noise * rng.gauss(),
            ]);
            ys.push(0);
            xs.push(vec![
                1.0 - t.cos() + noise * rng.gauss(),
                0.5 - t.sin() + noise * rng.gauss(),
            ]);
            ys.push(1);
        }
        (xs, ys)
    }

    /// Grid micro-clustering: each occupied `cell`-sized cell becomes one feature.
    /// Returns (features, point -> feature index).
    pub fn grid_micros(points: &[Vec<f64>], cell: f64) -> (Vec<Spherical<f64>>, Vec<usize>) {
        let mut map: HashMap<(i64, i64), usize> = HashMap::new();
        let mut cfs: Vec<Spherical<f64>> = Vec::new();
        let mut assign = vec![0usize; points.len()];
        for (i, p) in points.iter().enumerate() {
            let key = ((p[0] / cell).round() as i64, (p[1] / cell).round() as i64);
            let idx = *map.entry(key).or_insert_with(|| {
                cfs.push(Spherical::new(2));
                cfs.len() - 1
            });
            cfs[idx].push(p, 1.0);
            assign[i] = idx;
        }
        (cfs, assign)
    }
}
