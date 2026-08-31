//! Global clustering (BIRCH Phase 3) on the leaf clustering features of a CF-tree.
//!
//! Parametric heads: weighted k-means ([`kmeans`]) and diagonal GMM-EM ([`gmm_diagonal`]).
//! A density head (HDBSCAN-on-CF, [`hdbscan`]) clusters the leaf microclusters directly.

pub mod agglomerative;
pub mod bregman;
pub mod community;
pub mod dcdist;
pub mod fuzzy;
pub mod gmm;
pub mod gmm_toeplitz;
pub(crate) mod graph;
pub mod hdbscan;
pub mod hyperbolic;
pub mod kmeans;
pub(crate) mod knn;
pub mod kprototypes;
pub mod medoid;
pub mod mfa;
pub mod mppca;
pub mod nmf;
pub mod optics;
pub(crate) mod pca;
pub(crate) mod rng;
pub mod scalespace;
pub mod spectral;
pub mod vmf;
pub mod ward;
pub mod watson;
pub mod xmeans;

pub use agglomerative::{
    Agglomerative, BoundedDendrogram, CertificateError, Linkage, agglomerative, agglomerative_auto,
    certificate_radius, dendrogram_below,
};
pub use bregman::{BregmanMixture, bregman_agglomerative, bregman_em, bregman_kmeans};
pub use community::{Community, Objective, leiden};
pub use dcdist::{DcClustering, DcObjective, dc_clustering};
pub use fuzzy::{FuzzyCMeans, fuzzy_cmeans, fuzzy_cmeans_auto};
pub use gmm::{Gmm, GmmFull, gmm_diagonal, gmm_diagonal_auto, gmm_full, gmm_full_auto};
pub use gmm_toeplitz::{
    GmmToeplitz, gmm_toeplitz, gmm_toeplitz_auto, gmm_toeplitz_full, gmm_toeplitz_full_auto,
    gmm_toeplitz_gs, gmm_toeplitz_gs_auto,
};
pub use hdbscan::{Hdbscan, Selection, hdbscan, hdbscan_selected};
pub use hyperbolic::{
    HyperbolicKMeans, f64_working_radius, hyperbolic_kmeans, merge_increase, project_to_sheet,
};
pub use kmeans::{ConstraintError, KMeans, cop_kmeans, kmeans, kmeans_auto};
pub use kprototypes::{MixedCf, kprototypes, nearest_micro, summarize_mixed};
pub use medoid::{MedoidClustering, Pam, dyn_msc, kmedoids};
pub use mfa::{Mfa, mfa, mfa_auto};
pub use mppca::{Mppca, mppca, mppca_auto};
pub use optics::{Reachability, optics};
pub use scalespace::{ScaleSpace, scale_space};
pub use spectral::{Spectral, spectral};
pub use vmf::{Movmf, SphericalKMeans, movmf, movmf_auto, spherical_kmeans};
pub use ward::{WardHac, ward_hac, ward_hac_auto};
pub use watson::{Watson, watson, watson_auto};
pub use xmeans::xmeans;

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

    /// `n_true` isotropic Gaussian blobs in `d` dimensions on a random layout, returned already
    /// summarised as four leaves per blob — the summary a tree would build at a threshold near the
    /// blob radius, which keeps a head reading leaves rather than centroids. Separate from [`blobs`],
    /// which is 2-D only, because a `k`-selector's behaviour is a function of `d` and 2-D is the
    /// hardest case for anything that scores a split against a per-cluster penalty.
    pub fn blob_leaves(
        n_true: usize,
        d: usize,
        per: usize,
        seed: u64,
    ) -> (Vec<Spherical<f64>>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let centers: Vec<Vec<f64>> = (0..n_true)
            .map(|_| (0..d).map(|_| 20.0 * rng.gauss()).collect())
            .collect();
        let (mut micros, mut truth) = (Vec::new(), Vec::new());
        for (c, ctr) in centers.iter().enumerate() {
            for _ in 0..4 {
                let mut f = Spherical::new(d);
                for _ in 0..per / 4 {
                    let p: Vec<f64> = ctr.iter().map(|&m| m + rng.gauss()).collect();
                    f.push(&p, 1.0);
                }
                micros.push(f);
                truth.push(c);
            }
        }
        (micros, truth)
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
