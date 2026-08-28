//! Internal cluster-validity indices, evaluated on the leaf summary rather than on the points.
//!
//! Every index here is a function of within- and between-cluster *squared* distances, and that is
//! the whole reason they belong on cluster features: for any point `c`,
//!
//! ```text
//! Σ_{x ∈ leaf i} ‖x − c‖²  =  S_i + n_i‖μ_i − c‖²
//! ```
//!
//! exactly — no approximation, no sampling. So a sum of squared distances over the whole dataset is
//! a sum over `m` leaves, and an index built only from such sums costs `O(m·k·d)` instead of
//! `O(N·k·d)`, or `O(m·k·d)` instead of `O(N²·d)` for the silhouette family.
//!
//! **What is exact and what is not.** [`calinski_harabasz`] is exact: it is a ratio of two sums of
//! squared distances, and both are exact. [`davies_bouldin`] ships the **RMS** dispersion
//! `√(E‖x − c‖²)`, not the classical mean distance `E‖x − c‖`; the latter is not a function of a
//! cluster feature at all, and Jensen only bounds it (`E‖x−c‖ ≤ √(E‖x−c‖²)`), so this is a
//! deliberate variant rather than an approximation of the original. [`medoid_silhouette`] is a
//! per-*leaf* ratio weighted by leaf mass; a ratio of expectations is not the expectation of a
//! ratio, so it is the index *of the summary*, and it converges to the point-level value only as
//! the leaves shrink.
//!
//! **None of these can say "there is no structure here".** Schubert, *Stop using the elbow
//! criterion for k-means* (SIGKDD Explorations 25(1), 2023, arXiv 2212.12189), Table 1 shows the
//! distance-based indices hallucinating 3–22 clusters in pure noise where BIC correctly returns 1.
//! `calinski_harabasz` is undefined at `k = 1` (`B = 0` over `k − 1 = 0`), which is the same
//! limitation stated honestly: it can rank `k ≥ 2` against each other and nothing more. The
//! `n_clusters = 0` BIC path stays the authority on whether there is one cluster at all.

use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Per-cluster weight and weighted centroid, plus the grand weight and centroid.
struct Centroids<R> {
    weight: Vec<R>,
    centroid: Vec<Vec<R>>,
    total: R,
    grand: Vec<R>,
}

fn centroids<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    labels: &[usize],
    k: usize,
) -> Centroids<R> {
    let d = features[0].dim();
    let mut weight = vec![R::zero(); k];
    let mut centroid = vec![vec![R::zero(); d]; k];
    let mut grand = vec![R::zero(); d];
    let mut total = R::zero();
    for (f, &c) in features.iter().zip(labels) {
        let w = f.weight();
        weight[c] = weight[c] + w;
        total = total + w;
        for (acc, &x) in centroid[c].iter_mut().zip(f.mean()) {
            *acc = *acc + w * x;
        }
        for (acc, &x) in grand.iter_mut().zip(f.mean()) {
            *acc = *acc + w * x;
        }
    }
    for (c, w) in weight.iter().enumerate() {
        if *w > R::zero() {
            for x in &mut centroid[c] {
                *x = *x / *w;
            }
        }
    }
    if total > R::zero() {
        for x in &mut grand {
            *x = *x / total;
        }
    }
    Centroids {
        weight,
        centroid,
        total,
        grand,
    }
}

/// Total within-cluster sum of squares, per cluster. Exact: `Σ_i S_i + n_i‖μ_i − c‖²` is the sum
/// over the cluster's *points*, not over its leaf means.
fn within<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    labels: &[usize],
    k: usize,
    centroid: &[Vec<R>],
) -> Vec<R> {
    let mut w = vec![R::zero(); k];
    for (f, &c) in features.iter().zip(labels) {
        w[c] = w[c] + f.ssd() + f.weight() * sq_euclidean(f.mean(), &centroid[c]);
    }
    w
}

/// Calinski–Harabasz variance-ratio criterion, `(B / (k−1)) / (W / (N−k))`, higher is better.
///
/// Exact on cluster features, and equal to the value the same labelling would score on the raw
/// points. Returns `0.0` outside `2 ≤ k < N`, where the index is undefined — including `k = 1`,
/// which it structurally cannot rank (see the module docs).
pub fn calinski_harabasz<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    labels: &[usize],
    k: usize,
) -> f64 {
    if features.is_empty() || k < 2 {
        return 0.0;
    }
    let c = centroids(features, labels, k);
    let n = c.total.to_f64().unwrap_or(0.0);
    let kf = k as f64;
    if n <= kf {
        return 0.0;
    }
    let between: f64 = (0..k)
        .map(|j| {
            (c.weight[j] * sq_euclidean(&c.centroid[j], &c.grand))
                .to_f64()
                .unwrap_or(0.0)
        })
        .sum();
    let wcss: f64 = within(features, labels, k, &c.centroid)
        .iter()
        .map(|x| x.to_f64().unwrap_or(0.0))
        .sum();
    if wcss <= 0.0 {
        return f64::INFINITY;
    }
    (between / (kf - 1.0)) / (wcss / (n - kf))
}

/// Davies–Bouldin index in its RMS-dispersion form, lower is better.
///
/// `DB = (1/k) Σ_j max_{l≠j} (σ_j + σ_l) / ‖c_j − c_l‖` with `σ_j = √(E‖x − c_j‖²)`. The classical
/// index uses `E‖x − c_j‖`, which no cluster feature carries; see the module docs for why the RMS
/// form is shipped instead of a Jensen bound on the original.
pub fn davies_bouldin<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    labels: &[usize],
    k: usize,
) -> f64 {
    if features.is_empty() || k < 2 {
        return 0.0;
    }
    let c = centroids(features, labels, k);
    let wcss = within(features, labels, k, &c.centroid);
    let sigma: Vec<f64> = (0..k)
        .map(|j| {
            let w = c.weight[j].to_f64().unwrap_or(0.0);
            if w > 0.0 {
                (wcss[j].to_f64().unwrap_or(0.0) / w).max(0.0).sqrt()
            } else {
                0.0
            }
        })
        .collect();
    let live: Vec<usize> = (0..k).filter(|&j| c.weight[j] > R::zero()).collect();
    if live.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0;
    for &j in &live {
        let mut worst = 0.0f64;
        for &l in &live {
            if l == j {
                continue;
            }
            let sep = sq_euclidean(&c.centroid[j], &c.centroid[l])
                .to_f64()
                .unwrap_or(0.0)
                .max(0.0)
                .sqrt();
            // Coincident centroids are infinitely bad, and saying so beats dividing by zero.
            if sep <= 0.0 {
                return f64::INFINITY;
            }
            worst = worst.max((sigma[j] + sigma[l]) / sep);
        }
        acc += worst;
    }
    acc / live.len() as f64
}

/// Medoid silhouette on the leaf summary, in the squared-distance form; higher is better, `1.0` is
/// the ceiling.
///
/// Lenssen & Schubert, *Medoid silhouette clustering with automatic cluster number selection*
/// (Inf. Syst. 120, 2024): replace the `O(N²)` all-pairs silhouette with distances to `k` medoids,
/// `s = 1 − d(x, m_own) / d(x, m_nearest other)`, which is `O(N·k)`.
///
/// Two things are pinned down here that the paper leaves to the metric. The medoid of a cluster is
/// taken in the **squared** metric, where `Σ_{i'} n_{i'}‖μ_i − μ_{i'}‖² = n_j‖μ_i − c_j‖² + const`
/// makes the minimiser exactly the leaf nearest the centroid — an `O(m)` scan instead of the
/// `O(m²)` one the unsquared medoid needs. And the per-leaf distance is the *mean squared* distance
/// from that leaf's points to the medoid point, `‖μ_i − μ_m‖² + S_i/n_i`, which is exact. The
/// leaf's own scatter therefore keeps a leaf off zero even when it *is* the medoid, which is the
/// honest answer: its points are not at the medoid.
///
/// This is an approximation of the exact silhouette *by construction*, not by implementation
/// choice, and no richer leaf model would close the gap: a cluster feature carries the
/// permutation-invariant polynomials of degree ≤ 2 and the silhouette is not one of them. Two point
/// sets with bitwise-identical features can have different pairwise distance sets — see
/// `research/RESULTS-cf-boundary.md` and
/// `feature::tests::two_point_sets_can_share_a_feature_and_not_a_geometry`.
pub fn medoid_silhouette<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    labels: &[usize],
    k: usize,
) -> f64 {
    if features.is_empty() || k < 2 {
        return 0.0;
    }
    let c = centroids(features, labels, k);
    let mut medoid = vec![usize::MAX; k];
    let mut best = vec![R::infinity(); k];
    for (i, (f, &j)) in features.iter().zip(labels).enumerate() {
        let d = sq_euclidean(f.mean(), &c.centroid[j]);
        if d < best[j] {
            best[j] = d;
            medoid[j] = i;
        }
    }
    let live: Vec<usize> = (0..k).filter(|&j| medoid[j] != usize::MAX).collect();
    if live.len() < 2 {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut mass = 0.0;
    for (f, &j) in features.iter().zip(labels) {
        let w = f.weight().to_f64().unwrap_or(0.0);
        if w <= 0.0 {
            continue;
        }
        let spread = (f.ssd() / f.weight()).to_f64().unwrap_or(0.0);
        let to = |m: usize| {
            sq_euclidean(f.mean(), features[m].mean())
                .to_f64()
                .unwrap_or(0.0)
                + spread
        };
        let own = to(medoid[j]);
        let other = live
            .iter()
            .filter(|&&l| l != j)
            .map(|&l| to(medoid[l]))
            .fold(f64::INFINITY, f64::min);
        // Every distance carries the leaf's own spread, so `other` is zero only for a zero-width
        // leaf sitting exactly on a foreign medoid — a coincident-cluster degeneracy, score 0.
        let s = if other > 0.0 { 1.0 - own / other } else { 0.0 };
        acc += w * s;
        mass += w;
    }
    if mass > 0.0 { acc / mass } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::blobs;
    use crate::feature::Spherical;

    /// One leaf per point, so a leaf-summary index has to agree with the point-level one exactly.
    fn leaf_per_point(points: &[Vec<f64>]) -> Vec<Spherical<f64>> {
        points
            .iter()
            .map(|p| {
                let mut f = Spherical::new(p.len());
                f.push(p, 1.0);
                f
            })
            .collect()
    }

    /// Independent re-derivation of Calinski–Harabasz from the raw points, written from the
    /// textbook definition and sharing no code with [`calinski_harabasz`].
    fn reference_ch(points: &[Vec<f64>], labels: &[usize], k: usize) -> f64 {
        let d = points[0].len();
        let n = points.len();
        let mut grand = vec![0.0; d];
        for p in points {
            for (g, x) in grand.iter_mut().zip(p) {
                *g += x;
            }
        }
        for g in &mut grand {
            *g /= n as f64;
        }
        let mut cent = vec![vec![0.0; d]; k];
        let mut cnt = vec![0.0; k];
        for (p, &c) in points.iter().zip(labels) {
            cnt[c] += 1.0;
            for (g, x) in cent[c].iter_mut().zip(p) {
                *g += x;
            }
        }
        for (c, n_c) in cnt.iter().enumerate() {
            if *n_c > 0.0 {
                for g in &mut cent[c] {
                    *g /= n_c;
                }
            }
        }
        let mut b = 0.0;
        for (c, n_c) in cnt.iter().enumerate() {
            b += n_c
                * cent[c]
                    .iter()
                    .zip(&grand)
                    .map(|(a, g)| (a - g) * (a - g))
                    .sum::<f64>();
        }
        let mut w = 0.0;
        for (p, &c) in points.iter().zip(labels) {
            w += p
                .iter()
                .zip(&cent[c])
                .map(|(a, g)| (a - g) * (a - g))
                .sum::<f64>();
        }
        (b / (k as f64 - 1.0)) / (w / (n as f64 - k as f64))
    }

    #[test]
    fn calinski_harabasz_matches_the_point_level_definition() {
        let mut rng = SplitMix64::new(11);
        let centers = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
        let (pts, truth) = blobs(&mut rng, 240, &centers, 0.7);
        let feats = leaf_per_point(&pts);
        let got = calinski_harabasz(&feats, &truth, 3);
        let want = reference_ch(&pts, &truth, 3);
        assert!(
            (got - want).abs() < 1e-9 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }

    #[test]
    fn calinski_harabasz_is_the_same_on_merged_leaves_as_on_the_points() {
        // The claim the module rests on: pooling points into a cluster feature loses nothing,
        // because `S_i + n_i‖μ_i − c‖²` reconstructs the cluster's full sum of squares.
        let mut rng = SplitMix64::new(3);
        let centers = [[0.0, 0.0], [6.0, 0.0], [0.0, 6.0], [6.0, 6.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.5);
        let fine = leaf_per_point(&pts);
        // Pool each cluster's points into pairs of leaves, so the summary is genuinely coarser.
        let mut coarse: Vec<Spherical<f64>> = Vec::new();
        let mut coarse_labels: Vec<usize> = Vec::new();
        for c in 0..4 {
            for half in 0..2 {
                let mut f = Spherical::new(2);
                for (i, p) in pts.iter().enumerate() {
                    if truth[i] == c && i % 2 == half {
                        f.push(p, 1.0);
                    }
                }
                coarse.push(f);
                coarse_labels.push(c);
            }
        }
        let want = calinski_harabasz(&fine, &truth, 4);
        let got = calinski_harabasz(&coarse, &coarse_labels, 4);
        assert!(
            (got - want).abs() < 1e-8 * want,
            "coarse {got} vs fine {want}"
        );
    }

    #[test]
    fn calinski_harabasz_prefers_the_true_grouping_to_a_shuffled_one() {
        let mut rng = SplitMix64::new(7);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.6);
        let feats = leaf_per_point(&pts);
        let shuffled: Vec<usize> = (0..pts.len()).map(|i| i % 3).collect();
        assert!(calinski_harabasz(&feats, &truth, 3) > calinski_harabasz(&feats, &shuffled, 3));
    }

    #[test]
    fn calinski_harabasz_is_zero_where_it_is_undefined() {
        let mut rng = SplitMix64::new(1);
        let (pts, _) = blobs(&mut rng, 30, &[[0.0, 0.0]], 1.0);
        let feats = leaf_per_point(&pts);
        assert_eq!(calinski_harabasz(&feats, &vec![0; pts.len()], 1), 0.0);
        assert_eq!(calinski_harabasz::<f64, Spherical<f64>>(&[], &[], 3), 0.0);
    }

    #[test]
    fn davies_bouldin_and_medoid_silhouette_both_prefer_the_true_grouping() {
        let mut rng = SplitMix64::new(23);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.6);
        let feats = leaf_per_point(&pts);
        let shuffled: Vec<usize> = (0..pts.len()).map(|i| i % 3).collect();
        assert!(davies_bouldin(&feats, &truth, 3) < davies_bouldin(&feats, &shuffled, 3));
        assert!(medoid_silhouette(&feats, &truth, 3) > medoid_silhouette(&feats, &shuffled, 3));
    }

    #[test]
    fn the_medoid_is_the_leaf_nearest_its_centroid() {
        // The O(m) shortcut the squared metric buys: the squared-distance medoid of a cluster is
        // the leaf minimising ‖μ_i − c_j‖², which a brute-force weighted all-pairs scan confirms.
        let mut rng = SplitMix64::new(31);
        let centers = [[0.0, 0.0], [7.0, 2.0]];
        let (pts, truth) = blobs(&mut rng, 120, &centers, 0.9);
        let feats = leaf_per_point(&pts);
        for c in 0..2 {
            let members: Vec<usize> = (0..pts.len()).filter(|&i| truth[i] == c).collect();
            let brute = *members
                .iter()
                .min_by(|&&a, &&b| {
                    let cost = |i: usize| -> f64 {
                        members
                            .iter()
                            .map(|&j| sq_euclidean(feats[i].mean(), feats[j].mean()))
                            .sum()
                    };
                    cost(a).partial_cmp(&cost(b)).unwrap()
                })
                .unwrap();
            let cent = centroids(&feats, &truth, 2);
            let near = *members
                .iter()
                .min_by(|&&a, &&b| {
                    let d = |i: usize| sq_euclidean(feats[i].mean(), &cent.centroid[c]);
                    d(a).partial_cmp(&d(b)).unwrap()
                })
                .unwrap();
            assert_eq!(brute, near);
        }
    }

    #[test]
    fn medoid_silhouette_is_capped_at_one_and_falls_as_clusters_merge() {
        let mut rng = SplitMix64::new(19);
        let centers = [[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.4);
        let feats = leaf_per_point(&pts);
        let good = medoid_silhouette(&feats, &truth, 3);
        assert!(good > 0.9 && good <= 1.0, "{good}");
        // Fold two well-separated clusters together: the merged medoid sits between them, so every
        // member's own distance grows while the nearest foreign medoid does not.
        let merged: Vec<usize> = truth.iter().map(|&c| if c == 2 { 2 } else { 0 }).collect();
        assert!(medoid_silhouette(&feats, &merged, 3) < good);
    }
}
