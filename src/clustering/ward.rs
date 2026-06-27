//! Ward agglomerative hierarchical clustering on leaf clustering features.
//!
//! Builds the full dendrogram with the nearest-neighbour-chain algorithm — O(m²) time, O(m) extra
//! space — then cuts it. It is *exact* because the CF merge is exact (no Lance-Williams update
//! approximation): the Ward linkage between two clusters is the D4 variance-increase distance
//! `(n_a n_b / (n_a + n_b))·‖μ_a − μ_b‖²` ([`crate::distance::VarianceIncrease`]), and merging two
//! clusters is the exact CF merge. The chain emits reciprocal-NN merges in discovery order (a valid
//! topological order, but *not* globally height-sorted); [`dendrogram`] sorts them by height so the
//! fixed-`k` cut and the auto-`k` height-jump scan both operate on a proper horizontal cut.

use crate::distance::{CFDistance, VarianceIncrease};
use crate::feature::ClusterFeature;
use crate::types::Real;

/// Result of a Ward-HAC run over features.
pub struct WardHac {
    /// Cluster label per input feature (contiguous `0..k`).
    pub labels: Vec<usize>,
}

/// One agglomeration step: cluster `from` is merged into cluster `into` at Ward height `height`.
struct Merge<R> {
    into: usize,
    from: usize,
    height: R,
}

/// Full Ward dendrogram via nearest-neighbour chain; merges come out in non-decreasing height.
fn dendrogram<R: Real, C: ClusterFeature<R>>(features: &[C]) -> Vec<Merge<R>> {
    let m = features.len();
    let mut cf: Vec<C> = features.to_vec();
    let mut active = vec![true; m];
    let mut n_active = m;
    let mut chain: Vec<usize> = Vec::new();
    let mut merges: Vec<Merge<R>> = Vec::with_capacity(m.saturating_sub(1));

    while n_active > 1 {
        if chain.is_empty() {
            chain.push(active.iter().position(|&x| x).unwrap());
        }
        loop {
            let a = *chain.last().unwrap();
            // Nearest active cluster to `a` (excluding `a`); ties broken by smallest index.
            let mut b = usize::MAX;
            let mut best_d = R::infinity();
            for (j, &act) in active.iter().enumerate() {
                if act && j != a {
                    let d = VarianceIncrease.between(&cf[a], &cf[j]);
                    if d < best_d {
                        best_d = d;
                        b = j;
                    }
                }
            }
            if chain.len() >= 2 && chain[chain.len() - 2] == b {
                // a and b are reciprocal nearest neighbours → merge b into a.
                chain.pop();
                chain.pop();
                let other = cf[b].clone();
                cf[a].merge(&other);
                active[b] = false;
                n_active -= 1;
                merges.push(Merge {
                    into: a,
                    from: b,
                    height: best_d,
                });
                break;
            }
            chain.push(b);
        }
    }
    // The NN-chain emits reciprocal-NN merges in *discovery* order, which is a valid topological
    // order (a cluster's sub-merges precede it) but is NOT globally sorted by height. Cutting the
    // dendrogram at a fixed `k` (or scanning height jumps for auto-`k`) needs the merges in
    // non-decreasing height. Ward is monotonic (no inversions), so the height-sorted prefix is
    // downward-closed and the union-find replay in `labels_at` reconstructs the correct horizontal
    // cut regardless of the original discovery order.
    merges.sort_by(|x, y| {
        x.height
            .partial_cmp(&y.height)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merges
}

/// Union-find root with path compression.
fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

/// Apply the first `t` dendrogram merges and return contiguous `0..(m - t)` labels.
fn labels_at<R: Real>(m: usize, merges: &[Merge<R>], t: usize) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..m).collect();
    for mg in merges.iter().take(t) {
        let ra = uf_find(&mut parent, mg.into);
        let rb = uf_find(&mut parent, mg.from);
        if ra != rb {
            parent[rb] = ra;
        }
    }
    let mut label_of = vec![usize::MAX; m];
    let mut next = 0;
    let mut labels = vec![0usize; m];
    for (i, lab) in labels.iter_mut().enumerate() {
        let r = uf_find(&mut parent, i);
        if label_of[r] == usize::MAX {
            label_of[r] = next;
            next += 1;
        }
        *lab = label_of[r];
    }
    labels
}

/// Agglomeratively cluster `features` into `k` clusters by Ward linkage. `k` is clamped to
/// `[1, features.len()]`.
pub fn ward_hac<R: Real, C: ClusterFeature<R>>(features: &[C], k: usize) -> WardHac {
    let m = features.len();
    if m == 0 {
        return WardHac { labels: Vec::new() };
    }
    let k = k.max(1).min(m);
    let merges = dendrogram(features);
    WardHac {
        labels: labels_at(m, &merges, m - k),
    }
}

/// Ward-HAC with automatic cluster count: cut the dendrogram at the largest *relative* jump in
/// merge height within `[k_min, k_max]` (well-separated clusters are expensive to merge, so the
/// height spikes when the cut crosses a true cluster boundary).
pub fn ward_hac_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
) -> WardHac {
    let m = features.len();
    if m == 0 {
        return WardHac { labels: Vec::new() };
    }
    let merges = dendrogram(features);
    let k_hi = k_max.min(m).max(1);
    let k_lo = k_min.max(1).min(k_hi);

    // The merge that reduces k → k-1 is `merges[m - k]`; compare its height to the previous merge.
    // Valid only for k ∈ [2, m-1] (need both a next and a previous merge).
    let lo = k_lo.max(2);
    let hi = k_hi.min(m.saturating_sub(1));
    let mut best_k = k_lo;
    if lo <= hi {
        let tiny = R::from_f64(1e-12).unwrap();
        let mut best_score = R::neg_infinity();
        for k in lo..=hi {
            let t = m - k;
            let score = merges[t].height / merges[t - 1].height.max(tiny);
            if score > best_score {
                best_score = score;
                best_k = k;
            }
        }
    }
    let best_k = best_k.max(1).min(m);
    WardHac {
        labels: labels_at(m, &merges, m - best_k),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros};
    use std::collections::HashSet;

    #[test]
    fn ward_recovers_separated_blobs() {
        let mut rng = SplitMix64::new(7);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.6);
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let w = ward_hac(&micros, 4);
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| w.labels[m]).collect();
        let score = ari(&labels, &truth);
        assert!(score > 0.95, "ARI = {score}");
    }

    #[test]
    fn ward_clamps_k_at_both_ends() {
        let mut rng = SplitMix64::new(1);
        let (pts, _) = blobs(&mut rng, 12, &[[0.0, 0.0], [5.0, 5.0]], 0.3);
        let (micros, _) = grid_micros(&pts, 0.5);
        let m = micros.len();
        assert!(ward_hac(&micros, 1).labels.iter().all(|&l| l == 0));
        let w = ward_hac(&micros, m + 5);
        let distinct: HashSet<usize> = w.labels.iter().copied().collect();
        assert_eq!(distinct.len(), m);
    }

    #[test]
    fn ward_auto_k_recovers_cluster_count() {
        let mut rng = SplitMix64::new(9);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.6);
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let w = ward_hac_auto(&micros, 1, 8);
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| w.labels[m]).collect();
        let k: HashSet<usize> = labels.iter().copied().collect();
        assert_eq!(k.len(), 4, "selected k = {}", k.len());
        assert!(ari(&labels, &truth) > 0.95);
    }

    #[test]
    fn ward_handles_empty_input() {
        let empty: Vec<crate::feature::Spherical<f64>> = Vec::new();
        assert!(ward_hac(&empty, 3).labels.is_empty());
        assert!(ward_hac_auto(&empty, 2, 5).labels.is_empty());
    }

    /// Exact greedy weighted Ward (always merge the globally-minimum variance-increase pair).
    /// O(m^3) — for small `m` it is the ground truth the NN-chain must match.
    fn brute_ward<R: Real, C: ClusterFeature<R>>(features: &[C], k: usize) -> Vec<usize> {
        let m = features.len();
        let mut cf: Vec<C> = features.to_vec();
        let mut alive: Vec<usize> = (0..m).collect();
        let mut group: Vec<Vec<usize>> = (0..m).map(|i| vec![i]).collect();
        while alive.len() > k {
            let (mut bi, mut bj, mut bd) = (0usize, 0usize, R::infinity());
            for x in 0..alive.len() {
                for y in (x + 1)..alive.len() {
                    let (i, j) = (alive[x], alive[y]);
                    let d = VarianceIncrease.between(&cf[i], &cf[j]);
                    if d < bd {
                        bd = d;
                        bi = i;
                        bj = j;
                    }
                }
            }
            let other = cf[bj].clone();
            cf[bi].merge(&other);
            let gj = group[bj].clone();
            group[bi].extend(gj);
            alive.retain(|&z| z != bj);
        }
        let mut labels = vec![0usize; m];
        for (lab, &root) in alive.iter().enumerate() {
            for &p in &group[root] {
                labels[p] = lab;
            }
        }
        labels
    }

    #[test]
    fn ward_matches_bruteforce_on_singletons() {
        use crate::feature::Spherical;
        let mut rng = SplitMix64::new(3);
        let centers = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]];
        let (pts, _) = blobs(&mut rng, 16, &centers, 1.2); // overlapping → stresses merge order
        let feats: Vec<Spherical<f64>> = pts
            .iter()
            .map(|p| {
                let mut f = <Spherical<f64> as ClusterFeature<f64>>::new(2);
                f.push(p, 1.0);
                f
            })
            .collect();
        // Dendrogram heights must be non-decreasing after the fix (Ward has no inversions).
        let merges = dendrogram(&feats);
        assert!(
            merges.windows(2).all(|w| w[1].height + 1e-9 >= w[0].height),
            "dendrogram heights are not sorted"
        );
        // The NN-chain cut must equal exact greedy Ward at every k.
        for k in [2usize, 4, 7, 12] {
            let nn = ward_hac(&feats, k).labels;
            let bf = brute_ward(&feats, k);
            let score = ari(&nn, &bf);
            assert!(score > 0.999, "k={k}: NN-chain vs brute ARI = {score}");
        }
    }
}
