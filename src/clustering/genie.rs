//! Genie (Gagolewski, Bartoszuk & Cena 2016) over the leaf summary, for a told `k`.
//!
//! Single linkage cuts a minimum spanning tree at its `k − 1` heaviest edges, and on noisy data
//! those edges isolate stray points: the "clusters" it returns are one giant component and a few
//! outliers. Genie keeps the spanning tree and changes only the merge order. The edges are merged
//! in increasing order, as single linkage merges them, except that whenever the cluster sizes are
//! too unequal — their Gini index above a threshold `g` — the next merge must involve one of the
//! smallest clusters: the shortest unmerged edge with an endpoint in a cluster of the smallest
//! size. At `g = 1` the constraint never binds, and the cut is single linkage's; at `g = 0` every
//! merge grows a smallest cluster.
//!
//! **Sizes are masses.** A leaf summarises many points, so a cluster's size here is the point
//! count its leaves carry, not the number of leaves: the Gini index is the one the point-level
//! clustering would see, and a leaf of a thousand points is never mistaken for an outlier.
//!
//! The tree is the Euclidean MST of the leaf means (exact, or over the beam-search index at
//! `graph_degree > 0`, as the density heads build theirs). Written from the paper; no code of the
//! reference implementation was read.
//!
//! Against `leiden` with `leaf_graph="rows"` and `resolution="auto"`, the count given, five row
//! orders by seeds 0/1/2: a tie on the density fixtures of `bench/density_rivals.py` (worst-seed
//! mean ARI 0.831 against 0.833; it wins `aniso` and loses `blobs` and `varied`), and a lead on the
//! `graves` (NCA 0.910 against 0.726) and `other` (0.806 against 0.746) batteries of Gagolewski's
//! suite, where thin and noisy shapes are what the rows graph splits (`docs/USAGE.md`).

use super::hdbscan::{UnionFind, euclidean_mst};
use crate::feature::ClusterFeature;
use crate::types::Real;
use std::cmp::Ordering;

/// The Gini index of `sizes`, sorted ascending: `Σ_{i<j} |s_i − s_j| / ((n − 1) Σ s)`, in `[0, 1]`.
/// `0` when every size is equal, `1` when one holds everything.
fn gini(sizes: &[f64], total: f64) -> f64 {
    let n = sizes.len();
    if n < 2 || total <= 0.0 {
        return 0.0;
    }
    // With ascending order the pairwise sum telescopes: s_i appears with sign +1 against each of
    // the i − 1 smaller sizes and −1 against each of the n − i larger ones.
    let spread: f64 = sizes
        .iter()
        .enumerate()
        .map(|(i, &s)| (2.0 * (i + 1) as f64 - n as f64 - 1.0) * s)
        .sum();
    (spread / ((n - 1) as f64 * total)).clamp(0.0, 1.0)
}

/// Remove one copy of `value` from the ascending `sizes`.
fn remove_size(sizes: &mut Vec<f64>, value: f64) {
    let at = sizes
        .binary_search_by(|s| s.total_cmp(&value))
        .expect("every live cluster's mass is in the size list");
    sizes.remove(at);
}

/// Insert `value` into the ascending `sizes`.
fn insert_size(sizes: &mut Vec<f64>, value: f64) {
    let at = sizes.partition_point(|s| s.total_cmp(&value) == Ordering::Less);
    sizes.insert(at, value);
}

/// Cut the leaves into `k` clusters by Genie with Gini threshold `gini_threshold` (in `[0, 1]`)
/// over the leaf masses. Returns one label per leaf in `0..k`, numbered in order of first
/// appearance; `k` is clamped to `1..=m`.
///
/// Deterministic: the spanning tree's edges are ordered by weight and then by their endpoints, so
/// equal distances and equal masses resolve the same way on every run. `seed` is read only by the
/// beam-search index at `graph_degree > 0`. The merge loop is `O(m²)` in the leaf count, below the
/// `O(m²·d)` of the exact tree.
pub fn genie<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    gini_threshold: f64,
    graph_degree: usize,
    seed: u64,
) -> Vec<usize> {
    let m = features.len();
    if m == 0 {
        return Vec::new();
    }
    let k = k.clamp(1, m);
    if k == m {
        return (0..m).collect();
    }
    let mass: Vec<f64> = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap_or(0.0))
        .collect();
    let mu: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|v| v.to_f64().unwrap_or(0.0)).collect())
        .collect();
    let mut edges = euclidean_mst(&mu, graph_degree, seed);
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));

    let total: f64 = mass.iter().sum();
    let mut held = mass.clone();
    let mut sizes = mass.clone();
    sizes.sort_by(f64::total_cmp);
    let mut uf = UnionFind::new(m);
    let mut merged = vec![false; edges.len()];
    let mut next = 0;
    for _ in k..m {
        while merged[next] {
            next += 1;
        }
        let pick = if gini(&sizes, total) > gini_threshold {
            let low = sizes[0];
            // The edges before `next` are all merged, so the scan starts there; a spanning tree
            // joins two clusters with every edge left, and some cluster has the smallest size.
            (next..edges.len())
                .find(|&e| {
                    let (_, a, b) = edges[e];
                    !merged[e] && held[uf.find(a)].min(held[uf.find(b)]) == low
                })
                .expect("a smallest cluster has an unmerged edge while more than one remain")
        } else {
            next
        };
        merged[pick] = true;
        let (_, a, b) = edges[pick];
        let (ra, rb) = (uf.find(a), uf.find(b));
        let (wa, wb) = (held[ra], held[rb]);
        remove_size(&mut sizes, wa);
        remove_size(&mut sizes, wb);
        insert_size(&mut sizes, wa + wb);
        let root = uf.union(ra, rb);
        held[root] = wa + wb;
    }

    let mut label = vec![usize::MAX; m];
    let mut out = Vec::with_capacity(m);
    let mut fresh = 0;
    for i in 0..m {
        let r = uf.find(i);
        if label[r] == usize::MAX {
            label[r] = fresh;
            fresh += 1;
        }
        out.push(label[r]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::dcdist::{DcObjective, dc_clustering};
    use crate::clustering::testutil::ari;
    use crate::feature::{ClusterFeature, Spherical};

    fn weighted(pts: &[([f64; 2], f64)]) -> Vec<Spherical<f64>> {
        pts.iter()
            .map(|(p, w)| {
                let mut cf = Spherical::new(2);
                cf.push(p, *w);
                cf
            })
            .collect()
    }

    fn unit(pts: &[[f64; 2]]) -> Vec<Spherical<f64>> {
        weighted(&pts.iter().map(|&p| (p, 1.0)).collect::<Vec<_>>())
    }

    #[test]
    fn gini_is_zero_for_equal_sizes_and_one_for_a_single_holder() {
        assert_eq!(gini(&[3.0, 3.0, 3.0], 9.0), 0.0);
        assert_eq!(gini(&[0.0, 0.0, 6.0], 6.0), 1.0);
        // Σ_{i<j} |s_i − s_j| = 1 + 4 + 3 = 8 over (n − 1)·Σ s = 2·8: the definition, by hand.
        assert!((gini(&[1.0, 2.0, 5.0], 8.0) - 8.0 / 16.0).abs() < 1e-15);
        assert_eq!(gini(&[4.0], 4.0), 0.0);
    }

    /// Two dense runs on a line and one far straggler: single linkage spends its second cut on
    /// the straggler; Genie at a low threshold may not, because a one-point cluster next to two
    /// populous ones is exactly the inequality it refuses.
    fn runs_and_a_straggler() -> (Vec<Spherical<f64>>, Vec<usize>) {
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for i in 0..20 {
            pts.push([i as f64 * 0.1, 0.0]);
            truth.push(0);
        }
        for i in 0..20 {
            pts.push([5.0 + i as f64 * 0.1, 0.0]);
            truth.push(1);
        }
        pts.push([20.0, 0.0]);
        truth.push(1);
        (unit(&pts), truth)
    }

    #[test]
    fn single_linkage_isolates_the_straggler_and_genie_does_not() {
        let (leaves, truth) = runs_and_a_straggler();
        let single = genie(&leaves, 2, 1.0, 0, 0);
        assert_eq!(single.iter().filter(|&&l| l == single[40]).count(), 1);
        let g = genie(&leaves, 2, 0.1, 0, 0);
        assert_eq!(ari(&g, &truth), 1.0);
    }

    #[test]
    fn at_threshold_one_it_is_single_linkage() {
        // Single linkage is k-center in the dc ultrametric at `min_samples = 1`: delete the k − 1
        // heaviest MST edges. Two independent routes to the same partition.
        let mut pts = Vec::new();
        let mut s = 7u64;
        for _ in 0..60 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (s >> 11) as f64 / (1u64 << 53) as f64;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = (s >> 11) as f64 / (1u64 << 53) as f64;
            pts.push([x * 10.0, y * 10.0]);
        }
        let leaves = unit(&pts);
        for k in [2, 3, 5, 9] {
            let single = genie(&leaves, k, 1.0, 0, 0);
            let center = dc_clustering(&leaves, k, DcObjective::Center, 1, 0, 0).labels;
            assert_eq!(ari(&single, &center), 1.0, "k = {k}");
        }
    }

    #[test]
    fn the_gini_index_reads_masses_not_leaf_counts() {
        // Two heavy leaves 1 apart and a light one 3 further on. Single linkage joins the heavy
        // pair; by mass the sizes {1000, 1000, 1} are unequal, so the light leaf merges first.
        let far = weighted(&[
            ([0.0, 0.0], 1000.0),
            ([1.0, 0.0], 1000.0),
            ([4.0, 0.0], 1.0),
        ]);
        assert_eq!(genie(&far, 2, 1.0, 0, 0), vec![0, 0, 1]);
        assert_eq!(genie(&far, 2, 0.1, 0, 0), vec![0, 1, 1]);
        // The same geometry by leaf count: three equal sizes, Gini 0, single linkage's cut.
        let flat = weighted(&[([0.0, 0.0], 1.0), ([1.0, 0.0], 1.0), ([4.0, 0.0], 1.0)]);
        assert_eq!(genie(&flat, 2, 0.1, 0, 0), vec![0, 0, 1]);
    }

    #[test]
    fn every_smallest_cluster_competes_for_the_shortest_edge() {
        // Two light leaves, at 0 and at 4.5, are both smallest. The paper merges the shortest edge
        // with an endpoint in *any* smallest cluster: 4 → 4.5, not the first light leaf's 0 → 3.
        let tie = weighted(&[
            ([0.0, 0.0], 1.0),
            ([3.0, 0.0], 1000.0),
            ([4.0, 0.0], 1000.0),
            ([4.5, 0.0], 1.0),
        ]);
        assert_eq!(genie(&tie, 3, 0.1, 0, 0), vec![0, 1, 2, 2]);
    }

    #[test]
    fn it_returns_k_clusters_numbered_in_first_seen_order() {
        let (leaves, _) = runs_and_a_straggler();
        for k in 1..=leaves.len() {
            let labels = genie(&leaves, k, 0.3, 0, 0);
            let mut seen = Vec::new();
            for &l in &labels {
                if !seen.contains(&l) {
                    assert_eq!(l, seen.len(), "labels are numbered in first-seen order");
                    seen.push(l);
                }
            }
            assert_eq!(seen.len(), k);
        }
        // `k` is clamped to the leaf count, and an empty summary has no labels.
        assert_eq!(genie(&leaves, 0, 0.3, 0, 0), vec![0; leaves.len()]);
        assert_eq!(genie(&leaves, 1000, 0.3, 0, 0).len(), leaves.len());
        assert!(genie::<f64, Spherical<f64>>(&[], 3, 0.3, 0, 0).is_empty());
    }

    #[test]
    fn the_index_graph_tree_gives_the_exact_answer_on_separated_runs() {
        let (leaves, truth) = runs_and_a_straggler();
        assert_eq!(ari(&genie(&leaves, 2, 0.1, 8, 3), &truth), 1.0);
    }
}
