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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Keep the shorter of two `(height, index)` candidates, ties going to the smaller index.
///
/// Associative and commutative, and it reproduces the sequential scan's rule exactly — which is why
/// splitting the scan across threads cannot change a single merge. `<` on the height and `<` on the
/// index give a total order on the pair, so no chunking of the range can prefer a different `j`.
fn nearer<R: Real>(a: (R, usize), b: (R, usize)) -> (R, usize) {
    if b.0 < a.0 || (b.0 == a.0 && b.1 < a.1) {
        b
    } else {
        a
    }
}

/// Scalar reads below which the parallel scan's fork/join costs more than it saves.
///
/// The unit is `candidates × dim`, because one candidate costs a `dim`-vector read: gating on the
/// candidate count alone would fork on a 20-dimensional scan that is over before the threads start.
/// The chain shrinks the active set towards 1, so late scans fall back to the serial fold on their
/// own — which is the point, and what the first shape of this (fork on every scan, whatever its
/// length) got wrong: it returned **1.21×** on 8 threads.
#[cfg(feature = "parallel")]
const PAR_MIN_READS: usize = 1 << 15;

/// Nearest active cluster to `a`, excluding `a` itself: `(Ward height, index)`.
///
/// This scan is the whole of Ward's `O(m²)`: `m − 1` merges, each preceded by at least one pass over
/// the active set, each evaluation reading a `dim`-vector. At `m = 7250` leaves of `d = 784` it was
/// measured at **67 s** against a 6.8 s tree build — the head, not the build, is what a large
/// `max_leaves` fit waits on.
///
/// `active` is the *dense* list of surviving clusters, not a mask over all `m`: a chain that is
/// 90 % through its merges would otherwise pay a full `m`-length pass to find the 10 % still alive,
/// and rayon would split that pass into chunks that are mostly holes.
///
/// With a [`Prefix`] (`dim ≥` [`PRUNE_MIN_DIM`]) `hint`, the chain predecessor, is evaluated
/// first, so the scan starts from an incumbent no farther than the answer, and [`Prefix::exceeds`]
/// drops every candidate whose lower bound is already beyond it. A dropped candidate's height is
/// strictly greater than the incumbent's, so neither the minimum nor its smallest-index tie-break
/// can move, and every survivor's height is the one [`VarianceIncrease`] computes.
fn nearest_active<R: Real, C: ClusterFeature<R>>(
    cf: &[C],
    prefix: Option<&Prefix<R>>,
    active: &[usize],
    a: usize,
    hint: Option<usize>,
) -> (R, usize) {
    let Some(prefix) = prefix else {
        let eval = |&j: &usize| (VarianceIncrease.between(&cf[a], &cf[j]), j);
        #[cfg(feature = "parallel")]
        if active.len().saturating_mul(cf[a].dim()) >= PAR_MIN_READS {
            return active
                .par_iter()
                .filter(|&&j| j != a)
                .map(eval)
                .reduce(|| (R::infinity(), usize::MAX), nearer);
        }
        return active
            .iter()
            .filter(|&&j| j != a)
            .map(eval)
            .fold((R::infinity(), usize::MAX), nearer);
    };
    let exact = |j: usize| (VarianceIncrease.between(&cf[a], &cf[j]), j);
    let start = hint.map_or((R::infinity(), usize::MAX), exact);
    let na = cf[a].weight();
    let step = |best: (R, usize), &j: &usize| {
        if j == a || Some(j) == hint || prefix.exceeds(a, j, na, cf[j].weight(), best.0) {
            best
        } else {
            nearer(best, exact(j))
        }
    };
    #[cfg(feature = "parallel")]
    if active.len().saturating_mul(cf[a].dim()) >= PAR_MIN_READS {
        return active
            .par_iter()
            .fold(|| start, step)
            .reduce(|| start, nearer);
    }
    active.iter().fold(start, step)
}

/// Dimension from which the staged bound is used. Counted in retired instructions on the Ward head
/// over ten Gaussians (`m ≈ 900 / 1 800` leaves, 30 000 rows) it costs `+32 %` at `d = 16` and `+15 %`
/// at 64, and saves `43 %` at 784: below 128 a stage is shorter than one SIMD block and the checks
/// cost more than they skip. On MNIST-20k leaves (`d = 784`, 924 / 1 966 / 3 847 leaves) it saves
/// `−2 %`, `0 %` and costs `+5 %`, since a 64-coordinate stage costs about what its check does; what
/// it saves there is memory traffic. Timed on that head, it takes `0.54×` the time at 1 966 leaves on
/// one thread (L1 misses and cycles both `−45 %`) and at 3 847 leaves on eight, and `1.02×` at 1 966
/// on eight, where the means stay resident in L3.
const PRUNE_MIN_DIM: usize = 128;

/// Leaf means permuted into descending-variance order, for the staged lower bound on a Ward height.
///
/// Every staged prefix of `‖μ_a − μ_b‖²` sums non-negative terms, so in exact arithmetic it is at
/// most the full sum. In floating point the two are summed in different orders, so the prefix is
/// first lowered by a relative margin that covers both roundings, `γ_d` each (Higham, *Accuracy and
/// Stability*, §4.2), plus the absolute error gradual underflow can add. The lowered value is then
/// at most the full sum the kernel returns, and it is scaled by the same `· n_a · n_b / n_ab`
/// sequence [`VarianceIncrease`] applies; rounding is monotone, so the bound stays below the
/// height that sequence would compute.
struct Prefix<R> {
    means: Vec<Vec<R>>,
    order: Vec<u32>,
    stages: Vec<usize>,
    shrink: R,
    floor: R,
}

impl<R: Real> Prefix<R> {
    fn new<C: ClusterFeature<R>>(cf: &[C]) -> Option<Self> {
        let dim = cf.first().map_or(0, |f| f.dim());
        if dim < PRUNE_MIN_DIM {
            return None;
        }
        let raw: Vec<Vec<R>> = cf.iter().map(|f| f.mean().to_vec()).collect();
        let mass: Vec<R> = cf.iter().map(|f| f.weight()).collect();
        let order = crate::assign::variance_order(&raw, Some(&mass), dim);
        let block = crate::assign::default_block(dim);
        let stages = (1..)
            .map(|s| s * block)
            .take_while(|&at| at < dim)
            .chain(std::iter::once(dim))
            .collect();
        let d = R::from_usize(dim).unwrap();
        let mut prefix = Prefix {
            means: Vec::with_capacity(cf.len()),
            order,
            stages,
            shrink: R::one()
                - (d * R::from_f64(4.0).unwrap() + R::from_f64(16.0).unwrap()) * R::epsilon(),
            floor: (d + R::one()) * R::min_positive_value(),
        };
        prefix.means = raw.iter().map(|m| prefix.permuted(m)).collect();
        Some(prefix)
    }

    fn permuted(&self, mean: &[R]) -> Vec<R> {
        self.order.iter().map(|&j| mean[j as usize]).collect()
    }

    /// Re-read cluster `a`'s mean after a merge changed it.
    fn update(&mut self, a: usize, mean: &[R]) {
        self.means[a] = self.permuted(mean);
    }

    /// Whether the Ward height of `(a, b)` is certainly greater than `best`.
    fn exceeds(&self, a: usize, b: usize, na: R, nb: R, best: R) -> bool {
        let nab = na + nb;
        if nab <= R::zero() {
            return false;
        }
        let (x, y) = (&self.means[a], &self.means[b]);
        let mut acc = R::zero();
        let mut lo = 0;
        for &hi in &self.stages {
            acc = acc + crate::kernels::sq_euclidean(&x[lo..hi], &y[lo..hi]);
            lo = hi;
            if (acc * self.shrink - self.floor) * na * nb / nab > best {
                return true;
            }
        }
        false
    }
}

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
    // Kept ascending, and `remove` rather than `swap_remove` is what keeps it so. A chain that
    // restarts takes the smallest surviving index, as the mask scan it replaces did; restarting
    // somewhere else would reorder the merge list, and the height sort that follows is not a total
    // order, so equal heights would come out cut differently.
    let mut active: Vec<usize> = (0..m).collect();
    let mut prefix = Prefix::new(&cf);
    let mut chain: Vec<usize> = Vec::new();
    let mut merges: Vec<Merge<R>> = Vec::with_capacity(m.saturating_sub(1));

    while active.len() > 1 {
        if chain.is_empty() {
            chain.push(active[0]);
        }
        loop {
            let a = *chain.last().unwrap();
            let hint = chain.len().checked_sub(2).map(|i| chain[i]);
            let (best_d, b) = nearest_active(&cf, prefix.as_ref(), &active, a, hint);
            if chain.len() >= 2 && chain[chain.len() - 2] == b {
                // a and b are reciprocal nearest neighbours → merge b into a.
                chain.pop();
                chain.pop();
                let other = cf[b].clone();
                cf[a].merge(&other);
                if let Some(prefix) = prefix.as_mut() {
                    prefix.update(a, cf[a].mean());
                }
                if let Ok(at) = active.binary_search(&b) {
                    active.remove(at);
                }
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

/// Ward-HAC with automatic cluster count: score every horizontal cut in `[k_min, k_max]` by the
/// Calinski–Harabasz variance-ratio criterion and keep the best.
///
/// This used to cut at the largest *relative* jump in merge height, which is the elbow criterion
/// wearing a dendrogram — and Schubert, *Stop using the elbow criterion for k-means* (SIGKDD
/// Explorations 25(1), 2023) is a direct argument against it: the elbow has no null model, is not
/// scale-free, and on the paper's own examples picks a `k` that the variance ratio does not. CH is
/// the first alternative the paper names, it is exact on cluster features
/// ([`crate::validity::calinski_harabasz`]), and scoring 19 cuts costs `O(k_max·m·d)` — nothing
/// against the `O(m²)` dendrogram that already had to be built.
///
/// CH is undefined at `k = 1`, so this cannot report "no structure"; `n_clusters = 0` on a mixture
/// head, where BIC *can*, remains the criterion for that question.
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

    // `k_lo` is the fallback for the degenerate sweeps CH cannot score: a single feature, or a
    // range that collapses below k = 2.
    let mut best = labels_at(m, &merges, m - k_lo);
    let mut best_score = f64::NEG_INFINITY;
    for k in k_lo.max(2)..=k_hi {
        let labels = labels_at(m, &merges, m - k);
        let score = crate::validity::calinski_harabasz(features, &labels, k);
        if score > best_score {
            best_score = score;
            best = labels;
        }
    }
    WardHac { labels: best }
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
    fn auto_k_survives_the_nested_scales_the_elbow_criterion_cannot_see() {
        // Two far groups of two nearby subclusters each. The tallest *relative* jump in merge
        // height is the one that joins the far groups, so the elbow criterion this selector used
        // to run reports k = 2 on every seed; the variance ratio reports the true 4.
        for seed in [1u64, 2, 3] {
            let mut rng = SplitMix64::new(seed);
            let centers = [[0.0, 0.0], [2.0, 0.0], [30.0, 0.0], [32.0, 0.0]];
            let (pts, truth) = blobs(&mut rng, 400, &centers, 0.4);
            let (micros, point_to_micro) = grid_micros(&pts, 0.3);
            let w = ward_hac_auto(&micros, 1, 8);
            let labels: Vec<usize> = point_to_micro.iter().map(|&m| w.labels[m]).collect();
            let k: HashSet<usize> = labels.iter().copied().collect();
            assert_eq!(k.len(), 4, "seed {seed} selected k = {}", k.len());
            assert!(ari(&labels, &truth) > 0.95, "seed {seed}");
        }
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

    #[test]
    fn the_nearer_candidate_is_the_lower_height_then_the_lower_index_either_way_round() {
        for (a, b, want) in [
            ((2.0, 0), (1.0, 5), (1.0, 5)),
            ((1.0, 2), (1.0, 3), (1.0, 2)),
            ((1.0, 3), (1.0, 2), (1.0, 2)),
        ] {
            assert_eq!(nearer(a, b), want, "{a:?} {b:?}");
            assert_eq!(nearer(b, a), want, "{b:?} {a:?}");
        }
    }

    #[test]
    fn a_scan_long_enough_to_fork_matches_brute_force_ward() {
        // 200 leaves of 200 dimensions: the first scans read 40 000 scalars, past `PAR_MIN_READS`,
        // so the parallel arm decides the early merges whenever the feature is built in.
        use crate::feature::Spherical;
        let mut rng = SplitMix64::new(9);
        let feats: Vec<Spherical<f64>> = (0..200)
            .map(|i| {
                let p: Vec<f64> = (0..200)
                    .map(|j| f64::from(u8::from(j == i % 4)) * 6.0 + rng.gauss())
                    .collect();
                let mut f = <Spherical<f64> as ClusterFeature<f64>>::new(200);
                f.push(&p, 1.0);
                f
            })
            .collect();
        let score = ari(&ward_hac(&feats, 4).labels, &brute_ward(&feats, 4));
        assert!(score > 0.999, "NN-chain vs brute ARI = {score}");
    }

    #[test]
    fn auto_k_keeps_the_smallest_cut_among_equally_scored_ones() {
        // Two sites, two coincident leaves each: the cuts at 2 and 3 clusters both have zero
        // within-cluster scatter, so Calinski-Harabasz scores both +inf and only the tie rule
        // picks. It keeps the first, coarsest cut.
        use crate::feature::Spherical;
        let leaf = |x: f64, y: f64| {
            let mut cf = Spherical::<f64>::new(2);
            cf.push(&[x, y], 1.0);
            cf
        };
        let feats = vec![
            leaf(0.0, 0.0),
            leaf(0.0, 0.0),
            leaf(4.0, 1.0),
            leaf(4.0, 1.0),
        ];
        let labels = ward_hac_auto(&feats, 2, 4).labels;
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    /// The chain as it ran before the prefix bound: every candidate evaluated in full.
    fn full_scan_dendrogram<R: Real, C: ClusterFeature<R>>(features: &[C]) -> Vec<Merge<R>> {
        let mut cf: Vec<C> = features.to_vec();
        let mut active: Vec<usize> = (0..cf.len()).collect();
        let mut chain: Vec<usize> = Vec::new();
        let mut merges = Vec::new();
        while active.len() > 1 {
            if chain.is_empty() {
                chain.push(active[0]);
            }
            loop {
                let a = *chain.last().unwrap();
                let (best_d, b) = active
                    .iter()
                    .filter(|&&j| j != a)
                    .map(|&j| (VarianceIncrease.between(&cf[a], &cf[j]), j))
                    .fold((R::infinity(), usize::MAX), nearer);
                if chain.len() >= 2 && chain[chain.len() - 2] == b {
                    chain.pop();
                    chain.pop();
                    let other = cf[b].clone();
                    cf[a].merge(&other);
                    let at = active.binary_search(&b).unwrap();
                    active.remove(at);
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
        merges.sort_by(|x, y| x.height.partial_cmp(&y.height).unwrap());
        merges
    }

    fn same_merges<R: Real, C: ClusterFeature<R>>(features: &[C], what: &str) {
        let got = dendrogram::<R, C>(features);
        let want = full_scan_dendrogram::<R, C>(features);
        assert_eq!(got.len(), want.len(), "{what}");
        for (t, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g.into, g.from) == (w.into, w.from) && g.height.to_f64() == w.height.to_f64(),
                "{what}: merge {t} is ({}, {}) at {:?}, the full scan's ({}, {}) at {:?}",
                g.into,
                g.from,
                g.height,
                w.into,
                w.from,
                w.height
            );
        }
    }

    fn features<R: Real>(rows: &[Vec<f64>], mass: &[f64]) -> Vec<crate::feature::Spherical<R>> {
        rows.iter()
            .zip(mass)
            .map(|(row, &w)| {
                let x: Vec<R> = row.iter().map(|&v| R::from_f64(v).unwrap()).collect();
                let mut f = crate::feature::Spherical::new(x.len());
                f.push(&x, R::from_f64(w).unwrap());
                f
            })
            .collect()
    }

    #[test]
    fn the_pruned_chain_merges_exactly_as_the_full_scan() {
        let mut rng = SplitMix64::new(11);
        for dim in [2usize, 64, PRUNE_MIN_DIM, 150, 784] {
            let rows: Vec<Vec<f64>> = (0..240)
                .map(|i| {
                    let c = (i % 5) as f64 * 3.0;
                    (0..dim)
                        .map(|_| c + rng.gauss() * (1.0 + 4.0 * rng.next_f64()))
                        .collect()
                })
                .collect();
            let mass: Vec<f64> = (0..240)
                .map(|_| 1.0 + (rng.next_u64() % 50) as f64)
                .collect();
            same_merges::<f64, _>(&features::<f64>(&rows, &mass), &format!("f64, d = {dim}"));
            same_merges::<f32, _>(&features::<f32>(&rows, &mass), &format!("f32, d = {dim}"));
        }
    }

    #[test]
    fn the_pruned_chain_breaks_ties_exactly_as_the_full_scan() {
        // Small integer lattices with repeats: sums are exact, so heights tie to the bit, and a
        // bound that dropped a tied candidate would hand the merge to a larger index.
        let mut rng = SplitMix64::new(3);
        for dim in [2usize, 3, 130] {
            let rows: Vec<Vec<f64>> = (0..200)
                .map(|_| (0..dim).map(|_| (rng.next_u64() % 3) as f64).collect())
                .collect();
            for mass in [
                vec![1.0; 200],
                (0..200).map(|i| (1 + i % 4) as f64).collect(),
            ] {
                same_merges::<f64, _>(&features::<f64>(&rows, &mass), &format!("grid, d = {dim}"));
            }
        }
    }

    #[test]
    fn the_bound_never_rules_out_a_pair_at_its_own_height() {
        // `exceeds(a, b, …, h)` with `h` the height `VarianceIncrease` computes for the same pair
        // must be false, or a pruned candidate could have been the minimum. Coordinates of one
        // magnitude make the two summation orders round apart most often; a spread over 790 binary
        // orders reaches subnormal squares, where only the absolute floor holds.
        let mut rng = SplitMix64::new(5);
        let mut cases = 0usize;
        for (dim, spread) in [PRUNE_MIN_DIM, 200, 784, 3000]
            .into_iter()
            .flat_map(|d| [(d, 2u64), (d, 790)])
        {
            for _ in 0..400 {
                let draw = |rng: &mut SplitMix64| -> f64 {
                    let e = (rng.next_u64() % spread) as i32 - (spread as i32 * 2) / 3;
                    let sign = if rng.next_u64() % 2 == 0 { 1.0 } else { -1.0 };
                    sign * (1.0 + rng.next_f64())
                        * 2f64.powi(e)
                        * if rng.next_u64() % 9 == 0 { 1e-20 } else { 1.0 }
                };
                let rows: Vec<Vec<f64>> = (0..2)
                    .map(|_| (0..dim).map(|_| draw(&mut rng)).collect())
                    .collect();
                let mass = [
                    1.0 + (rng.next_u64() % 30) as f64,
                    1.0 + (rng.next_u64() % 30) as f64,
                ];
                let feats = features::<f64>(&rows, &mass);
                let prefix = Prefix::new(&feats).unwrap();
                let h: f64 = VarianceIncrease.between(&feats[0], &feats[1]);
                assert!(
                    !prefix.exceeds(0, 1, mass[0], mass[1], h),
                    "d = {dim}: the pair is ruled out at its own height {h:e}"
                );
                cases += 1;
            }
        }
        assert_eq!(cases, 3200);
    }
}
