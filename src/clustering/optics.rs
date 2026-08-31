//! The OPTICS reachability plot over the leaf summary — a density **diagnostic**, not a head.
//!
//! OPTICS (Ankerst, Breunig, Kriegel & Sander 1999) does not return a partition. It returns an
//! *ordering* of the objects and, for each, the distance at which it was reached: plot the second
//! against the first and clusters appear as valleys, separated by peaks at the distance you would
//! have to walk to leave one and enter the next. It is the one output in this crate that answers
//! "what does the density structure look like" rather than "which cluster is this in", which is why
//! it ships as a curve alongside [`gap_statistic`](../../python/betula_cluster/tuning.py) rather
//! than as a `method=`.
//!
//! ## It is the head's own hierarchy, written sideways
//!
//! With no `ε` cutoff, OPTICS's priority-queue sweep **is** Prim's algorithm on the reachability
//! graph, and the reachability values it emits are the weights of the spanning-tree edges in the
//! order Prim added them. So the plot is not an approximation of the density structure HDBSCAN\*
//! extracts — it is the same minimum spanning tree, read in a different order.
//!
//! This module makes that structural rather than incidental: it calls
//! [`mutual_reachability`](super::hdbscan::mutual_reachability), the same function
//! [`hdbscan`](super::hdbscan::hdbscan) builds its hierarchy from, and then walks the resulting MST.
//! Two consequences the tests assert directly:
//!
//! - `reachability[1..]` is a **permutation of the MST edge weights**. Every peak in the plot is a
//!   merge height in the HDBSCAN\* dendrogram, and every merge height is a peak.
//! - Cutting the plot at `ε` — split wherever the reachability exceeds it — gives exactly the
//!   connected components of the MST under `ε`, which is DBSCAN\* at that `ε` once the leaves whose
//!   core distance exceeds `ε` are dropped as noise.
//!
//! The deviation from the 1999 paper is deliberate and is what buys that. Ankerst et al. use the
//! *asymmetric* reachability `max(core(q), d(q, p))` of the point `q` that pulled `p` in; this uses
//! the **mutual** reachability `max(core(p), core(q), d(p, q))` that HDBSCAN\* is defined over. The
//! asymmetric form would give a plot that merely resembles the shipped head's hierarchy. The
//! symmetric one gives its transcript.
//!
//! ## What the leaf summary means for it
//!
//! One position per leaf, not per point, and `min_samples` is counted in points throughout — a leaf
//! contributes its mass, exactly as in [`hdbscan`](super::hdbscan::hdbscan). So the plot's *width*
//! is the leaf budget and its *height* is in the data's own units. A valley 3 leaves wide can hold
//! a hundred thousand points; read the mass, which is returned alongside, before reading the width.

use super::hdbscan::mutual_reachability;
use crate::feature::ClusterFeature;
use crate::types::Real;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// An OPTICS reachability plot over the leaf micro-clusters.
pub struct Reachability {
    /// Leaf indices in the order the sweep reached them. A permutation of `0..m`.
    pub order: Vec<usize>,
    /// Reachability at each *position* of [`Reachability::order`], so `reachability[i]` belongs to
    /// `order[i]`. The first entry is `f64::INFINITY`: nothing reached the starting leaf.
    pub reachability: Vec<f64>,
    /// Core distance per leaf, in the leaves' own indexing — the radius enclosing `min_samples`
    /// points' worth of mass, counting the leaf's own.
    pub core: Vec<f64>,
    /// Mass per leaf, in the leaves' own indexing. A valley's height is a distance; its *weight* is
    /// this, and on a summary the two are not interchangeable.
    pub mass: Vec<f64>,
}

/// The reachability plot of `features` under the mutual reachability HDBSCAN\* uses.
///
/// `min_samples` is the core-distance neighbourhood in **points**; `graph_degree` is `0` for the
/// exact complete graph or a positive floor for the bounded-degree kNN index, with the same meaning
/// and the same `seed` as [`hdbscan_with`](super::hdbscan::hdbscan_with) — pass the values your fit
/// used, or the plot describes a different graph than the head did.
pub fn optics<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    min_samples: usize,
    graph_degree: usize,
    seed: u64,
) -> Reachability {
    let m = features.len();
    let mu: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|v| v.to_f64().unwrap_or(0.0)).collect())
        .collect();
    let mass: Vec<f64> = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap_or(0.0))
        .collect();
    if m == 0 {
        return Reachability {
            order: Vec::new(),
            reachability: Vec::new(),
            core: Vec::new(),
            mass,
        };
    }
    if m == 1 {
        return Reachability {
            order: vec![0],
            reachability: vec![f64::INFINITY],
            core: vec![0.0],
            mass,
        };
    }
    let (mst, core) = mutual_reachability(m, &mu, &mass, min_samples, graph_degree, seed);
    let (order, reachability) = prim_order(m, &mst);
    Reachability {
        order,
        reachability,
        core,
        mass,
    }
}

/// Prim over the spanning tree itself, from leaf `0`.
///
/// Running Prim on a graph and on that graph's own MST pick the same edges — at every step the
/// lightest edge crossing the cut is a tree edge — so this emits the order OPTICS would have emitted
/// over the full reachability graph, at `O(m log m)` instead of `O(m²)`. Ties are broken by leaf
/// index, which is what makes the plot reproducible across runs on the same summary.
fn prim_order(m: usize, mst: &[(f64, usize, usize)]) -> (Vec<usize>, Vec<f64>) {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for &(w, a, b) in mst {
        adj[a].push((b, w));
        adj[b].push((a, w));
    }
    let mut seen = vec![false; m];
    let mut order = Vec::with_capacity(m);
    let mut reach = Vec::with_capacity(m);
    // (weight, vertex): `Reverse` on a tuple orders by weight then by index, which is the tie-break.
    let mut heap: BinaryHeap<Reverse<(OrderedF64, usize)>> = BinaryHeap::new();
    let mut start = 0usize;
    while order.len() < m {
        // The tree is connected by construction, so this outer loop runs once; it is here so that a
        // disconnected input yields every component rather than silently truncating the plot.
        while start < m && seen[start] {
            start += 1;
        }
        if start >= m {
            break;
        }
        heap.push(Reverse((OrderedF64(f64::INFINITY), start)));
        while let Some(Reverse((OrderedF64(w), v))) = heap.pop() {
            if seen[v] {
                continue;
            }
            seen[v] = true;
            order.push(v);
            reach.push(w);
            for &(u, uw) in &adj[v] {
                if !seen[u] {
                    heap.push(Reverse((OrderedF64(uw), u)));
                }
            }
        }
    }
    (order, reach)
}

/// A total order on the reachability weights, which are finite distances plus the leading infinity.
///
/// `f64` is only `PartialOrd`, and a `BinaryHeap` needs `Ord`. `total_cmp` is the right total order
/// here rather than a NaN-panicking wrapper: a NaN weight would mean a NaN coordinate reached the
/// tree, and sorting it to one end is a legible plot rather than an abort.
#[derive(PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::hdbscan::hdbscan;
    use crate::clustering::rng::SplitMix64;
    use crate::feature::Spherical;

    fn leaves(mu: &[[f64; 2]]) -> Vec<Spherical<f64>> {
        mu.iter()
            .map(|p| {
                let mut cf = Spherical::new(2);
                cf.push(p, 1.0);
                cf
            })
            .collect()
    }

    fn blob_leaves(rng: &mut SplitMix64, per: usize, centres: &[[f64; 2]]) -> Vec<Spherical<f64>> {
        let mut out = Vec::new();
        for c in centres {
            for _ in 0..per {
                let mut cf = Spherical::new(2);
                cf.push(
                    &[
                        c[0] + 0.6 * (rng.next_f64() - 0.5),
                        c[1] + 0.6 * (rng.next_f64() - 0.5),
                    ],
                    1.0,
                );
                out.push(cf);
            }
        }
        out
    }

    /// Union-find components of the MST under `eps`, computed straight from the tree. The reference
    /// the plot has to reproduce.
    fn components_below(m: usize, mst: &[(f64, usize, usize)], eps: f64) -> Vec<usize> {
        let mut parent: Vec<usize> = (0..m).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for &(w, a, b) in mst {
            if w <= eps {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra] = rb;
                }
            }
        }
        (0..m).map(|i| find(&mut parent, i)).collect()
    }

    fn mst_of(features: &[Spherical<f64>], min_samples: usize) -> Vec<(f64, usize, usize)> {
        let mu: Vec<Vec<f64>> = features.iter().map(|f| f.mean().to_vec()).collect();
        let mass: Vec<f64> = features.iter().map(|f| f.weight()).collect();
        mutual_reachability(features.len(), &mu, &mass, min_samples, 0, 0).0
    }

    /// Cut the plot at `eps` the way the docs promise: a new segment wherever the reachability
    /// exceeds it. This is the extraction the Python wrapper ships; asserting it here is what ties
    /// the two together.
    fn segments_at(plot: &Reachability, eps: f64) -> Vec<usize> {
        let mut label = vec![0usize; plot.order.len()];
        let mut current = 0usize;
        for (i, &leaf) in plot.order.iter().enumerate() {
            if i > 0 && plot.reachability[i] > eps {
                current += 1;
            }
            label[leaf] = current;
        }
        label
    }

    fn same_partition(a: &[usize], b: &[usize]) -> bool {
        use std::collections::HashMap;
        let mut fwd: HashMap<usize, usize> = HashMap::new();
        let mut rev: HashMap<usize, usize> = HashMap::new();
        for (&x, &y) in a.iter().zip(b) {
            if *fwd.entry(x).or_insert(y) != y || *rev.entry(y).or_insert(x) != x {
                return false;
            }
        }
        true
    }

    #[test]
    fn the_plot_is_a_permutation_of_the_hierarchy_that_hdbscan_cuts() {
        // The whole claim of the module: every peak is a merge height and every merge height is a
        // peak, because the plot is the same spanning tree walked in Prim order. Compare the two
        // multisets rather than the two orders, which is what "the same tree, written sideways"
        // means precisely.
        let mut rng = SplitMix64::new(3);
        let feats = blob_leaves(&mut rng, 25, &[[0.0, 0.0], [7.0, 0.0], [0.0, 7.0]]);
        let plot = optics::<f64, _>(&feats, 5, 0, 0);
        assert_eq!(plot.order.len(), feats.len());
        let mut seen = plot.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..feats.len()).collect::<Vec<_>>());
        assert!(plot.reachability[0].is_infinite());

        let mut from_plot: Vec<f64> = plot.reachability[1..].to_vec();
        let mut from_mst: Vec<f64> = mst_of(&feats, 5).iter().map(|&(w, _, _)| w).collect();
        from_plot.sort_by(f64::total_cmp);
        from_mst.sort_by(f64::total_cmp);
        assert_eq!(from_plot.len(), from_mst.len());
        for (a, b) in from_plot.iter().zip(&from_mst) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn cutting_the_plot_at_eps_is_the_hierarchy_cut_at_eps() {
        // The second half of the claim, and the one a caller acts on: splitting the plot wherever
        // the reachability rises above `eps` must give the components the dendrogram has at that
        // height. Swept over the whole range of the plot so the equality is not a coincidence at
        // one threshold.
        let mut rng = SplitMix64::new(4);
        let feats = blob_leaves(
            &mut rng,
            20,
            &[[0.0, 0.0], [6.0, 0.0], [3.0, 6.0], [12.0, 6.0]],
        );
        let plot = optics::<f64, _>(&feats, 4, 0, 0);
        let mst = mst_of(&feats, 4);
        let hi = plot.reachability[1..].iter().cloned().fold(0.0, f64::max);
        for step in 0..=20 {
            let eps = hi * step as f64 / 20.0;
            assert!(
                same_partition(
                    &segments_at(&plot, eps),
                    &components_below(feats.len(), &mst, eps)
                ),
                "the plot and the tree disagree at eps = {eps}"
            );
        }
    }

    #[test]
    fn the_valleys_are_the_clusters_hdbscan_finds() {
        // The acceptance bar for the diagnostic: on a fixture the head resolves, the deepest three
        // valleys have to be the head's three clusters. Read by cutting at the largest reachability
        // that still leaves three segments, which is what an eye does to the plot.
        let mut rng = SplitMix64::new(5);
        let feats = blob_leaves(&mut rng, 30, &[[0.0, 0.0], [9.0, 0.0], [0.0, 9.0]]);
        let got = hdbscan(&feats, 5, 5);
        assert_eq!(
            got.n_clusters, 3,
            "the fixture must be one the head resolves"
        );

        let plot = optics::<f64, _>(&feats, 5, 0, 0);
        let mut peaks: Vec<f64> = plot.reachability[1..].to_vec();
        peaks.sort_by(f64::total_cmp);
        // Below the two largest merge heights the tree has exactly three components.
        let eps = peaks[peaks.len() - 3];
        let cut = segments_at(&plot, eps);
        assert_eq!(
            cut.iter().collect::<std::collections::HashSet<_>>().len(),
            3
        );
        let labels: Vec<usize> = got.labels.iter().map(|&l| l as usize).collect();
        assert!(
            same_partition(&cut, &labels),
            "the three deepest valleys are not the head's three clusters"
        );
    }

    #[test]
    fn a_gap_in_the_data_is_a_peak_in_the_plot_and_a_dense_run_is_a_valley() {
        // Two dense runs 20 apart on a line: 9 of the 10 steps are the within-run spacing and one is
        // the crossing. The plot must say so — a single peak an order of magnitude above the floor.
        let mut mu: Vec<[f64; 2]> = (0..6).map(|i| [i as f64, 0.0]).collect();
        mu.extend((0..6).map(|i| [20.0 + i as f64, 0.0]));
        let plot = optics::<f64, _>(&leaves(&mu), 1, 0, 0);
        let peaks = &plot.reachability[1..];
        let big = peaks.iter().filter(|&&w| w > 5.0).count();
        assert_eq!(big, 1, "expected exactly one crossing: {peaks:?}");
        assert!(peaks.iter().filter(|&&w| w <= 5.0).all(|&w| w < 1.5));
    }

    #[test]
    fn the_core_distance_convention_is_the_heads_own() {
        // `min_samples` counts the leaf's own mass (Campello Def. 3.1), so at 1 every core distance
        // is zero and the mutual reachability collapses to the plain distance — the plot becomes the
        // single-linkage one. A head and a diagnostic that disagreed here would be describing
        // different neighbourhoods.
        let feats = leaves(&[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0]]);
        let plot = optics::<f64, _>(&feats, 1, 0, 0);
        assert!(plot.core.iter().all(|&c| c == 0.0));
        assert_eq!(plot.order, vec![0, 1, 2, 3]);
        assert_eq!(&plot.reachability[1..], &[1.0, 2.0, 4.0]);

        // At `min_samples = 2` each leaf needs one neighbour, so its core distance is the gap to it,
        // and the reachability floors at the larger of the two ends' gaps.
        let wide = optics::<f64, _>(&feats, 2, 0, 0);
        assert_eq!(wide.core, vec![1.0, 1.0, 2.0, 4.0]);
        assert_eq!(&wide.reachability[1..], &[1.0, 2.0, 4.0]);
    }

    #[test]
    fn the_bounded_degree_graph_gives_the_same_plot_when_it_holds_every_edge() {
        // The index is an approximation of the complete graph, and the one place to check that is
        // where it cannot be approximating: a degree at the leaf count reproduces the exact plot.
        let mut rng = SplitMix64::new(6);
        let feats = blob_leaves(&mut rng, 12, &[[0.0, 0.0], [8.0, 0.0]]);
        let exact = optics::<f64, _>(&feats, 4, 0, 0);
        let graph = optics::<f64, _>(&feats, 4, feats.len() - 1, 7);
        assert_eq!(exact.core, graph.core);
        let mut a: Vec<f64> = exact.reachability[1..].to_vec();
        let mut b: Vec<f64> = graph.reachability[1..].to_vec();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-12, "{x} vs {y}");
        }
    }

    #[test]
    fn the_degenerate_inputs_answer_rather_than_panic() {
        let empty: Vec<Spherical<f64>> = Vec::new();
        let plot = optics::<f64, _>(&empty, 5, 0, 0);
        assert!(plot.order.is_empty() && plot.reachability.is_empty());

        let one = leaves(&[[1.0, 2.0]]);
        let plot = optics::<f64, _>(&one, 5, 0, 0);
        assert_eq!(plot.order, vec![0]);
        assert!(plot.reachability[0].is_infinite());

        // Coincident leaves: every distance is zero, so the plot is flat at zero after the first.
        let same = leaves(&[[3.0, 3.0]; 5]);
        let plot = optics::<f64, _>(&same, 2, 0, 0);
        assert!(plot.reachability[1..].iter().all(|&w| w == 0.0));
        assert_eq!(plot.mass.len(), 5);
    }
}
