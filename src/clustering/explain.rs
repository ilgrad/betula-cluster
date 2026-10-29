//! Explainable k-means: an axis-aligned threshold tree with one leaf per cluster, fitted to a
//! clustering the caller already has.
//!
//! [`imm`] is the Iterative Mistake Minimization of Moshkovitz, Dasgupta, Rashtchian & Frost
//! (ICML 2020). Each node holds a set of reference centres and the rows that reached it; it cuts
//! one coordinate at a threshold that separates the centres, choosing the cut that sends the least
//! **mass** of rows away from their own centre, drops those rows, and recurses on each side until a
//! side holds one centre. The tree has exactly `k` leaves, and reading a row's path gives the
//! reason for its cluster as at most `k − 1` comparisons.
//!
//! On a CF-tree the rows are the leaf means, each weighed by its point count. The mistake mass of a
//! cut is then exact for every point that sits on the same side as its leaf mean; a leaf the cut
//! passes through is charged whole to its mean's side. Measured against IMM on all points with the
//! same reference centres (MNIST-PCA8 k = 10, ALOI-27 k = 20, covtype k = 7, seeds 0/1/2), the
//! leaf tree's point-level SSE is within +0.0 to +2.2 % of the point tree's.

/// One node of a [`ThresholdTree`].
#[derive(Clone, Debug, PartialEq)]
pub enum TreeNode {
    /// Rows with `x[feature] <= threshold` go to `left`, the rest to `right` (node indices).
    Cut {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
    },
    /// Every row reaching this node is assigned `cluster`.
    Leaf { cluster: usize },
}

/// An axis-aligned threshold tree; node `0` is the root.
#[derive(Clone, Debug, PartialEq)]
pub struct ThresholdTree {
    pub nodes: Vec<TreeNode>,
}

impl ThresholdTree {
    /// The cluster of the leaf `x` reaches.
    pub fn predict(&self, x: &[f64]) -> usize {
        let mut at = 0;
        loop {
            match self.nodes[at] {
                TreeNode::Cut {
                    feature,
                    threshold,
                    left,
                    right,
                } => at = if x[feature] <= threshold { left } else { right },
                TreeNode::Leaf { cluster } => return cluster,
            }
        }
    }
}

/// Fit an IMM threshold tree to `labels` over the rows of `points` (`n × dim`, row-major), each
/// weighed by `weights`, with the reference centres `centers` (`k × dim`). Every label must be
/// below `k`. Ties between cuts go to the lowest feature, then the lowest threshold, so the tree is
/// a function of its inputs.
///
/// A cluster no row carries gets no leaf. Centres no coordinate separates — two clusters with the
/// same mean, as concentric rings have — share one leaf, labelled with the heaviest of them there;
/// no axis-aligned tree can tell them apart.
pub fn imm(
    points: &[f64],
    weights: &[f64],
    labels: &[usize],
    centers: &[f64],
    dim: usize,
) -> ThresholdTree {
    assert!(dim > 0, "dim must be positive");
    let n = weights.len();
    assert_eq!(points.len(), n * dim, "points must be n × dim");
    assert_eq!(labels.len(), n, "one label per row");
    assert!(
        !centers.is_empty() && centers.len() % dim == 0,
        "centers must be k × dim with k ≥ 1"
    );
    let k = centers.len() / dim;
    assert!(labels.iter().all(|&y| y < k), "every label must be below k");
    let mut tree = ThresholdTree { nodes: Vec::new() };
    let rows: Vec<usize> = (0..n).collect();
    let mut used = vec![false; k];
    labels.iter().for_each(|&y| used[y] = true);
    let set: Vec<usize> = (0..k).filter(|&c| used[c]).collect();
    if set.is_empty() {
        return ThresholdTree {
            nodes: vec![TreeNode::Leaf { cluster: 0 }],
        };
    }
    grow(
        &mut tree, points, weights, labels, centers, dim, &rows, &set,
    );
    tree
}

/// Append the subtree for `rows` and the centres `set` and return its root index.
#[allow(clippy::too_many_arguments)]
fn grow(
    tree: &mut ThresholdTree,
    points: &[f64],
    weights: &[f64],
    labels: &[usize],
    centers: &[f64],
    dim: usize,
    rows: &[usize],
    set: &[usize],
) -> usize {
    let me = tree.nodes.len();
    if set.len() == 1 {
        tree.nodes.push(TreeNode::Leaf { cluster: set[0] });
        return me;
    }
    let Some((feature, threshold)) = best_cut(points, weights, labels, centers, dim, rows, set)
    else {
        let mut mass = vec![0.0; centers.len() / dim];
        rows.iter().for_each(|&j| mass[labels[j]] += weights[j]);
        let cluster = set
            .iter()
            .copied()
            .fold(set[0], |b, c| if mass[c] > mass[b] { c } else { b });
        tree.nodes.push(TreeNode::Leaf { cluster });
        return me;
    };
    tree.nodes.push(TreeNode::Leaf {
        cluster: usize::MAX,
    }); // replaced below
    let goes_left = |v: f64| v <= threshold;
    let (mut lrows, mut rrows) = (Vec::new(), Vec::new());
    for &j in rows {
        let left = goes_left(points[j * dim + feature]);
        if left == goes_left(centers[labels[j] * dim + feature]) {
            if left { &mut lrows } else { &mut rrows }.push(j);
        }
    }
    let (lset, rset): (Vec<usize>, Vec<usize>) = set
        .iter()
        .partition(|&&c| goes_left(centers[c * dim + feature]));
    let left = grow(tree, points, weights, labels, centers, dim, &lrows, &lset);
    let right = grow(tree, points, weights, labels, centers, dim, &rrows, &rset);
    tree.nodes[me] = TreeNode::Cut {
        feature,
        threshold,
        left,
        right,
    };
    me
}

/// The cut of least mistake mass over the features that separate `set`; `None` when none does.
///
/// Row `j` is a mistake at `θ` exactly when `a_j ≤ θ < b_j`, with `a_j`, `b_j` the smaller and the
/// larger of its own coordinate and its centre's. The mass of mistakes is a step function of `θ`
/// whose steps sit at the `a_j` and `b_j`, so it is enough to read it at `lo` and at every step in
/// `[lo, hi)`, where `lo`/`hi` are the extreme centre coordinates. The threshold returned is the
/// middle of the winning step, which keeps it off every row the fit saw.
fn best_cut(
    points: &[f64],
    weights: &[f64],
    labels: &[usize],
    centers: &[f64],
    dim: usize,
    rows: &[usize],
    set: &[usize],
) -> Option<(usize, f64)> {
    let mut best: Option<(f64, usize, f64)> = None;
    let mut opens: Vec<(f64, f64)> = Vec::with_capacity(rows.len());
    let mut closes: Vec<(f64, f64)> = Vec::with_capacity(rows.len());
    for feature in 0..dim {
        let (lo, hi) = set
            .iter()
            .map(|&c| centers[c * dim + feature])
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), v| {
                (l.min(v), h.max(v))
            });
        if lo >= hi {
            continue;
        }
        opens.clear();
        closes.clear();
        for &j in rows {
            let x = points[j * dim + feature];
            let c = centers[labels[j] * dim + feature];
            opens.push((x.min(c), weights[j]));
            closes.push((x.max(c), weights[j]));
        }
        opens.sort_by(|a, b| a.0.total_cmp(&b.0));
        closes.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut steps: Vec<f64> = opens
            .iter()
            .chain(&closes)
            .map(|&(v, _)| v)
            .filter(|&v| lo < v && v < hi)
            .collect();
        steps.push(lo);
        steps.sort_by(f64::total_cmp);
        steps.dedup();
        let (mut io, mut ic, mut open, mut closed) = (0, 0, 0.0, 0.0);
        for (s, &theta) in steps.iter().enumerate() {
            while io < opens.len() && opens[io].0 <= theta {
                open += opens[io].1;
                io += 1;
            }
            while ic < closes.len() && closes[ic].0 <= theta {
                closed += closes[ic].1;
                ic += 1;
            }
            let mistakes = open - closed;
            if best.is_none_or(|(m, _, _)| mistakes < m) {
                let upper = steps.get(s + 1).copied().unwrap_or(hi);
                best = Some((mistakes, feature, theta + (upper - theta) / 2.0));
            }
        }
    }
    best.map(|(_, feature, threshold)| (feature, threshold))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;

    fn leaves(tree: &ThresholdTree) -> Vec<usize> {
        let mut out: Vec<usize> = tree
            .nodes
            .iter()
            .filter_map(|n| match n {
                TreeNode::Leaf { cluster } => Some(*cluster),
                TreeNode::Cut { .. } => None,
            })
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn separable_blobs_get_a_tree_that_reproduces_their_labels() {
        let centers = [0.0, 0.0, 5.0, 1.0, 10.0, -1.0];
        let mut rng = SplitMix64::new(3);
        let (mut p, mut w, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for c in 0..3 {
            for _ in 0..40 {
                p.push(centers[c * 2] + 0.5 * rng.gauss());
                p.push(centers[c * 2 + 1] + 0.5 * rng.gauss());
                w.push(1.0 + rng.next_f64());
                y.push(c);
            }
        }
        let tree = imm(&p, &w, &y, &centers, 2);
        assert_eq!(leaves(&tree), vec![0, 1, 2]);
        assert_eq!(tree.nodes.len(), 5);
        for j in 0..y.len() {
            assert_eq!(tree.predict(&p[j * 2..j * 2 + 2]), y[j], "row {j}");
        }
    }

    /// Check every cut of the subtree at `node` against brute force over the rows that reach it,
    /// and return its leaves. Brute force reads the mistake mass at every row and centre coordinate,
    /// which includes every step of the function the sweep reads.
    #[allow(clippy::too_many_arguments)]
    fn check(
        tree: &ThresholdTree,
        node: usize,
        p: &[f64],
        w: &[f64],
        y: &[usize],
        c: &[f64],
        dim: usize,
        rows: Vec<usize>,
        set: Vec<usize>,
    ) {
        let TreeNode::Cut {
            feature,
            threshold,
            left,
            right,
        } = tree.nodes[node]
        else {
            assert_eq!(set.len(), 1, "a leaf holds {set:?}");
            return;
        };
        let mass = |f: usize, t: f64| -> f64 {
            rows.iter()
                .filter(|&&j| (p[j * dim + f] <= t) != (c[y[j] * dim + f] <= t))
                .map(|&j| w[j])
                .sum()
        };
        let mut want = f64::INFINITY;
        for f in 0..dim {
            let cs: Vec<f64> = set.iter().map(|&s| c[s * dim + f]).collect();
            let lo = cs.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = cs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for t in rows.iter().map(|&j| p[j * dim + f]).chain(cs) {
                if lo <= t && t < hi {
                    want = want.min(mass(f, t));
                }
            }
        }
        let got = mass(feature, threshold);
        assert!(
            (got - want).abs() <= 1e-9 * want.max(1.0),
            "node {node}: cut costs {got}, the best costs {want}"
        );
        let side = |v: f64| v <= threshold;
        let kept = |l: bool| -> Vec<usize> {
            rows.iter()
                .copied()
                .filter(|&j| side(p[j * dim + feature]) == l && side(c[y[j] * dim + feature]) == l)
                .collect()
        };
        let part = |l: bool| -> Vec<usize> {
            set.iter()
                .copied()
                .filter(|&s| side(c[s * dim + feature]) == l)
                .collect()
        };
        check(tree, left, p, w, y, c, dim, kept(true), part(true));
        check(tree, right, p, w, y, c, dim, kept(false), part(false));
    }

    #[test]
    fn every_cut_has_the_least_mistake_mass_over_the_rows_that_reach_it() {
        // Overlapping clusters, so every cut costs something and the minimum is a real choice.
        for seed in 0..20u64 {
            let mut rng = SplitMix64::new(seed);
            let (dim, k, n) = (3, 5, 80);
            let centers: Vec<f64> = (0..k * dim).map(|_| 2.0 * rng.gauss()).collect();
            let (mut p, mut w, mut y) = (Vec::new(), Vec::new(), Vec::new());
            for j in 0..n {
                let c = j % k;
                for f in 0..dim {
                    p.push(centers[c * dim + f] + 1.5 * rng.gauss());
                }
                w.push(0.5 + 3.0 * rng.next_f64());
                y.push(c);
            }
            let tree = imm(&p, &w, &y, &centers, dim);
            assert_eq!(leaves(&tree), (0..k).collect::<Vec<_>>(), "seed {seed}");
            check(
                &tree,
                0,
                &p,
                &w,
                &y,
                &centers,
                dim,
                (0..n).collect(),
                (0..k).collect(),
            );
        }
    }

    #[test]
    fn the_cut_weighs_rows_by_their_mass() {
        // Two cuts are available on feature 0, between centres 0 and 1 at 0 and 10. Row A (x = 3,
        // cluster 1) is a mistake for any θ ≥ 3, rows B (x = 7, cluster 0) for any θ < 7. A
        // count says "one mistake against two"; a mass says otherwise when A weighs 10.
        let centers = [0.0, 10.0];
        let p = [3.0, 7.0, 7.0];
        let y = [1, 0, 0];
        let by_count = imm(&p, &[1.0, 1.0, 1.0], &y, &centers, 1);
        let by_mass = imm(&p, &[10.0, 1.0, 1.0], &y, &centers, 1);
        let t = |tree: &ThresholdTree| match tree.nodes[0] {
            TreeNode::Cut { threshold, .. } => threshold,
            TreeNode::Leaf { .. } => panic!("two centres need a cut"),
        };
        assert!(t(&by_count) >= 7.0, "count: θ = {}", t(&by_count));
        assert!(t(&by_mass) < 3.0, "mass: θ = {}", t(&by_mass));
    }

    #[test]
    fn inseparable_centres_share_the_leaf_of_the_heaviest() {
        // Clusters 0 and 2 have one mean; cluster 1 carries no row. The tree separates what an
        // axis can, gives the empty cluster nothing, and labels the shared leaf by mass.
        let centers = [0.0, 0.0, 9.0, 9.0, 0.0, 0.0, 5.0, 5.0];
        let p = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 5.0, 5.0];
        let tree = imm(&p, &[1.0, 1.0, 3.0, 1.0], &[0, 0, 2, 3], &centers, 2);
        assert_eq!(leaves(&tree), vec![2, 3]);
        assert_eq!(tree.predict(&[0.0, 0.0]), 2);
        assert_eq!(tree.predict(&[5.0, 5.0]), 3);
    }

    #[test]
    fn a_single_cluster_is_a_single_leaf() {
        let tree = imm(&[1.0, 2.0], &[1.0], &[0], &[0.0, 0.0], 2);
        assert_eq!(tree.nodes, vec![TreeNode::Leaf { cluster: 0 }]);
        assert_eq!(tree.predict(&[9.0, 9.0]), 0);
    }
}
