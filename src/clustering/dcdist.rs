//! Exact `k`-center and `k`-median in the density-connectivity ultrametric over the leaves.
//!
//! The density-connectivity distance (Beer, Draganov, Hohma, Jahn, Frey & Assent, KDD 2023) is the
//! minimax distance taken over mutual reachability: `dc(a, b)` is the largest edge on the unique
//! path between `a` and `b` in the mutual-reachability minimum spanning tree. It is an
//! **ultrametric**, and in it DBSCAN\*, `k`-center and spectral clustering coincide. This module is
//! the `k`-center and `k`-median half of that equivalence, over the leaf summary.
//!
//! The point of it, next to [`hdbscan`](super::hdbscan): **you ask for `k`, not for
//! `min_cluster_size`.** The density head discovers its own cluster count from the persistence of
//! the same tree; here you name the count and get the provably optimal partition of the leaves
//! under it. Same tree — [`mutual_reachability`](super::hdbscan::mutual_reachability) is shared, as
//! it is with [`optics`](super::optics) — different question asked of it.
//!
//! ## Why both objectives are exact, and why they answer differently
//!
//! **`k`-center.** Minimise `max_i dc(i, c(i))`. In the ultrametric the optimum is reached by
//! deleting the `k − 1` heaviest MST edges: the components that fall out are the clusters, the
//! optimum is the heaviest surviving edge, and any leaf of a component serves as its centre because
//! every leaf of a component is exactly the component's diameter away from the farthest other. No
//! search, `O(m log m)`.
//!
//! **`k`-median.** Minimise `Σ_i w_i · dc(i, c(i))` with `w_i` the leaf's mass. The exactness comes
//! from one observation: in an ultrametric induced by a dendrogram, `dc(i, c)` is the *height of the
//! lowest common ancestor* of `i` and `c`, so a leaf's service cost depends only on **which subtree
//! its nearest centre falls in**, never on which centre it is. That collapses the choice of `k`
//! centres into a knapsack over the dendrogram, solved bottom-up in `O(m·k)`:
//!
//! ```text
//! f(v, j) = cost of the leaves under v when exactly j >= 1 centres sit inside v
//! f(leaf, 1) = 0
//! f(v, j)    = min ( f(a, j) + h_v·W_b ,            // all j centres left of the split
//!                    f(b, j) + h_v·W_a ,            // all j centres right of it
//!                    min_{1<=i<j} f(a, i) + f(b, j-i) )
//! ```
//!
//! with `h_v` the merge height and `W_x` the subtree mass. `j >= 1` is not a restriction: if a
//! subtree holds a centre then every leaf under it is served from inside it, because its LCA with
//! any outside centre is higher. The `h_v·W_x` terms are the leaves of the centre-free child, all of
//! which sit at exactly `h_v` from every centre on the other side. Draganov et al. (NeurIPS 2025)
//! build the same recursion for every `k` at once; this crate wants one `k` at a time and so keeps
//! the plain `O(m·k)` form.
//!
//! The tests do not take that on trust: below `m = 12` both objectives are checked against brute
//! force over every `C(m, k)` centre set, scored on an independently computed dc-dist matrix.
//!
//! ## What the leaf summary does to it, and what it does not fix
//!
//! `k`-median weighs each leaf by its mass; `k`-center is a maximum and therefore **mass-blind by
//! construction**. On a CF summary that is not a detail: an outlier is a low-mass leaf, and to a
//! maximum a leaf of mass 1 and a leaf of mass 10 000 are the same object, so `k`-center spends its
//! budget isolating outliers into singleton clusters. This is the mass-based answer the CURE probe
//! asked for and did not find in shrinkage — it is not a repair of `k`-center but a different
//! objective, and the two are published side by side with the measurement that separates them.
//!
//! **No noise label.** Both objectives partition; every leaf gets a cluster. If you want `-1`, the
//! question you are asking is DBSCAN\*'s and [`hdbscan`](super::hdbscan) is the head for it — over
//! this very same tree.
//!
//! Neither objective admits an automatic `k`: both costs fall monotonically in `k` to 0 at `k = m`,
//! exactly as [`kmedoids`](super::medoid::kmedoids)'s deviation does.

use super::hdbscan::{UnionFind, mutual_reachability};
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Absent parent / absent child. The dendrogram is a tree, so one sentinel covers both.
const NONE: usize = usize::MAX;

/// Which optimum to take in the density-connectivity ultrametric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum DcObjective {
    /// Minimise the largest dc-dist any leaf sits from its centre. Exact by deleting the `k − 1`
    /// heaviest MST edges — and mass-blind, because a maximum cannot see a weight.
    Center,
    /// Minimise the mass-weighted total dc-dist. Exact by a dendrogram knapsack, and the only one of
    /// the two that knows a singleton leaf from a populated one.
    Median,
}

/// An optimal `k`-clustering of the leaves in the density-connectivity ultrametric.
pub struct DcClustering {
    /// Cluster label per leaf, in `0..k`. Never `-1`: both objectives partition.
    pub labels: Vec<usize>,
    /// Leaf index of each cluster's centre, indexed by label.
    pub centers: Vec<usize>,
    /// The objective's optimum — the largest served dc-dist for [`DcObjective::Center`], the
    /// mass-weighted total for [`DcObjective::Median`].
    pub cost: f64,
}

/// Optimal `k`-center / `k`-median over the leaf summary in the dc ultrametric.
///
/// `min_samples` and `graph_degree` are [`hdbscan`](super::hdbscan::hdbscan)'s: the core distance
/// counts a leaf's mass, and `graph_degree > 0` bounds the neighbour pass with the approximate
/// proximity graph, which changes the tree and therefore the optimum it is optimal *for*.
/// `k` is clamped to `1..=m`.
pub fn dc_clustering<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    objective: DcObjective,
    min_samples: usize,
    graph_degree: usize,
    seed: u64,
) -> DcClustering {
    let m = features.len();
    let mass: Vec<f64> = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap_or(0.0))
        .collect();
    if m == 0 {
        return DcClustering {
            labels: Vec::new(),
            centers: Vec::new(),
            cost: 0.0,
        };
    }
    let kk = k.clamp(1, m);
    if kk == m {
        return DcClustering {
            labels: (0..m).collect(),
            centers: (0..m).collect(),
            cost: 0.0,
        };
    }
    let mu: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|v| v.to_f64().unwrap_or(0.0)).collect())
        .collect();
    let (mst, _core) = mutual_reachability(m, &mu, &mass, min_samples, graph_degree, seed);
    match objective {
        DcObjective::Center => k_center(m, kk, &mst, &mass),
        DcObjective::Median => {
            let tree = dendrogram(m, &mass, &mst);
            let (centers, cost) = k_median(&tree, kk);
            let labels = assign(&tree, &centers, &mu);
            DcClustering {
                labels,
                centers,
                cost,
            }
        }
    }
}

/// Delete the `k − 1` heaviest MST edges; the surviving components are the optimum.
///
/// The partition this returns is also the nearest-centre partition of the centres it returns: a
/// within-component dc-dist is a surviving edge and a cross-component one is a deleted edge, and
/// every deleted edge is heavier. So there is nothing to reconcile between the fit and its centres.
fn k_center(m: usize, k: usize, mst: &[(f64, usize, usize)], mass: &[f64]) -> DcClustering {
    let mut edges = mst.to_vec();
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    let keep = (m - k).min(edges.len());
    let mut uf = UnionFind::new(m);
    for &(_, a, b) in &edges[..keep] {
        uf.union(a, b);
    }
    let cost = if keep == 0 { 0.0 } else { edges[keep - 1].0 };

    let mut label_of = vec![NONE; m];
    let mut labels = vec![0usize; m];
    let mut centers: Vec<usize> = Vec::new();
    for i in 0..m {
        let root = uf.find(i);
        let lab = if label_of[root] == NONE {
            label_of[root] = centers.len();
            centers.push(i);
            centers.len() - 1
        } else {
            label_of[root]
        };
        labels[i] = lab;
        // Any leaf of a component is an optimal centre for it, so the choice is free; take the
        // heaviest, which is the one a caller reading `centers` would have wanted named.
        if mass[i] > mass[centers[lab]] {
            centers[lab] = i;
        }
    }
    DcClustering {
        labels,
        centers,
        cost,
    }
}

/// The single-linkage dendrogram over the MST: `2m − 1` nodes, leaves `0..m`.
struct Dendrogram {
    m: usize,
    /// Merge height per node; `0` at a leaf.
    height: Vec<f64>,
    /// Children per node; `[NONE, NONE]` at a leaf.
    kids: Vec<[usize; 2]>,
    parent: Vec<usize>,
    /// Total mass under each node.
    mass: Vec<f64>,
    /// Leaf count under each node.
    size: Vec<usize>,
    /// Leaves in depth-first order.
    order: Vec<usize>,
    /// Half-open slice of `order` covered by each node.
    span: Vec<(usize, usize)>,
    root: usize,
}

/// Kruskal over the MST: merging in increasing weight *is* the single-linkage dendrogram, and every
/// node is created after both of its children, so every later pass can run as a forward loop.
fn dendrogram(m: usize, mass: &[f64], mst: &[(f64, usize, usize)]) -> Dendrogram {
    let mut edges = mst.to_vec();
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));

    let n = 2 * m - 1;
    let mut height = Vec::with_capacity(n);
    let mut kids = Vec::with_capacity(n);
    let mut msum = Vec::with_capacity(n);
    let mut size = Vec::with_capacity(n);
    height.extend(std::iter::repeat_n(0.0, m));
    kids.extend(std::iter::repeat_n([NONE; 2], m));
    msum.extend_from_slice(mass);
    size.extend(std::iter::repeat_n(1usize, m));

    let mut uf = UnionFind::new(m);
    let mut node_of: Vec<usize> = (0..m).collect();
    for (w, a, b) in edges {
        let (ra, rb) = (uf.find(a), uf.find(b));
        if ra == rb {
            continue;
        }
        let (na, nb) = (node_of[ra], node_of[rb]);
        let v = height.len();
        height.push(w);
        kids.push([na, nb]);
        msum.push(msum[na] + msum[nb]);
        size.push(size[na] + size[nb]);
        node_of[uf.union(a, b)] = v;
    }
    let root = node_of[uf.find(0)];

    let mut parent = vec![NONE; height.len()];
    for (v, ch) in kids.iter().enumerate() {
        if ch[0] != NONE {
            parent[ch[0]] = v;
            parent[ch[1]] = v;
        }
    }

    // Iterative, because a single-linkage dendrogram over a chain is `m` deep.
    let mut order = Vec::with_capacity(m);
    let mut span = vec![(0usize, 0usize); height.len()];
    let mut stack = vec![(root, false)];
    while let Some((v, closing)) = stack.pop() {
        if closing {
            span[v].1 = order.len();
        } else if kids[v][0] == NONE {
            span[v] = (order.len(), order.len() + 1);
            order.push(v);
        } else {
            span[v].0 = order.len();
            stack.push((v, true));
            stack.push((kids[v][1], false));
            stack.push((kids[v][0], false));
        }
    }

    Dendrogram {
        m,
        height,
        kids,
        parent,
        mass: msum,
        size,
        order,
        span,
        root,
    }
}

/// Which way the knapsack sent the centres at a node.
#[derive(Clone, Copy)]
enum Pick {
    Left,
    Right,
    Split(usize),
}

/// The `O(m·k)` dendrogram knapsack, and the centre set it chose.
fn k_median(d: &Dendrogram, k: usize) -> (Vec<usize>, f64) {
    let n = d.height.len();
    let cap: Vec<usize> = (0..n).map(|v| k.min(d.size[v])).collect();
    let mut f: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut pick: Vec<Vec<Pick>> = Vec::with_capacity(n);

    for v in 0..n {
        let c = cap[v];
        if d.kids[v][0] == NONE {
            f.push(vec![f64::INFINITY, 0.0]);
            pick.push(vec![Pick::Left; 2]);
            continue;
        }
        let (a, b) = (d.kids[v][0], d.kids[v][1]);
        let h = d.height[v];
        let mut fv = vec![f64::INFINITY; c + 1];
        let mut pv = vec![Pick::Left; c + 1];
        for j in 1..=c {
            // At least one arm always applies, so `best` never stays infinite: `j <= cap[a]` gives
            // the first, `j <= cap[b]` the second, and if neither holds then `j > size_a` and
            // `j > size_b`, which puts `ja = size_a` inside the split loop's range and its guard.
            let mut best = f64::INFINITY;
            let mut how = Pick::Left;
            if j <= cap[a] {
                let t = f[a][j] + h * d.mass[b];
                if t < best {
                    best = t;
                    how = Pick::Left;
                }
            }
            if j <= cap[b] {
                let t = f[b][j] + h * d.mass[a];
                if t < best {
                    best = t;
                    how = Pick::Right;
                }
            }
            for ja in 1..j {
                if ja <= cap[a] && j - ja <= cap[b] {
                    let t = f[a][ja] + f[b][j - ja];
                    if t < best {
                        best = t;
                        how = Pick::Split(ja);
                    }
                }
            }
            fv[j] = best;
            pv[j] = how;
        }
        f.push(fv);
        pick.push(pv);
    }

    let mut centers = Vec::with_capacity(k);
    let mut stack = vec![(d.root, k)];
    while let Some((v, j)) = stack.pop() {
        if d.kids[v][0] == NONE {
            centers.push(v);
            continue;
        }
        match pick[v][j] {
            Pick::Left => stack.push((d.kids[v][0], j)),
            Pick::Right => stack.push((d.kids[v][1], j)),
            Pick::Split(ja) => {
                stack.push((d.kids[v][0], ja));
                stack.push((d.kids[v][1], j - ja));
            }
        }
    }
    centers.sort_unstable();
    (centers, f[d.root][k])
}

/// `dc(i, c)` for every leaf `i`, from one walk up the dendrogram.
///
/// The sibling subtrees along `c`'s root path partition every other leaf, and each such subtree sits
/// at exactly the height of the node that joined it — so every entry is written exactly once and the
/// whole vector costs `O(m)`, not `O(m log m)` lookups.
fn dc_from(d: &Dendrogram, c: usize, out: &mut [f64]) {
    let mut cur = c;
    while d.parent[cur] != NONE {
        let p = d.parent[cur];
        let sib = if d.kids[p][0] == cur {
            d.kids[p][1]
        } else {
            d.kids[p][0]
        };
        let (lo, hi) = d.span[sib];
        for &leaf in &d.order[lo..hi] {
            out[leaf] = d.height[p];
        }
        cur = p;
    }
    out[c] = 0.0;
}

/// Nearest centre in the ultrametric, ties broken by Euclidean distance then by centre index.
///
/// Ties are structural rather than accidental: when a leaf's lowest centre-bearing ancestor holds
/// several centres, the ultrametric puts all of them at the same distance, and the objective is
/// indifferent between them. It is the *partition* that is not, so the tie-break is named here and
/// is deterministic — the reported `cost` is the optimum whichever way it falls.
fn assign(d: &Dendrogram, centers: &[usize], mu: &[Vec<f64>]) -> Vec<usize> {
    let mut best = vec![f64::INFINITY; d.m];
    let mut lab = vec![0usize; d.m];
    let mut dist = vec![0.0f64; d.m];
    for (idx, &c) in centers.iter().enumerate() {
        dc_from(d, c, &mut dist);
        for i in 0..d.m {
            if dist[i] < best[i] {
                best[i] = dist[i];
                lab[i] = idx;
            } else if dist[i] == best[i]
                && sq_euclidean(&mu[i], &mu[c]) < sq_euclidean(&mu[i], &mu[centers[lab[i]]])
            {
                lab[i] = idx;
            }
        }
    }
    lab
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::testutil::ari;
    use crate::feature::{ClusterFeature, Spherical};

    const MS: usize = 2;

    fn leaves(pts: &[[f64; 2]], w: f64) -> Vec<Spherical<f64>> {
        pts.iter()
            .map(|p| {
                let mut cf = Spherical::new(2);
                cf.push(p, w);
                cf
            })
            .collect()
    }

    fn weighted(pts: &[([f64; 2], f64)]) -> Vec<Spherical<f64>> {
        pts.iter()
            .map(|(p, w)| {
                let mut cf = Spherical::new(2);
                cf.push(p, *w);
                cf
            })
            .collect()
    }

    /// The dc-dist matrix, computed independently of the dendrogram: Floyd-Warshall in the minimax
    /// semiring over the MST. Slow and obviously correct, which is the point.
    fn dc_matrix(m: usize, mst: &[(f64, usize, usize)]) -> Vec<Vec<f64>> {
        let mut d = vec![vec![f64::INFINITY; m]; m];
        for (i, row) in d.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        for &(w, a, b) in mst {
            d[a][b] = d[a][b].min(w);
            d[b][a] = d[a][b];
        }
        for k in 0..m {
            for i in 0..m {
                for j in 0..m {
                    let via = d[i][k].max(d[k][j]);
                    if via < d[i][j] {
                        d[i][j] = via;
                    }
                }
            }
        }
        d
    }

    fn mst_of(feats: &[Spherical<f64>], min_samples: usize) -> (usize, Vec<(f64, usize, usize)>) {
        let m = feats.len();
        let mu: Vec<Vec<f64>> = feats.iter().map(|f| f.mean().to_vec()).collect();
        let mass: Vec<f64> = feats.iter().map(|f| f.weight()).collect();
        let (mst, _) = mutual_reachability(m, &mu, &mass, min_samples, 0, 0);
        (m, mst)
    }

    /// Every `k`-subset of the leaves, scored on the independent dc matrix.
    fn brute(d: &[Vec<f64>], mass: &[f64], k: usize, objective: DcObjective) -> f64 {
        let m = d.len();
        let mut best = f64::INFINITY;
        let mut idx: Vec<usize> = (0..k).collect();
        loop {
            let mut acc = 0.0f64;
            for i in 0..m {
                let near = idx.iter().map(|&c| d[i][c]).fold(f64::INFINITY, f64::min);
                match objective {
                    DcObjective::Center => acc = acc.max(near),
                    DcObjective::Median => acc += mass[i] * near,
                }
            }
            best = best.min(acc);
            // Next combination in lexicographic order.
            let mut p = k;
            while p > 0 && idx[p - 1] == m - k + p - 1 {
                p -= 1;
            }
            if p == 0 {
                return best;
            }
            idx[p - 1] += 1;
            for q in p..k {
                idx[q] = idx[q - 1] + 1;
            }
        }
    }

    #[test]
    fn both_objectives_match_brute_force_over_every_centre_set() {
        // 11 leaves, uneven masses, in three loose groups plus a far outlier.
        let pts = [
            ([0.0, 0.0], 40.0),
            ([0.4, 0.1], 12.0),
            ([0.1, 0.5], 7.0),
            ([5.0, 0.0], 30.0),
            ([5.3, 0.4], 3.0),
            ([5.1, -0.3], 25.0),
            ([0.0, 9.0], 18.0),
            ([0.5, 9.2], 9.0),
            ([-0.3, 8.7], 21.0),
            ([2.5, 4.5], 1.0),
            ([40.0, 40.0], 1.0),
        ];
        let feats = weighted(&pts);
        let mass: Vec<f64> = pts.iter().map(|(_, w)| *w).collect();
        let (m, mst) = mst_of(&feats, MS);
        let d = dc_matrix(m, &mst);
        for k in 1..=6 {
            for objective in [DcObjective::Center, DcObjective::Median] {
                let got = dc_clustering(&feats, k, objective, MS, 0, 0);
                let want = brute(&d, &mass, k, objective);
                assert!(
                    (got.cost - want).abs() < 1e-12,
                    "k={k} {objective:?}: {} vs brute {want}",
                    got.cost
                );
            }
        }
    }

    #[test]
    fn the_reported_cost_is_the_cost_of_the_reported_partition() {
        // A head that answers one partition and scores another is the defect this asserts against.
        let pts = [
            ([0.0, 0.0], 5.0),
            ([0.3, 0.2], 11.0),
            ([4.0, 0.0], 2.0),
            ([4.2, 0.3], 30.0),
            ([4.1, -0.4], 6.0),
            ([0.0, 7.0], 14.0),
            ([0.4, 7.3], 3.0),
            ([9.0, 9.0], 1.0),
        ];
        let feats = weighted(&pts);
        let mass: Vec<f64> = pts.iter().map(|(_, w)| *w).collect();
        let (m, mst) = mst_of(&feats, MS);
        let d = dc_matrix(m, &mst);
        for k in 1..=5 {
            for objective in [DcObjective::Center, DcObjective::Median] {
                let got = dc_clustering(&feats, k, objective, MS, 0, 0);
                let realised = (0..m).fold(0.0f64, |acc, i| {
                    let served = d[i][got.centers[got.labels[i]]];
                    match objective {
                        DcObjective::Center => acc.max(served),
                        DcObjective::Median => acc + mass[i] * served,
                    }
                });
                assert!(
                    (realised - got.cost).abs() < 1e-12,
                    "k={k} {objective:?}: partition costs {realised}, head reported {}",
                    got.cost
                );
            }
        }
    }

    #[test]
    fn k_center_is_the_hierarchy_cut_at_the_k_th_heaviest_edge() {
        let pts: Vec<[f64; 2]> = (0..30)
            .map(|i| {
                let g = i / 10;
                [
                    g as f64 * 12.0 + (i % 10) as f64 * 0.3,
                    (i % 3) as f64 * 0.2,
                ]
            })
            .collect();
        let feats = leaves(&pts, 4.0);
        let (m, mst) = mst_of(&feats, MS);
        let mut w: Vec<f64> = mst.iter().map(|e| e.0).collect();
        w.sort_by(f64::total_cmp);
        for k in 1..6 {
            let got = dc_clustering(&feats, k, DcObjective::Center, MS, 0, 0);
            assert_eq!(got.centers.len(), k);
            assert_eq!(got.labels.iter().copied().max().unwrap() + 1, k);
            // The optimum is the heaviest surviving edge; with `k − 1` deleted that is `w[m − k − 1]`.
            assert!((got.cost - w[m - k - 1]).abs() < 1e-12);
        }
    }

    #[test]
    fn the_median_objective_sees_a_mass_the_center_objective_cannot() {
        // Three populated groups and one single-point leaf far away. At k = 3, `center` must spend a
        // whole cluster on the outlier — a maximum has no weight to trade against — while `median`
        // weighs 1 unit of mass against thousands and keeps the groups.
        let mut pts: Vec<([f64; 2], f64)> = Vec::new();
        for g in 0..3 {
            for i in 0..6 {
                pts.push(([g as f64 * 10.0 + i as f64 * 0.2, 0.0], 500.0));
            }
        }
        pts.push(([0.0, 60.0], 1.0));
        let feats = weighted(&pts);
        let truth: Vec<usize> = (0..18).map(|i| i / 6).chain(std::iter::once(0)).collect();

        let center = dc_clustering(&feats, 3, DcObjective::Center, MS, 0, 0);
        let median = dc_clustering(&feats, 3, DcObjective::Median, MS, 0, 0);
        assert!(ari(&median.labels, &truth) > 0.99);
        assert!(ari(&center.labels, &truth) < 0.5);
        // ...and the reason, stated as the cluster the outlier is alone in.
        let outlier = center.labels[18];
        assert_eq!(center.labels.iter().filter(|&&l| l == outlier).count(), 1);
    }

    #[test]
    fn the_ultrametric_finds_the_shapes_a_centroid_head_cannot() {
        // Two interlocking arcs: density-connected, not linearly separable, no convex centre.
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for i in 0..40 {
            let t = std::f64::consts::PI * i as f64 / 39.0;
            pts.push([t.cos(), t.sin()]);
            truth.push(0usize);
        }
        for i in 0..40 {
            let t = std::f64::consts::PI * i as f64 / 39.0;
            pts.push([1.0 + t.cos(), -t.sin() + 0.4]);
            truth.push(1usize);
        }
        let feats = leaves(&pts, 3.0);
        for objective in [DcObjective::Center, DcObjective::Median] {
            let got = dc_clustering(&feats, 2, objective, MS, 0, 0);
            assert!(
                ari(&got.labels, &truth) > 0.99,
                "{objective:?} scored {}",
                ari(&got.labels, &truth)
            );
        }
    }

    #[test]
    fn the_cost_falls_monotonically_in_k_which_is_why_there_is_no_auto_k() {
        let pts: Vec<[f64; 2]> = (0..25)
            .map(|i| [(i % 5) as f64 * 3.0, (i / 5) as f64 * 3.0])
            .collect();
        let feats = leaves(&pts, 2.0);
        for objective in [DcObjective::Center, DcObjective::Median] {
            let mut prev = f64::INFINITY;
            for k in 1..=10 {
                let got = dc_clustering(&feats, k, objective, MS, 0, 0);
                assert!(got.cost <= prev + 1e-12, "{objective:?} rose at k={k}");
                prev = got.cost;
            }
        }
    }

    #[test]
    fn every_centre_is_a_leaf_of_the_cluster_it_serves() {
        let pts: Vec<[f64; 2]> = (0..40)
            .map(|i| [(i % 8) as f64 * 1.7, (i / 8) as f64 * 4.0])
            .collect();
        let feats = leaves(&pts, 6.0);
        for objective in [DcObjective::Center, DcObjective::Median] {
            let got = dc_clustering(&feats, 5, objective, MS, 0, 0);
            assert_eq!(got.centers.len(), 5);
            for (lab, &c) in got.centers.iter().enumerate() {
                assert_eq!(got.labels[c], lab);
            }
        }
    }

    #[test]
    fn a_chain_dendrogram_does_not_overflow_the_stack() {
        // Single linkage on evenly spaced points builds a dendrogram `m` deep; the traversals are
        // iterative for exactly this input.
        let pts: Vec<[f64; 2]> = (0..4000).map(|i| [i as f64, 0.0]).collect();
        let feats = leaves(&pts, 1.0);
        let got = dc_clustering(&feats, 7, DcObjective::Median, MS, 0, 0);
        assert_eq!(got.centers.len(), 7);
        assert_eq!(got.labels.len(), 4000);
    }

    #[test]
    fn the_degenerate_inputs_answer_rather_than_panic() {
        let empty: Vec<Spherical<f64>> = Vec::new();
        let got = dc_clustering(&empty, 3, DcObjective::Median, MS, 0, 0);
        assert!(got.labels.is_empty() && got.centers.is_empty() && got.cost == 0.0);

        let one = leaves(&[[1.0, 2.0]], 5.0);
        let got = dc_clustering(&one, 4, DcObjective::Center, MS, 0, 0);
        assert_eq!(got.labels, vec![0]);
        assert_eq!(got.centers, vec![0]);

        // `k` above the leaf count is the identity partition, at zero cost, for both objectives.
        let four = leaves(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [9.0, 9.0]], 2.0);
        for objective in [DcObjective::Center, DcObjective::Median] {
            let got = dc_clustering(&four, 99, objective, MS, 0, 0);
            assert_eq!(got.labels, vec![0, 1, 2, 3]);
            assert_eq!(got.cost, 0.0);
        }
        // `k = 0` is clamped to one cluster rather than answering an empty partition.
        let got = dc_clustering(&four, 0, DcObjective::Median, MS, 0, 0);
        assert_eq!(got.centers.len(), 1);
        assert!(got.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn the_bounded_degree_graph_gives_the_same_answer_when_it_holds_every_edge() {
        let pts: Vec<[f64; 2]> = (0..24)
            .map(|i| [(i % 6) as f64 * 2.0, (i / 6) as f64 * 5.0])
            .collect();
        let feats = leaves(&pts, 3.0);
        for objective in [DcObjective::Center, DcObjective::Median] {
            let exact = dc_clustering(&feats, 4, objective, MS, 0, 0);
            let dense = dc_clustering(&feats, 4, objective, MS, 23, 0);
            assert_eq!(exact.labels, dense.labels);
            assert!((exact.cost - dense.cost).abs() < 1e-12);
        }
    }
}
