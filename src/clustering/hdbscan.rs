//! HDBSCAN* on leaf clustering features — the density / topological Phase-3b head.
//!
//! Each leaf feature is a weighted point. We build the mutual-reachability graph (single-linkage
//! robustified by a `min_samples` core distance), whose 0-dimensional persistence is the
//! single-linkage hierarchy; clusters are then extracted by **mass-weighted stability** (excess
//! of mass), labelling low-stability points as noise (`-1`). This finds non-convex /
//! variable-density clusters and chooses the number of clusters automatically.
//!
//! Working precision is `f64` for the graph/topology math regardless of `R`.

use crate::feature::ClusterFeature;
use crate::types::Real;

/// Result of an HDBSCAN run.
pub struct Hdbscan {
    /// Cluster label per feature; `-1` is noise.
    pub labels: Vec<i64>,
    /// Number of clusters found.
    pub n_clusters: usize,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) -> usize {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return ra;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => {
                self.parent[ra] = rb;
                rb
            }
            std::cmp::Ordering::Greater => {
                self.parent[rb] = ra;
                ra
            }
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
                ra
            }
        }
    }
}

fn collect_leaves(nd: usize, m: usize, children: &[(usize, usize)], out: &mut Vec<usize>) {
    if nd < m {
        out.push(nd);
        return;
    }
    collect_leaves(children[nd].0, m, children, out);
    collect_leaves(children[nd].1, m, children, out);
}

fn new_cluster(
    birth: &mut Vec<f64>,
    stab: &mut Vec<f64>,
    kids: &mut Vec<Vec<usize>>,
    b: f64,
) -> usize {
    birth.push(b);
    stab.push(0.0);
    kids.push(Vec::new());
    birth.len() - 1
}

/// Cluster `features` with HDBSCAN*. `min_samples` sets the core-distance neighbourhood and
/// `min_cluster_size` the smallest admissible cluster.
pub fn hdbscan<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    min_samples: usize,
    min_cluster_size: usize,
) -> Hdbscan {
    let m = features.len();
    if m == 0 {
        return Hdbscan {
            labels: vec![],
            n_clusters: 0,
        };
    }
    if m == 1 {
        return Hdbscan {
            labels: vec![0],
            n_clusters: 1,
        };
    }

    let mu: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|v| v.to_f64().unwrap()).collect())
        .collect();
    let mass: Vec<f64> = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap())
        .collect();
    let dist = |i: usize, j: usize| -> f64 { crate::kernels::sq_euclidean(&mu[i], &mu[j]).sqrt() };

    // core distance = distance to the k-th nearest neighbour
    let k = min_samples.clamp(1, m - 1);
    let mut core = vec![0.0f64; m];
    for (i, ci) in core.iter_mut().enumerate() {
        let mut ds: Vec<f64> = (0..m).filter(|&j| j != i).map(|j| dist(i, j)).collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        *ci = ds[k - 1];
    }
    let mreach = |i: usize, j: usize| -> f64 { core[i].max(core[j]).max(dist(i, j)) };

    // Prim minimum spanning tree over mutual reachability
    let mut in_tree = vec![false; m];
    let mut best = vec![f64::INFINITY; m];
    let mut parent = vec![usize::MAX; m];
    best[0] = 0.0;
    let mut mst: Vec<(f64, usize, usize)> = Vec::with_capacity(m - 1);
    for _ in 0..m {
        let mut u = usize::MAX;
        let mut bu = f64::INFINITY;
        for v in 0..m {
            if !in_tree[v] && best[v] < bu {
                bu = best[v];
                u = v;
            }
        }
        if u == usize::MAX {
            break;
        }
        in_tree[u] = true;
        if parent[u] != usize::MAX {
            mst.push((best[u], parent[u], u));
        }
        for v in 0..m {
            if !in_tree[v] {
                let w = mreach(u, v);
                if w < best[v] {
                    best[v] = w;
                    parent[v] = u;
                }
            }
        }
    }

    // single-linkage dendrogram: leaves 0..m, merges m..2m-1
    let total = 2 * m;
    let mut children: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); total];
    let mut node_dist = vec![0.0f64; total];
    let mut node_mass = vec![0.0f64; total];
    let mut node_size = vec![0usize; total];
    for i in 0..m {
        node_mass[i] = mass[i];
        node_size[i] = 1;
    }
    let mut comp_node: Vec<usize> = (0..m).collect();
    let mut uf = UnionFind::new(m);
    let mut next = m;
    mst.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for &(w, a, b) in &mst {
        let (ra, rb) = (uf.find(a), uf.find(b));
        let (na, nb) = (comp_node[ra], comp_node[rb]);
        let id = next;
        next += 1;
        children[id] = (na, nb);
        node_dist[id] = w;
        node_mass[id] = node_mass[na] + node_mass[nb];
        node_size[id] = node_size[na] + node_size[nb];
        let r = uf.union(ra, rb);
        comp_node[r] = id;
    }
    let root = next - 1;

    // condense + mass-weighted stability
    let lam = |nd: usize| -> f64 {
        if node_dist[nd] > 0.0 {
            1.0 / node_dist[nd]
        } else {
            f64::INFINITY
        }
    };
    let mut birth = Vec::new();
    let mut stab = Vec::new();
    let mut kids: Vec<Vec<usize>> = Vec::new();
    let mut point_cluster = vec![0usize; m];
    new_cluster(&mut birth, &mut stab, &mut kids, 0.0); // root cluster 0

    let mut stack = vec![(root, 0usize)];
    while let Some((nd, c)) = stack.pop() {
        if nd < m {
            continue; // single point — stays in c
        }
        let (l, r) = children[nd];
        let split = lam(nd);
        let lbig = node_size[l] >= min_cluster_size;
        let rbig = node_size[r] >= min_cluster_size;
        if lbig && rbig {
            stab[c] += (split - birth[c]) * node_mass[nd];
            let cl = new_cluster(&mut birth, &mut stab, &mut kids, split);
            let cr = new_cluster(&mut birth, &mut stab, &mut kids, split);
            kids[c].push(cl);
            kids[c].push(cr);
            let mut lp = Vec::new();
            collect_leaves(l, m, &children, &mut lp);
            for &p in &lp {
                point_cluster[p] = cl;
            }
            let mut rp = Vec::new();
            collect_leaves(r, m, &children, &mut rp);
            for &p in &rp {
                point_cluster[p] = cr;
            }
            stack.push((l, cl));
            stack.push((r, cr));
        } else if lbig {
            let mut rp = Vec::new();
            collect_leaves(r, m, &children, &mut rp);
            for &p in &rp {
                stab[c] += (split - birth[c]) * mass[p];
            }
            stack.push((l, c));
        } else if rbig {
            let mut lp = Vec::new();
            collect_leaves(l, m, &children, &mut lp);
            for &p in &lp {
                stab[c] += (split - birth[c]) * mass[p];
            }
            stack.push((r, c));
        } else {
            let mut all = Vec::new();
            collect_leaves(nd, m, &children, &mut all);
            for &p in &all {
                stab[c] += (split - birth[c]) * mass[p];
            }
        }
    }
    let n_cl = birth.len();

    // excess-of-mass selection (root cluster 0 is never selected on its own)
    let mut selected = vec![false; n_cl];
    let mut prop = stab.clone();
    for c in (1..n_cl).rev() {
        let child_stab: f64 = kids[c].iter().map(|&cc| prop[cc]).sum();
        if kids[c].is_empty() || stab[c] >= child_stab {
            selected[c] = true;
            let mut ds = kids[c].clone();
            while let Some(x) = ds.pop() {
                selected[x] = false;
                ds.extend(kids[x].iter().copied());
            }
            prop[c] = stab[c];
        } else {
            prop[c] = child_stab;
        }
    }

    // dense labels for the selected clusters
    let mut cl_parent = vec![usize::MAX; n_cl];
    for (c, kc) in kids.iter().enumerate() {
        for &cc in kc {
            cl_parent[cc] = c;
        }
    }
    let mut label_of = vec![-1i64; n_cl];
    let mut next_label = 0i64;
    for c in 0..n_cl {
        if selected[c] {
            label_of[c] = next_label;
            next_label += 1;
        }
    }
    let mut labels = vec![-1i64; m];
    for (p, lab) in labels.iter_mut().enumerate() {
        let mut c = point_cluster[p];
        loop {
            if selected[c] {
                *lab = label_of[c];
                break;
            }
            if cl_parent[c] == usize::MAX {
                break;
            }
            c = cl_parent[c];
        }
    }

    Hdbscan {
        labels,
        n_clusters: next_label as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, grid_micros, two_moons};
    use crate::feature::Spherical;

    #[test]
    fn hdbscan_separates_two_moons() {
        // k-means/GMM cannot split moons; density/topology can.
        let mut rng = SplitMix64::new(7);
        let (pts, truth) = two_moons(&mut rng, 700, 0.07);
        let (micros, point_to_micro) = grid_micros(&pts, 0.1);
        let res = hdbscan(&micros, 5, 5);
        assert!(res.n_clusters >= 2, "n_clusters = {}", res.n_clusters);
        let labels: Vec<usize> = point_to_micro
            .iter()
            .map(|&mi| {
                if res.labels[mi] < 0 {
                    usize::MAX
                } else {
                    res.labels[mi] as usize
                }
            })
            .collect();
        let score = ari(&labels, &truth);
        assert!(
            score > 0.7,
            "ARI = {score}, n_clusters = {}",
            res.n_clusters
        );
    }

    #[test]
    fn hdbscan_empty_and_single_point() {
        use crate::feature::{ClusterFeature, Spherical};
        let empty: Vec<Spherical<f64>> = Vec::new();
        let r0 = hdbscan(&empty, 5, 5);
        assert!(r0.labels.is_empty() && r0.n_clusters == 0);
        let mut one = Spherical::<f64>::new(2);
        one.push(&[0.0, 0.0], 1.0);
        let r1 = hdbscan(&[one], 5, 5);
        assert_eq!(r1.labels, vec![0]);
        assert_eq!(r1.n_clusters, 1);
    }

    #[test]
    fn hdbscan_labels_isolated_point_as_noise() {
        use crate::feature::{ClusterFeature, Spherical};
        // A micro-cluster in a sparse region must be labelled noise (-1), not forced into a cluster —
        // HDBSCAN*'s defining behaviour vs k-means / GMM.
        let micro = |mx: f64, my: f64| {
            let mut c = Spherical::<f64>::new(2);
            c.push(&[mx, my], 1.0);
            c
        };
        let mut feats: Vec<Spherical<f64>> = Vec::new();
        for (x, y) in [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1), (0.05, 0.05)] {
            feats.push(micro(x, y)); // dense group A near the origin
        }
        for (x, y) in [
            (10.0, 0.0),
            (10.1, 0.0),
            (10.0, 0.1),
            (10.1, 0.1),
            (10.05, 0.05),
        ] {
            feats.push(micro(x, y)); // dense group B near (10, 0)
        }
        feats.push(micro(5.0, 20.0)); // an isolated micro-cluster (index 10)

        let res = hdbscan(&feats, 2, 3);
        assert_eq!(res.labels[10], -1, "isolated micro must be noise");
        assert!(
            res.labels[0] >= 0 && res.labels[5] >= 0,
            "dense groups must cluster"
        );
        assert_ne!(
            res.labels[0], res.labels[5],
            "A and B are distinct clusters"
        );
        assert_eq!(res.n_clusters, 2);
    }

    /// HDBSCAN* re-derived from Campello, Moulavi & Sander (2013). The library builds its minimum
    /// spanning tree with Prim; this one uses Kruskal over every pair, so the two agree only if the
    /// mutual-reachability metric underneath them agrees. From there: the single-linkage dendrogram,
    /// the condensed tree — a merge whose two sides both reach `min_cluster_size` births two
    /// clusters, anything smaller falls out of its parent and pays stability `(λ_split − λ_birth)`
    /// per unit of the mass that left — and excess-of-mass selection, keeping a cluster whenever its
    /// own stability is at least the total its descendants can claim.
    fn reference_hdbscan(
        mu: &[Vec<f64>],
        mass: &[f64],
        min_samples: usize,
        min_cluster_size: usize,
    ) -> Vec<i64> {
        let m = mu.len();
        let dist = |i: usize, j: usize| -> f64 {
            mu[i]
                .iter()
                .zip(&mu[j])
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        };
        let k = min_samples.clamp(1, m - 1);
        let core: Vec<f64> = (0..m)
            .map(|i| {
                let mut ds: Vec<f64> = (0..m).filter(|&j| j != i).map(|j| dist(i, j)).collect();
                ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                ds[k - 1]
            })
            .collect();

        fn root_of(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut c = x;
            while p[c] != c {
                let nxt = p[c];
                p[c] = r;
                c = nxt;
            }
            r
        }

        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..m {
            for j in (i + 1)..m {
                edges.push((core[i].max(core[j]).max(dist(i, j)), i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut uf: Vec<usize> = (0..m).collect();
        let mut mst: Vec<(f64, usize, usize)> = Vec::new();
        for (w, a, b) in edges {
            let (ra, rb) = (root_of(&mut uf, a), root_of(&mut uf, b));
            if ra != rb {
                uf[ra] = rb;
                mst.push((w, a, b));
            }
        }

        let total = 2 * m;
        let mut children = vec![(usize::MAX, usize::MAX); total];
        let mut node_dist = vec![0.0f64; total];
        let mut node_mass = vec![0.0f64; total];
        let mut node_size = vec![0usize; total];
        for i in 0..m {
            node_mass[i] = mass[i];
            node_size[i] = 1;
        }
        let mut uf: Vec<usize> = (0..m).collect();
        let mut comp_node: Vec<usize> = (0..m).collect();
        let mut next = m;
        for &(w, a, b) in &mst {
            let (ra, rb) = (root_of(&mut uf, a), root_of(&mut uf, b));
            let (na, nb) = (comp_node[ra], comp_node[rb]);
            children[next] = (na, nb);
            node_dist[next] = w;
            node_mass[next] = node_mass[na] + node_mass[nb];
            node_size[next] = node_size[na] + node_size[nb];
            uf[ra] = rb;
            comp_node[rb] = next;
            next += 1;
        }
        let root = next - 1;

        let leaves = |nd: usize, children: &[(usize, usize)]| -> Vec<usize> {
            let mut out = Vec::new();
            let mut st = vec![nd];
            while let Some(x) = st.pop() {
                if x < m {
                    out.push(x);
                } else {
                    st.push(children[x].0);
                    st.push(children[x].1);
                }
            }
            out
        };

        let (mut birth, mut stab, mut kids) = (vec![0.0f64], vec![0.0f64], vec![Vec::new()]);
        let mut point_cluster = vec![0usize; m];
        let mut stack = vec![(root, 0usize)];
        while let Some((nd, c)) = stack.pop() {
            if nd < m {
                continue;
            }
            let (l, r) = children[nd];
            let split = if node_dist[nd] > 0.0 {
                1.0 / node_dist[nd]
            } else {
                f64::INFINITY
            };
            let (lbig, rbig) = (
                node_size[l] >= min_cluster_size,
                node_size[r] >= min_cluster_size,
            );
            let fresh =
                |b: f64, birth: &mut Vec<f64>, stab: &mut Vec<f64>, kids: &mut Vec<Vec<usize>>| {
                    birth.push(b);
                    stab.push(0.0);
                    kids.push(Vec::new());
                    birth.len() - 1
                };
            if lbig && rbig {
                stab[c] += (split - birth[c]) * node_mass[nd];
                let cl = fresh(split, &mut birth, &mut stab, &mut kids);
                let cr = fresh(split, &mut birth, &mut stab, &mut kids);
                kids[c].push(cl);
                kids[c].push(cr);
                for p in leaves(l, &children) {
                    point_cluster[p] = cl;
                }
                for p in leaves(r, &children) {
                    point_cluster[p] = cr;
                }
                stack.push((l, cl));
                stack.push((r, cr));
            } else if lbig {
                for p in leaves(r, &children) {
                    stab[c] += (split - birth[c]) * mass[p];
                }
                stack.push((l, c));
            } else if rbig {
                for p in leaves(l, &children) {
                    stab[c] += (split - birth[c]) * mass[p];
                }
                stack.push((r, c));
            } else {
                for p in leaves(nd, &children) {
                    stab[c] += (split - birth[c]) * mass[p];
                }
            }
        }

        let n_cl = birth.len();
        let mut selected = vec![false; n_cl];
        let mut prop = stab.clone();
        for c in (1..n_cl).rev() {
            let child_stab: f64 = kids[c].iter().map(|&cc| prop[cc]).sum();
            if kids[c].is_empty() || stab[c] >= child_stab {
                selected[c] = true;
                let mut ds = kids[c].clone();
                while let Some(x) = ds.pop() {
                    selected[x] = false;
                    ds.extend(kids[x].iter().copied());
                }
                prop[c] = stab[c];
            } else {
                prop[c] = child_stab;
            }
        }
        let mut cl_parent = vec![usize::MAX; n_cl];
        for (c, kc) in kids.iter().enumerate() {
            for &cc in kc {
                cl_parent[cc] = c;
            }
        }
        let mut label_of = vec![-1i64; n_cl];
        let mut nl = 0i64;
        for c in 0..n_cl {
            if selected[c] {
                label_of[c] = nl;
                nl += 1;
            }
        }
        (0..m)
            .map(|p| {
                let mut c = point_cluster[p];
                loop {
                    if selected[c] {
                        return label_of[c];
                    }
                    if cl_parent[c] == usize::MAX {
                        return -1;
                    }
                    c = cl_parent[c];
                }
            })
            .collect()
    }

    /// Nested structure at three scales: two tight blobs a short hop apart, a third far away, and
    /// two stragglers between them. A flat fixture never exercises the condensed tree at all.
    fn nested_micros() -> Vec<Spherical<f64>> {
        let mut rng = SplitMix64::new(1234);
        let mut pts: Vec<Vec<f64>> = Vec::new();
        for c in [[0.0, 0.0], [1.4, 0.0], [9.0, 6.0]] {
            for _ in 0..40 {
                pts.push(vec![c[0] + 0.25 * rng.gauss(), c[1] + 0.25 * rng.gauss()]);
            }
        }
        pts.push(vec![4.5, 3.0]);
        pts.push(vec![5.5, 3.6]);
        grid_micros(&pts, 0.4).0
    }

    #[test]
    fn hdbscan_matches_an_independent_reference_partition() {
        let feats = nested_micros();
        let mu: Vec<Vec<f64>> = feats.iter().map(|f| f.mean().to_vec()).collect();
        let mass: Vec<f64> = feats.iter().map(|f| f.weight()).collect();
        for (ms, mcs) in [(1usize, 3usize), (2, 3), (3, 5), (5, 8), (2, 12)] {
            let got = hdbscan(&feats, ms, mcs);
            let want = reference_hdbscan(&mu, &mass, ms, mcs);
            assert_eq!(
                got.labels.iter().map(|&l| l < 0).collect::<Vec<_>>(),
                want.iter().map(|&l| l < 0).collect::<Vec<_>>(),
                "min_samples {ms}, min_cluster_size {mcs}: the noise set differs"
            );
            let a: Vec<usize> = got.labels.iter().map(|&l| (l + 1) as usize).collect();
            let b: Vec<usize> = want.iter().map(|&l| (l + 1) as usize).collect();
            assert!(
                (ari(&a, &b) - 1.0).abs() < 1e-12,
                "min_samples {ms}, min_cluster_size {mcs}: {:?} vs {:?}",
                got.labels,
                want
            );
            assert_eq!(
                got.n_clusters,
                want.iter()
                    .filter(|&&l| l >= 0)
                    .map(|&l| l as usize)
                    .max()
                    .map_or(0, |x| x + 1),
                "min_samples {ms}, min_cluster_size {mcs}: cluster count"
            );
        }
    }
}
