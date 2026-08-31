//! HDBSCAN* on leaf clustering features — the density / topological Phase-3b head.
//!
//! Each leaf feature is a weighted point. We build the mutual-reachability graph (single-linkage
//! robustified by a `min_samples` core distance), whose 0-dimensional persistence is the
//! single-linkage hierarchy; clusters are then extracted by **mass-weighted stability** (excess
//! of mass), labelling low-stability points as noise (`-1`). This finds non-convex /
//! variable-density clusters and chooses the number of clusters automatically.
//!
//! Flattening that hierarchy is a separate decision from building it, and [`Selection`] carries the
//! two rules on offer: excess of mass at a minimum cluster size the caller names, or PLSCAN, which
//! reads the size off the hierarchy's own leaf-cluster barcode. Which one is better depends on how
//! the clusters differ — measured, and stated with its regime, in the `PLSCAN` section of
//! `bench/RESULTS.md`.
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
    /// Minimum cluster size the flattening used, in points. Echoes the argument under
    /// [`Selection::ExcessOfMass`]; under [`Selection::Persistence`] it is the size the run *chose*,
    /// and is the diagnostic worth logging — everything else about that arm follows from it.
    pub selected_size: f64,
}

/// How a flat clustering is read off the mutual-reachability hierarchy.
///
/// The two arms differ in what they do with the minimum cluster size, not in how the hierarchy is
/// built: both see the same mutual-reachability MST and the same core distances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// HDBSCAN\*'s excess of mass at a **fixed** minimum cluster size, counted in points
    /// (Campello, Moulavi & Sander 2015).
    ExcessOfMass {
        /// Smallest admissible cluster, in points.
        min_cluster_size: usize,
    },
    /// PLSCAN: the minimum cluster size is **chosen, not given** (Bot, McInnes & Aerts,
    /// *Persistent Multiscale Density-based Clustering*, arXiv:2512.16558).
    ///
    /// Raising the minimum cluster size never changes where the hierarchy merges, only which
    /// branches are pruned, so one dendrogram already contains every clustering the parameter can
    /// produce. Each segment therefore has a size interval `(s_min, s_max]` over which it is a
    /// *leaf* cluster, and the total length of those intervals alive at a given size rates that
    /// size. The run reports the clustering at the size that maximises it.
    ///
    /// `min_samples` is the floor of the search, matching the paper's `m_c ≥ k`.
    Persistence,
}

pub(crate) struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    pub(crate) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    pub(crate) fn union(&mut self, a: usize, b: usize) -> usize {
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

/// The leaves under `nd`, where ids below `m` are leaves and the rest are merges.
///
/// Expands each node at most once. On a dendrogram that costs nothing — every node is the child of
/// exactly one merge, so nothing is ever skipped — but it is what bounds the walk when `children`
/// is *not* a tree: a merge whose two sides are the same node expands into two copies of it, and
/// two copies of its children, for `2^depth` visits and an output that outgrows memory. Ids always
/// decrease on the way down, so the blowup is re-visiting rather than cycling and a depth or
/// downward-only bound would not catch it.
fn collect_leaves(nd: usize, m: usize, children: &[(usize, usize)]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut seen = vec![false; children.len()];
    let mut stack = vec![nd];
    while let Some(x) = stack.pop() {
        if x < m {
            out.push(x);
        } else if !std::mem::replace(&mut seen[x], true) {
            stack.push(children[x].0);
            stack.push(children[x].1);
        }
    }
    out
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

/// Radius around each of `m` objects that encloses `min_samples` points' worth of `mass`,
/// **counting the object itself** — so `min_samples = 1` is 0 everywhere and mutual reachability
/// degenerates to `dist`.
///
/// The self-inclusion convention is a genuine split in the field and is therefore chosen here rather
/// than assumed: Campello, Moulavi & Sander's Def. 3.1, ELKI (whose parameter says "including this
/// point") and `sklearn.cluster.HDBSCAN` all include the object; `scikit-learn-contrib/hdbscan`
/// excludes it, so the same argument there means one neighbour more.
///
/// On unit `mass` this is exactly the `min_samples`-th smallest distance. Weighted, it is the
/// smallest radius whose enclosed weight reaches `min_samples`, which is the same density estimate
/// asked of a summary rather than of the points it stands for. When the total mass never reaches
/// `min_samples` the radius saturates at the farthest object, matching the unweighted clamp to `m`.
fn core_distances(
    m: usize,
    min_samples: usize,
    mass: &[f64],
    dist: impl Fn(usize, usize) -> f64,
) -> Vec<f64> {
    let need = min_samples.max(1) as f64;
    (0..m)
        .map(|i| {
            let mut ds: Vec<(f64, f64)> = (0..m).map(|j| (dist(i, j), mass[j])).collect();
            ds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut enclosed = 0.0;
            let mut radius = 0.0;
            for (d, w) in ds {
                enclosed += w;
                radius = d;
                if enclosed >= need {
                    break;
                }
            }
            radius
        })
        .collect()
}

/// Out-degree the proximity graph actually gets, given the caller's request.
///
/// `min_samples` is counted in **points** and a leaf carries `mass[i]` of them, so a graph that must
/// bound the core distances needs about `min_samples / mean_mass` *leaves* per neighbourhood, not
/// `min_samples` of them. Okkels et al. state the requirement as `Ω(minPts)` for unit-weight data;
/// this is that requirement translated into the currency the head is actually counting in. The
/// caller's number is a floor, never a ceiling — asking for too small a degree silently truncates
/// every core distance, which is the one approximation with no upper bound on its error.
fn graph_degree_for(requested: usize, min_samples: usize, mass: &[f64]) -> usize {
    let m = mass.len();
    let mean = mass.iter().sum::<f64>() / m as f64;
    let need = if mean > 0.0 {
        (min_samples as f64 / mean).ceil() as usize
    } else {
        min_samples
    };
    requested.max(need).clamp(1, m - 1)
}

/// [`core_distances`] read off a proximity graph instead of the complete one.
///
/// Identical accumulation, identical self-inclusion convention; the only difference is that the walk
/// stops at the end of `adj[i]` rather than at the end of the dataset. When the neighbourhood's mass
/// never reaches `min_samples` the radius saturates at the farthest *neighbour*, which
/// **under**estimates the true core distance — the opposite direction from the overestimate the
/// paper warns about, and the reason [`graph_degree_for`] raises the degree rather than trusting the
/// caller's.
fn core_distances_from_graph(
    min_samples: usize,
    mass: &[f64],
    adj: &[Vec<(usize, f64)>],
) -> Vec<f64> {
    let need = min_samples.max(1) as f64;
    adj.iter()
        .enumerate()
        .map(|(i, list)| {
            let mut enclosed = mass[i];
            let mut radius = 0.0;
            if enclosed < need {
                for &(j, d) in list {
                    enclosed += mass[j];
                    radius = d;
                    if enclosed >= need {
                        break;
                    }
                }
            }
            radius
        })
        .collect()
}

/// Exact MST of the mutual-reachability weights **restricted to the graph's edges** — Kruskal over
/// `O(m · degree)` candidates rather than Prim over `O(m²)`.
///
/// The graph is connected by construction: every vertex `i ≥ 1` takes at least one random shortcut
/// into `0..i` during the build, so induction on `i` gives a path to vertex 0. The trailing branch
/// that links leftover components is therefore unreachable, and exists so that a future change to
/// the index degrades into a coarse hierarchy rather than into a silently truncated one.
fn mst_over_graph(
    m: usize,
    adj: &[Vec<(usize, f64)>],
    mreach: &impl Fn(usize, usize) -> f64,
) -> Vec<(f64, usize, usize)> {
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for (i, list) in adj.iter().enumerate() {
        for &(j, _) in list {
            if i < j {
                edges.push((mreach(i, j), i, j));
            }
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then((a.1, a.2).cmp(&(b.1, b.2))));
    let mut uf = UnionFind::new(m);
    let mut mst: Vec<(f64, usize, usize)> = Vec::with_capacity(m - 1);
    let mut heaviest = 0.0f64;
    for (w, a, b) in edges {
        heaviest = heaviest.max(w);
        if uf.find(a) != uf.find(b) {
            uf.union(a, b);
            mst.push((w, a, b));
        }
    }
    let mut anchor = usize::MAX;
    for v in 0..m {
        if mst.len() + 1 >= m {
            break;
        }
        if anchor == usize::MAX {
            anchor = v;
        } else if uf.find(anchor) != uf.find(v) {
            uf.union(anchor, v);
            mst.push((heaviest, anchor, v));
        }
    }
    mst
}

/// The mutual-reachability MST over the leaf centroids, and the core distances it was built from.
///
/// Factored out because it is the object *two* readouts share. HDBSCAN\*'s hierarchy is this tree's
/// 0-dimensional persistence; the OPTICS reachability plot in [`optics`](super::optics) is the same
/// tree written down in Prim order. Building it once, here, is what makes the second a readout of
/// the first rather than a second opinion about it.
pub(crate) fn mutual_reachability(
    m: usize,
    mu: &[Vec<f64>],
    mass: &[f64],
    min_samples: usize,
    graph_degree: usize,
    seed: u64,
) -> (Vec<(f64, usize, usize)>, Vec<f64>) {
    let dist = |i: usize, j: usize| -> f64 { crate::kernels::sq_euclidean(&mu[i], &mu[j]).sqrt() };
    if graph_degree == 0 {
        let core = core_distances(m, min_samples, mass, dist);
        let mst = prim_complete(m, &|i, j| core[i].max(core[j]).max(dist(i, j)));
        (mst, core)
    } else {
        let degree = graph_degree_for(graph_degree, min_samples, mass);
        let adj = crate::clustering::knn::build(m, degree, seed, dist);
        let core = core_distances_from_graph(min_samples, mass, &adj);
        let mst = {
            let mreach = |i: usize, j: usize| -> f64 { core[i].max(core[j]).max(dist(i, j)) };
            mst_over_graph(m, &adj, &mreach)
        };
        (mst, core)
    }
}

/// Cluster `features` with HDBSCAN*. `min_samples` sets the core-distance neighbourhood and
/// `min_cluster_size` the smallest admissible cluster.
///
/// Both are counted in **points**, never in features: a feature contributes its `weight()`, so the
/// two arguments mean the same thing whether they are handed one feature per point or a leaf
/// summary of a million of them. On unit weights every quantity below is an integer count and the
/// behaviour is the textbook one.
///
/// `min_samples` **counts the feature's own mass**, matching Campello's Def. 3.1,
/// `sklearn.cluster.HDBSCAN` and ELKI, so `min_samples = 1` leaves every core distance at 0 and
/// HDBSCAN\* degenerates to single linkage. `scikit-learn-contrib/hdbscan` uses the exclusive
/// convention, where the same argument means one neighbour more.
pub fn hdbscan<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    min_samples: usize,
    min_cluster_size: usize,
) -> Hdbscan {
    hdbscan_with(features, min_samples, min_cluster_size, 0, 0)
}

/// [`hdbscan`], with the complete mutual-reachability graph optionally replaced by a bounded-degree
/// proximity graph of out-degree `graph_degree` (`0` keeps the exact quadratic path).
///
/// This is Okkels et al.'s **two-pass** shape (Inf. Syst. 142 (2026) 102768, Algorithm 4): build the
/// approximate neighbour graph, read the core distances off it, take an exact MST of *that* graph.
/// Two-pass rather than one-pass because it is the variant their evaluation reports as reaching
/// cophenetic correlation ≥ 0.9 on both datasets while one-pass saturates at 0.7 — and because its
/// edge count is fixed at `O(m · degree)` where one-pass lets an expansion queue grow quadratically.
///
/// The price it names is a degree requirement of `Ω(minPts)`, since the core distances are bounded
/// directly from the graph. Here `minPts` is counted in points and a leaf carries many, so the
/// requirement is translated into leaves before it is enforced: see [`graph_degree_for`].
pub fn hdbscan_with<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    min_samples: usize,
    min_cluster_size: usize,
    graph_degree: usize,
    seed: u64,
) -> Hdbscan {
    hdbscan_selected(
        features,
        min_samples,
        Selection::ExcessOfMass { min_cluster_size },
        graph_degree,
        seed,
    )
}

/// [`hdbscan_with`], with the flattening rule chosen explicitly — see [`Selection`].
///
/// Everything up to the mutual-reachability MST is shared, so the two rules are genuinely comparable:
/// they differ only in how the same hierarchy is cut.
pub fn hdbscan_selected<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    min_samples: usize,
    selection: Selection,
    graph_degree: usize,
    seed: u64,
) -> Hdbscan {
    let m = features.len();
    let size_of = |s: Selection| match s {
        Selection::ExcessOfMass { min_cluster_size } => min_cluster_size as f64,
        Selection::Persistence => min_samples as f64,
    };
    if m == 0 {
        return Hdbscan {
            labels: vec![],
            n_clusters: 0,
            selected_size: size_of(selection),
        };
    }
    if m == 1 {
        return Hdbscan {
            labels: vec![0],
            n_clusters: 1,
            selected_size: size_of(selection),
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
    let (mst, _core) = mutual_reachability(m, &mu, &mass, min_samples, graph_degree, seed);

    match selection {
        Selection::ExcessOfMass { min_cluster_size } => from_mst(m, &mass, mst, min_cluster_size),
        Selection::Persistence => from_mst_persistence(m, &mass, mst, min_samples as f64),
    }
}

/// The size interval `(s_min, s_max]` over which each dendrogram segment is a **leaf cluster**.
///
/// Reading it off one dendrogram is the whole trick, and it rests on a monotonicity: raising the
/// minimum cluster size never moves a merge, it only prunes branches that fail to reach the size.
/// So a segment `c` under a split of `p` into `(l, r)` exists exactly while
/// `m_c ≤ min(mass(l), mass(r))` — above that the split is not admitted and `c` is absorbed into
/// `p` — and it is a *leaf* once every split inside it has likewise stopped being admitted, which is
/// `max` over its own split threshold and its children's. That `max` is monotone up the tree, so the
/// root's `s_min` is the global maximum; the root is capped there and consequently never a leaf
/// cluster, which is the same convention as the excess-of-mass arm's "the root is never selected".
///
/// Segments with `s_min ≥ s_max` are leaf clusters at no size at all and drop out of everything below.
fn leaf_barcode(m: usize, d: &Dendrogram) -> (Vec<f64>, Vec<f64>) {
    let total = d.children.len();
    let mut s_min = vec![0.0f64; total];
    let mut s_max = vec![f64::INFINITY; total];
    for nd in m..=d.root {
        let (l, r) = d.children[nd];
        let split = d.node_mass[l].min(d.node_mass[r]);
        s_max[l] = split;
        s_max[r] = split;
        // A single feature never splits, so its `s_min` stays at zero: a leaf heavy enough to carry
        // the minimum size on its own is a cluster on its own, which weighted leaves make reachable
        // and unit-weight points do not.
        s_min[nd] = split.max(s_min[l]).max(s_min[r]);
    }
    s_max[d.root] = s_min[d.root];
    (s_min, s_max)
}

/// Total leaf-cluster persistence at every size worth testing.
///
/// Only sizes where a segment is born or dies can be local maxima of the total, so the candidates
/// are the barcode's own endpoints. A segment alive on `[s_min, s_max)` contributes its whole
/// lifetime `s_max − s_min` at every one of them, accumulated through a difference array.
fn persistence_trace(
    root: usize,
    s_min: &[f64],
    s_max: &[f64],
    floor: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut cuts: Vec<f64> = Vec::with_capacity(2 * (root + 1));
    for nd in 0..=root {
        // `floor` bounds the search, not the barcode: a size below it is not on offer, so a segment's
        // lifetime is scored from the floor up. That clamp is what makes a single feature behave the
        // way the paper's points do — on unit weights `s_max ≤ 1 < floor` and it drops out entirely.
        if s_min[nd].max(floor) < s_max[nd] {
            cuts.push(s_min[nd].max(floor));
            cuts.push(s_max[nd]);
        }
    }
    cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cuts.dedup();
    if cuts.is_empty() {
        return (cuts, Vec::new());
    }
    // `(s_min, s_max]`: the first candidate strictly above `s_min`, the first strictly above `s_max`.
    let after = |v: f64| cuts.partition_point(|&c| c <= v);
    let mut diff = vec![0.0f64; cuts.len() + 1];
    for nd in 0..=root {
        let birth = s_min[nd].max(floor);
        if birth < s_max[nd] {
            let p = s_max[nd] - birth;
            diff[after(birth)] += p;
            diff[after(s_max[nd])] -= p;
        }
    }
    let mut trace = Vec::with_capacity(cuts.len());
    let mut acc = 0.0;
    for &d in &diff[..cuts.len()] {
        acc += d;
        trace.push(acc);
    }
    (cuts, trace)
}

/// Turn the mutual-reachability MST into labels by [`Selection::Persistence`].
fn from_mst_persistence(
    m: usize,
    mass: &[f64],
    mst: Vec<(f64, usize, usize)>,
    floor: f64,
) -> Hdbscan {
    let d = dendrogram(m, mass, mst);
    let (s_min, s_max) = leaf_barcode(m, &d);
    let (cuts, trace) = persistence_trace(d.root, &s_min, &s_max, floor);

    // Ties go to the smallest size, i.e. the finer clustering: a tie means two thresholds rate the
    // hierarchy equally well, and the coarser one is the one that has already thrown structure away.
    let Some(cut) = trace
        .iter()
        .enumerate()
        .fold(None::<(usize, f64)>, |best, (i, &t)| match best {
            Some((_, b)) if b >= t => best,
            _ => Some((i, t)),
        })
        .map(|(i, _)| cuts[i])
    else {
        // No segment is a leaf cluster at any size — a hierarchy with nothing to flatten.
        return Hdbscan {
            labels: vec![-1; m],
            n_clusters: 0,
            selected_size: floor,
        };
    };

    // Leaf clusters at `cut` are disjoint by construction — a leaf has no surviving descendant — so
    // labelling is one downward pass: a selected segment names a cluster, everything below inherits,
    // and anything reaching a feature without passing one is noise. Merge ids increase with distance,
    // so descending ids visits every parent before its children.
    let mut node_label = vec![-1i64; d.children.len()];
    let mut next_label = 0i64;
    for nd in (0..=d.root).rev() {
        if s_min[nd] < cut && cut <= s_max[nd] {
            node_label[nd] = next_label;
            next_label += 1;
        }
        if nd >= m {
            let (l, r) = d.children[nd];
            node_label[l] = node_label[nd];
            node_label[r] = node_label[nd];
        }
    }

    Hdbscan {
        labels: node_label[..m].to_vec(),
        n_clusters: next_label as usize,
        selected_size: cut,
    }
}

/// Prim's MST over the complete graph — `O(m²)` edge weights, the exact path.
fn prim_complete(m: usize, weight: &impl Fn(usize, usize) -> f64) -> Vec<(f64, usize, usize)> {
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
                let w = weight(u, v);
                if w < best[v] {
                    best[v] = w;
                    parent[v] = u;
                }
            }
        }
    }
    mst
}

/// Turn the mutual-reachability MST into labels: single-linkage dendrogram, condensation by
/// mass-weighted stability, then excess-of-mass selection.
/// Single-linkage dendrogram over the mutual-reachability MST: ids `0..m` are the features, ids
/// `m..` the merges, in increasing merge distance — so a node's id always exceeds its children's.
struct Dendrogram {
    children: Vec<(usize, usize)>,
    node_dist: Vec<f64>,
    node_mass: Vec<f64>,
    root: usize,
}

fn dendrogram(m: usize, mass: &[f64], mut mst: Vec<(f64, usize, usize)>) -> Dendrogram {
    let total = 2 * m;
    let mut children: Vec<(usize, usize)> = vec![(usize::MAX, usize::MAX); total];
    let mut node_dist = vec![0.0f64; total];
    let mut node_mass = vec![0.0f64; total];
    node_mass[..m].copy_from_slice(&mass[..m]);
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
        let r = uf.union(ra, rb);
        comp_node[r] = id;
    }
    Dendrogram {
        children,
        node_dist,
        node_mass,
        root: next - 1,
    }
}

fn from_mst(
    m: usize,
    mass: &[f64],
    mst: Vec<(f64, usize, usize)>,
    min_cluster_size: usize,
) -> Hdbscan {
    let Dendrogram {
        children,
        node_dist,
        node_mass,
        root,
    } = dendrogram(m, mass, mst);
    let total = children.len();

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

    // Each of the `2m` dendrogram nodes is condensed once. That is a property of the tree, not a
    // safety margin: a merge whose two sides are the same node would otherwise push it twice under
    // two different cluster ids, doubling `stack`, `birth`, `stab` and `kids` at every level.
    let mut condensed = vec![false; total];
    let mut stack = vec![(root, 0usize)];
    while let Some((nd, c)) = stack.pop() {
        if nd < m {
            continue; // single point — stays in c
        }
        if std::mem::replace(&mut condensed[nd], true) {
            continue;
        }
        let (l, r) = children[nd];
        let split = lam(nd);
        let want = min_cluster_size as f64;
        let lbig = node_mass[l] >= want;
        let rbig = node_mass[r] >= want;
        if lbig && rbig {
            stab[c] += (split - birth[c]) * node_mass[nd];
            let cl = new_cluster(&mut birth, &mut stab, &mut kids, split);
            let cr = new_cluster(&mut birth, &mut stab, &mut kids, split);
            kids[c].push(cl);
            kids[c].push(cr);
            for p in collect_leaves(l, m, &children) {
                point_cluster[p] = cl;
            }
            for p in collect_leaves(r, m, &children) {
                point_cluster[p] = cr;
            }
            stack.push((l, cl));
            stack.push((r, cr));
        } else if lbig {
            for p in collect_leaves(r, m, &children) {
                stab[c] += (split - birth[c]) * mass[p];
            }
            stack.push((l, c));
        } else if rbig {
            for p in collect_leaves(l, m, &children) {
                stab[c] += (split - birth[c]) * mass[p];
            }
            stack.push((r, c));
        } else {
            for p in collect_leaves(nd, m, &children) {
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
        selected_size: min_cluster_size as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros, two_moons};
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
        let core: Vec<f64> = super::core_distances(m, min_samples, mass, dist);

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
        node_mass[..m].copy_from_slice(&mass[..m]);
        let mut uf: Vec<usize> = (0..m).collect();
        let mut comp_node: Vec<usize> = (0..m).collect();
        let mut next = m;
        for &(w, a, b) in &mst {
            let (ra, rb) = (root_of(&mut uf, a), root_of(&mut uf, b));
            let (na, nb) = (comp_node[ra], comp_node[rb]);
            children[next] = (na, nb);
            node_dist[next] = w;
            node_mass[next] = node_mass[na] + node_mass[nb];
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
                node_mass[l] >= min_cluster_size as f64,
                node_mass[r] >= min_cluster_size as f64,
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

    #[test]
    fn a_walk_over_a_node_that_is_its_own_two_children_stays_linear_instead_of_doubling() {
        // The shape a broken union-find produces: every merge joins the same component to itself,
        // so `children[id] = (id - 1, id - 1)` all the way down to one leaf. Ids still decrease, so
        // the walk never cycles — it re-visits, and re-visiting doubles. Twenty levels is 2^20
        // leaves out of a three-leaf dendrogram, which is small enough to fail an assertion rather
        // than the machine; the mutant that first produced this shape reached 17 GB.
        let m = 3;
        let mut children = vec![(usize::MAX, usize::MAX); m];
        children.push((0, 0));
        for id in m + 1..m + 20 {
            children.push((id - 1, id - 1));
        }
        let root = children.len() - 1;

        let out = collect_leaves(root, m, &children);
        // Length first: the doubling version returns 2^20 of these, and asserting on the vector
        // itself would print every one of them.
        assert_eq!(
            out.len(),
            2,
            "one leaf, reached once down each of the two sides"
        );
        assert!(out.iter().all(|&p| p == 0));
    }

    #[test]
    fn min_samples_counts_the_object_itself() {
        // Four points on a line at 0, 1, 3, 6, so every neighbour rank is a distinct number and the
        // two conventions cannot coincide by accident. Sorted distances, self included:
        //
        //   from 0: 0 1 3 6      from 1: 0 1 2 5      from 3: 0 2 3 3      from 6: 0 3 5 6
        //
        // The inclusive convention (Campello Def. 3.1, ELKI, `sklearn.cluster.HDBSCAN`) reads off
        // column `min_samples`; the exclusive one (`scikit-learn-contrib/hdbscan`) drops the leading
        // zero and reads one column further. `min_samples = 1` is the sharpest discriminator of the
        // two: inclusive gives 0 everywhere, which makes mutual reachability the plain distance.
        const LINE: [f64; 4] = [0.0, 1.0, 3.0, 6.0];
        let d = |i: usize, j: usize| (LINE[i] - LINE[j]).abs();
        let unit = [1.0; 4];

        assert_eq!(
            core_distances(4, 1, &unit, d),
            vec![0.0, 0.0, 0.0, 0.0],
            "min_samples = 1 must be the object itself, i.e. core distance 0"
        );
        assert_eq!(
            core_distances(4, 2, &unit, d),
            vec![1.0, 1.0, 2.0, 3.0],
            "min_samples = 2 must be the nearest *other* object"
        );
        assert_eq!(core_distances(4, 3, &unit, d), vec![3.0, 2.0, 3.0, 5.0]);
        // Asking for more neighbours than exist saturates rather than panicking.
        assert_eq!(
            core_distances(4, 99, &unit, d),
            core_distances(4, 4, &unit, d)
        );
    }

    #[test]
    fn a_core_radius_encloses_min_samples_points_of_mass_not_min_samples_features() {
        // Same line, but object 1 now stands for eight points and the rest for one each. A radius
        // that has to enclose four *points* therefore stops at object 1 for everyone who reaches it,
        // where the feature-counting rule would have kept walking to the fourth-nearest feature.
        //
        //   from 0: (0, w1) (1, w8) -> 1+8 = 9 >= 4 at distance 1, and 1 alone is not enough
        //   from 1: (0, w8)         -> 8 >= 4 at distance 0
        //   from 3: (0, w1) (2, w8) -> 9 >= 4 at distance 2
        //   from 6: (0, w1) (3, w1) (5, w8) -> 10 >= 4 at distance 5, object 1 being 5 away
        const LINE: [f64; 4] = [0.0, 1.0, 3.0, 6.0];
        let d = |i: usize, j: usize| (LINE[i] - LINE[j]).abs();
        let heavy = [1.0, 8.0, 1.0, 1.0];

        assert_eq!(
            core_distances(4, 4, &heavy, d),
            vec![1.0, 0.0, 2.0, 5.0],
            "the radius must be driven by enclosed mass"
        );
        // The fixture can see the difference: on unit weights the same call reads off the fourth
        // column instead, and disagrees in every position but one.
        assert_eq!(core_distances(4, 4, &[1.0; 4], d), vec![6.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn min_cluster_size_admits_a_cluster_on_the_points_it_holds_not_the_features_it_spans() {
        // Two well-separated groups of two leaves, each leaf standing for 50 points. Asking for
        // clusters of at least 60 points is a question the data answers twice over — 100 points a
        // side — but only four features exist, so a rule that counted features would find `4 < 60`
        // at every split, birth no cluster at all, and return every leaf as noise with no warning.
        // That silent all-noise return is the failure this fixture exists to forbid.
        use crate::feature::{ClusterFeature, Spherical};
        let leaf = |x: f64, w: usize| {
            let mut c = Spherical::<f64>::new(2);
            for _ in 0..w {
                c.push(&[x, 0.0], 1.0);
            }
            c
        };
        let feats = vec![leaf(0.0, 50), leaf(0.1, 50), leaf(10.0, 50), leaf(10.1, 50)];

        let res = hdbscan(&feats, 1, 60);
        assert_eq!(res.n_clusters, 2, "labels = {:?}", res.labels);
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[2], res.labels[3]);
        assert_ne!(res.labels[0], res.labels[2]);

        // The same call one point past the total mass of either side finds nothing, which is what
        // "too large" is supposed to look like: 201 points cannot be had from 100.
        assert_eq!(hdbscan(&feats, 1, 201).n_clusters, 0);
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

    /// Two sub-blobs of `per` leaves each, `sep` apart, inside one loose parent cloud. Sweeping
    /// `sep` walks the condensed tree across the point where the children's stability overtakes
    /// their parent's, which is the only thing the excess-of-mass integral decides.
    fn split_fixture(sep: f64, seed: u64) -> Vec<Spherical<f64>> {
        let mut rng = SplitMix64::new(seed);
        let mut pts: Vec<Vec<f64>> = Vec::new();
        for s in [-0.5 * sep, 0.5 * sep] {
            for _ in 0..45 {
                pts.push(vec![s + 0.45 * rng.gauss(), 0.45 * rng.gauss()]);
            }
        }
        for _ in 0..12 {
            pts.push(vec![6.0 + 0.3 * rng.gauss(), 5.0 + 0.3 * rng.gauss()]);
        }
        grid_micros(&pts, 0.35).0
    }

    #[test]
    fn the_excess_of_mass_boundary_sits_where_the_reference_puts_it() {
        // Comparing one partition is a coarse probe: the stability integral can be corrupted and
        // still select the same clusters on a fixture whose answer is obvious. Walking the
        // separation moves the decision, and where it moves is decided entirely by
        // `Σ (λ_split − λ_birth) · mass` -- both the per-node form and the per-point one.
        let mut got = Vec::new();
        let mut want = Vec::new();
        for step in 0..16 {
            let sep = 0.6 + 0.28 * step as f64;
            let feats = split_fixture(sep, 91);
            let mu: Vec<Vec<f64>> = feats.iter().map(|f| f.mean().to_vec()).collect();
            let mass: Vec<f64> = feats.iter().map(|f| f.weight()).collect();
            let w = reference_hdbscan(&mu, &mass, 3, 4);
            let g = hdbscan(&feats, 3, 4);
            // The count is the coarsest reading of the sweep. Compare the partition itself at every
            // step: a corrupted stability integral can hold the count and still move which points
            // land in which cluster, which is the failure this fixture exists to see.
            let a: Vec<usize> = g.labels.iter().map(|&l| (l + 1) as usize).collect();
            let b: Vec<usize> = w.iter().map(|&l| (l + 1) as usize).collect();
            assert_eq!(
                g.labels.iter().map(|&l| l < 0).collect::<Vec<_>>(),
                w.iter().map(|&l| l < 0).collect::<Vec<_>>(),
                "sep {sep}: the noise set differs, {:?} vs {w:?}",
                g.labels
            );
            assert!(
                (ari(&a, &b) - 1.0).abs() < 1e-12,
                "sep {sep}: the partition differs, {:?} vs {w:?}",
                g.labels
            );
            got.push(g.n_clusters);
            want.push(
                w.iter()
                    .filter(|&&l| l >= 0)
                    .map(|&l| l as usize + 1)
                    .max()
                    .unwrap_or(0),
            );
        }
        assert!(
            want.windows(2).any(|w| w[0] != w[1]),
            "the sweep never crosses a boundary: {want:?}"
        );
        assert_eq!(got, want, "the excess-of-mass boundary moved");
    }

    /// A small ring of leaves, so the graph path is exercised on something with a known shape and
    /// few enough objects that a complete graph is within the search's expansion budget.
    fn ring(m: usize, per_leaf: f64) -> Vec<Spherical<f64>> {
        use crate::feature::ClusterFeature;
        (0..m)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / m as f64;
                let mut f = Spherical::<f64>::new(2);
                f.push(&[a.cos(), a.sin()], per_leaf);
                f
            })
            .collect()
    }

    /// Leaves at `i^1.3` on a line: irregular spacing, so every pairwise distance is distinct and a
    /// float comparison against the exact path can be exact rather than approximate.
    fn irregular_line(m: usize, per_leaf: f64) -> Vec<Spherical<f64>> {
        use crate::feature::ClusterFeature;
        (0..m)
            .map(|i| {
                let mut f = Spherical::<f64>::new(1);
                f.push(&[(i as f64).powf(1.3)], per_leaf);
                f
            })
            .collect()
    }

    #[test]
    fn a_complete_proximity_graph_reproduces_the_exact_labels() {
        // The gate on the whole approximation: with the degree at `m - 1` the graph is the complete
        // one, so the two code paths differ only in *how* they reach the same edge set. Anything but
        // an identical labelling means the sparse path has a bug rather than an approximation.
        let mut rng = SplitMix64::new(4);
        let (pts, _) = two_moons(&mut rng, 120, 0.05);
        let (micros, _) = grid_micros(&pts, 0.25);
        let m = micros.len();
        assert!(m > 8 && m <= 64, "fixture must fit the expansion cap: {m}");
        let exact = hdbscan(&micros, 4, 3);
        let graph = hdbscan_with(&micros, 4, 3, m - 1, 1);
        assert_eq!(graph.n_clusters, exact.n_clusters);
        // Up to relabelling, not label for label: Prim emits its edges in insertion order and
        // Kruskal in weight order, so a tie between two equal mutual-reachability edges can be
        // broken the other way and renumber the clusters. The partition is the invariant.
        let as_usize = |v: &[i64]| -> Vec<usize> {
            v.iter()
                .map(|&l| if l < 0 { usize::MAX } else { l as usize })
                .collect()
        };
        let score = ari(&as_usize(&graph.labels), &as_usize(&exact.labels));
        assert_eq!(
            score, 1.0,
            "the complete-graph path found a different partition"
        );
    }

    #[test]
    fn the_graph_path_recovers_the_two_moons_partition() {
        let mut rng = SplitMix64::new(7);
        let (pts, truth) = two_moons(&mut rng, 700, 0.07);
        let (micros, point_to_micro) = grid_micros(&pts, 0.1);
        let res = hdbscan_with(&micros, 5, 5, 12, 3);
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
    fn the_degree_floor_is_counted_in_leaves_not_in_points() {
        // 100 leaves of 10 points each: reaching 50 points takes 5 leaves, so a request for 2 is
        // raised to 5 and a request for 9 is left alone. On unit mass the same `min_samples` needs
        // 50 leaves, and the floor says so.
        let heavy = vec![10.0f64; 100];
        assert_eq!(graph_degree_for(2, 50, &heavy), 5);
        assert_eq!(graph_degree_for(9, 50, &heavy), 9);
        assert_eq!(graph_degree_for(2, 50, &vec![1.0f64; 100]), 50);
        // …and it can never exceed the number of other objects there are.
        assert_eq!(graph_degree_for(1000, 50, &heavy), 99);
    }

    #[test]
    fn core_distances_off_a_complete_adjacency_match_the_exact_ones() {
        use crate::feature::ClusterFeature;
        let micros = irregular_line(24, 3.0);
        let mu: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let mass = vec![3.0f64; 24];
        let dist = |i: usize, j: usize| crate::kernels::sq_euclidean(&mu[i], &mu[j]).sqrt();
        let adj = crate::clustering::knn::build(24, 23, 2, dist);
        for min_samples in [1usize, 3, 7, 200] {
            assert_eq!(
                core_distances_from_graph(min_samples, &mass, &adj),
                core_distances(24, min_samples, &mass, dist),
                "min_samples = {min_samples}"
            );
        }
    }

    #[test]
    fn the_graph_mst_spans_every_leaf() {
        let micros = ring(60, 1.0);
        let mu: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let dist = |i: usize, j: usize| crate::kernels::sq_euclidean(&mu[i], &mu[j]).sqrt();
        for degree in [2usize, 4, 10] {
            let adj = crate::clustering::knn::build(60, degree, 6, dist);
            let mst = mst_over_graph(60, &adj, &dist);
            assert_eq!(mst.len(), 59, "degree = {degree}");
            let mut uf = UnionFind::new(60);
            for &(_, a, b) in &mst {
                assert_ne!(uf.find(a), uf.find(b), "the MST has a cycle");
                uf.union(a, b);
            }
        }
    }

    /// Blobs whose sizes span an order of magnitude, summarised at grid cell `cell`. One minimum
    /// cluster size cannot cover that range, which is the regime the persistence arm exists for.
    fn uneven(seed: u64, cell: f64) -> (Vec<Spherical<f64>>, Vec<usize>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let sizes = [400usize, 200, 70, 25];
        let centres = [[0.0, 0.0], [14.0, 0.0], [7.0, 12.0], [21.0, 12.0]];
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, (&n, ctr)) in sizes.iter().zip(centres.iter()).enumerate() {
            for _ in 0..n {
                pts.push(vec![ctr[0] + 0.9 * rng.gauss(), ctr[1] + 0.9 * rng.gauss()]);
                truth.push(c);
            }
        }
        let (micros, assign) = grid_micros(&pts, cell);
        (micros, assign, truth)
    }

    /// Four clusters of equal size but spreads from 0.4 to 3.2 — the variable-density regime, where a
    /// single minimum cluster size is the classic weak spot of excess of mass.
    fn uneven_density(seed: u64, cell: f64) -> (Vec<Spherical<f64>>, Vec<usize>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let spreads = [0.4f64, 0.8, 1.6, 3.2];
        let centres = [[0.0, 0.0], [16.0, 0.0], [8.0, 16.0], [26.0, 16.0]];
        let mut pts = Vec::new();
        let mut truth = Vec::new();
        for (c, (&sd, ctr)) in spreads.iter().zip(centres.iter()).enumerate() {
            for _ in 0..175 {
                pts.push(vec![ctr[0] + sd * rng.gauss(), ctr[1] + sd * rng.gauss()]);
                truth.push(c);
            }
        }
        let (micros, assign) = grid_micros(&pts, cell);
        (micros, assign, truth)
    }

    fn mst_of(micros: &[Spherical<f64>], min_samples: usize) -> (usize, Vec<f64>, Dendrogram) {
        let m = micros.len();
        let mass: Vec<f64> = micros.iter().map(ClusterFeature::weight).collect();
        let mu: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let dist = |i: usize, j: usize| crate::kernels::sq_euclidean(&mu[i], &mu[j]).sqrt();
        let core = core_distances(m, min_samples, &mass, dist);
        let mst = prim_complete(m, &|i, j| core[i].max(core[j]).max(dist(i, j)));
        let d = dendrogram(m, &mass, mst);
        (m, mass, d)
    }

    /// Independent re-derivation of "which segments are leaf clusters at minimum size `t`": prune the
    /// dendrogram directly. A split is admitted only when both sides carry `t` points; otherwise the
    /// cluster follows the surviving side and keeps the identity of the node it entered on. The
    /// clusters that never split are the leaf clusters. The barcode claims to answer this for every
    /// `t` at once, off one traversal, which is the whole reason it exists.
    fn reference_leaf_clusters(m: usize, d: &Dendrogram, t: f64) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![d.root];
        while let Some(top) = stack.pop() {
            let mut nd = top;
            loop {
                if nd < m {
                    out.push(top);
                    break;
                }
                let (l, r) = d.children[nd];
                let (ml, mr) = (d.node_mass[l], d.node_mass[r]);
                if ml >= t && mr >= t {
                    stack.push(l);
                    stack.push(r);
                    break;
                } else if ml >= t {
                    nd = l;
                } else if mr >= t {
                    nd = r;
                } else {
                    out.push(top);
                    break;
                }
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn the_leaf_barcode_answers_for_every_size_what_pruning_answers_for_one() {
        for seed in [7u64, 11, 23] {
            let (micros, _, _) = uneven(seed, 0.5);
            let (m, _, d) = mst_of(&micros, 3);
            let (s_min, s_max) = leaf_barcode(m, &d);
            for t in 1..=700 {
                let t = t as f64;
                let mut want = reference_leaf_clusters(m, &d, t);
                // The top-level cluster is the whole dataset and is never a cluster of its own —
                // the same convention the excess-of-mass arm applies to its root.
                want.retain(|&nd| nd != d.root);
                let got: Vec<usize> = (0..=d.root)
                    .filter(|&nd| s_min[nd] < t && t <= s_max[nd])
                    .collect();
                assert_eq!(got, want, "seed {seed}, minimum cluster size {t}");
            }
        }
    }

    #[test]
    fn the_persistence_arm_takes_no_minimum_cluster_size_and_still_finds_the_blobs() {
        let mut rng = SplitMix64::new(7);
        let centres = [[0.0, 0.0], [10.0, 0.0], [5.0, 9.0], [16.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 200, &centres, 0.7);
        let (micros, assign) = grid_micros(&pts, 0.5);
        for min_samples in [3usize, 10, 40] {
            let res = hdbscan_selected(&micros, min_samples, Selection::Persistence, 0, 0);
            assert_eq!(res.n_clusters, 4, "min_samples = {min_samples}");
            let labels: Vec<usize> = assign
                .iter()
                .map(|&i| (res.labels[i] + 1) as usize)
                .collect();
            assert!(
                ari(&labels, &truth) > 0.99,
                "min_samples = {min_samples}: ARI = {}",
                ari(&labels, &truth)
            );
            assert!(
                res.selected_size >= min_samples as f64,
                "the chosen size must respect its floor"
            );
        }
    }

    #[test]
    fn the_persistence_arm_answers_the_degenerate_inputs() {
        let empty: Vec<Spherical<f64>> = Vec::new();
        let res = hdbscan_selected(&empty, 3, Selection::Persistence, 0, 0);
        assert_eq!(res.n_clusters, 0);
        assert!(res.labels.is_empty());

        let mut one = Spherical::<f64>::new(2);
        one.push(&[0.0, 0.0], 1.0);
        let res = hdbscan_selected(std::slice::from_ref(&one), 3, Selection::Persistence, 0, 0);
        assert_eq!(res.labels, vec![0]);

        // Two features: the only merge is the root, which is never a cluster, so everything is noise
        // rather than a spurious pair.
        let mut two = one.clone();
        two.push(&[5.0, 5.0], 1.0);
        let pair = [one, two];
        let res = hdbscan_selected(&pair, 3, Selection::Persistence, 0, 0);
        assert_eq!(res.n_clusters, 0);
        assert_eq!(res.labels, vec![-1, -1]);
    }

    #[test]
    fn selected_size_reports_the_argument_under_excess_of_mass_and_the_choice_under_persistence() {
        let (micros, _, _) = uneven(7, 0.5);
        let eom = hdbscan_selected(
            &micros,
            3,
            Selection::ExcessOfMass {
                min_cluster_size: 17,
            },
            0,
            0,
        );
        assert_eq!(eom.selected_size, 17.0);
        let per = hdbscan_selected(&micros, 3, Selection::Persistence, 0, 0);
        assert_ne!(
            per.selected_size, 3.0,
            "the floor is a floor, not the answer"
        );
    }

    /// The measurement behind the `PLSCAN` section of `bench/RESULTS.md`. Not a test — run it with
    /// `cargo test --lib measure_persistence_against_excess_of_mass -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn measure_persistence_against_excess_of_mass() {
        type Fixture = fn(u64, f64) -> (Vec<Spherical<f64>>, Vec<usize>, Vec<usize>);
        let sweep = [3usize, 5, 10, 15, 25, 40, 60, 100];
        let fixtures: [(&str, Fixture); 2] = [("size", uneven), ("density", uneven_density)];
        println!("fixture  cell  leaves  mass  |  EOM mean  spread  |  persistence mean  spread");
        for (tag, make) in fixtures {
            for cell in [0.02f64, 0.1, 0.25, 0.5, 1.0] {
                for seed in [7u64, 11, 23, 31, 47] {
                    let (micros, assign, truth) = make(seed, cell);
                    let mass = micros.iter().map(ClusterFeature::weight).sum::<f64>()
                        / micros.len() as f64;
                    let score = |sel: &dyn Fn(usize) -> Hdbscan| -> (f64, f64) {
                        let a: Vec<f64> = sweep
                            .iter()
                            .map(|&ms| {
                                let r = sel(ms);
                                let l: Vec<usize> =
                                    assign.iter().map(|&i| (r.labels[i] + 1) as usize).collect();
                                ari(&l, &truth)
                            })
                            .collect();
                        let hi = a.iter().cloned().fold(f64::MIN, f64::max);
                        let lo = a.iter().cloned().fold(f64::MAX, f64::min);
                        (a.iter().sum::<f64>() / a.len() as f64, hi - lo)
                    };
                    let (em, es) = score(&|ms| {
                        hdbscan_selected(
                            &micros,
                            ms,
                            Selection::ExcessOfMass {
                                min_cluster_size: ms,
                            },
                            0,
                            0,
                        )
                    });
                    let (pm, ps) =
                        score(&|ms| hdbscan_selected(&micros, ms, Selection::Persistence, 0, 0));
                    println!(
                        "{tag:8} {cell:4}  {:5}  {mass:4.2}  |  {em:.3}  {es:.3}  |  {pm:.3}  {ps:.3}   (seed {seed})",
                        micros.len()
                    );
                }
            }
        }
    }
}
