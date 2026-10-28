//! Community detection on the CF-tree leaf microclusters (graph clustering) via the **Leiden**
//! algorithm (Traag, Waltman & van Eck 2019).
//!
//! Builds the shared self-tuning k-NN affinity graph over the leaf means (see [`crate::clustering::
//! graph`]) and optimizes a quality function over it with Leiden's three phases per level:
//!
//! 1. **local moving** — greedily move nodes between communities to raise the quality (as in Louvain);
//! 2. **refinement** — inside each community, rebuild sub-communities by merging singletons *along
//!    edges*, so every sub-community is connected by construction (this is the fix Leiden adds over
//!    Louvain, which can leave communities internally disconnected or badly connected);
//! 3. **aggregation** — collapse each refined sub-community to a super-node and seed the next level
//!    from the pre-refinement partition, which lets Leiden re-split and reach higher quality.
//!
//! It **discovers the community count** — no `k`. Two quality functions: **modularity** (γ = 1 is the
//! Newman-Girvan default; the resolution `γ` trades community count against size but has a resolution
//! limit) and **CPM** (Constant Potts Model — resolution-limit-free, but `γ` is an absolute density
//! threshold on the edge-weight scale). Pure Rust, no eigensolver.

use crate::clustering::graph::{knn_affinity, knn_affinity_geo, log_covariances, tangent_bases};
use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::types::Real;
use std::collections::HashMap;

/// Quality function optimized by [`leiden`].
#[derive(Clone, Copy)]
pub enum Objective {
    /// Newman-Girvan modularity with resolution `γ` (has a resolution limit).
    Modularity,
    /// Constant Potts Model with resolution `γ` (resolution-limit-free; `γ` is a density threshold).
    Cpm,
}

/// What a leaf is in the graph Leiden partitions.
///
/// A leaf summarises `n_i` points, and the leaves of one tree are very unequal in mass. With
/// [`LeafGraph::Unit`] every leaf is one node of size 1 joined by its affinity `w_ij`, so the
/// objective is not that of any point graph: a singleton leaf counts as much as a leaf of a
/// thousand points. [`LeafGraph::Mass`] partitions the quotient of the point graph instead
/// (Traag, Waltman & van Eck 2019, App. A): each pair of points across two leaves carries their
/// leaves' affinity, so the edge is `n_i·n_j·w_ij`, and the node size is `n_i`, which is the point
/// count the CPM null model charges for. The within-leaf edges are left out: the kernel inside a
/// leaf needs a spread the leaf means do not set, and the two forms that model it measured worse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LeafGraph {
    /// One node of size 1 per leaf, joined by the leaf-mean affinity — the graph of 1.x.
    #[default]
    Unit,
    /// The point graph's quotient over the leaves: edges `n_i·n_j·w_ij`, node size `n_i`.
    Mass,
}

/// Result of a community-detection run: one community label per input microcluster.
pub struct Community {
    /// Community index per input feature.
    pub labels: Vec<usize>,
}

/// A weighted undirected working graph. `adj` holds only inter-node (external) edges; internal edges
/// created by aggregation are folded into `degree` / `size` (a super-node inherits the sums of its
/// members), so `2m` and the node-count total are invariant across levels.
struct Graph {
    adj: Vec<Vec<(usize, f64)>>,
    degree: Vec<f64>, // weighted degree — modularity null model
    size: Vec<f64>,   // node count — CPM null model
    two_m: f64,
}

impl Graph {
    fn from_adj(adj: Vec<Vec<(usize, f64)>>) -> Self {
        let size = vec![1.0; adj.len()];
        Self::from_adj_sized(adj, size)
    }

    fn from_adj_sized(adj: Vec<Vec<(usize, f64)>>, size: Vec<f64>) -> Self {
        let degree: Vec<f64> = adj
            .iter()
            .map(|r| r.iter().map(|&(_, w)| w).sum())
            .collect();
        let two_m = degree.iter().sum();
        Self {
            adj,
            degree,
            size,
            two_m,
        }
    }
    fn len(&self) -> usize {
        self.adj.len()
    }
}

/// Cast an `m × d × d` (or `m × d × r`) tensor to `f64` for the affinity graph.
fn to_f64_tensors<R: Real>(t: Vec<Vec<Vec<R>>>) -> Vec<Vec<Vec<f64>>> {
    t.into_iter()
        .map(|m| {
            m.into_iter()
                .map(|row| row.into_iter().map(|x| x.to_f64().unwrap()).collect())
                .collect()
        })
        .collect()
}

/// Leiden community detection on the leaf microclusters. `resolution` is `γ`; `k` is not used.
/// `cov_weight > 0` adds a log-Euclidean **covariance/shape** term and `tangent_weight > 0` a
/// Grassmann **tangent-subspace** term (rank `tangent_rank`) to the microcluster affinity, so
/// communities agree in centroid, shape, and manifold orientation (GeoBETULA; best with
/// `feature="full"`). Both `0` reproduce the plain centroid affinity.
///
/// One Leiden pass. [`leiden_iterated`] repeats it, and its `max_passes = 1` is this function.
#[allow(clippy::too_many_arguments)]
pub fn leiden<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    resolution: f64,
    objective: Objective,
    seed: u64,
    cov_weight: f64,
    tangent_weight: f64,
    tangent_rank: usize,
) -> Community {
    leiden_iterated(
        features,
        resolution,
        objective,
        seed,
        cov_weight,
        tangent_weight,
        tangent_rank,
        1,
    )
}

/// [`leiden`] repeated: every pass after the first starts its first local move from the previous
/// pass's partition instead of from singletons, as igraph's `n_iterations` does. It stops after
/// `max_passes` passes, or earlier at the first pass that returns the partition it was given;
/// `0` is read as `1`. Pass `p` runs on its own seed, derived from `seed` and `p`, and the first
/// pass uses `seed` itself, so `max_passes = 1` returns exactly what [`leiden`] returns.
///
/// One pass stops at a local optimum of its last aggregate graph; Leiden's guarantees (Traag,
/// Waltman & van Eck 2019) are about the iterated algorithm. On the three-community stochastic
/// block model at four times the detection threshold, a second pass lifts the ARI from about 0.6
/// to about 0.9 (see `docs/MATH.md`). The work is linear in the passes run.
#[allow(clippy::too_many_arguments)]
pub fn leiden_iterated<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    resolution: f64,
    objective: Objective,
    seed: u64,
    cov_weight: f64,
    tangent_weight: f64,
    tangent_rank: usize,
    max_passes: usize,
) -> Community {
    leiden_on(
        features,
        resolution,
        objective,
        seed,
        cov_weight,
        tangent_weight,
        tangent_rank,
        max_passes,
        LeafGraph::Unit,
    )
}

/// [`leiden_iterated`] on the leaf graph `graph` names. [`LeafGraph::Unit`] is
/// [`leiden_iterated`] itself, bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn leiden_on<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    resolution: f64,
    objective: Objective,
    seed: u64,
    cov_weight: f64,
    tangent_weight: f64,
    tangent_rank: usize,
    max_passes: usize,
    graph: LeafGraph,
) -> Community {
    let n = features.len();
    if n <= 1 {
        return Community { labels: vec![0; n] };
    }
    let mut base = affinity(features, cov_weight, tangent_weight, tangent_rank);
    let sizes = match graph {
        LeafGraph::Unit => None,
        LeafGraph::Mass => {
            let mass: Vec<f64> = features
                .iter()
                .map(|f| f.weight().to_f64().unwrap())
                .collect();
            mass_quotient(&mut base, &mass);
            Some(mass)
        }
    };
    Community {
        labels: detect_passes_sized(
            &base,
            sizes.as_deref(),
            resolution,
            objective,
            seed,
            max_passes,
        )
        .0,
    }
}

/// The self-tuning kNN affinity graph over the leaf means, with the geometry terms whose weights
/// are positive.
fn affinity<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    cov_weight: f64,
    tangent_weight: f64,
    tangent_rank: usize,
) -> Vec<Vec<(usize, f64)>> {
    let centers: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|&x| x.to_f64().unwrap()).collect())
        .collect();
    let log_covs = (cov_weight > 0.0).then(|| to_f64_tensors(log_covariances(features)));
    let tangents =
        (tangent_weight > 0.0).then(|| to_f64_tensors(tangent_bases(features, tangent_rank)));
    if log_covs.is_some() || tangents.is_some() {
        knn_affinity_geo::<f64>(
            &centers,
            log_covs.as_deref().map(|lc| (lc, cov_weight)),
            tangents.as_deref().map(|t| (t, tangent_weight)),
        )
    } else {
        knn_affinity::<f64>(&centers)
    }
}

/// Gain of adding a node (degree `ki`, size `si`, edge weight `w_to` into community `c`) to `c`,
/// whose current totals are `tot_deg` / `tot_size`. Constant terms shared across candidates cancel,
/// so the argmax of this is the argmax of ΔQ.
#[allow(clippy::too_many_arguments)]
fn gain(
    obj: Objective,
    gamma: f64,
    two_m: f64,
    w_to: f64,
    ki: f64,
    si: f64,
    tot_deg: f64,
    tot_size: f64,
) -> f64 {
    match obj {
        Objective::Modularity => w_to - gamma * tot_deg * ki / two_m,
        Objective::Cpm => w_to - gamma * tot_size * si,
    }
}

fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    order
}

/// Seed of pass `pass` (counted from 0). The first pass keeps the caller's seed, which is what
/// makes one pass bit-identical to [`detect`]; the odd Weyl increment keeps the later passes off the
/// `seed + level` seeds a pass already spends on its levels.
/// Turn a leaf graph into the quotient of its point graph: every edge `w_ij` becomes
/// `n_i·n_j·w_ij`, the sum of `w_ij` over the point pairs it stands for. With node sizes `n_i`, a
/// partition of the leaves then scores exactly what the same partition of their points scores.
fn mass_quotient(base: &mut [Vec<(usize, f64)>], mass: &[f64]) {
    for (i, row) in base.iter_mut().enumerate() {
        for (j, w) in row.iter_mut() {
            *w *= mass[i] * mass[*j];
        }
    }
}

fn pass_seed(seed: u64, pass: usize) -> u64 {
    seed.wrapping_add((pass as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

/// Up to `max_passes` Leiden passes (at least one), each but the first started from the previous
/// partition; stops early at the first pass that returns its input. Returns the partition and the
/// number of passes run.
#[cfg(test)]
fn detect_passes(
    base: &[Vec<(usize, f64)>],
    gamma: f64,
    obj: Objective,
    seed: u64,
    max_passes: usize,
) -> (Vec<usize>, usize) {
    detect_passes_sized(base, None, gamma, obj, seed, max_passes)
}

/// [`detect_passes`] over nodes of the given sizes; `None` is size 1 for every node.
fn detect_passes_sized(
    base: &[Vec<(usize, f64)>],
    sizes: Option<&[f64]>,
    gamma: f64,
    obj: Objective,
    seed: u64,
    max_passes: usize,
) -> (Vec<usize>, usize) {
    let singletons: Vec<usize> = (0..base.len()).collect();
    let mut labels = detect_from_sized(base, sizes, gamma, obj, seed, &singletons);
    let mut passes = 1;
    while passes < max_passes {
        let next = detect_from_sized(base, sizes, gamma, obj, pass_seed(seed, passes), &labels);
        passes += 1;
        // Both sides are in first-seen order, so equal vectors are equal partitions.
        if next == labels {
            break;
        }
        labels = next;
    }
    (labels, passes)
}

/// Multi-level Leiden: local-move → refine → aggregate (seeded from the local-move partition),
/// until aggregation no longer coarsens the graph.
#[cfg(test)]
fn detect(base: &[Vec<(usize, f64)>], gamma: f64, obj: Objective, seed: u64) -> Vec<usize> {
    let singletons: Vec<usize> = (0..base.len()).collect();
    detect_from(base, gamma, obj, seed, &singletons)
}

/// [`detect`] with the first local move started from `start` instead of from singletons.
#[cfg(test)]
fn detect_from(
    base: &[Vec<(usize, f64)>],
    gamma: f64,
    obj: Objective,
    seed: u64,
    start: &[usize],
) -> Vec<usize> {
    detect_from_sized(base, None, gamma, obj, seed, start)
}

/// [`detect_from`] over nodes of the given sizes; `None` is size 1 for every node.
fn detect_from_sized(
    base: &[Vec<(usize, f64)>],
    sizes: Option<&[f64]>,
    gamma: f64,
    obj: Objective,
    seed: u64,
    start: &[usize],
) -> Vec<usize> {
    let n = base.len();
    let mut membership: Vec<usize> = (0..n).collect(); // original node → current super-node
    let mut g = match sizes {
        Some(s) => Graph::from_adj_sized(base.to_vec(), s.to_vec()),
        None => Graph::from_adj(base.to_vec()),
    };
    let mut init: Vec<usize> = start.to_vec(); // seed partition for the current local-move
    let mut level = 0u64;
    loop {
        let part = one_level(&g, obj, gamma, seed.wrapping_add(level), &init);
        let refined = refine(&g, &part, obj, gamma, seed.wrapping_add(level));
        for m in membership.iter_mut() {
            *m = refined[*m];
        }
        let n_ref = refined.iter().max().map_or(0, |&x| x + 1);
        if n_ref == g.len() {
            break; // refinement did not coarsen the graph ⇒ converged
        }
        let (next, coarse_seed) = aggregate(&g, &part, &refined, n_ref);
        g = next;
        init = coarse_seed;
        level += 1;
    }
    relabel(&membership)
}

/// One local-moving pass, started from partition `init`: each node greedily joins the neighbouring
/// community with the largest quality gain, iterating until no node moves.
fn one_level(g: &Graph, obj: Objective, gamma: f64, seed: u64, init: &[usize]) -> Vec<usize> {
    let n = g.len();
    let mut comm = init.to_vec();
    let nc = comm.iter().max().map_or(0, |&x| x + 1);
    let mut tot_deg = vec![0.0; nc];
    let mut tot_size = vec![0.0; nc];
    for i in 0..n {
        tot_deg[comm[i]] += g.degree[i];
        tot_size[comm[i]] += g.size[i];
    }
    let order = shuffled(n, seed);
    let mut moved = true;
    while moved {
        moved = false;
        for &i in &order {
            let ci = comm[i];
            let (ki, si) = (g.degree[i], g.size[i]);
            let mut w_to: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &g.adj[i] {
                *w_to.entry(comm[j]).or_insert(0.0) += w;
            }
            tot_deg[ci] -= ki;
            tot_size[ci] -= si;
            let score = |c: usize, td: &[f64], ts: &[f64]| {
                gain(
                    obj,
                    gamma,
                    g.two_m,
                    w_to.get(&c).copied().unwrap_or(0.0),
                    ki,
                    si,
                    td[c],
                    ts[c],
                )
            };
            let mut best_c = ci;
            let mut best = score(ci, &tot_deg, &tot_size);
            // Community order, not `HashMap` order -- see the note in `refine`.
            let mut cands: Vec<usize> = w_to.keys().copied().collect();
            cands.sort_unstable();
            for c in cands {
                let s = score(c, &tot_deg, &tot_size);
                if s > best {
                    best = s;
                    best_c = c;
                }
            }
            tot_deg[best_c] += ki;
            tot_size[best_c] += si;
            comm[i] = best_c;
            if best_c != ci {
                moved = true;
            }
        }
    }
    relabel(&comm)
}

/// Refinement: within each community of `part`, grow sub-communities by merging *singleton* nodes
/// along edges into a same-community sub-community of positive gain. Because a node is only ever
/// merged into a sub-community it has an edge to, every refined sub-community is connected.
fn refine(g: &Graph, part: &[usize], obj: Objective, gamma: f64, seed: u64) -> Vec<usize> {
    let n = g.len();
    let mut refined: Vec<usize> = (0..n).collect();
    let mut singleton = vec![true; n];
    let mut tot_deg = g.degree.clone();
    let mut tot_size = g.size.clone();
    for &v in &shuffled(n, seed ^ 0x9e37_79b9) {
        if !singleton[v] {
            continue; // only merge nodes still alone in their refined sub-community
        }
        let cv = part[v];
        let (kv, sv) = (g.degree[v], g.size[v]);
        let mut w_to: HashMap<usize, f64> = HashMap::new();
        for &(j, w) in &g.adj[v] {
            if part[j] == cv {
                *w_to.entry(refined[j]).or_insert(0.0) += w; // only sub-communities inside cv
            }
        }
        tot_deg[refined[v]] -= kv;
        tot_size[refined[v]] -= sv;
        let mut best_c = refined[v];
        let mut best = 0.0; // require a strictly positive gain to leave the singleton
        // Scan the candidates in sub-community order: `w_to` is a `HashMap`, whose iteration order
        // depends on a per-instance random seed, and the strict `>` below keeps whichever tied
        // candidate came first. Left unordered, two identical calls disagree.
        let mut cands: Vec<(usize, f64)> = w_to.into_iter().collect();
        cands.sort_by_key(|&(c, _)| c);
        for (c, w) in cands {
            let s = gain(obj, gamma, g.two_m, w, kv, sv, tot_deg[c], tot_size[c]);
            if s > best {
                best = s;
                best_c = c;
            }
        }
        tot_deg[best_c] += kv;
        tot_size[best_c] += sv;
        if best_c != refined[v] {
            refined[v] = best_c;
            singleton[best_c] = false; // the target sub-community is no longer a lone singleton
        }
    }
    relabel(&refined)
}

/// Aggregate each refined sub-community to a super-node; return the aggregated graph and the seed
/// partition for the next level (each super-node keeps the *pre-refinement* community of its members,
/// so the next local-move can re-split them).
fn aggregate(g: &Graph, part: &[usize], refined: &[usize], nr: usize) -> (Graph, Vec<usize>) {
    let mut degree = vec![0.0; nr];
    let mut size = vec![0.0; nr];
    let mut coarse = vec![usize::MAX; nr];
    let mut acc: Vec<HashMap<usize, f64>> = vec![HashMap::new(); nr];
    for (i, row) in g.adj.iter().enumerate() {
        let ri = refined[i];
        degree[ri] += g.degree[i];
        size[ri] += g.size[i];
        coarse[ri] = part[i]; // all members of a refined sub-community share one coarse community
        for &(j, w) in row {
            let rj = refined[j];
            if ri != rj {
                *acc[ri].entry(rj).or_insert(0.0) += w;
            }
        }
    }
    // Sorted by neighbour index: the row order decides the order the weighted degree is summed
    // in, and floating-point addition is not associative, so an unordered row makes `degree` --
    // and every gain computed from it -- depend on the `HashMap` seed.
    let adj: Vec<Vec<(usize, f64)>> = acc
        .into_iter()
        .map(|m| {
            let mut row: Vec<(usize, f64)> = m.into_iter().collect();
            row.sort_by_key(|&(j, _)| j);
            row
        })
        .collect();
    let two_m = g.two_m;
    (
        Graph {
            adj,
            degree,
            size,
            two_m,
        },
        relabel(&coarse),
    )
}

/// Map arbitrary ids to a contiguous `0..k` in first-seen order.
fn relabel(labels: &[usize]) -> Vec<usize> {
    let mut map: HashMap<usize, usize> = HashMap::new();
    labels
        .iter()
        .map(|&l| {
            let next = map.len();
            *map.entry(l).or_insert(next)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros};
    use std::collections::HashSet;

    fn n_distinct(labels: &[usize]) -> usize {
        labels.iter().copied().collect::<HashSet<_>>().len()
    }

    fn modularity(base: &[Vec<(usize, f64)>], labels: &[usize], gamma: f64) -> f64 {
        let deg: Vec<f64> = base
            .iter()
            .map(|r| r.iter().map(|&(_, w)| w).sum())
            .collect();
        let two_m: f64 = deg.iter().sum();
        let internal: f64 = (0..base.len())
            .map(|i| {
                base[i]
                    .iter()
                    .filter(|&&(j, _)| labels[i] == labels[j])
                    .map(|&(_, w)| w)
                    .sum::<f64>()
            })
            .sum();
        let mut cd: HashMap<usize, f64> = HashMap::new();
        for (i, &l) in labels.iter().enumerate() {
            *cd.entry(l).or_insert(0.0) += deg[i];
        }
        (internal - gamma * cd.values().map(|&s| s * s).sum::<f64>() / two_m) / two_m
    }

    /// True if every community is connected in the base graph (Leiden's guarantee).
    fn all_communities_connected(base: &[Vec<(usize, f64)>], labels: &[usize]) -> bool {
        let n = labels.len();
        let mut seen = vec![false; n];
        for s in 0..n {
            if seen[s] {
                continue;
            }
            let mut stack = vec![s];
            seen[s] = true;
            let mut members = 1;
            while let Some(u) = stack.pop() {
                for &(v, _) in &base[u] {
                    if !seen[v] && labels[v] == labels[u] {
                        seen[v] = true;
                        members += 1;
                        stack.push(v);
                    }
                }
            }
            // the BFS from s covered one connected same-label component; it must equal the whole
            // community for `s`, i.e. no other node shares s's label outside this component.
            if members != labels.iter().filter(|&&l| l == labels[s]).count() {
                return false;
            }
        }
        true
    }

    #[test]
    fn leiden_discovers_separated_blobs_without_k() {
        let mut rng = SplitMix64::new(4);
        let centers = [[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.5);
        let (micros, p2m) = grid_micros(&pts, 1.0);
        let labels = leiden(&micros, 1.0, Objective::Modularity, 1, 0.0, 0.0, 2).labels;
        assert_eq!(n_distinct(&labels), 3);
        let pred: Vec<usize> = p2m.iter().map(|&m| labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.95);
    }

    #[test]
    fn leiden_communities_are_connected_and_beat_trivial() {
        let mut rng = SplitMix64::new(1);
        let centers = [[0.0, 0.0], [5.0, 0.0]];
        let (pts, _t) = blobs(&mut rng, 300, &centers, 1.0);
        let (micros, _) = grid_micros(&pts, 0.5);
        let centers_f: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let base = knn_affinity::<f64>(&centers_f);
        let labels = leiden(&micros, 1.0, Objective::Modularity, 1, 0.0, 0.0, 2).labels;
        // Leiden's key guarantee: every community is internally connected.
        assert!(all_communities_connected(&base, &labels));
        // …and the checker is not vacuous — two far, non-adjacent nodes sharing a label is caught.
        let mut broken = vec![1usize; micros.len()];
        broken[0] = 0;
        broken[micros.len() - 1] = 0;
        assert!(!all_communities_connected(&base, &broken));
        let q = modularity(&base, &labels, 1.0);
        assert!(q > modularity(&base, &vec![0; micros.len()], 1.0));
        assert!(q > modularity(&base, &(0..micros.len()).collect::<Vec<_>>(), 1.0));
    }

    #[test]
    fn leiden_resolution_controls_granularity() {
        let mut rng = SplitMix64::new(2);
        let (pts, _t) = blobs(&mut rng, 400, &[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], 0.9);
        let (micros, _) = grid_micros(&pts, 0.4);
        let coarse =
            n_distinct(&leiden(&micros, 0.5, Objective::Modularity, 1, 0.0, 0.0, 2).labels);
        let fine = n_distinct(&leiden(&micros, 2.0, Objective::Modularity, 1, 0.0, 0.0, 2).labels);
        assert!(
            fine >= coarse,
            "higher γ should not yield fewer communities ({fine} vs {coarse})"
        );
    }

    #[test]
    fn leiden_cpm_objective_partitions_blobs() {
        let mut rng = SplitMix64::new(3);
        let centers = [[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.5);
        let (micros, p2m) = grid_micros(&pts, 1.0);
        let labels = leiden(&micros, 0.05, Objective::Cpm, 1, 0.0, 0.0, 2).labels;
        let pred: Vec<usize> = p2m.iter().map(|&m| labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.9);
    }

    #[test]
    fn leiden_single_feature_is_one_community() {
        let (micros, _) = grid_micros(&[vec![1.0, 2.0]], 1.0);
        assert_eq!(
            leiden(&micros, 1.0, Objective::Modularity, 1, 0.0, 0.0, 2).labels,
            vec![0]
        );
    }

    /// Two triangles joined by a single edge: the modularity-optimal partition is the two triangles,
    /// and the join is the one edge a refinement step must not swallow.
    fn two_triangles() -> Vec<Vec<(usize, f64)>> {
        vec![
            vec![(1, 1.0), (2, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(0, 1.0), (1, 1.0), (3, 1.0)],
            vec![(2, 1.0), (4, 1.0), (5, 1.0)],
            vec![(3, 1.0), (5, 1.0)],
            vec![(3, 1.0), (4, 1.0)],
        ]
    }

    #[test]
    fn gain_is_the_objective_specific_null_model_correction() {
        // Modularity: w_to − γ·tot_deg·k_i / 2m = 3 − 1·5·4/10 = 1.
        let m = gain(Objective::Modularity, 1.0, 10.0, 3.0, 4.0, 9.0, 5.0, 7.0);
        assert!((m - 1.0).abs() < 1e-12, "modularity gain = {m}");
        // The CPM branch must ignore degree and 2m entirely: w_to − γ·tot_size·s_i = 3 − 0.5·2·2 = 1.
        // `ki`, `tot_deg` and `two_m` are deliberately set to values that would move a modularity
        // score, so a branch that reaches for them is caught rather than silently agreeing.
        let c = gain(Objective::Cpm, 0.5, 10.0, 3.0, 9.0, 2.0, 7.0, 2.0);
        assert!((c - 1.0).abs() < 1e-12, "CPM gain = {c}");
        // Resolution scales only the penalty.
        let m2 = gain(Objective::Modularity, 2.0, 10.0, 3.0, 4.0, 9.0, 5.0, 7.0);
        assert!((m2 - (-1.0)).abs() < 1e-12, "doubled resolution = {m2}");
    }

    #[test]
    fn shuffled_is_a_permutation_that_depends_on_the_seed() {
        for n in [0usize, 1, 2, 9, 64] {
            let o = shuffled(n, 42);
            assert_eq!(o.len(), n);
            let mut sorted = o.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..n).collect::<Vec<_>>(),
                "n = {n} is not a permutation"
            );
        }
        assert_eq!(shuffled(32, 7), shuffled(32, 7), "not deterministic");
        assert_ne!(shuffled(32, 7), shuffled(32, 8), "seed is ignored");
        assert_ne!(
            shuffled(32, 7),
            (0..32).collect::<Vec<_>>(),
            "nothing was shuffled"
        );
    }

    #[test]
    fn pass_seed_keeps_the_callers_seed_first_and_then_avoids_the_level_seeds() {
        for seed in [0u64, 7, u64::MAX - 3] {
            assert_eq!(pass_seed(seed, 0), seed, "pass 0 must reuse seed {seed}");
            let later: Vec<u64> = (1..8).map(|p| pass_seed(seed, p)).collect();
            for (i, &s) in later.iter().enumerate() {
                assert!(
                    (0..64).all(|level| s != seed.wrapping_add(level)),
                    "pass {} of seed {seed} reuses a level seed",
                    i + 1
                );
                assert!(!later[..i].contains(&s), "pass {} repeats a seed", i + 1);
            }
        }
    }

    #[test]
    fn relabel_compacts_ids_in_first_seen_order() {
        assert_eq!(relabel(&[7, 7, 3, 7, 3, 9]), vec![0, 0, 1, 0, 1, 2]);
        assert_eq!(relabel(&[]), Vec::<usize>::new());
        assert_eq!(relabel(&[5]), vec![0]);
    }

    #[test]
    fn one_level_finds_the_two_triangles_and_beats_the_trivial_partitions() {
        let base = two_triangles();
        let g = Graph::from_adj(base.clone());
        let init: Vec<usize> = (0..6).collect();
        let part = one_level(&g, Objective::Modularity, 1.0, 3, &init);
        assert_eq!(
            relabel(&part),
            vec![0, 0, 0, 1, 1, 1],
            "partition = {part:?}"
        );

        // Independently scored: the recovered partition must beat both degenerate ones.
        let q = modularity(&base, &part, 1.0);
        assert!(q > modularity(&base, &[0; 6], 1.0), "lost to one community");
        assert!(q > modularity(&base, &init, 1.0), "lost to all-singletons");
    }

    #[test]
    fn refine_never_crosses_a_community_boundary() {
        // The defining property of Leiden's refinement: every refined sub-community lies inside
        // exactly one community of the input partition. Violating it silently lets the aggregation
        // step merge nodes the local move had separated.
        let g = Graph::from_adj(two_triangles());
        let part = vec![0, 0, 0, 1, 1, 1];
        for seed in 0..16u64 {
            let refined = refine(&g, &part, Objective::Modularity, 1.0, seed);
            assert_eq!(refined.len(), 6);
            let mut owner: HashMap<usize, usize> = HashMap::new();
            for (i, &r) in refined.iter().enumerate() {
                match owner.get(&r) {
                    Some(&c) => assert_eq!(
                        c, part[i],
                        "seed {seed}: sub-community {r} spans two communities"
                    ),
                    None => {
                        owner.insert(r, part[i]);
                    }
                }
            }
            assert!(
                n_distinct(&refined) >= 2,
                "seed {seed}: refinement collapsed both communities into one"
            );
        }
    }

    #[test]
    fn aggregate_preserves_total_mass_and_drops_self_loops() {
        let g = Graph::from_adj(two_triangles());
        let part = vec![0, 0, 0, 1, 1, 1];
        let refined = vec![0, 0, 0, 1, 1, 1];
        let (next, coarse) = aggregate(&g, &part, &refined, 2);

        assert_eq!(next.len(), 2, "one super-node per refined sub-community");
        assert_eq!(coarse.len(), 2, "one seed community per super-node");
        assert_eq!(
            coarse,
            vec![0, 1],
            "super-nodes lost their coarse community"
        );

        let before: f64 = g.degree.iter().sum();
        let after: f64 = next.degree.iter().sum();
        assert!((before - after).abs() < 1e-12, "degree {before} -> {after}");
        assert!((next.two_m - g.two_m).abs() < 1e-12, "2m changed");
        let size_before: f64 = g.size.iter().sum();
        assert!(
            (next.size.iter().sum::<f64>() - size_before).abs() < 1e-12,
            "node count changed"
        );

        // Only the single joining edge survives as an inter-super-node edge, once per direction.
        for (i, row) in next.adj.iter().enumerate() {
            assert!(row.iter().all(|&(j, _)| j != i), "self-loop at {i}");
        }
        assert_eq!(next.adj[0], vec![(1, 1.0)]);
        assert_eq!(next.adj[1], vec![(0, 1.0)]);
    }
    /// Fisher--Yates from the top down, drawing `j` from `0..=i`, written out rather than reused.
    fn reference_shuffle(n: usize, seed: u64) -> (Vec<usize>, SplitMix64) {
        let mut order: Vec<usize> = (0..n).collect();
        let mut rng = SplitMix64::new(seed);
        let mut i = n;
        while i > 1 {
            i -= 1;
            let draw = rng.next_u64();
            let j = (draw % (i as u64 + 1)) as usize;
            let (a, b) = (order[i], order[j]);
            order[i] = b;
            order[j] = a;
        }
        (order, rng)
    }

    #[test]
    fn shuffled_matches_an_independent_fisher_yates_draw_for_draw() {
        // Checking that the output is *a* permutation cannot see a wrong draw range: `% i` instead
        // of `% (i + 1)` still permutes, it just never leaves an element in place at the top.
        for n in [0usize, 1, 2, 3, 9, 64] {
            for seed in [0u64, 1, 42, 0xdead_beef] {
                let (want, want_rng) = reference_shuffle(n, seed);
                assert_eq!(shuffled(n, seed), want, "n = {n}, seed = {seed}");
                // The two consumed the same number of draws, so the stream ends in the same place.
                let mut got_rng = SplitMix64::new(seed);
                for _ in 1..n {
                    got_rng.next_u64();
                }
                let mut want_rng = want_rng;
                assert_eq!(
                    got_rng.next_u64(),
                    want_rng.next_u64(),
                    "n = {n}, seed = {seed}: draw count"
                );
            }
        }
    }

    /// Objective value of a whole partition, summed over communities from the graph itself --
    /// the quantity the local move is hill-climbing, independent of any incremental bookkeeping.
    fn quality(g: &Graph, obj: Objective, gamma: f64, comm: &[usize]) -> f64 {
        let nc = comm.iter().max().map_or(0, |&x| x + 1);
        let (mut inner, mut deg, mut size) = (vec![0.0; nc], vec![0.0; nc], vec![0.0; nc]);
        for i in 0..g.len() {
            deg[comm[i]] += g.degree[i];
            size[comm[i]] += g.size[i];
            for &(j, w) in &g.adj[i] {
                if comm[j] == comm[i] {
                    inner[comm[i]] += w;
                }
            }
        }
        match obj {
            Objective::Modularity => (0..nc)
                .map(|c| inner[c] / g.two_m - gamma * (deg[c] / g.two_m).powi(2))
                .sum(),
            Objective::Cpm => (0..nc)
                .map(|c| 0.5 * inner[c] - 0.5 * gamma * size[c] * (size[c] - 1.0))
                .sum(),
        }
    }

    /// The defining property of [`LeafGraph::Mass`]: a partition of the leaves scores on the
    /// quotient graph exactly what it scores on the point graph it summarises — each leaf `i`
    /// expanded into `n_i` points, every cross-leaf point pair joined at `w_ij`, none inside a leaf.
    /// Both objectives, several partitions, unequal masses.
    #[test]
    fn the_mass_graph_scores_a_leaf_partition_as_the_point_graph_scores_its_points() {
        let leaves = irregular_graph();
        let mass: Vec<f64> = (0..leaves.len())
            .map(|i| (1 + (i * 7) % 5) as f64)
            .collect();
        let mut quotient = leaves.clone();
        mass_quotient(&mut quotient, &mass);
        let q = Graph::from_adj_sized(quotient, mass.clone());

        let first: Vec<usize> = mass
            .iter()
            .scan(0, |next, &m| {
                let at = *next;
                *next += m as usize;
                Some(at)
            })
            .collect();
        let leaf_of: Vec<usize> = (0..leaves.len())
            .flat_map(|i| std::iter::repeat_n(i, mass[i] as usize))
            .collect();
        let expanded: Vec<Vec<(usize, f64)>> = leaf_of
            .iter()
            .map(|&i| {
                let mut row = Vec::new();
                for &(j, w) in &leaves[i] {
                    row.extend((0..mass[j] as usize).map(|t| (first[j] + t, w)));
                }
                row
            })
            .collect();
        let p = Graph::from_adj(expanded);

        let mut rng = SplitMix64::new(5);
        for _ in 0..6 {
            let part: Vec<usize> = (0..leaves.len())
                .map(|_| (rng.next_u64() % 3) as usize)
                .collect();
            let points: Vec<usize> = leaf_of.iter().map(|&i| part[i]).collect();
            for (obj, gamma) in [(Objective::Modularity, 1.0), (Objective::Cpm, 0.05)] {
                let (a, b) = (
                    quality(&q, obj, gamma, &part),
                    quality(&p, obj, gamma, &points),
                );
                assert!((a - b).abs() <= 1e-12 * b.abs().max(1.0), "{a} vs {b}");
            }
        }
    }

    /// At unit mass the quotient is the unit graph, so the two must agree label for label.
    #[test]
    fn the_mass_graph_of_unit_leaves_is_the_unit_graph() {
        let mut rng = SplitMix64::new(4);
        let centers = [[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]];
        let (pts, _) = blobs(&mut rng, 120, &centers, 1.5);
        let singles: Vec<crate::feature::Spherical<f64>> = pts
            .iter()
            .map(|p| {
                let mut f = crate::feature::Spherical::new(2);
                f.push(p, 1.0);
                f
            })
            .collect();
        for obj in [Objective::Modularity, Objective::Cpm] {
            let run = |graph| leiden_on(&singles, 1.0, obj, 3, 0.0, 0.0, 2, 4, graph).labels;
            assert_eq!(run(LeafGraph::Unit), run(LeafGraph::Mass));
        }
    }

    /// Four planted communities of five nodes joined in a ring by weak edges. The within-community
    /// weights carry a deterministic jitter so no two candidate moves score exactly alike: the
    /// local move breaks ties by `HashMap` iteration order, which a test must not depend on.
    fn planted_ring() -> (Vec<Vec<(usize, f64)>>, Vec<usize>) {
        let n = 20;
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let link = |adj: &mut Vec<Vec<(usize, f64)>>, i: usize, j: usize, w: f64| {
            adj[i].push((j, w));
            adj[j].push((i, w));
        };
        for c in 0..4 {
            for a in 0..5 {
                for b in (a + 1)..5 {
                    let (i, j) = (c * 5 + a, c * 5 + b);
                    link(&mut adj, i, j, 1.0 + 0.013 * ((i * 7 + j * 11) % 7) as f64);
                }
            }
        }
        for c in 0..4 {
            link(
                &mut adj,
                c * 5 + 4,
                ((c + 1) % 4) * 5,
                0.11 + 0.007 * c as f64,
            );
        }
        let planted: Vec<usize> = (0..n).map(|i| i / 5).collect();
        (adj, planted)
    }

    /// The local move's contract: it returns a partition no single node can improve on, and it
    /// scores at least as well as the planted answer. A corrupted running total of `tot_deg` /
    /// `tot_size` still produces *a* partition -- it just stops being an optimum of the objective
    /// the totals are supposed to describe.
    #[test]
    fn one_level_returns_a_partition_no_single_move_can_improve() {
        let (base, planted) = planted_ring();
        let g = Graph::from_adj(base.clone());
        let init: Vec<usize> = (0..g.len()).collect();
        for (name, obj) in [
            ("modularity", Objective::Modularity),
            ("cpm", Objective::Cpm),
        ] {
            for &gamma in &[0.25_f64, 0.5, 1.0] {
                for seed in [1u64, 17, 99] {
                    let part = one_level(&g, obj, gamma, seed, &init);
                    let ctx = format!("{name} gamma={gamma} seed={seed}");
                    let q = quality(&g, obj, gamma, &part);
                    assert!(
                        q >= quality(&g, obj, gamma, &planted) - 1e-9,
                        "{ctx}: lost to the planted partition ({q} vs {})",
                        quality(&g, obj, gamma, &planted)
                    );
                    for i in 0..g.len() {
                        let mut targets: Vec<usize> =
                            g.adj[i].iter().map(|&(j, _)| part[j]).collect();
                        targets.sort_unstable();
                        targets.dedup();
                        for c in targets {
                            if c == part[i] {
                                continue;
                            }
                            let mut alt = part.clone();
                            alt[i] = c;
                            let alt_q = quality(&g, obj, gamma, &relabel(&alt));
                            assert!(
                                alt_q <= q + 1e-9,
                                "{ctx}: moving {i} to {c} improves {q} -> {alt_q}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Resolution has to reach the local move through the null-model totals. Totals stuck at zero
    /// (or negated) leave the gain resolution-free, and the granularity stops responding.
    #[test]
    fn the_resolution_moves_the_local_move_granularity() {
        let (base, _) = planted_ring();
        let g = Graph::from_adj(base);
        let init: Vec<usize> = (0..g.len()).collect();
        for (name, obj) in [
            ("modularity", Objective::Modularity),
            ("cpm", Objective::Cpm),
        ] {
            let counts: Vec<usize> = [0.05_f64, 0.5, 2.0, 8.0, 40.0]
                .iter()
                .map(|&gamma| n_distinct(&one_level(&g, obj, gamma, 5, &init)))
                .collect();
            assert!(
                counts.windows(2).all(|w| w[0] <= w[1]),
                "{name}: granularity is not monotone in the resolution: {counts:?}"
            );
            assert!(
                counts.first() < counts.last(),
                "{name}: the resolution changed nothing: {counts:?}"
            );
        }
    }

    /// A node leaves its singleton only for a *strictly* positive gain. Unit weights and unit node
    /// sizes under CPM at `gamma = 1` make the first merge score exactly `1 - 1·1·1 = 0`, which is
    /// the one input that tells `>` and `>=` apart.
    #[test]
    fn refine_requires_a_strictly_positive_gain_to_leave_a_singleton() {
        let g = Graph::from_adj(two_triangles());
        let part = vec![0, 0, 0, 1, 1, 1];
        for seed in 0..8u64 {
            assert_eq!(
                refine(&g, &part, Objective::Cpm, 1.0, seed),
                (0..6).collect::<Vec<_>>(),
                "seed {seed}: a zero-gain merge was taken"
            );
        }
        // Just below the break-even resolution the same merges are strictly profitable.
        let merged = refine(&g, &part, Objective::Cpm, 0.9, 3);
        assert!(
            n_distinct(&merged) < 6,
            "nothing merged at gamma = 0.9: {merged:?}"
        );
    }

    /// Every refined sub-community must be connected inside the graph -- the property that makes
    /// Leiden's aggregation safe, and the one a corrupted running total breaks first.
    #[test]
    fn every_refined_sub_community_is_connected() {
        let (base, planted) = planted_ring();
        let g = Graph::from_adj(base.clone());
        for (name, obj) in [
            ("modularity", Objective::Modularity),
            ("cpm", Objective::Cpm),
        ] {
            for &gamma in &[0.25_f64, 1.0] {
                for seed in 0..6u64 {
                    let refined = refine(&g, &planted, obj, gamma, seed);
                    assert!(
                        all_communities_connected(&base, &refined),
                        "{name} gamma={gamma} seed={seed}: {refined:?} is disconnected"
                    );
                }
            }
        }
    }
    /// Microclusters whose centres lie on a ring -- so a centre-only affinity links each to its
    /// ring neighbours -- but whose spread alternates between two orthogonal orientations. Only a
    /// geometry-aware affinity can see the second structure.
    fn oriented_ring() -> Vec<crate::feature::Full<f64>> {
        (0..24)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / 24.0;
                let (cx, cy) = (5.0 * t.cos(), 5.0 * t.sin());
                let th = if i % 2 == 0 {
                    0.0
                } else {
                    std::f64::consts::FRAC_PI_2
                };
                let mut cf = crate::feature::Full::new(2);
                for k in 0..12 {
                    let s = (k as f64 - 5.5) * 0.25;
                    let u = ((k % 3) as f64 - 1.0) * 0.03;
                    cf.push(
                        &[
                            cx + s * th.cos() - u * th.sin(),
                            cy + s * th.sin() + u * th.cos(),
                        ],
                        1.0,
                    );
                }
                cf
            })
            .collect()
    }

    /// `leiden` reaches for the log-covariance and tangent terms exactly when their weights are
    /// positive, and for neither otherwise. Nothing in the Rust suite exercised the switch, so a
    /// guard that fired on the wrong side -- or demanded both weights instead of either -- silently
    /// dropped the geometry the caller asked for.
    #[test]
    fn leiden_reaches_for_the_geometry_terms_exactly_when_their_weights_are_positive() {
        let feats = oriented_ring();
        let centers: Vec<Vec<f64>> = feats.iter().map(|f| f.mean().to_vec()).collect();
        let lc = to_f64_tensors(log_covariances(&feats));
        // Rank 1: in a 2-D space a rank-2 tangent basis spans everything, so every pair of
        // subspaces is at Grassmann distance 0 and the term contributes nothing.
        let tg = to_f64_tensors(tangent_bases(&feats, 1));
        let (w, seed, gamma, rank) = (0.8_f64, 7u64, 1.0_f64, 1usize);

        let routes = [
            (
                (0.0, 0.0),
                detect(
                    &knn_affinity::<f64>(&centers),
                    gamma,
                    Objective::Modularity,
                    seed,
                ),
            ),
            (
                (w, 0.0),
                detect(
                    &knn_affinity_geo::<f64>(&centers, Some((&lc, w)), None),
                    gamma,
                    Objective::Modularity,
                    seed,
                ),
            ),
            (
                (0.0, w),
                detect(
                    &knn_affinity_geo::<f64>(&centers, None, Some((&tg, w))),
                    gamma,
                    Objective::Modularity,
                    seed,
                ),
            ),
            (
                (w, w),
                detect(
                    &knn_affinity_geo::<f64>(&centers, Some((&lc, w)), Some((&tg, w))),
                    gamma,
                    Objective::Modularity,
                    seed,
                ),
            ),
        ];
        for ((cw, tw), want) in &routes {
            assert_eq!(
                &leiden(&feats, gamma, Objective::Modularity, seed, *cw, *tw, rank).labels,
                want,
                "cov_weight = {cw}, tangent_weight = {tw}"
            );
        }
        // Each geometry route has to actually move the answer, or the assertions above are
        // vacuous -- a guard that never fires reads exactly like one that always does.
        assert_ne!(routes[0].1, routes[1].1, "cov_weight changed nothing");
        assert_ne!(routes[0].1, routes[2].1, "tangent_weight changed nothing");
        assert_ne!(routes[0].1, routes[3].1, "neither weight changed anything");
    }
    /// Two identical calls must agree. They did not: `refine` picked its target sub-community by
    /// scanning a `HashMap`, whose iteration order comes from a per-instance random seed, and the
    /// strict `>` kept whichever tied candidate happened to come first. `leiden`'s `seed` only
    /// ever controlled the node visit order, so nothing pinned the tie-break.
    #[test]
    fn detect_is_reproducible_within_a_process() {
        let (base, _) = planted_ring();
        assert_eq!(
            detect(&base, 1.0, Objective::Modularity, 7),
            detect(&base, 1.0, Objective::Modularity, 7),
            "planted ring: same seed, same graph, different answer"
        );
        // A symmetric ring of microclusters is where the ties are: every node sees the same gain
        // toward either side of it.
        let feats = oriented_ring();
        let centers: Vec<Vec<f64>> = feats.iter().map(|f| f.mean().to_vec()).collect();
        let g = knn_affinity::<f64>(&centers);
        assert_eq!(
            detect(&g, 1.0, Objective::Modularity, 7),
            detect(&g, 1.0, Objective::Modularity, 7),
            "oriented ring: same seed, same graph, different answer"
        );
        assert_eq!(
            leiden(&feats, 1.0, Objective::Modularity, 7, 0.0, 0.0, 2).labels,
            detect(&g, 1.0, Objective::Modularity, 7),
            "leiden disagrees with its own pipeline"
        );
    }
    /// A weighted random graph with a wide degree spread and no planted structure, so the
    /// null-model correction -- not the raw edge weight -- decides almost every move. `planted_ring`
    /// cannot serve here: its blocks are so much denser than the links between them that the greedy
    /// recovers them whatever the totals say, which is exactly why a corrupted total hid in it.
    fn irregular_graph() -> Vec<Vec<(usize, f64)>> {
        let n = 22;
        let mut rng = SplitMix64::new(0xc0ff_ee11);
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for a in 0..n {
            for b in (a + 1)..n {
                let p = (92 - 3 * (a + b) as i64).clamp(6, 92) as u64;
                if rng.next_u64() % 100 < p {
                    let w = 0.2 + (rng.next_u64() % 20) as f64 * 0.1;
                    adj[a].push((b, w));
                    adj[b].push((a, w));
                }
            }
        }
        adj
    }

    /// `one_level` with the null-model totals recomputed from the whole graph at every node instead
    /// of carried incrementally. Same visit order, same candidate order, same strict `>`; the only
    /// difference is that nothing is remembered between steps, so a corrupted running total shows
    /// up as a different greedy trajectory rather than as a partition that still looks plausible.
    fn reference_one_level(
        g: &Graph,
        obj: Objective,
        gamma: f64,
        seed: u64,
        init: &[usize],
    ) -> Vec<usize> {
        let n = g.len();
        let mut comm = init.to_vec();
        let order = shuffled(n, seed);
        loop {
            let mut moved = false;
            for &i in &order {
                let ci = comm[i];
                let nc = comm.iter().max().map_or(0, |&x| x + 1);
                let (mut td, mut ts) = (vec![0.0; nc], vec![0.0; nc]);
                for v in 0..n {
                    if v != i {
                        td[comm[v]] += g.degree[v];
                        ts[comm[v]] += g.size[v];
                    }
                }
                let mut w_to: std::collections::BTreeMap<usize, f64> =
                    std::collections::BTreeMap::new();
                for &(j, w) in &g.adj[i] {
                    *w_to.entry(comm[j]).or_insert(0.0) += w;
                }
                let sc = |c: usize| {
                    gain(
                        obj,
                        gamma,
                        g.two_m,
                        w_to.get(&c).copied().unwrap_or(0.0),
                        g.degree[i],
                        g.size[i],
                        td[c],
                        ts[c],
                    )
                };
                let mut best_c = ci;
                let mut best = sc(ci);
                for &c in w_to.keys() {
                    let s = sc(c);
                    if s > best {
                        best = s;
                        best_c = c;
                    }
                }
                if best_c != ci {
                    comm[i] = best_c;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        relabel(&comm)
    }

    /// `refine` with the same treatment: sub-community totals recomputed rather than carried.
    fn reference_refine(
        g: &Graph,
        part: &[usize],
        obj: Objective,
        gamma: f64,
        seed: u64,
    ) -> Vec<usize> {
        let n = g.len();
        let mut refined: Vec<usize> = (0..n).collect();
        let mut singleton = vec![true; n];
        for &v in &shuffled(n, seed ^ 0x9e37_79b9) {
            if !singleton[v] {
                continue;
            }
            let cv = part[v];
            let (mut td, mut ts) = (vec![0.0; n], vec![0.0; n]);
            for u in 0..n {
                if u != v {
                    td[refined[u]] += g.degree[u];
                    ts[refined[u]] += g.size[u];
                }
            }
            let mut w_to: std::collections::BTreeMap<usize, f64> =
                std::collections::BTreeMap::new();
            for &(j, w) in &g.adj[v] {
                if part[j] == cv {
                    *w_to.entry(refined[j]).or_insert(0.0) += w;
                }
            }
            let mut best_c = refined[v];
            let mut best = 0.0;
            for (&c, &w) in &w_to {
                let s = gain(obj, gamma, g.two_m, w, g.degree[v], g.size[v], td[c], ts[c]);
                if s > best {
                    best = s;
                    best_c = c;
                }
            }
            if best_c != refined[v] {
                refined[v] = best_c;
                singleton[best_c] = false;
            }
        }
        relabel(&refined)
    }

    /// The running totals are the whole null model. Nothing compared them against a recomputation,
    /// so an initial total left at zero -- or negated, or divided instead of subtracted -- still
    /// produced a partition that reads as plausible: connected, resolution-responsive, and a local
    /// optimum of *something*. It is just no longer a local optimum of the stated objective.
    #[test]
    fn the_running_null_model_totals_match_a_from_scratch_recomputation() {
        let (ring, _) = planted_ring();
        let mut moved_any = 0usize;
        for (gname, base) in [("planted ring", ring), ("irregular", irregular_graph())] {
            let g = Graph::from_adj(base);
            let n = g.len();
            // Three seed partitions, because an initial total is only a genuine sum when a
            // community holds more than one node -- from all-singletons the first pass rebuilds
            // them anyway, and `detect` seeds every level above the first from the previous one.
            let inits: [(&str, Vec<usize>); 3] = [
                ("singletons", (0..n).collect()),
                ("blocks", (0..n).map(|i| i / 5).collect()),
                ("halves", (0..n).map(|i| i / (n / 2)).collect()),
            ];
            for (name, obj) in [
                ("modularity", Objective::Modularity),
                ("cpm", Objective::Cpm),
            ] {
                for &gamma in &[0.1_f64, 0.5, 1.0, 3.0] {
                    for seed in [1u64, 17, 99, 2024] {
                        for (iname, init) in &inits {
                            let ctx =
                                format!("{gname} {name} gamma={gamma} seed={seed} init={iname}");
                            let part = one_level(&g, obj, gamma, seed, init);
                            assert_eq!(
                                part,
                                reference_one_level(&g, obj, gamma, seed, init),
                                "{ctx}: local move"
                            );
                            assert_eq!(
                                refine(&g, &part, obj, gamma, seed),
                                reference_refine(&g, &part, obj, gamma, seed),
                                "{ctx}: refinement"
                            );
                            if n_distinct(&part) < n {
                                moved_any += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(
            moved_any > 16,
            "the sweep hardly ever left the all-singleton partition: {moved_any}"
        );
    }

    /// A symmetric 3-community sparse SBM: rate `a/n` inside a community, `b/n` between, labels in
    /// contiguous node-id blocks. Each block pair gets exactly `round(pairs · rate)` distinct edges
    /// drawn uniformly -- a G(n, m)-style SBM, whose edge count differs from the Bernoulli one by
    /// O(√m) and in nothing detection depends on.
    fn sbm3(n: usize, a: f64, b: f64, seed: u64) -> (Vec<Vec<(usize, f64)>>, Vec<usize>) {
        let block = n / 3;
        let n = 3 * block;
        let mut rng = SplitMix64::new(seed);
        let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for p in 0..3 {
            for q in p..3 {
                let (pairs, rate) = if p == q {
                    ((block * (block - 1) / 2) as f64, a)
                } else {
                    ((block * block) as f64, b)
                };
                let m = (pairs * rate / n as f64).round() as usize;
                let mut drawn = 0;
                while drawn < m {
                    let u = p * block + (rng.next_u64() % block as u64) as usize;
                    let v = q * block + (rng.next_u64() % block as u64) as usize;
                    if u != v && seen.insert((u.min(v), u.max(v))) {
                        adj[u].push((v, 1.0));
                        adj[v].push((u, 1.0));
                        drawn += 1;
                    }
                }
            }
        }
        (adj, (0..n).map(|i| i / block).collect())
    }

    /// `(a, b)` at signal `s = (a − b)²/(3(a + 2b))` and mean degree `c = (a + 2b)/3`. Weak recovery
    /// is possible iff `s > 1` (the Kesten–Stigum line, exact for three communities).
    fn sbm_rates(c: f64, s: f64) -> (f64, f64) {
        let t = (c * s).sqrt();
        (c + 2.0 * t, c - t)
    }

    /// `detect` on the same graph with its node ids shuffled, labels mapped back: the leak-free
    /// reference for a graph whose communities sit in contiguous id blocks.
    fn detect_shuffled(base: &[Vec<(usize, f64)>], seed: u64) -> Vec<usize> {
        let perm = shuffled(base.len(), seed ^ 0x5eed);
        let mut moved: Vec<Vec<(usize, f64)>> = vec![Vec::new(); base.len()];
        for (i, row) in base.iter().enumerate() {
            moved[perm[i]] = row.iter().map(|&(j, w)| (perm[j], w)).collect();
        }
        let lab = detect(&moved, 1.0, Objective::Modularity, seed);
        perm.iter().map(|&p| lab[p]).collect()
    }

    /// Planted-partition controls on the symmetric sparse 3-community SBM at mean degree 6, its
    /// communities in contiguous node-id blocks. Below the Kesten–Stigum line `s = 1` every
    /// estimator's expected overlap tends to chance (three-state reconstruction preprint, OpenAI,
    /// 2026-09-25, Corollary 1.2), so at `s = 0.6` an ARI above the finite-n floor can only have come
    /// from the node ids. Far above it the core has to find the blocks.
    ///
    /// Thresholds from `measure_sbm_detection` and these seeds at n = 3000: the floor at `s = 0.6`
    /// reads 0.0015 / 0.0015 / 0.0021 (graph seeds 0..40: mean 0.0021, sd 0.0007, max 0.0037) and
    /// scales like 5/n (median 0.0048 at n = 1500, which is why n is not smaller), with shuffled ids
    /// reading the same; `s = 5` reads 0.8466 / 0.8715 / 0.8863. A local move started from id blocks
    /// of 100 nodes reads 0.0070–0.0138 over those 40 graphs and fails here; blocks of 50
    /// (0.0022–0.0061) overlap the floor, so a leak that weak is below what n = 3000 can resolve.
    #[test]
    fn leiden_on_a_planted_partition_neither_leaks_node_ids_nor_misses_a_strong_signal() {
        const N: usize = 3000;
        for seed in 0u64..3 {
            let (a, b) = sbm_rates(6.0, 0.6);
            let (base, truth) = sbm3(N, a, b, seed);
            let below = ari(&detect(&base, 1.0, Objective::Modularity, seed), &truth);
            assert!(
                below < 0.005,
                "s = 0.6, seed {seed}: ARI {below:.4} is above the no-information floor"
            );
            let (a, b) = sbm_rates(6.0, 5.0);
            let (base, truth) = sbm3(N, a, b, seed);
            let above = ari(&detect(&base, 1.0, Objective::Modularity, seed), &truth);
            assert!(
                above > 0.75,
                "s = 5, seed {seed}: ARI {above:.4}, the planted blocks were not recovered"
            );
        }
    }

    /// FNV-1a over the labels, so a test can pin a partition in one number.
    fn fnv1a(labels: &[usize]) -> u64 {
        labels.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &l| {
            (l as u64).to_le_bytes().iter().fold(h, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
            })
        })
    }

    /// One pass is the 1.0 algorithm, bit for bit. The digests were taken from `detect` before the
    /// pass loop existed, on unit-weight SBM graphs, so they hold on every platform: the graph
    /// carries no floating-point input that a SIMD kernel could round differently.
    #[test]
    fn one_pass_keeps_the_released_labels() {
        let pinned: [(f64, u64, Objective, f64, u64); 6] = [
            (0.6, 0, Objective::Modularity, 1.0, 0xacd4_88c3_27b4_c126),
            (4.0, 0, Objective::Modularity, 1.0, 0xf830_2a6c_8052_92ee),
            (4.0, 1, Objective::Modularity, 1.0, 0x9b7f_ec32_5986_ec27),
            (4.0, 2, Objective::Modularity, 1.0, 0xdba1_5c79_663d_c2a1),
            (5.0, 1, Objective::Modularity, 1.0, 0x2f52_25be_bce7_128c),
            (4.0, 0, Objective::Cpm, 0.002, 0xd77d_3088_b623_8840),
        ];
        for (s, seed, obj, gamma, want) in pinned {
            let (a, b) = sbm_rates(6.0, s);
            let (base, _) = sbm3(3000, a, b, seed);
            let (one, passes) = detect_passes(&base, gamma, obj, seed, 1);
            assert_eq!(fnv1a(&one), want, "s = {s}, seed = {seed}: one pass moved");
            assert_eq!(passes, 1);
            assert_eq!(
                detect_passes(&base, gamma, obj, seed, 0),
                (one, 1),
                "0 is read as 1"
            );
        }
    }

    /// Passes after the first, on the 3-community SBM at mean degree 6 and n = 3000. One pass reads
    /// ARI 0.587 / 0.697 / 0.657 at `s = 4` (graph seeds 0/1/2); two read 0.897 / 0.893 / 0.931, and
    /// until stable 0.908 / 0.933 / 0.933 after 12 / 11 / 7 passes. Modularity never falls from one
    /// pass to the next, and the last pass returns its own input. Below the threshold (`s = 0.6`)
    /// iterating raises modularity by fitting noise, 0.385 to 0.423, and recovers nothing: ARI stays
    /// at the 0.002 floor through the 19--21 passes it takes to settle. The negative control runs
    /// to the end on one graph only, to keep the debug-build suite fast.
    #[test]
    fn later_passes_recover_what_one_pass_misses_and_never_lower_the_modularity() {
        const N: usize = 3000;
        let cases = [
            (4.0, 0u64, true),
            (4.0, 1, true),
            (4.0, 2, true),
            (0.6, 0, true),
            (0.6, 1, false),
            (0.6, 2, false),
        ];
        for (s, seed, to_the_end) in cases {
            let (a, b) = sbm_rates(6.0, s);
            let (base, truth) = sbm3(N, a, b, seed);
            let ctx = format!("s = {s}, seed = {seed}");
            let signal = s > 1.0;
            let mut labels = detect(&base, 1.0, Objective::Modularity, seed);
            let mut q = modularity(&base, &labels, 1.0);
            let mut passes = 1;
            loop {
                let next = detect_from(
                    &base,
                    1.0,
                    Objective::Modularity,
                    pass_seed(seed, passes),
                    &labels,
                );
                passes += 1;
                let q_next = modularity(&base, &next, 1.0);
                assert!(
                    q_next >= q - 1e-12,
                    "{ctx}: pass {passes} lowered Q {q} -> {q_next}"
                );
                q = q_next;
                let stable = next == labels;
                labels = next;
                if passes == 2 {
                    let two = ari(&labels, &truth);
                    assert!(
                        if signal { two >= 0.85 } else { two < 0.005 },
                        "{ctx}: two passes read ARI {two:.4}"
                    );
                    if !to_the_end {
                        break;
                    }
                }
                if stable {
                    break;
                }
                assert!(passes < 60, "{ctx}: not stable after {passes} passes");
            }
            if !to_the_end {
                continue;
            }
            let last = ari(&labels, &truth);
            assert!(
                if signal { last >= 0.88 } else { last < 0.005 },
                "{ctx}: until stable reads ARI {last:.4}"
            );
            if seed == 0 {
                assert_eq!(
                    detect_passes(&base, 1.0, Objective::Modularity, seed, 1000),
                    (labels, passes),
                    "{ctx}: the pass loop disagrees with the reference loop"
                );
            }
        }
    }

    /// ARI, modularity and community count after one pass, two passes and until stable, per graph
    /// seed, on the 3-community SBM at mean degree 6. Also reports whether modularity ever fell
    /// from one pass to the next.
    ///
    /// `cargo test --release --lib clustering::community::tests::measure_sbm_passes -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn measure_sbm_passes() {
        let ns: Vec<usize> = std::env::var("SBM_NS")
            .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or_else(|_| vec![3000, 30000]);
        for n in ns {
            for s in [0.6, 3.0, 4.0, 5.0] {
                let (a, b) = sbm_rates(6.0, s);
                for seed in 0u64..3 {
                    let (base, truth) = sbm3(n, a, b, seed);
                    let mut labels = detect(&base, 1.0, Objective::Modularity, seed);
                    let mut row = format!("n={n} s={s:.1} seed={seed}");
                    let mut q_prev = modularity(&base, &labels, 1.0);
                    let mut fell = 0usize;
                    row += &format!(
                        " | p1 ARI {:+.4} Q {:.4} k {}",
                        ari(&labels, &truth),
                        q_prev,
                        n_distinct(&labels)
                    );
                    let mut passes = 1;
                    while passes < 200 {
                        let next = detect_from(
                            &base,
                            1.0,
                            Objective::Modularity,
                            pass_seed(seed, passes),
                            &labels,
                        );
                        passes += 1;
                        let q = modularity(&base, &next, 1.0);
                        if q < q_prev - 1e-12 {
                            fell += 1;
                        }
                        q_prev = q;
                        let stable = next == labels;
                        labels = next;
                        if passes == 2 {
                            row += &format!(
                                " | p2 ARI {:+.4} Q {:.4} k {}",
                                ari(&labels, &truth),
                                q,
                                n_distinct(&labels)
                            );
                        }
                        if stable {
                            break;
                        }
                    }
                    row += &format!(
                        " | stable@{passes} ARI {:+.4} Q {:.4} k {} | Q fell {fell}x",
                        ari(&labels, &truth),
                        q_prev,
                        n_distinct(&labels)
                    );
                    println!("{row}");
                }
            }
        }
    }

    /// Modularity-Leiden on the 3-community SBM along the signal `s` at mean degree 6: median over
    /// graph seeds 0/1/2 (the `detect` seed is the graph seed) of the ARI with the node ids as drawn
    /// (communities in contiguous blocks), the ARI with the ids shuffled, and the community count.
    ///
    /// `cargo test --release --lib clustering::community::tests::measure_sbm_detection -- --ignored --nocapture`
    #[test]
    #[ignore = "measurement harness, not a check"]
    fn measure_sbm_detection() {
        let median = |mut v: Vec<f64>| {
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        for n in [1500usize, 3000, 6000, 12000, 30000] {
            println!("\nn = {n}, mean degree 6");
            println!("    s       a       b   ARI(blocks)  ARI(shuffled)  communities");
            for s in [0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
                let (a, b) = sbm_rates(6.0, s);
                let (mut blocks, mut shuf, mut comms) = (Vec::new(), Vec::new(), Vec::new());
                for seed in 0u64..3 {
                    let (base, truth) = sbm3(n, a, b, seed);
                    let lab = detect(&base, 1.0, Objective::Modularity, seed);
                    blocks.push(ari(&lab, &truth));
                    comms.push(n_distinct(&lab) as f64);
                    shuf.push(ari(&detect_shuffled(&base, seed), &truth));
                }
                println!(
                    "{s:>5.1} {a:>7.3} {b:>7.3} {:>+12.4} {:>+14.4} {:>12}",
                    median(blocks),
                    median(shuf),
                    median(comms)
                );
            }
        }
    }
}
