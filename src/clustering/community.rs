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
        let degree: Vec<f64> = adj
            .iter()
            .map(|r| r.iter().map(|&(_, w)| w).sum())
            .collect();
        let two_m = degree.iter().sum();
        let size = vec![1.0; adj.len()];
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
    let n = features.len();
    if n <= 1 {
        return Community { labels: vec![0; n] };
    }
    let centers: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|&x| x.to_f64().unwrap()).collect())
        .collect();
    let log_covs = (cov_weight > 0.0).then(|| to_f64_tensors(log_covariances(features)));
    let tangents =
        (tangent_weight > 0.0).then(|| to_f64_tensors(tangent_bases(features, tangent_rank)));
    let base = if log_covs.is_some() || tangents.is_some() {
        knn_affinity_geo::<f64>(
            &centers,
            log_covs.as_deref().map(|lc| (lc, cov_weight)),
            tangents.as_deref().map(|t| (t, tangent_weight)),
        )
    } else {
        knn_affinity::<f64>(&centers)
    };
    Community {
        labels: detect(&base, resolution, objective, seed),
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

/// Multi-level Leiden: local-move → refine → aggregate (seeded from the local-move partition),
/// until aggregation no longer coarsens the graph.
fn detect(base: &[Vec<(usize, f64)>], gamma: f64, obj: Objective, seed: u64) -> Vec<usize> {
    let n = base.len();
    let mut membership: Vec<usize> = (0..n).collect(); // original node → current super-node
    let mut g = Graph::from_adj(base.to_vec());
    let mut init: Vec<usize> = (0..n).collect(); // seed partition for the current local-move
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
            for &c in w_to.keys() {
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
        for (&c, &w) in &w_to {
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
    let adj = acc.into_iter().map(|m| m.into_iter().collect()).collect();
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
}
