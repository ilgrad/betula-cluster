//! Community detection on the CF-tree leaf microclusters (graph clustering).
//!
//! Builds the shared self-tuning k-NN affinity graph over the leaf means (see [`crate::clustering::
//! graph`]) and runs **Louvain modularity maximization** (Blondel et al. 2008): greedy local moving
//! of nodes between communities to increase modularity, then aggregation of each community into a
//! super-node, repeated until modularity stops improving. Unlike the parametric heads it needs **no
//! `k`** — the community count is discovered from the graph structure.
//!
//! Each returned community is guaranteed internally connected: after Louvain converges, any community
//! that is disconnected in the affinity graph is split into its connected components. This is the key
//! correctness property the Leiden refinement (Traag et al. 2019) adds over plain Louvain, obtained
//! here with a single post-hoc pass (cheap and exact on the small microcluster graph).

use crate::clustering::graph::knn_affinity;
use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::types::Real;
use std::collections::HashMap;

/// Resolution `γ` of the modularity null model. `1.0` is the classic Newman-Girvan modularity;
/// higher favours more, smaller communities.
const RESOLUTION: f64 = 1.0;

/// Result of a community-detection run: one community label per input microcluster.
pub struct Community {
    /// Community index per input feature.
    pub labels: Vec<usize>,
}

/// A weighted undirected working graph. `adj` holds only inter-node (external) edges; internal edges
/// created by aggregation are folded into `degree` (a community's degree is the sum of its members').
struct Graph {
    adj: Vec<Vec<(usize, f64)>>,
    degree: Vec<f64>,
    two_m: f64, // = 2·(total edge weight); invariant across aggregation levels
}

impl Graph {
    fn from_adj(adj: Vec<Vec<(usize, f64)>>) -> Self {
        let degree: Vec<f64> = adj
            .iter()
            .map(|r| r.iter().map(|&(_, w)| w).sum())
            .collect();
        let two_m = degree.iter().sum();
        Self { adj, degree, two_m }
    }
}

/// Louvain community detection on the leaf microclusters; `k` is not used (the count is discovered).
pub fn louvain<R: Real, C: ClusterFeature<R>>(features: &[C], seed: u64) -> Community {
    let n = features.len();
    if n <= 1 {
        return Community { labels: vec![0; n] };
    }
    let centers: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.mean().iter().map(|&x| x.to_f64().unwrap()).collect())
        .collect();
    let base = knn_affinity::<f64>(&centers);
    let labels = enforce_connectivity(&detect(&base, seed), &base);
    Community { labels }
}

/// Multi-level Louvain: local-move to a fixpoint, aggregate, repeat until no move improves modularity.
fn detect(base: &[Vec<(usize, f64)>], seed: u64) -> Vec<usize> {
    let n = base.len();
    let mut membership: Vec<usize> = (0..n).collect(); // original node → current community
    let mut g = Graph::from_adj(base.to_vec());
    let mut level = 0u64;
    loop {
        let (part, improved) = one_level(&g, seed.wrapping_add(level));
        if !improved {
            break;
        }
        for m in membership.iter_mut() {
            *m = part[*m];
        }
        let c = part.iter().max().map_or(0, |&x| x + 1);
        g = aggregate(&g, &part, c);
        level += 1;
    }
    relabel(&membership)
}

/// One local-moving pass over the current graph: each node greedily joins the neighbouring community
/// with the largest modularity gain, iterating until no node moves. Returns the (relabelled) partition
/// and whether any node changed community.
fn one_level(g: &Graph, seed: u64) -> (Vec<usize>, bool) {
    let n = g.adj.len();
    let mut comm: Vec<usize> = (0..n).collect();
    let mut sigma_tot = g.degree.clone(); // Σ degree per community (each node its own to start)
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize; // Fisher-Yates, seeded visit order
        order.swap(i, j);
    }
    let mut improved = false;
    let mut moved = true;
    while moved {
        moved = false;
        for &i in &order {
            let ci = comm[i];
            let ki = g.degree[i];
            let mut w_to: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &g.adj[i] {
                *w_to.entry(comm[j]).or_insert(0.0) += w;
            }
            // Remove i from its community before scoring so every candidate is compared equally.
            // Gain of joining community `c` is `w_to[c] − γ·Σtot[c]·k_i / 2m`; rejoining `ci` (now
            // possibly empty ⇒ the "stay isolated" option, gain 0) is the baseline.
            sigma_tot[ci] -= ki;
            let gain = |c: usize| -> f64 {
                w_to.get(&c).copied().unwrap_or(0.0) - RESOLUTION * sigma_tot[c] * ki / g.two_m
            };
            let mut best_c = ci;
            let mut best_gain = gain(ci);
            for &c in w_to.keys() {
                let g_c = gain(c);
                if g_c > best_gain {
                    best_gain = g_c;
                    best_c = c;
                }
            }
            sigma_tot[best_c] += ki;
            comm[i] = best_c;
            if best_c != ci {
                moved = true;
                improved = true;
            }
        }
    }
    (relabel(&comm), improved)
}

/// Aggregate each community into a super-node: external inter-community edges are summed; internal
/// edges are dropped from `adj` but preserved in `degree` (member-degree sum), keeping `2m` invariant.
fn aggregate(g: &Graph, part: &[usize], c: usize) -> Graph {
    let mut degree = vec![0.0; c];
    let mut acc: Vec<HashMap<usize, f64>> = vec![HashMap::new(); c];
    for (i, row) in g.adj.iter().enumerate() {
        let ci = part[i];
        degree[ci] += g.degree[i];
        for &(j, w) in row {
            let cj = part[j];
            if ci != cj {
                *acc[ci].entry(cj).or_insert(0.0) += w;
            }
        }
    }
    let adj = acc.into_iter().map(|m| m.into_iter().collect()).collect();
    Graph {
        adj,
        degree,
        two_m: g.two_m,
    }
}

/// Split any disconnected community into its connected components (in the base graph), so every
/// returned community is internally connected. BFS assigns each same-label component a fresh id.
fn enforce_connectivity(labels: &[usize], base: &[Vec<(usize, f64)>]) -> Vec<usize> {
    let n = labels.len();
    let mut out = vec![usize::MAX; n];
    let mut next = 0;
    for s in 0..n {
        if out[s] != usize::MAX {
            continue;
        }
        out[s] = next;
        let mut stack = vec![s];
        while let Some(u) = stack.pop() {
            for &(v, _) in &base[u] {
                if out[v] == usize::MAX && labels[v] == labels[u] {
                    out[v] = next;
                    stack.push(v);
                }
            }
        }
        next += 1;
    }
    out
}

/// Map arbitrary community ids to a contiguous `0..k` in first-seen order.
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

    fn modularity(base: &[Vec<(usize, f64)>], labels: &[usize]) -> f64 {
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
        let mut comm_deg: HashMap<usize, f64> = HashMap::new();
        for (i, &l) in labels.iter().enumerate() {
            *comm_deg.entry(l).or_insert(0.0) += deg[i];
        }
        let null: f64 = comm_deg.values().map(|&s| s * s).sum();
        (internal - RESOLUTION * null / two_m) / two_m
    }

    #[test]
    fn louvain_discovers_separated_blobs_without_k() {
        // Community detection on compact groups: the count is found from the graph, not supplied.
        let mut rng = SplitMix64::new(4);
        let centers = [[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]];
        let (pts, truth) = blobs(&mut rng, 300, &centers, 0.5);
        let (micros, point_to_micro) = grid_micros(&pts, 1.0);
        let labels = louvain(&micros, 1).labels;
        assert_eq!(n_distinct(&labels), 3); // three communities discovered, no k given
        let pred: Vec<usize> = point_to_micro.iter().map(|&m| labels[m]).collect();
        assert!(ari(&pred, &truth) > 0.95);
    }

    #[test]
    fn louvain_beats_trivial_partitions_on_modularity() {
        // A single connected cloud with community structure: Louvain splits it (exercising the
        // multi-level cross-community aggregation) into a partition of strictly higher modularity
        // than either trivial partition.
        let mut rng = SplitMix64::new(1);
        let centers = [[0.0, 0.0], [5.0, 0.0]];
        let (pts, _t) = blobs(&mut rng, 300, &centers, 1.0);
        let (micros, _) = grid_micros(&pts, 0.5);
        let centers_f: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let base = knn_affinity::<f64>(&centers_f);
        let labels = louvain(&micros, 1).labels;
        assert!(n_distinct(&labels) >= 2);
        let all_one = vec![0usize; micros.len()];
        let singletons: Vec<usize> = (0..micros.len()).collect();
        let q = modularity(&base, &labels);
        assert!(q > modularity(&base, &all_one));
        assert!(q > modularity(&base, &singletons));
    }

    #[test]
    fn louvain_single_feature_is_one_community() {
        let (micros, _) = grid_micros(&[vec![1.0, 2.0]], 1.0);
        assert_eq!(louvain(&micros, 1).labels, vec![0]);
    }
}
