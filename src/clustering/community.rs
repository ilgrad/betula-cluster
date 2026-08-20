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
}
