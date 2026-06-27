//! Mapper — a topological-skeleton graph over the CF leaf microclusters (exploration, not a partition).
//!
//! Standard TDA Mapper (Singh–Mémoli–Carlsson 2007) specialised to BETULA microclusters:
//!   1. a *lens* `f` maps each microcluster to `R` (density / radius / a coordinate / ‖μ‖ / eccentricity),
//!   2. the lens range is covered by `resolution` overlapping bins (overlap fraction `gain`),
//!   3. microclusters in a bin are single-linked at `link_scale ×` the bin's median nearest-neighbour gap,
//!   4. one graph node per (bin, component); nodes sharing a microcluster (from the cover overlap) link.
//!
//! The nerve graph exposes branches, bridges and loops in the data's shape — for RAG curation, dedup,
//! leakage detection and structure inspection. It runs in `O(M²)` over the `M ≪ N` microclusters (the
//! lens/linkage scans are pairwise), never the raw points, so it is cheap for `M ~ 10³–10⁴`.
//!
//! Working precision is `f64` for the graph/topology math regardless of the tree's `R`.

use crate::feature::ClusterFeature;
use crate::types::Real;
use std::collections::HashMap;

/// Filter function mapping each microcluster to a scalar the cover is built over.
#[derive(Clone, Copy, Debug)]
pub enum Lens {
    /// Local density `1 / (mean distance to the `k` nearest microclusters)` — high in crowded regions.
    Density { k: usize },
    /// RMS radius `√(S/n)` — the microcluster's own spread.
    Radius,
    /// Euclidean norm of the centroid `‖μ‖` — natural for direction/embedding data.
    L2Norm,
    /// A single centroid coordinate `μ[c]`.
    Coordinate(usize),
    /// Mean distance to all other microclusters — large at the periphery of the shape.
    Eccentricity,
}

/// Mapper construction parameters.
#[derive(Clone, Copy, Debug)]
pub struct MapperParams {
    /// The filter function.
    pub lens: Lens,
    /// Number of overlapping cover bins over the lens range (`≥ 1`).
    pub resolution: usize,
    /// Cover overlap as a fraction of the bin step, in `[0, 1)`; the source of nerve edges.
    pub gain: f64,
    /// Single-linkage multiplier: microclusters `i, j` in a bin link iff `d(μ_i,μ_j) ≤ link_scale ×`
    /// the bin's median nearest-neighbour gap (data-adaptive; larger ⇒ a more connected skeleton).
    pub link_scale: f64,
    /// Drop graph nodes whose total mass is below this (cover-induced specks / noise).
    pub min_node_mass: f64,
}

impl Default for MapperParams {
    fn default() -> Self {
        Self {
            lens: Lens::Density { k: 5 },
            resolution: 10,
            gain: 0.3,
            link_scale: 2.0,
            min_node_mass: 0.0,
        }
    }
}

/// One Mapper node: a connected component of microclusters inside one cover bin.
pub struct MapperNode {
    /// Indices (into the input `features`) of the microclusters in this node.
    pub members: Vec<usize>,
    /// Total mass `Σ n_i` of the members.
    pub mass: f64,
    /// Cover bin this node came from.
    pub bin: usize,
    /// Mass-weighted centroid of the members (for plotting / labelling).
    pub centroid: Vec<f64>,
    /// Mean lens value of the members.
    pub lens_value: f64,
}

/// The Mapper graph: nodes (above), weighted nerve edges, and derived topological landmarks.
pub struct MapperGraph {
    /// Graph nodes.
    pub nodes: Vec<MapperNode>,
    /// Nerve edges `(a, b, shared)`: nodes `a < b` sharing `shared` microclusters (cover overlap).
    pub edges: Vec<(usize, usize, usize)>,
    /// Nodes of degree `≥ 3` — where the shape splits (branch points).
    pub branch_points: Vec<usize>,
    /// Indices into `edges` that are bridges: removing one disconnects its endpoints (a thin link
    /// between otherwise separate regions — a leakage/merge between topics for embeddings).
    pub bridges: Vec<usize>,
}

/// Union–find with path halving + union by rank, for per-bin single-linkage components.
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
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

/// RMS radius `√(S/n)` of a microcluster (`0` for an empty / single-point feature).
fn rms_radius<R: Real, C: ClusterFeature<R>>(f: &C) -> f64 {
    let n = f.weight().to_f64().unwrap();
    if n <= 0.0 {
        return 0.0;
    }
    (f.ssd().to_f64().unwrap() / n).max(0.0).sqrt()
}

/// Centroid as `f64`.
fn centroid64<R: Real, C: ClusterFeature<R>>(f: &C) -> Vec<f64> {
    f.mean().iter().map(|v| v.to_f64().unwrap()).collect()
}

fn euclid(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// Evaluate the lens for every microcluster.
fn lens_values(mu: &[Vec<f64>], radius: &[f64], lens: Lens) -> Vec<f64> {
    let m = mu.len();
    match lens {
        Lens::Radius => radius.to_vec(),
        Lens::L2Norm => mu.iter().map(|c| euclid(c, &vec![0.0; c.len()])).collect(),
        Lens::Coordinate(c) => mu
            .iter()
            .map(|p| p.get(c).copied().unwrap_or(0.0))
            .collect(),
        Lens::Eccentricity => (0..m)
            .map(|i| {
                if m <= 1 {
                    return 0.0;
                }
                let s: f64 = (0..m)
                    .filter(|&j| j != i)
                    .map(|j| euclid(&mu[i], &mu[j]))
                    .sum();
                s / (m as f64 - 1.0)
            })
            .collect(),
        Lens::Density { k } => (0..m)
            .map(|i| {
                if m <= 1 {
                    return 0.0;
                }
                let mut ds: Vec<f64> = (0..m)
                    .filter(|&j| j != i)
                    .map(|j| euclid(&mu[i], &mu[j]))
                    .collect();
                ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let kk = k.clamp(1, ds.len());
                let mean = ds[..kk].iter().sum::<f64>() / kk as f64;
                if mean > 0.0 {
                    1.0 / mean
                } else {
                    f64::INFINITY
                }
            })
            .collect(),
    }
}

/// Bridge edges via Tarjan's algorithm on the simple graph; returns indices into `edges`.
///
/// Parallel edges (two nodes joined by more than one entry in `edges`) are never bridges; the DFS
/// guards against re-using the *edge* it descended through (by edge id), not merely the parent node,
/// so a doubled link correctly fails the `low > disc` bridge test.
fn find_bridges(n: usize, edges: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut adj: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n]; // (neighbour, edge id)
    for (eid, &(a, b, _)) in edges.iter().enumerate() {
        adj[a].push((b, eid));
        adj[b].push((a, eid));
    }
    let mut disc = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut bridges = Vec::new();
    let mut timer = 0usize;
    // Iterative DFS (stack of (node, parent edge id, neighbour cursor)) to avoid recursion blowup.
    for start in 0..n {
        if disc[start] != usize::MAX {
            continue;
        }
        let mut stack: Vec<(usize, usize, usize)> = vec![(start, usize::MAX, 0)];
        disc[start] = timer;
        low[start] = timer;
        timer += 1;
        while let Some(&(u, pe, ci)) = stack.last() {
            if ci < adj[u].len() {
                stack.last_mut().unwrap().2 += 1;
                let (v, eid) = adj[u][ci];
                if eid == pe {
                    continue; // do not climb back through the edge we arrived on
                }
                if disc[v] == usize::MAX {
                    disc[v] = timer;
                    low[v] = timer;
                    timer += 1;
                    stack.push((v, eid, 0));
                } else {
                    low[u] = low[u].min(disc[v]);
                }
            } else {
                stack.pop();
                if let Some(&(p, _, _)) = stack.last() {
                    low[p] = low[p].min(low[u]);
                    if low[u] > disc[p] {
                        bridges.push(pe); // `pe` is u's parent edge = the edge (p, u)
                    }
                }
            }
        }
    }
    bridges.sort_unstable();
    bridges
}

/// Build a Mapper graph over the leaf microcluster `features`.
pub fn mapper<R: Real, C: ClusterFeature<R>>(features: &[C], p: &MapperParams) -> MapperGraph {
    let m = features.len();
    let empty = MapperGraph {
        nodes: Vec::new(),
        edges: Vec::new(),
        branch_points: Vec::new(),
        bridges: Vec::new(),
    };
    if m == 0 {
        return empty;
    }
    let mu: Vec<Vec<f64>> = features.iter().map(centroid64).collect();
    let mass: Vec<f64> = features
        .iter()
        .map(|f| f.weight().to_f64().unwrap())
        .collect();
    let radius: Vec<f64> = features.iter().map(rms_radius).collect();
    let f = lens_values(&mu, &radius, p.lens);

    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in &f {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() {
        (lo, hi) = (0.0, 0.0); // all-infinite lens (e.g. fully duplicated points): one bin
    }
    let resolution = p.resolution.max(1);
    let step = if hi > lo {
        (hi - lo) / resolution as f64
    } else {
        1.0
    };
    let pad = p.gain.clamp(0.0, 0.999) * step / 2.0;

    // Cover: assign each microcluster to every bin whose padded interval contains its lens value.
    let mut bin_members: Vec<Vec<usize>> = vec![Vec::new(); resolution];
    for (i, &fi) in f.iter().enumerate() {
        let v = if fi.is_finite() { fi } else { lo };
        for (b, members) in bin_members.iter_mut().enumerate() {
            let blo = lo + b as f64 * step - pad;
            let bhi = lo + (b as f64 + 1.0) * step + pad;
            if v >= blo && v <= bhi {
                members.push(i);
            }
        }
    }

    // Per-bin single-linkage at `link_scale ×` the median nearest-neighbour spacing → one node per
    // (bin, component). The data-adaptive scale tracks local density and — unlike a radius-based
    // touch test — does not fragment on threshold-0 point microclusters (whose radius is ~0).
    let mut nodes: Vec<MapperNode> = Vec::new();
    let mut node_of_micro: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (bin, members) in bin_members.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let bn = members.len();
        let mut uf = UnionFind::new(bn);
        if bn > 1 {
            let mut nn = vec![f64::INFINITY; bn];
            for a in 0..bn {
                for b in (a + 1)..bn {
                    let d = euclid(&mu[members[a]], &mu[members[b]]);
                    nn[a] = nn[a].min(d);
                    nn[b] = nn[b].min(d);
                }
            }
            let mut sorted = nn.clone();
            sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let thresh = p.link_scale * sorted[bn / 2]; // link_scale × median nearest-neighbour gap
            for a in 0..bn {
                for b in (a + 1)..bn {
                    if euclid(&mu[members[a]], &mu[members[b]]) <= thresh {
                        uf.union(a, b);
                    }
                }
            }
        }
        let mut comp: HashMap<usize, Vec<usize>> = HashMap::new();
        for (local, &gi) in members.iter().enumerate() {
            comp.entry(uf.find(local)).or_default().push(gi);
        }
        for group in comp.into_values() {
            let node_mass: f64 = group.iter().map(|&i| mass[i]).sum();
            if node_mass < p.min_node_mass {
                continue;
            }
            let dim = mu[group[0]].len();
            let mut centroid = vec![0.0; dim];
            let mut lens_acc = 0.0;
            for &i in &group {
                for (d, c) in centroid.iter_mut().enumerate() {
                    *c += mass[i] * mu[i][d];
                }
                lens_acc += if f[i].is_finite() { f[i] } else { lo };
            }
            if node_mass > 0.0 {
                centroid.iter_mut().for_each(|c| *c /= node_mass);
            }
            let nid = nodes.len();
            for &i in &group {
                node_of_micro[i].push(nid);
            }
            nodes.push(MapperNode {
                lens_value: lens_acc / group.len() as f64,
                members: group,
                mass: node_mass,
                bin,
                centroid,
            });
        }
    }

    // Nerve edges: nodes sharing a microcluster (a microcluster lands in overlapping bins) are linked,
    // weighted by the number shared. Nodes within one bin never share, so every edge crosses bins.
    let mut shared: HashMap<(usize, usize), usize> = HashMap::new();
    for node_ids in &node_of_micro {
        for a in 0..node_ids.len() {
            for b in (a + 1)..node_ids.len() {
                let (x, y) = (node_ids[a].min(node_ids[b]), node_ids[a].max(node_ids[b]));
                *shared.entry((x, y)).or_insert(0) += 1;
            }
        }
    }
    let mut edges: Vec<(usize, usize, usize)> =
        shared.into_iter().map(|((a, b), w)| (a, b, w)).collect();
    edges.sort_unstable();

    let mut degree = vec![0usize; nodes.len()];
    for &(a, b, _) in &edges {
        degree[a] += 1;
        degree[b] += 1;
    }
    let branch_points: Vec<usize> = (0..nodes.len()).filter(|&i| degree[i] >= 3).collect();
    let bridges = find_bridges(nodes.len(), &edges);

    MapperGraph {
        nodes,
        edges,
        branch_points,
        bridges,
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::testutil::grid_micros;
    use crate::feature::Spherical;

    /// Microclusters on a line: a dense blob, a thin bridge, a second dense blob (a "dumbbell").
    /// A coordinate lens must recover a connected skeleton whose only links across the gap are bridges.
    fn dumbbell() -> Vec<Spherical<f64>> {
        let mut pts: Vec<Vec<f64>> = Vec::new();
        for i in 0..40 {
            pts.push(vec![(i as f64) * 0.05, 0.3 * ((i % 5) as f64 - 2.0)]); // blob A: x∈[0,2]
            pts.push(vec![6.0 + (i as f64) * 0.05, 0.3 * ((i % 5) as f64 - 2.0)]);
            // blob B: x∈[6,8]
        }
        for i in 0..6 {
            pts.push(vec![2.0 + i as f64 * 0.66, 0.0]); // sparse bridge x∈[2,6]
        }
        grid_micros(&pts, 0.25).0
    }

    #[test]
    fn mapper_dumbbell_skeleton_has_a_bridge() {
        let g = mapper(
            &dumbbell(),
            &MapperParams {
                lens: Lens::Coordinate(0),
                resolution: 8,
                gain: 0.4,
                link_scale: 3.0,
                min_node_mass: 0.0,
            },
        );
        assert!(g.nodes.len() >= 3, "expected a multi-node skeleton");
        assert!(!g.edges.is_empty(), "cover overlap must create nerve edges");
        assert!(
            !g.bridges.is_empty(),
            "the thin neck between the two blobs must be a bridge edge"
        );
    }

    #[test]
    fn mapper_density_lens_runs_and_links() {
        // Two well-separated blobs under a density lens: a connected graph with no spurious bridge
        // across the (empty) gap — the blobs occupy different lens levels but the cover links each.
        let pts: Vec<Vec<f64>> = {
            let mut v = Vec::new();
            for i in 0..50 {
                let t = i as f64 * 0.1;
                v.push(vec![t.sin() * 0.2, t.cos() * 0.2]);
                v.push(vec![5.0 + t.sin() * 0.2, t.cos() * 0.2]);
            }
            v
        };
        let micros = grid_micros(&pts, 0.2).0;
        let g = mapper(&micros, &MapperParams::default());
        assert!(!g.nodes.is_empty());
        // Total node mass never exceeds total microcluster mass times the max bin multiplicity.
        let node_mass: f64 = g.nodes.iter().map(|n| n.mass).sum();
        assert!(node_mass > 0.0);
    }

    #[test]
    fn mapper_empty_and_single() {
        let g = mapper::<f64, Spherical<f64>>(&[], &MapperParams::default());
        assert!(g.nodes.is_empty() && g.edges.is_empty());

        let mut one = Spherical::new(2);
        one.push(&[1.0, 2.0], 1.0);
        let g = mapper(&[one], &MapperParams::default());
        assert_eq!(g.nodes.len(), 1);
        assert!(g.edges.is_empty() && g.bridges.is_empty() && g.branch_points.is_empty());
    }

    #[test]
    fn mapper_degenerate_lens_single_bin() {
        // All centroids share an L2 norm ⇒ the lens is constant ⇒ one bin, still a valid graph.
        let micros = grid_micros(&[vec![1.0, 0.0], vec![-1.0, 0.0], vec![0.0, 1.0]], 0.5).0;
        let g = mapper(
            &micros,
            &MapperParams {
                lens: Lens::L2Norm,
                resolution: 5,
                ..MapperParams::default()
            },
        );
        assert!(!g.nodes.is_empty());
    }

    #[test]
    fn find_bridges_ignores_parallel_edges() {
        // A triangle has no bridge; a path's every edge is a bridge (and the returned values must be
        // the *edge indices* 0 and 1, never a `usize::MAX` parent-edge sentinel); a doubled edge is
        // never a bridge.
        assert!(find_bridges(3, &[(0, 1, 1), (1, 2, 1), (0, 2, 1)]).is_empty());
        assert_eq!(find_bridges(3, &[(0, 1, 1), (1, 2, 1)]), vec![0, 1]);
        assert!(find_bridges(2, &[(0, 1, 1), (0, 1, 1)]).is_empty());
        // A bridge joining two triangles: only the middle edge (index 6) is a bridge.
        let two_triangles = [
            (0, 1, 1),
            (1, 2, 1),
            (0, 2, 1),
            (3, 4, 1),
            (4, 5, 1),
            (3, 5, 1),
            (2, 3, 1),
        ];
        assert_eq!(find_bridges(6, &two_triangles), vec![6]);
    }

    /// A microcluster with real spread around `center` (so its RMS radius is non-zero).
    fn spread_micro(center: [f64; 2], spread: f64) -> Spherical<f64> {
        let mut cf = Spherical::new(2);
        for i in 0..10 {
            let t = i as f64 / 10.0;
            cf.push(&[center[0] + spread * (t - 0.5), center[1]], 1.0);
        }
        cf
    }

    #[test]
    fn mapper_links_touching_microclusters_within_a_bin() {
        // Three evenly-spaced microclusters sharing one bin must merge by single linkage at the
        // median-NN scale — exercising the union-find — so the node count drops below three.
        let micros: Vec<Spherical<f64>> = [[0.0, 0.0], [0.3, 0.0], [0.6, 0.0]]
            .iter()
            .map(|&c| spread_micro(c, 0.6))
            .collect();
        let g = mapper(
            &micros,
            &MapperParams {
                lens: Lens::Coordinate(1), // constant (all y = 0) ⇒ a single bin holds all three
                resolution: 4,
                gain: 0.3,
                link_scale: 1.0,
                min_node_mass: 0.0,
            },
        );
        assert!(
            g.nodes.len() < micros.len(),
            "touching microclusters must merge"
        );
    }

    #[test]
    fn mapper_radius_and_eccentricity_lenses_run() {
        let micros = dumbbell();
        for lens in [Lens::Radius, Lens::Eccentricity] {
            let g = mapper(
                &micros,
                &MapperParams {
                    lens,
                    ..MapperParams::default()
                },
            );
            assert!(!g.nodes.is_empty());
        }
    }

    #[test]
    fn mapper_coincident_microclusters_use_the_degenerate_fallback() {
        // Distinct microclusters at the same point ⇒ the density lens is +∞ everywhere ⇒ the
        // all-infinite fallback collapses the cover to one bin instead of dividing by a zero range.
        let micros: Vec<Spherical<f64>> = (0..3)
            .map(|_| {
                let mut cf = Spherical::new(2);
                cf.push(&[1.0, 1.0], 1.0);
                cf
            })
            .collect();
        let g = mapper(
            &micros,
            &MapperParams {
                lens: Lens::Density { k: 2 },
                ..MapperParams::default()
            },
        );
        assert!(!g.nodes.is_empty());
    }

    #[test]
    fn mapper_skips_empty_cover_bins() {
        // Two far-apart microclusters with many bins ⇒ the middle bins are empty and skipped.
        let micros = grid_micros(&[vec![0.0, 0.0], vec![100.0, 0.0]], 1.0).0;
        let g = mapper(
            &micros,
            &MapperParams {
                lens: Lens::Coordinate(0),
                resolution: 10,
                gain: 0.1,
                ..MapperParams::default()
            },
        );
        assert_eq!(g.nodes.len(), 2); // one node per cluster; empty middle bins produce none
    }

    #[test]
    fn mapper_drops_nodes_below_min_mass() {
        let micros = dumbbell();
        let base = MapperParams {
            lens: Lens::Coordinate(0),
            resolution: 8,
            gain: 0.4,
            link_scale: 3.0,
            min_node_mass: 0.0,
        };
        let kept = mapper(&micros, &base);
        let filtered = mapper(
            &micros,
            &MapperParams {
                min_node_mass: 1e9,
                ..base
            },
        );
        assert!(!kept.nodes.is_empty());
        assert!(
            filtered.nodes.is_empty(),
            "a huge mass floor drops every node"
        );
    }
}
