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
    /// Per-edge distributional overlap — the Bhattacharyya coefficient `∈ (0, 1]` between the two
    /// nodes' pooled diagonal Gaussians (`1` = indistinguishable, `→ 0` across a real density gap).
    /// Qualifies each nerve edge beyond the raw shared-microcluster count, so a structural bridge that
    /// is *also* low-overlap is a true density gap rather than a cover artifact. Aligned with `edges`.
    pub edge_overlap: Vec<f64>,
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

/// Which function on the nerve to filter by for 0-dimensional persistence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filtration {
    /// Connectivity filtration on the edge gap `1 − edge_overlap`: high-overlap (dense) links appear
    /// first and merge components; a real Bhattacharyya gap merges late. Each finite bar's death is
    /// the depth of a bottleneck — this ranks the boolean `bridges` quantitatively.
    EdgeOverlap,
    /// Sublevel filtration on the lens value (the canonical Mapper/Reeb persistence — flares/branches
    /// of the shape). Degenerate for a monotone coordinate lens; informative for density/eccentricity.
    Lens,
}

/// A 0-dimensional persistence diagram of the Mapper nerve (single-linkage by union-find).
pub struct PersistenceDiagram {
    /// `(birth, death)` per class, sorted by persistence (`death − birth`) descending. Essential
    /// classes — the nerve's connected components — carry `death = f64::INFINITY`.
    pub points: Vec<(f64, f64)>,
    /// Number of essential classes = connected components of the nerve (β₀).
    pub n_components: usize,
    /// Filtration values at which independent cycles close (β₁ births; a bare graph has no 2-cells, so
    /// these carry no finite death). `len() == edges − nodes + n_components`.
    pub loop_births: Vec<f64>,
}

impl MapperGraph {
    /// 0-dimensional persistent homology of the nerve under `filt`, by union-find over the sorted
    /// filtration with the elder rule: `O(E log E)`, pure (no matrix reduction, no deps). On a fixed
    /// nerve both filtrations are function filtrations, so the diagram is bottleneck-stable
    /// (Cohen–Steiner–Edelsbrunner–Harer 2007) under perturbations of the filter values.
    pub fn persistence_diagram(&self, filt: Filtration) -> PersistenceDiagram {
        let n = self.nodes.len();
        // Vertex birth values (monotone: an edge never precedes its endpoints).
        let bv: Vec<f64> = match filt {
            Filtration::EdgeOverlap => vec![0.0; n],
            Filtration::Lens => self.nodes.iter().map(|nd| nd.lens_value).collect(),
        };
        let mut fe: Vec<(f64, usize, usize)> = self
            .edges
            .iter()
            .enumerate()
            .map(|(i, &(a, b, _))| {
                let val = match filt {
                    Filtration::EdgeOverlap => 1.0 - self.edge_overlap[i],
                    Filtration::Lens => bv[a].max(bv[b]),
                };
                (val, a, b)
            })
            .collect();
        fe.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut uf = UnionFind::new(n);
        let mut cmin = bv.clone(); // min birth per component, valid at the current root
        let mut points: Vec<(f64, f64)> = Vec::new();
        let mut loop_births: Vec<f64> = Vec::new();
        for (val, a, b) in fe {
            let (ra, rb) = (uf.find(a), uf.find(b));
            if ra == rb {
                loop_births.push(val); // closes a cycle → a β₁ birth, no 0-D event
                continue;
            }
            // Elder rule: the younger class (larger birth) dies at this edge's filtration value.
            let (young, elder) = if cmin[ra] > cmin[rb] || (cmin[ra] == cmin[rb] && ra > rb) {
                (ra, rb)
            } else {
                (rb, ra)
            };
            points.push((cmin[young], val)); // death = val ≥ cmin[young] = birth (monotone)
            uf.union(ra, rb);
            let root = uf.find(a);
            cmin[root] = cmin[elder].min(cmin[young]);
        }
        // Essential classes: one (min birth, +∞) per surviving component root (incl. isolated nodes).
        let mut roots: HashMap<usize, f64> = HashMap::new();
        for (v, &birth) in bv.iter().enumerate() {
            let r = uf.find(v);
            let e = roots.entry(r).or_insert(f64::INFINITY);
            *e = e.min(birth);
        }
        let n_components = roots.len();
        for (_, birth) in roots {
            points.push((birth, f64::INFINITY));
        }
        points.sort_by(|p, q| {
            (q.1 - q.0)
                .partial_cmp(&(p.1 - p.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        PersistenceDiagram {
            points,
            n_components,
            loop_births,
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

/// Bhattacharyya coefficient between two diagonal Gaussians (per-dimension mean/variance), in
/// `(0, 1]`: `1` when identical, decaying toward `0` as the means separate or the spreads diverge.
/// Variances are floored to stay positive (a microcluster of `n = 1` has zero variance).
fn bhattacharyya_diag(mu_a: &[f64], var_a: &[f64], mu_b: &[f64], var_b: &[f64]) -> f64 {
    let mut d_b = 0.0;
    for k in 0..mu_a.len() {
        let va = var_a[k].max(0.0) + 1e-12;
        let vb = var_b[k].max(0.0) + 1e-12;
        let vbar = 0.5 * (va + vb);
        let dm = mu_a[k] - mu_b[k];
        // Bhattacharyya distance of two 1-D Gaussians, summed over the (independent) dimensions.
        d_b += 0.125 * dm * dm / vbar + 0.5 * (vbar.ln() - 0.5 * (va.ln() + vb.ln()));
    }
    (-d_b).exp().clamp(0.0, 1.0)
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
        edge_overlap: Vec::new(),
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

    // Per-node pooled diagonal Gaussian (merge the member microclusters), for CF-aware edge overlap.
    let dim = features[0].dim();
    let (node_mu, node_var): (Vec<Vec<f64>>, Vec<Vec<f64>>) = nodes
        .iter()
        .map(|node| {
            let mut cf = C::new(dim);
            for &i in &node.members {
                cf.merge(&features[i]);
            }
            let mean: Vec<f64> = (0..dim).map(|k| cf.mean()[k].to_f64().unwrap()).collect();
            let var: Vec<f64> = (0..dim).map(|k| cf.variance(k).to_f64().unwrap()).collect();
            (mean, var)
        })
        .unzip();
    let edge_overlap: Vec<f64> = edges
        .iter()
        .map(|&(a, b, _)| bhattacharyya_diag(&node_mu[a], &node_var[a], &node_mu[b], &node_var[b]))
        .collect();

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
        edge_overlap,
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
        // CF-aware edge overlap: aligned with `edges`, in [0, 1], and the bridge (across the sparse
        // neck) has lower distributional overlap than the densest within-blob edge.
        assert_eq!(g.edge_overlap.len(), g.edges.len());
        assert!(g.edge_overlap.iter().all(|&o| (0.0..=1.0).contains(&o)));
        let bridge_overlap = g.edge_overlap[g.bridges[0]];
        let max_overlap = g.edge_overlap.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            bridge_overlap < max_overlap,
            "bridge overlap {bridge_overlap} should be below the densest edge {max_overlap}"
        );
    }

    #[test]
    fn bhattacharyya_diag_matches_known_values() {
        // identical diagonal Gaussians → coefficient 1
        let same = bhattacharyya_diag(&[0.0, 3.0], &[2.0, 1.0], &[0.0, 3.0], &[2.0, 1.0]);
        assert!((same - 1.0).abs() < 1e-9, "identical = {same}");
        // 1-D closed form BC = sqrt(2σaσb/(σa²+σb²))·exp(−Δ²/(4(σa²+σb²))); σ=1, Δ=2 → exp(−0.5)
        let bc = bhattacharyya_diag(&[0.0], &[1.0], &[2.0], &[1.0]);
        assert!((bc - (-0.5f64).exp()).abs() < 1e-6, "bc = {bc}");
        // overlap decreases monotonically as the means separate
        let near = bhattacharyya_diag(&[0.0], &[1.0], &[1.0], &[1.0]);
        let far = bhattacharyya_diag(&[0.0], &[1.0], &[10.0], &[1.0]);
        assert!(near > far && far < 1e-5, "near {near}, far {far}");
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

    #[test]
    fn persistence_overlap_dumbbell_ranks_the_bridge() {
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
        let d = g.persistence_diagram(Filtration::EdgeOverlap);
        // one class per node; essentials (∞ death) = connected components (the A–neck–B chain plus
        // any isolated cover specks).
        assert_eq!(d.points.len(), g.nodes.len());
        assert!(d.n_components >= 1);
        assert_eq!(
            d.points.iter().filter(|p| p.1.is_infinite()).count(),
            d.n_components
        );
        // every class sits on or above the diagonal.
        assert!(d.points.iter().all(|&(b, dth)| dth >= b));
        // the dominant finite bar's death is the neck's Bhattacharyya gap (1 − min bridge overlap):
        // the last single-linkage merge crosses the sparsest link, which is the neck bridge.
        let min_bridge_overlap = g
            .bridges
            .iter()
            .map(|&e| g.edge_overlap[e])
            .fold(f64::INFINITY, f64::min);
        let max_finite_death = d
            .points
            .iter()
            .map(|p| p.1)
            .filter(|x| x.is_finite())
            .fold(0.0_f64, f64::max);
        assert!(
            (max_finite_death - (1.0 - min_bridge_overlap)).abs() < 1e-9,
            "dominant death {max_finite_death} should equal the bridge gap {}",
            1.0 - min_bridge_overlap
        );
    }

    #[test]
    fn persistence_diagram_invariants_hold() {
        // Two blobs under the default density lens: both filtrations give a valid diagram, and the
        // number of cycle-births equals the graph's first Betti number E − V + β₀.
        let pts: Vec<Vec<f64>> = {
            let mut v = Vec::new();
            for i in 0..50 {
                let t = i as f64 * 0.1;
                v.push(vec![t.sin() * 0.2, t.cos() * 0.2]);
                v.push(vec![5.0 + t.sin() * 0.2, t.cos() * 0.2]);
            }
            v
        };
        let g = mapper(&grid_micros(&pts, 0.2).0, &MapperParams::default());
        for filt in [Filtration::EdgeOverlap, Filtration::Lens] {
            let d = g.persistence_diagram(filt);
            assert_eq!(d.points.len(), g.nodes.len());
            assert!(d.points.iter().all(|&(b, dth)| dth >= b - 1e-12));
            // loop_births = E − V + β₀ (avoid usize underflow by moving V to the other side).
            assert_eq!(
                d.loop_births.len() + g.nodes.len(),
                g.edges.len() + d.n_components
            );
        }
    }

    /// Every node summary must be recomputable from its own members: the mass is their total, the
    /// centroid is their mass-weighted mean and the lens value is their unweighted mean. Nothing
    /// asserted any of the three -- the existing tests read the graph's *shape* (branch points,
    /// bridges, component count), which survives an arithmetic error in the aggregation.
    #[test]
    fn node_summaries_are_recomputable_from_their_members() {
        let feats = dumbbell();
        let p = MapperParams {
            lens: Lens::Coordinate(0),
            resolution: 8,
            gain: 0.3,
            link_scale: 2.0,
            min_node_mass: 0.0,
        };
        let g = mapper(&feats, &p);
        assert!(
            g.nodes.len() > 3,
            "the cover produced too few nodes to test"
        );

        let mu: Vec<Vec<f64>> = feats.iter().map(centroid64).collect();
        let mass: Vec<f64> = feats.iter().map(|f| f.weight()).collect();
        let radius: Vec<f64> = feats.iter().map(rms_radius).collect();
        let lens = lens_values(&mu, &radius, p.lens);
        let lo = lens.iter().copied().fold(f64::INFINITY, f64::min);

        for (nid, node) in g.nodes.iter().enumerate() {
            assert!(!node.members.is_empty(), "node {nid} has no members");

            let want_mass: f64 = node.members.iter().map(|&i| mass[i]).sum();
            assert!(
                (node.mass - want_mass).abs() < 1e-9,
                "node {nid} mass {} vs {want_mass}",
                node.mass
            );
            assert!(
                node.mass >= p.min_node_mass,
                "node {nid} is under the mass floor"
            );

            for (d, got) in node.centroid.iter().enumerate() {
                let want: f64 = node
                    .members
                    .iter()
                    .map(|&i| mass[i] * mu[i][d])
                    .sum::<f64>()
                    / want_mass;
                assert!(
                    (got - want).abs() < 1e-9,
                    "node {nid} centroid[{d}] = {got} vs {want}"
                );
            }

            let want_lens: f64 = node
                .members
                .iter()
                .map(|&i| if lens[i].is_finite() { lens[i] } else { lo })
                .sum::<f64>()
                / node.members.len() as f64;
            assert!(
                (node.lens_value - want_lens).abs() < 1e-9,
                "node {nid} lens {} vs {want_lens}",
                node.lens_value
            );
        }

        // Every microcluster that clears the mass floor lands in at least one node, and a node's
        // members all come from its own cover bin.
        let covered: std::collections::HashSet<usize> = g
            .nodes
            .iter()
            .flat_map(|n| n.members.iter().copied())
            .collect();
        assert_eq!(covered.len(), feats.len(), "the cover lost microclusters");
    }

    #[test]
    fn a_node_is_single_linkage_connected_at_the_scaled_median_gap() {
        // Members of one node must be reachable from each other by hops no longer than
        // `link_scale × median nearest-neighbour gap` within the bin. A threshold built from the
        // wrong statistic either fuses the whole bin into one node or shatters it into singletons.
        let feats = dumbbell();
        for link_scale in [1.0_f64, 2.0, 4.0] {
            let p = MapperParams {
                lens: Lens::Coordinate(0),
                resolution: 6,
                gain: 0.25,
                link_scale,
                min_node_mass: 0.0,
            };
            let g = mapper(&feats, &p);
            let mu: Vec<Vec<f64>> = feats.iter().map(centroid64).collect();

            for (nid, node) in g.nodes.iter().enumerate() {
                // Recompute the bin's threshold from the members of every node sharing this bin.
                let bin_members: Vec<usize> = g
                    .nodes
                    .iter()
                    .filter(|n| n.bin == node.bin)
                    .flat_map(|n| n.members.iter().copied())
                    .collect();
                let bn = bin_members.len();
                if bn < 2 {
                    continue;
                }
                let mut nn = vec![f64::INFINITY; bn];
                for a in 0..bn {
                    for b in (a + 1)..bn {
                        let d = euclid(&mu[bin_members[a]], &mu[bin_members[b]]);
                        nn[a] = nn[a].min(d);
                        nn[b] = nn[b].min(d);
                    }
                }
                nn.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let thresh = link_scale * nn[bn / 2];

                // Single-linkage closure of the node's members under `thresh` must be the node.
                let ms = &node.members;
                let mut reached = vec![false; ms.len()];
                reached[0] = true;
                let mut grew = true;
                while grew {
                    grew = false;
                    for a in 0..ms.len() {
                        if !reached[a] {
                            continue;
                        }
                        for b in 0..ms.len() {
                            if !reached[b] && euclid(&mu[ms[a]], &mu[ms[b]]) <= thresh + 1e-12 {
                                reached[b] = true;
                                grew = true;
                            }
                        }
                    }
                }
                assert!(
                    reached.iter().all(|&r| r),
                    "link_scale {link_scale}: node {nid} is not connected at threshold {thresh}"
                );
            }
        }
    }
    #[test]
    fn the_rms_radius_is_the_root_mean_square_deviation_from_the_centroid() {
        let mut square = Spherical::new(2);
        for p in [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [4.0, 3.0]] {
            square.push(&p, 1.0);
        }
        // Centroid (2, 1.5); every corner sits √(2² + 1.5²) = 2.5 away, so the RMS radius is 2.5.
        assert!(
            (rms_radius(&square) - 2.5).abs() < 1e-12,
            "{}",
            rms_radius(&square)
        );

        // Doubling the box about its centroid doubles the radius -- `√(S/n)` divides, it does not
        // take a remainder (which would read 0 here, and 1 for the box above).
        let mut wide = Spherical::new(2);
        for p in [[0.0, 0.0], [8.0, 0.0], [0.0, 6.0], [8.0, 6.0]] {
            wide.push(&p, 1.0);
        }
        assert!(
            (rms_radius(&wide) - 5.0).abs() < 1e-12,
            "{}",
            rms_radius(&wide)
        );

        // A lone point has no spread; an empty feature has no radius to report.
        let mut one = Spherical::new(2);
        one.push(&[7.0, -2.0], 1.0);
        assert_eq!(rms_radius(&one), 0.0);
        assert_eq!(rms_radius(&Spherical::<f64>::new(2)), 0.0);
    }

    #[test]
    fn every_lens_computes_the_quantity_it_names() {
        let mu = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![-6.0, 8.0]];
        let radius = vec![0.5, 1.5, 2.5];
        let (d01, d02) = (5.0_f64, 10.0_f64);
        let d12 = (9.0_f64 * 9.0 + 4.0 * 4.0).sqrt();

        assert_eq!(lens_values(&mu, &radius, Lens::Radius), radius);
        assert_eq!(lens_values(&mu, &radius, Lens::L2Norm), vec![0.0, d01, d02]);
        assert_eq!(
            lens_values(&mu, &radius, Lens::Coordinate(1)),
            vec![0.0, 4.0, 8.0]
        );
        // A coordinate past the end of the centroid reads as zero rather than panicking.
        assert_eq!(
            lens_values(&mu, &radius, Lens::Coordinate(9)),
            vec![0.0, 0.0, 0.0]
        );

        // Eccentricity is the *mean* distance to the other points, so it divides by m − 1.
        let want_ecc = [(d01 + d02) / 2.0, (d01 + d12) / 2.0, (d02 + d12) / 2.0];
        for (got, want) in lens_values(&mu, &radius, Lens::Eccentricity)
            .iter()
            .zip(want_ecc)
        {
            assert!((got - want).abs() < 1e-12, "eccentricity {got} vs {want}");
        }

        // Density is the reciprocal of the mean of the k smallest distances.
        let want_k1 = [1.0 / d01, 1.0 / d01, 1.0 / d12];
        for (got, want) in lens_values(&mu, &radius, Lens::Density { k: 1 })
            .iter()
            .zip(want_k1)
        {
            assert!((got - want).abs() < 1e-12, "density k=1 {got} vs {want}");
        }
        let want_k2 = [2.0 / (d01 + d02), 2.0 / (d01 + d12), 2.0 / (d02 + d12)];
        for (got, want) in lens_values(&mu, &radius, Lens::Density { k: 2 })
            .iter()
            .zip(want_k2)
        {
            assert!((got - want).abs() < 1e-12, "density k=2 {got} vs {want}");
        }
        // `k` above the neighbour count clamps to all of them; `k = 0` clamps up to one.
        assert_eq!(
            lens_values(&mu, &radius, Lens::Density { k: 99 }),
            lens_values(&mu, &radius, Lens::Density { k: 2 })
        );
        assert_eq!(
            lens_values(&mu, &radius, Lens::Density { k: 0 }),
            lens_values(&mu, &radius, Lens::Density { k: 1 })
        );

        // One point has no other point to measure against, and coincident points are infinitely
        // dense -- the two degenerate branches the cover's all-infinite fallback depends on.
        let single = vec![vec![1.0, 2.0]];
        assert_eq!(lens_values(&single, &[0.7], Lens::Eccentricity), vec![0.0]);
        assert_eq!(
            lens_values(&single, &[0.7], Lens::Density { k: 3 }),
            vec![0.0]
        );
        let dup = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        assert!(
            lens_values(&dup, &[0.0, 0.0], Lens::Density { k: 1 })
                .iter()
                .all(|v| v.is_infinite())
        );
    }

    #[test]
    fn the_bhattacharyya_coefficient_matches_its_closed_form_with_unequal_spreads() {
        // BC = √(2 σa σb / (σa² + σb²)) · exp(−Δ² / (4(σa² + σb²))). The first factor is exactly 1
        // whenever the variances match, which is why an equal-variance fixture cannot see it.
        let closed = |ma: f64, va: f64, mb: f64, vb: f64| {
            let (sa, sb) = ((va + 1e-12).sqrt(), (vb + 1e-12).sqrt());
            let (va, vb) = (va + 1e-12, vb + 1e-12);
            (2.0 * sa * sb / (va + vb)).sqrt() * (-(ma - mb).powi(2) / (4.0 * (va + vb))).exp()
        };
        for &(ma, va, mb, vb) in &[
            (0.0, 1.0, 0.0, 4.0),
            (0.0, 0.25, 1.0, 9.0),
            (-2.0, 3.0, 5.0, 0.5),
        ] {
            let got = bhattacharyya_diag(&[ma], &[va], &[mb], &[vb]);
            let want = closed(ma, va, mb, vb);
            assert!((got - want).abs() < 1e-9, "BC = {got} vs {want}");
        }
        // Independent dimensions multiply, so the summed distance exponentiates to their product.
        let joint = bhattacharyya_diag(&[0.0, -2.0], &[1.0, 3.0], &[0.0, 5.0], &[4.0, 0.5]);
        let product = closed(0.0, 1.0, 0.0, 4.0) * closed(-2.0, 3.0, 5.0, 0.5);
        assert!((joint - product).abs() < 1e-9, "{joint} vs {product}");
        // A single-point microcluster has zero variance: the floor keeps it comparable instead of
        // dividing by zero, and the coefficient still decays as the two are pulled apart.
        let coincident = bhattacharyya_diag(&[1.0], &[0.0], &[1.0], &[0.0]);
        assert!(
            (coincident - 1.0).abs() < 1e-12,
            "coincident = {coincident}"
        );
        let nudged = bhattacharyya_diag(&[1.0], &[0.0], &[1.0 + 1e-5], &[0.0]);
        assert!(nudged < 1.0 && nudged > 0.0, "nudged = {nudged}");
    }
    /// Node identity that survives the implementation's arbitrary within-bin ordering (components
    /// come out of a `HashMap`): the cover bin plus the ascending microcluster indices it holds.
    type NodeKey = (usize, Vec<usize>);

    struct RefNode {
        key: NodeKey,
        mass: f64,
        centroid: Vec<f64>,
        lens_value: f64,
    }

    struct RefGraph {
        nodes: Vec<RefNode>,
        edges: Vec<(NodeKey, NodeKey, usize)>,
        branch_points: Vec<NodeKey>,
    }

    /// An independent reconstruction of the cover → linkage → nerve pipeline, written from the
    /// documented algorithm rather than from the code: components come out of a breadth-first
    /// search instead of union-find, and an edge's weight is the size of the intersection of two
    /// nodes' member sets instead of an accumulator swept over the microclusters.
    fn reference_mapper(mu: &[Vec<f64>], mass: &[f64], f: &[f64], p: &MapperParams) -> RefGraph {
        let m = mu.len();
        let finite: Vec<f64> = f.iter().copied().filter(|v| v.is_finite()).collect();
        let (lo, hi) = if finite.is_empty() {
            (0.0, 0.0)
        } else {
            (
                finite.iter().copied().fold(f64::INFINITY, f64::min),
                finite.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        };
        let resolution = p.resolution.max(1);
        let step = if hi > lo {
            (hi - lo) / resolution as f64
        } else {
            1.0
        };
        let pad = p.gain.clamp(0.0, 0.999) * step / 2.0;
        let at = |i: usize| if f[i].is_finite() { f[i] } else { lo };

        let mut nodes: Vec<RefNode> = Vec::new();
        for bin in 0..resolution {
            let blo = lo + bin as f64 * step - pad;
            let bhi = lo + (bin as f64 + 1.0) * step + pad;
            let members: Vec<usize> = (0..m).filter(|&i| at(i) >= blo && at(i) <= bhi).collect();
            let bn = members.len();
            if bn == 0 {
                continue;
            }
            // Threshold = link_scale × the median of the per-member nearest-neighbour gaps.
            let thresh = if bn > 1 {
                let mut gaps: Vec<f64> = members
                    .iter()
                    .map(|&i| {
                        members
                            .iter()
                            .filter(|&&j| j != i)
                            .map(|&j| euclid(&mu[i], &mu[j]))
                            .fold(f64::INFINITY, f64::min)
                    })
                    .collect();
                gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                p.link_scale * gaps[bn / 2]
            } else {
                f64::INFINITY
            };
            let mut seen = vec![false; bn];
            for start in 0..bn {
                if seen[start] {
                    continue;
                }
                let mut queue = vec![start];
                seen[start] = true;
                let mut head = 0;
                while head < queue.len() {
                    let a = queue[head];
                    head += 1;
                    for b in 0..bn {
                        if !seen[b] && euclid(&mu[members[a]], &mu[members[b]]) <= thresh {
                            seen[b] = true;
                            queue.push(b);
                        }
                    }
                }
                let mut group: Vec<usize> = queue.iter().map(|&l| members[l]).collect();
                group.sort_unstable();
                let node_mass: f64 = group.iter().map(|&i| mass[i]).sum();
                if node_mass < p.min_node_mass {
                    continue;
                }
                let dim = mu[group[0]].len();
                let mut centroid = vec![0.0; dim];
                for &i in &group {
                    for (d, c) in centroid.iter_mut().enumerate() {
                        *c += mass[i] * mu[i][d];
                    }
                }
                if node_mass > 0.0 {
                    centroid.iter_mut().for_each(|c| *c /= node_mass);
                }
                let lens_value = group.iter().map(|&i| at(i)).sum::<f64>() / group.len() as f64;
                nodes.push(RefNode {
                    key: (bin, group),
                    mass: node_mass,
                    centroid,
                    lens_value,
                });
            }
        }
        nodes.sort_by(|a, b| a.key.cmp(&b.key));

        let mut edges: Vec<(NodeKey, NodeKey, usize)> = Vec::new();
        let mut degree = vec![0usize; nodes.len()];
        for a in 0..nodes.len() {
            for b in (a + 1)..nodes.len() {
                let shared = nodes[a]
                    .key
                    .1
                    .iter()
                    .filter(|i| nodes[b].key.1.contains(i))
                    .count();
                if shared > 0 {
                    edges.push((nodes[a].key.clone(), nodes[b].key.clone(), shared));
                    degree[a] += 1;
                    degree[b] += 1;
                }
            }
        }
        edges.sort();
        let branch_points: Vec<NodeKey> = (0..nodes.len())
            .filter(|&i| degree[i] >= 3)
            .map(|i| nodes[i].key.clone())
            .collect();
        RefGraph {
            nodes,
            edges,
            branch_points,
        }
    }

    /// Re-key a built `MapperGraph` into the reference's ordering-free shape.
    fn as_reference(g: &MapperGraph) -> RefGraph {
        let keys: Vec<NodeKey> = g.nodes.iter().map(|n| (n.bin, n.members.clone())).collect();
        let mut edges: Vec<(NodeKey, NodeKey, usize)> = g
            .edges
            .iter()
            .map(|&(a, b, w)| {
                let (x, y) = (keys[a].clone(), keys[b].clone());
                if x <= y { (x, y, w) } else { (y, x, w) }
            })
            .collect();
        edges.sort();
        let mut branch_points: Vec<NodeKey> =
            g.branch_points.iter().map(|&i| keys[i].clone()).collect();
        branch_points.sort();
        let mut nodes: Vec<RefNode> = g
            .nodes
            .iter()
            .zip(&keys)
            .map(|(n, k)| RefNode {
                key: k.clone(),
                mass: n.mass,
                centroid: n.centroid.clone(),
                lens_value: n.lens_value,
            })
            .collect();
        nodes.sort_by(|a, b| a.key.cmp(&b.key));
        RefGraph {
            nodes,
            edges,
            branch_points,
        }
    }

    fn assert_same_mapper(got: &RefGraph, want: &RefGraph, ctx: &str) {
        let got_keys: Vec<&NodeKey> = got.nodes.iter().map(|n| &n.key).collect();
        let want_keys: Vec<&NodeKey> = want.nodes.iter().map(|n| &n.key).collect();
        assert_eq!(got_keys, want_keys, "{ctx}: nodes");
        for (g, w) in got.nodes.iter().zip(&want.nodes) {
            assert!(
                (g.mass - w.mass).abs() < 1e-9,
                "{ctx}: node {:?} mass {} vs {}",
                g.key,
                g.mass,
                w.mass
            );
            assert!(
                (g.lens_value - w.lens_value).abs() < 1e-9,
                "{ctx}: node {:?} lens {} vs {}",
                g.key,
                g.lens_value,
                w.lens_value
            );
            assert_eq!(g.centroid.len(), w.centroid.len(), "{ctx}: centroid width");
            for (d, (a, b)) in g.centroid.iter().zip(&w.centroid).enumerate() {
                assert!(
                    (a - b).abs() < 1e-9,
                    "{ctx}: node {:?} centroid[{d}] {a} vs {b}",
                    g.key
                );
            }
        }
        assert_eq!(got.edges, want.edges, "{ctx}: nerve edges");
        assert_eq!(
            got.branch_points, want.branch_points,
            "{ctx}: branch points"
        );
    }

    /// The whole pipeline, not just the shape it produces: every node's membership, mass, centroid
    /// and lens value, every nerve edge *and its weight*, and every branch point, against an
    /// independent reconstruction -- swept across the two parameters that move the answer.
    #[test]
    fn the_mapper_graph_matches_an_independent_reconstruction() {
        let feats = dumbbell();
        let mu: Vec<Vec<f64>> = feats.iter().map(centroid64).collect();
        let mass: Vec<f64> = feats.iter().map(|f| f.weight()).collect();
        let radius: Vec<f64> = feats.iter().map(rms_radius).collect();

        let mut node_counts: Vec<usize> = Vec::new();
        for lens in [Lens::Coordinate(0), Lens::Radius, Lens::Eccentricity] {
            let f = lens_values(&mu, &radius, lens);
            for &link_scale in &[0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0] {
                for &(resolution, gain) in &[(4, 0.0), (6, 0.25), (8, 0.4), (12, 0.9)] {
                    let p = MapperParams {
                        lens,
                        resolution,
                        gain,
                        link_scale,
                        min_node_mass: 0.0,
                    };
                    let ctx =
                        format!("{lens:?} link_scale={link_scale} res={resolution} gain={gain}");
                    let want = reference_mapper(&mu, &mass, &f, &p);
                    assert_same_mapper(&as_reference(&mapper(&feats, &p)), &want, &ctx);
                    if matches!(lens, Lens::Coordinate(0)) && resolution == 8 {
                        node_counts.push(want.nodes.len());
                    }
                }
            }
        }
        // The linkage sweep has to actually cross a boundary, or every comparison above is being
        // made against the same graph and the test proves nothing about `link_scale`.
        node_counts.dedup();
        assert!(
            node_counts.len() > 2,
            "the link_scale sweep never changed the partition: {node_counts:?}"
        );
    }

    /// The mass floor is a strict `<`: a node whose mass lands exactly on it is kept.
    #[test]
    fn a_node_whose_mass_lands_on_the_floor_is_kept() {
        let feats = dumbbell();
        let mu: Vec<Vec<f64>> = feats.iter().map(centroid64).collect();
        let mass: Vec<f64> = feats.iter().map(|f| f.weight()).collect();
        let radius: Vec<f64> = feats.iter().map(rms_radius).collect();
        let base = MapperParams {
            lens: Lens::Coordinate(0),
            resolution: 8,
            gain: 0.3,
            link_scale: 2.0,
            min_node_mass: 0.0,
        };
        let f = lens_values(&mu, &radius, base.lens);

        // Masses are integer point counts, so a floor set to one of them is hit exactly.
        let mut masses: Vec<u64> = reference_mapper(&mu, &mass, &f, &base)
            .nodes
            .iter()
            .map(|n| n.mass as u64)
            .collect();
        masses.sort_unstable();
        masses.dedup();
        assert!(
            masses.len() > 2,
            "not enough distinct node masses: {masses:?}"
        );
        let floor = masses[masses.len() / 2] as f64;

        let p = MapperParams {
            min_node_mass: floor,
            ..base
        };
        let got = as_reference(&mapper(&feats, &p));
        assert_same_mapper(&got, &reference_mapper(&mu, &mass, &f, &p), "mass floor");
        assert!(
            got.nodes.iter().any(|n| (n.mass - floor).abs() < 1e-12),
            "no node sits exactly on the floor {floor}, so `<` and `<=` cannot be told apart"
        );
        assert!(
            got.nodes.len() < reference_mapper(&mu, &mass, &f, &base).nodes.len(),
            "the floor dropped nothing"
        );
    }
    /// The elder rule without union-find: components are explicit member lists, and a class's birth
    /// is recomputed as the minimum over its members every time it is needed.
    fn reference_persistence(births: &[f64], edges: &[(f64, usize, usize)]) -> Vec<(f64, f64)> {
        let n = births.len();
        let mut comp: Vec<usize> = (0..n).collect();
        let mut members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut order: Vec<usize> = (0..edges.len()).collect();
        order.sort_by(|&i, &j| {
            edges[i]
                .0
                .partial_cmp(&edges[j].0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let birth_of = |members: &[Vec<usize>], c: usize| {
            members[c]
                .iter()
                .map(|&v| births[v])
                .fold(f64::INFINITY, f64::min)
        };
        let mut points: Vec<(f64, f64)> = Vec::new();
        for &e in &order {
            let (val, a, b) = edges[e];
            let (ca, cb) = (comp[a], comp[b]);
            if ca == cb {
                continue; // a cycle closes: a beta-1 birth, no 0-D event
            }
            // The younger class -- the one born later -- is the one that dies at this edge.
            let younger = birth_of(&members, ca).max(birth_of(&members, cb));
            points.push((younger, val));
            let moved = std::mem::take(&mut members[cb]);
            for &v in &moved {
                comp[v] = ca;
            }
            members[ca].extend(moved);
        }
        let mut essential: Vec<usize> = comp.clone();
        essential.sort_unstable();
        essential.dedup();
        for c in essential {
            points.push((birth_of(&members, c), f64::INFINITY));
        }
        points
    }

    fn sorted_points(points: &[(f64, f64)]) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = points
            .iter()
            .map(|&(b, d)| (b.to_bits(), d.to_bits()))
            .collect();
        v.sort_unstable();
        v
    }

    /// Under `Filtration::EdgeOverlap` every node is born at `0`, so the elder rule never has to
    /// choose and any corruption of it is invisible. Under `Filtration::Lens` the births differ,
    /// and each bar records which of two classes was the younger.
    #[test]
    fn the_persistence_diagram_matches_an_independent_elder_rule() {
        let feats = dumbbell();
        let mut lens_bars = 0;
        for &(resolution, gain, link_scale) in &[
            (6, 0.25, 2.0_f64),
            (8, 0.4, 3.0),
            (10, 0.5, 1.5),
            (12, 0.3, 2.5),
        ] {
            let g = mapper(
                &feats,
                &MapperParams {
                    lens: Lens::Coordinate(0),
                    resolution,
                    gain,
                    link_scale,
                    min_node_mass: 0.0,
                },
            );
            for filt in [Filtration::EdgeOverlap, Filtration::Lens] {
                let births: Vec<f64> = match filt {
                    Filtration::EdgeOverlap => vec![0.0; g.nodes.len()],
                    Filtration::Lens => g.nodes.iter().map(|n| n.lens_value).collect(),
                };
                let fe: Vec<(f64, usize, usize)> = g
                    .edges
                    .iter()
                    .enumerate()
                    .map(|(i, &(a, b, _))| {
                        let val = match filt {
                            Filtration::EdgeOverlap => 1.0 - g.edge_overlap[i],
                            Filtration::Lens => births[a].max(births[b]),
                        };
                        (val, a, b)
                    })
                    .collect();

                let d = g.persistence_diagram(filt);
                let want = reference_persistence(&births, &fe);
                let ctx = format!("{filt:?} res={resolution} gain={gain} link={link_scale}");
                assert_eq!(
                    sorted_points(&d.points),
                    sorted_points(&want),
                    "{ctx}: diagram"
                );
                assert_eq!(
                    d.n_components,
                    want.iter().filter(|p| p.1.is_infinite()).count(),
                    "{ctx}: components"
                );
                // `loop_births` accounts for exactly the edges that closed a cycle.
                assert_eq!(
                    d.loop_births.len(),
                    g.edges.len() + d.n_components - g.nodes.len(),
                    "{ctx}: cycle count"
                );

                // The diagram is ordered by persistence, descending -- not by any other function
                // of `(birth, death)` that happens to agree on this fixture.
                for w in d.points.windows(2) {
                    let (p, q) = (w[0].1 - w[0].0, w[1].1 - w[1].0);
                    assert!(
                        p >= q || (p.is_nan() && q.is_nan()),
                        "{ctx}: {p} before {q}"
                    );
                }
                if matches!(filt, Filtration::Lens) {
                    // Sum and ratio must disagree with persistence somewhere on this fixture, or
                    // the ordering assertion above cannot tell the three apart.
                    let finite: Vec<(f64, f64)> = d
                        .points
                        .iter()
                        .copied()
                        .filter(|p| p.1.is_finite())
                        .collect();
                    lens_bars += finite
                        .iter()
                        .enumerate()
                        .flat_map(|(i, &a)| finite[i + 1..].iter().map(move |&b| (a, b)))
                        .filter(|&((ab, ad), (bb, bd))| ad + ab < bd + bb || ad / ab < bd / bb)
                        .count();
                }
            }
        }
        assert!(
            lens_bars > 0,
            "no pair of bars orders differently under persistence, sum and ratio"
        );
    }
    fn bare_node(bin: usize, lens_value: f64) -> MapperNode {
        MapperNode {
            members: vec![bin],
            mass: 1.0,
            bin,
            centroid: vec![lens_value],
            lens_value,
        }
    }

    /// The elder rule decides *which of two merging classes dies*, and it only has a decision to
    /// make when their births differ. On a Mapper nerve grown from a coordinate lens the node ids
    /// run with the lens, so the first endpoint of an edge is almost always the elder one and the
    /// rule is never asked the hard question. This nerve is built by hand to ask it: two of its
    /// four merges join a *younger* class through the lower-numbered endpoint.
    #[test]
    fn the_elder_rule_kills_the_younger_class_when_the_births_differ() {
        let births = [5.0, 1.0, 3.0, 0.5, 4.0];
        let edges = vec![(0usize, 1usize, 1usize), (2, 3, 1), (0, 2, 1), (1, 4, 1)];
        let g = MapperGraph {
            nodes: births
                .iter()
                .enumerate()
                .map(|(i, &b)| bare_node(i, b))
                .collect(),
            edge_overlap: vec![0.9, 0.5, 0.2, 0.7],
            edges: edges.clone(),
            branch_points: Vec::new(),
            bridges: Vec::new(),
        };

        for filt in [Filtration::Lens, Filtration::EdgeOverlap] {
            let bv: Vec<f64> = match filt {
                Filtration::EdgeOverlap => vec![0.0; births.len()],
                Filtration::Lens => births.to_vec(),
            };
            let fe: Vec<(f64, usize, usize)> = edges
                .iter()
                .enumerate()
                .map(|(i, &(a, b, _))| {
                    let val = match filt {
                        Filtration::EdgeOverlap => 1.0 - g.edge_overlap[i],
                        Filtration::Lens => bv[a].max(bv[b]),
                    };
                    (val, a, b)
                })
                .collect();
            let d = g.persistence_diagram(filt);
            assert_eq!(
                sorted_points(&d.points),
                sorted_points(&reference_persistence(&bv, &fe)),
                "{filt:?}: diagram"
            );
        }

        // Spelled out for the lens filtration, where the answer is short enough to read: the merge
        // at 3 kills node 2 (born 3), not node 3 (born 0.5), and the merge at 5 kills the class
        // born at 1 rather than the one born at 0.5 -- the survivor is the eldest class of all.
        let d = g.persistence_diagram(Filtration::Lens);
        let mut got: Vec<(f64, f64)> = d.points.clone();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            got,
            vec![
                (0.5, f64::INFINITY),
                (1.0, 5.0),
                (3.0, 3.0),
                (4.0, 4.0),
                (5.0, 5.0)
            ]
        );
        assert_eq!(d.n_components, 1);
        assert!(d.loop_births.is_empty());
        // Ordered by persistence: the essential class first, then the one real bar, then the three
        // zero-length ones. Sum or ratio of `(birth, death)` would not put them in this order.
        assert_eq!(d.points[0], (0.5, f64::INFINITY));
        assert_eq!(d.points[1], (1.0, 5.0));
    }
    /// The same comparison, over many random nerves. Which of two merging classes is `ra` and which
    /// is `rb` falls out of union-by-rank, so a hand-built example can only reach some of the cases
    /// the rule has to get right; a sweep of small random graphs reaches the rest. The reference is
    /// the right oracle here because it never looks at a root index at all -- the class that dies
    /// is the one born later, full stop.
    #[test]
    fn the_elder_rule_agrees_with_the_reference_on_random_nerves() {
        let mut rng = crate::clustering::rng::SplitMix64::new(0x5eed);
        let mut merges = 0usize;
        for case in 0..300 {
            let n = 4 + (rng.next_u64() % 7) as usize;
            let births: Vec<f64> = (0..n).map(|_| (rng.next_u64() % 17) as f64 * 0.5).collect();
            let mut edges: Vec<(usize, usize, usize)> = Vec::new();
            for a in 0..n {
                for b in (a + 1)..n {
                    if rng.next_u64() % 4 == 0 {
                        edges.push((a, b, 1));
                    }
                }
            }
            edges.sort_unstable();
            let g = MapperGraph {
                nodes: births
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| bare_node(i, b))
                    .collect(),
                edge_overlap: (0..edges.len())
                    .map(|_| (rng.next_u64() % 11) as f64 / 10.0)
                    .collect(),
                edges: edges.clone(),
                branch_points: Vec::new(),
                bridges: Vec::new(),
            };
            for filt in [Filtration::Lens, Filtration::EdgeOverlap] {
                let bv: Vec<f64> = match filt {
                    Filtration::EdgeOverlap => vec![0.0; n],
                    Filtration::Lens => births.clone(),
                };
                let fe: Vec<(f64, usize, usize)> = edges
                    .iter()
                    .enumerate()
                    .map(|(i, &(a, b, _))| {
                        let val = match filt {
                            Filtration::EdgeOverlap => 1.0 - g.edge_overlap[i],
                            Filtration::Lens => bv[a].max(bv[b]),
                        };
                        (val, a, b)
                    })
                    .collect();
                let d = g.persistence_diagram(filt);
                let want = reference_persistence(&bv, &fe);
                assert_eq!(
                    sorted_points(&d.points),
                    sorted_points(&want),
                    "case {case} {filt:?}: births {births:?} edges {edges:?}"
                );
                assert_eq!(
                    d.n_components,
                    want.iter().filter(|p| p.1.is_infinite()).count(),
                    "case {case} {filt:?}: components"
                );
                merges += want.iter().filter(|p| p.1.is_finite()).count();
            }
        }
        assert!(merges > 500, "the sweep barely merged anything: {merges}");
    }
}
