//! Flat bounded-degree approximate k-NN graph over the leaf means — the index that takes the
//! density head off its complete graph.
//!
//! **Flat, not hierarchical.** Thordsen & Schubert (SISAP 2025) report that in high dimension the
//! HNSW layer stack buys little, that approximating an RNG/SSG is futile there, and that a *capped*
//! beam search is both theoretically motivated and faster in practice. So this is the bottom layer
//! of an HNSW and nothing above it, which removes the layer machinery from the build entirely.
//!
//! **What replaces the upper layers.** Okkels et al. (Inf. Syst. 142 (2026) 102768) Algorithm 4
//! builds the approximate `minPts`-NN graph, *keeps the higher-layer edges*, and takes an exact MST
//! of the union — the long edges are what let the MST bridge separated clusters. With no hierarchy
//! there are no higher layers, so the long edges come instead from a few uniformly random out-edges
//! per vertex, which is the same role played by a construction the search can never reach locally.
//!
//! Nothing here is copied from `eth42/GraphIndexAPI` or `CamillaOkkels/HSSL`: both carry no licence
//! file, which is all-rights-reserved. The algorithms are re-derived from the printed pseudocode.

use crate::clustering::rng::SplitMix64;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// A `(distance, vertex)` pair ordered by distance, then by vertex so the order is total and the
/// build is reproducible. `total_cmp` rather than `partial_cmp().unwrap()`: the inputs are finite by
/// the time they reach here, and a total order costs nothing to ask for.
#[derive(Clone, Copy, PartialEq)]
struct Key(f64, usize);

impl Eq for Key {}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Candidate-list width of the construction beam search, as a multiple of the degree. Wider search
/// costs time and buys recall; `2×` is the smallest multiple that kept the MNIST cophenetic
/// correlation at 1.0000 in `local/scratch/graph_single_linkage.py`.
const BEAM_FACTOR: usize = 2;

/// Hard cap on how many vertices one search may expand.
///
/// This is the *capped* in "capped beam search". An uncapped beam search backtracks until its
/// expansion queue is exhausted, which on a graph whose local geometry lies (the normal case in high
/// dimension) degenerates towards a scan: the queue keeps producing candidates that are nearer than
/// the current worst result and the search never stops early. Capping expansions bounds the build at
/// `O(m · cap · degree)` distance evaluations with no data-dependent tail, and the recall it gives
/// up is recovered by the random out-edges rather than by more backtracking.
const MAX_EXPANSIONS: usize = 64;

/// Uniformly random extra out-edges per vertex. Three is the smallest count for which every fixture
/// in the module tests stays connected; they are the flat stand-in for an HNSW's upper layers, and
/// without them the MST has no candidate edge that bridges two well-separated clusters.
pub const RANDOM_OUT_EDGES: usize = 3;

/// The graph under construction: bounded-degree proximity edges, plus the random shortcuts held
/// outside the degree cap so a random draw can never evict a true neighbour.
struct Graph {
    adj: Vec<Vec<(usize, f64)>>,
    long: Vec<Vec<usize>>,
}

/// Reusable search scratch, so the build allocates one bitset rather than one per insertion.
/// `touched` records what to clear, which keeps the reset proportional to the search, not to `m`.
struct Scratch {
    visited: Vec<bool>,
    touched: Vec<usize>,
}

impl Scratch {
    /// Marks `v` and reports whether it was already marked.
    fn seen(&mut self, v: usize) -> bool {
        if std::mem::replace(&mut self.visited[v], true) {
            return true;
        }
        self.touched.push(v);
        false
    }

    fn reset(&mut self) {
        for &t in &self.touched {
            self.visited[t] = false;
        }
        self.touched.clear();
    }
}

/// Beam search for the `ef` nearest already-inserted vertices to `q`, expanding at most
/// [`MAX_EXPANSIONS`] of them. Returns the candidate list sorted by ascending distance.
fn search(
    q: usize,
    ef: usize,
    entries: &[usize],
    g: &Graph,
    dist: &impl Fn(usize, usize) -> f64,
    scratch: &mut Scratch,
) -> Vec<(usize, f64)> {
    let mut best: BinaryHeap<Key> = BinaryHeap::new(); // max-heap: the worst result is on top
    let mut queue: BinaryHeap<Reverse<Key>> = BinaryHeap::new(); // min-heap: expand nearest first
    for &e in entries {
        if scratch.seen(e) {
            continue;
        }
        let d = dist(q, e);
        queue.push(Reverse(Key(d, e)));
        best.push(Key(d, e));
    }
    let mut expansions = 0usize;
    while let Some(Reverse(Key(d, u))) = queue.pop() {
        // Standard stop rule: nothing left to expand can improve a full result set.
        if best.len() >= ef && d > best.peek().map_or(f64::INFINITY, |k| k.0) {
            break;
        }
        expansions += 1;
        if expansions > MAX_EXPANSIONS {
            break;
        }
        let neighbours = g.adj[u]
            .iter()
            .map(|&(v, _)| v)
            .chain(g.long[u].iter().copied());
        for v in neighbours {
            if scratch.seen(v) {
                continue;
            }
            let dv = dist(q, v);
            if best.len() < ef || dv < best.peek().map_or(f64::INFINITY, |k| k.0) {
                queue.push(Reverse(Key(dv, v)));
                best.push(Key(dv, v));
                if best.len() > ef {
                    best.pop();
                }
            }
        }
    }
    let mut out: Vec<(usize, f64)> = best.into_iter().map(|Key(d, v)| (v, d)).collect();
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    out
}

/// Insert `v` into `list` keeping it sorted by distance and no longer than `degree`.
fn offer(list: &mut Vec<(usize, f64)>, v: usize, d: f64, degree: usize) {
    if list.iter().any(|&(u, _)| u == v) {
        return;
    }
    let at = list.partition_point(|&(_, dd)| dd < d);
    if at >= degree {
        return;
    }
    list.insert(at, (v, d));
    list.truncate(degree);
}

/// Undirected view of a directed adjacency: `i—j` is kept if **either** endpoint kept it.
///
/// The out-degree cap has to be enforced per vertex, so an edge one endpoint keeps is routinely
/// evicted by the other; taking the union rather than the intersection is what makes the result an
/// undirected graph without throwing that asymmetry away. Per-vertex length is therefore *not*
/// bounded by `degree` — a hub can collect many reverse edges — but the **total** edge count still
/// is, at `m · (degree + 2·RANDOM_OUT_EDGES)` directed edges, which is what the complexity argument
/// needs. Each shortcut is recorded at both of its endpoints during the build, hence the factor 2.
fn symmetrise(out: &[Vec<(usize, f64)>]) -> Vec<Vec<(usize, f64)>> {
    let m = out.len();
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    for (i, list) in out.iter().enumerate() {
        for &(j, d) in list {
            adj[i].push((j, d));
            adj[j].push((i, d));
        }
    }
    for list in &mut adj {
        list.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        list.dedup_by_key(|&mut (j, _)| j);
    }
    adj
}

/// Build a symmetric bounded-degree proximity graph over `m` objects under `dist`.
///
/// Vertices are inserted one at a time. Each new vertex first takes [`RANDOM_OUT_EDGES`] uniformly
/// random shortcuts into the part of the graph already built, then searches for its `degree` nearest
/// and offers itself back to each of them — the offer may be refused, as in HNSW, because the
/// neighbour's own list is already full of closer vertices. The whole thing is symmetrised by union
/// at the end.
///
/// **The shortcuts are taken before the search, not after the build.** A capped beam search on a
/// purely local graph cannot reach a distant region at all: on `m` points along a line with degree
/// `k`, each expansion advances at most `k/2` positions, so [`MAX_EXPANSIONS`] expansions reach
/// `O(k·MAX_EXPANSIONS)` and every vertex past that is invisible. The random edges are what make the
/// diameter logarithmic, which is the property the cap needs in order to be cheap rather than blind.
/// They are also held outside the degree cap, so a random draw can never evict a true neighbour:
/// their purpose is to be the edge no local search would ever return.
///
/// The build is `O(m · MAX_EXPANSIONS · degree)` distance evaluations, with no data-dependent tail,
/// against the `O(m²)` of the complete graph it replaces.
pub fn build(
    m: usize,
    degree: usize,
    seed: u64,
    dist: impl Fn(usize, usize) -> f64,
) -> Vec<Vec<(usize, f64)>> {
    let degree = degree.clamp(1, m.saturating_sub(1).max(1));
    let ef = (BEAM_FACTOR * degree).min(m);
    let mut g = Graph {
        adj: vec![Vec::new(); m],
        long: vec![Vec::new(); m],
    };
    let mut rng = SplitMix64::new(seed);
    let mut scratch = Scratch {
        visited: vec![false; m],
        touched: Vec::new(),
    };

    for i in 1..m {
        // Entry points: vertex 0 is always reachable, plus one random already-inserted vertex, which
        // is what stops a long insertion chain from anchoring the whole graph on one region.
        let extra = (rng.next_u64() % i as u64) as usize;
        let entries = [0usize, extra];
        let found = search(i, ef, &entries, &g, &dist, &mut scratch);
        scratch.reset();
        for &(j, d) in found.iter().take(degree) {
            debug_assert_ne!(j, i, "a vertex was reachable before it was inserted");
            offer(&mut g.adj[i], j, d, degree);
            offer(&mut g.adj[j], i, d, degree);
        }
        // Shortcuts are published at both endpoints, and only *after* the search. Publishing the
        // reverse direction first puts `i` into `long[j]` while `i` is still the query, so the
        // search walks back to `i` and returns it as its own nearest neighbour at distance zero —
        // a self-loop, which then truncates `i`'s core-distance walk one neighbour early. The
        // shortcuts among `0..i` are already published, and they are all this search can use.
        for _ in 0..RANDOM_OUT_EDGES {
            let j = (rng.next_u64() % i as u64) as usize;
            g.long[i].push(j);
            g.long[j].push(i);
        }
    }

    for (i, shortcuts) in g.long.iter().enumerate() {
        for &j in shortcuts {
            let d = dist(i, j);
            g.adj[i].push((j, d));
        }
    }
    symmetrise(&g.adj)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `m` points on a line, so nearest neighbours are known by construction.
    fn line() -> impl Fn(usize, usize) -> f64 {
        |i: usize, j: usize| ((i as f64) - (j as f64)).abs()
    }

    fn grid(side: usize) -> impl Fn(usize, usize) -> f64 {
        move |i: usize, j: usize| {
            let (xi, yi) = ((i % side) as f64, (i / side) as f64);
            let (xj, yj) = ((j % side) as f64, (j / side) as f64);
            ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt()
        }
    }

    fn components(adj: &[Vec<(usize, f64)>]) -> usize {
        let m = adj.len();
        let mut seen = vec![false; m];
        let mut n = 0;
        for s in 0..m {
            if seen[s] {
                continue;
            }
            n += 1;
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(u) = stack.pop() {
                for &(v, _) in &adj[u] {
                    if !std::mem::replace(&mut seen[v], true) {
                        stack.push(v);
                    }
                }
            }
        }
        n
    }

    #[test]
    fn the_graph_is_symmetric_and_connected() {
        for (m, degree) in [(2usize, 1usize), (17, 4), (200, 8), (400, 6)] {
            let adj = build(m, degree, 11, line());
            assert_eq!(components(&adj), 1, "m = {m}, degree = {degree}");
            for (i, list) in adj.iter().enumerate() {
                for &(j, d) in list {
                    let back = adj[j].iter().find(|&&(u, _)| u == i);
                    assert!(back.is_some(), "edge {i}->{j} has no reverse");
                    assert_eq!(back.unwrap().1, d);
                }
            }
        }
    }

    #[test]
    fn on_a_line_the_two_true_neighbours_are_always_found() {
        // A one-dimensional fixture is where an approximate index has no excuse: the nearest
        // neighbours of `i` are `i±1`, and any degree ≥ 2 must recover both for every interior point.
        let m = 300;
        let adj = build(m, 6, 3, line());
        for (i, list) in adj.iter().enumerate().take(m - 1).skip(1) {
            for want in [i - 1, i + 1] {
                assert!(
                    list.iter().any(|&(j, _)| j == want),
                    "vertex {i} lost neighbour {want}"
                );
            }
        }
    }

    #[test]
    fn adjacency_is_sorted_and_the_total_edge_count_is_bounded() {
        let degree = 8;
        let m = 256;
        let adj = build(m, degree, 5, grid(16));
        // The bound the complexity argument rests on is on the *total*, not on any one list: an
        // out-degree cap of `degree` plus `2·RANDOM_OUT_EDGES` shortcuts (each recorded at both
        // endpoints during the build), all counted twice again by the symmetrisation.
        let total: usize = adj.iter().map(|l| l.len()).sum();
        assert!(
            total <= 2 * m * (degree + 2 * RANDOM_OUT_EDGES),
            "total = {total}"
        );
        for list in &adj {
            assert!(list.windows(2).all(|w| w[0].1 <= w[1].1));
            let mut ids: Vec<usize> = list.iter().map(|&(j, _)| j).collect();
            ids.sort_unstable();
            let before = ids.len();
            ids.dedup();
            assert_eq!(
                ids.len(),
                before,
                "duplicate neighbour in an adjacency list"
            );
        }
    }

    #[test]
    fn the_same_seed_builds_the_same_graph_and_a_different_one_does_not() {
        let a = build(150, 5, 42, grid(13));
        let b = build(150, 5, 42, grid(13));
        let c = build(150, 5, 43, grid(13));
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn recall_against_the_exact_neighbours_is_high_on_a_grid() {
        // The measurable property an approximate index has to be held to. On a 20×20 grid with
        // degree 10, the graph must recover most of each vertex's true 10 nearest neighbours.
        let side = 20;
        let m = side * side;
        let degree = 10;
        let d = grid(side);
        let adj = build(m, degree, 9, &d);
        let mut hits = 0usize;
        for (i, list) in adj.iter().enumerate() {
            let mut all: Vec<(usize, f64)> =
                (0..m).filter(|&j| j != i).map(|j| (j, d(i, j))).collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let truth: Vec<usize> = all.into_iter().take(degree).map(|(j, _)| j).collect();
            hits += truth
                .iter()
                .filter(|&&j| list.iter().any(|&(u, _)| u == j))
                .count();
        }
        let recall = hits as f64 / (m * degree) as f64;
        assert!(recall > 0.85, "recall = {recall}");
    }
}
