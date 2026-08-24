//! Generic agglomerative clustering on leaf clustering features: UPGMA, WPGMA, UPGMC, WPGMC.
//!
//! [`ward`](super::ward) drives Ward alone with the nearest-neighbour chain, which is only correct
//! for a *reducible* linkage. Centroid and median linkage are not reducible — they invert — so the
//! chain cannot drive them, and this module uses Anderberg's algorithm instead: one global minimum
//! per step over a nearest-neighbour cache repaired lazily. `O(m²·d)` expected, `O(m³·d)` on an
//! adversarial input that invalidates every cached neighbour at every step, `O(m·d)` extra space.
//!
//! ## One accumulator for five linkages
//!
//! Write `α` for the weights a cluster gives its member leaves (`Σ α = 1`). Every linkage here is a
//! function of exactly two per-cluster quantities:
//!
//! ```text
//! m_A = Σ α_i μ_i                        the α-weighted mean of the leaf means
//! V_A = Σ α_i (S_i/n_i + ‖μ_i − m_A‖²)   the α-weighted mean squared radius about it
//! ```
//!
//! and both compose *exactly* under a merge with child weights `(w_a, w_b)`, `β = w_a/(w_a + w_b)`:
//!
//! ```text
//! m_AB = β m_A + (1−β) m_B
//! V_AB = β V_A + (1−β) V_B + β(1−β)‖m_A − m_B‖²
//! ```
//!
//! The `V` recurrence is König–Huygens rearranged so that every term is non-negative. It never
//! forms `Σ α‖μ‖² − ‖m‖²`, which is precisely the catastrophic cancellation BETULA exists to avoid.
//!
//! The five linkages are then two independent choices. **How children are weighted** — by mass
//! (`w = n`: the "U" family, UPGMA/UPGMC/Ward) or equally (`w = 1`: the "W" family, WPGMA/WPGMC).
//! **What is measured** — `‖Δm‖² + V_a + V_b` (the average linkages), `‖Δm‖²` (the centroid ones),
//! or `2·n_a n_b/(n_a + n_b)·‖Δm‖²` (Ward).
//!
//! The W family is why this driver is not parameterised over [`CFDistance`](crate::distance)
//! directly: a [`ClusterFeature`] merge is mass-weighted by construction, so no cluster feature can
//! represent a cluster whose children were combined equally regardless of their size.
//!
//! ## The correspondence with `CFDistance`, and the factor two
//!
//! On mass weights the accumulator *is* the merged cluster feature (`m = μ`, `V = S/n`), so the U
//! family reproduces three of the measures in [`crate::distance`] exactly, all on **squared**
//! distances:
//!
//! | Linkage | `CFDistance` |
//! |---|---|
//! | UPGMA ([`Linkage::Average`]) | `D2²` — [`AverageIntercluster`](crate::distance::AverageIntercluster) |
//! | UPGMC ([`Linkage::Centroid`]) | `D0²` — [`CentroidEuclidean`](crate::distance::CentroidEuclidean) |
//! | Ward ([`Linkage::Ward`]) | `2·D4²` — [`VarianceIncrease`](crate::distance::VarianceIncrease) |
//!
//! The factor two on Ward is not decoration: it is what puts all five linkages on one scale, where
//! each of them reduces to the plain squared distance between two points on single-point leaves.
//! Both facts are asserted in the tests.
//!
//! ## Inversions
//!
//! Centroid and median linkage can merge at a height *below* one of their children's — a real
//! property of the linkage, not a defect. Merges are therefore emitted in agglomeration order and
//! cut by prefix, never sorted by height the way [`ward`](super::ward) can afford to.

use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Which linkage the driver agglomerates under. Names follow SciPy's `linkage(method=…)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub enum Linkage {
    /// UPGMA — mean squared distance over all cross-cluster point pairs; children weighted by mass.
    Average,
    /// WPGMA (McQuitty) — UPGMA with the two children weighted equally, whatever their sizes.
    Weighted,
    /// UPGMC — squared distance between mass-weighted centroids. Admits inversions.
    Centroid,
    /// WPGMC (median) — squared distance between dyadic midpoints. Admits inversions.
    Median,
    /// Ward — `2·D4²`, the doubled variance increase. Provided for the cross-check; the shipped
    /// `method="ward"` head keeps the `O(m²)`-guaranteed nearest-neighbour chain in
    /// [`ward`](super::ward).
    Ward,
}

impl Linkage {
    /// Fraction of the merged cluster's weight contributed by `a`.
    fn beta<R: Real>(self, a: &Node<R>, b: &Node<R>) -> R {
        let half = R::from_f64(0.5).unwrap();
        match self {
            Linkage::Weighted | Linkage::Median => half,
            Linkage::Average | Linkage::Centroid | Linkage::Ward => {
                let total = a.mass + b.mass;
                if total > R::zero() {
                    a.mass / total
                } else {
                    half
                }
            }
        }
    }

    /// Linkage value between two live clusters.
    fn value<R: Real>(self, a: &Node<R>, b: &Node<R>) -> R {
        let d2 = sq_euclidean(&a.mean, &b.mean);
        match self {
            Linkage::Average | Linkage::Weighted => d2 + a.spread + b.spread,
            Linkage::Centroid | Linkage::Median => d2,
            Linkage::Ward => {
                let total = a.mass + b.mass;
                if total > R::zero() {
                    R::from_f64(2.0).unwrap() * a.mass * b.mass / total * d2
                } else {
                    d2
                }
            }
        }
    }
}

/// A live cluster: the two quantities of the module docs plus the mass the U family weights by.
struct Node<R: Real> {
    mass: R,
    mean: Vec<R>,
    spread: R,
}

impl<R: Real> Node<R> {
    fn leaf<C: ClusterFeature<R>>(cf: &C) -> Self {
        let mass = cf.weight();
        let spread = if mass > R::zero() {
            (cf.ssd() / mass).max(R::zero())
        } else {
            R::zero()
        };
        Node {
            mass,
            mean: cf.mean().to_vec(),
            spread,
        }
    }

    /// Absorb `other` under `linkage`. `spread` is updated before `mean`, which it reads.
    fn absorb(&mut self, other: &Self, linkage: Linkage) {
        let beta = linkage.beta(self, other);
        let rest = R::one() - beta;
        let d2 = sq_euclidean(&self.mean, &other.mean);
        self.spread = beta * self.spread + rest * other.spread + beta * rest * d2;
        for (x, &y) in self.mean.iter_mut().zip(&other.mean) {
            *x = beta * *x + rest * y;
        }
        self.mass = self.mass + other.mass;
    }
}

/// One agglomeration step: cluster `from` is merged into cluster `into` at linkage `height`.
pub(crate) struct Merge<R> {
    pub(crate) into: usize,
    pub(crate) from: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) height: R,
}

/// Full dendrogram by Anderberg's algorithm; merges come out in agglomeration order.
fn dendrogram<R: Real, C: ClusterFeature<R>>(features: &[C], linkage: Linkage) -> Vec<Merge<R>> {
    let mut node: Vec<Node<R>> = features.iter().map(Node::leaf).collect();
    anderberg(
        &mut node,
        |a, b| linkage.value(a, b),
        |nodes, a, b| {
            let absorbed = Node {
                mass: nodes[b].mass,
                mean: nodes[b].mean.clone(),
                spread: nodes[b].spread,
            };
            nodes[a].absorb(&absorbed, linkage);
        },
    )
}

/// Anderberg's algorithm over an arbitrary cluster representation.
///
/// One global minimum per step over a lazily repaired nearest-neighbour cache: `O(m²·d)` expected,
/// `O(m³·d)` adversarial, `O(m·d)` space. It assumes **nothing** about the linkage — in particular
/// not reducibility — which is why it, rather than the nearest-neighbour chain in
/// [`ward`](super::ward), is what the non-reducible linkages have to use. Centroid and median need
/// it because they invert (Müllner 2011); Bregman-Ward needs it because it is not reducible for
/// `d ≥ 2` outside squared Euclidean (`docs/adr/002-bregman-ward-anderberg.md`).
///
/// `dist` is the linkage value between two live clusters. `absorb(nodes, a, b)` merges `b` into `a`
/// in place; it takes the whole slice and both indices rather than two references so the caller
/// owns the split borrow, which is where every representation differs.
pub(crate) fn anderberg<R: Real, N>(
    node: &mut [N],
    dist: impl Fn(&N, &N) -> R,
    mut absorb: impl FnMut(&mut [N], usize, usize),
) -> Vec<Merge<R>> {
    let m = node.len();
    let mut alive = vec![true; m];
    let mut nn = vec![usize::MAX; m];
    let mut nnd = vec![R::infinity(); m];

    // `nn[i]` is the nearest live cluster to `i`; the relation is not symmetric, so every live
    // cluster keeps its own. The global minimum of `nnd` over live clusters is the closest pair.
    let rescan = |node: &[N], alive: &[bool], i: usize| -> (usize, R) {
        let mut best = usize::MAX;
        let mut best_d = R::infinity();
        for (j, &live) in alive.iter().enumerate() {
            if live && j != i {
                let d = dist(&node[i], &node[j]);
                if d < best_d {
                    best_d = d;
                    best = j;
                }
            }
        }
        (best, best_d)
    };

    for i in 0..m {
        let (b, d) = rescan(node, &alive, i);
        nn[i] = b;
        nnd[i] = d;
    }

    let mut merges: Vec<Merge<R>> = Vec::with_capacity(m.saturating_sub(1));
    for _ in 1..m {
        let mut a = usize::MAX;
        let mut best = R::infinity();
        for (i, &live) in alive.iter().enumerate() {
            if live && nn[i] != usize::MAX && nnd[i] < best {
                best = nnd[i];
                a = i;
            }
        }
        if a == usize::MAX {
            break;
        }
        let b = nn[a];

        absorb(node, a, b);
        alive[b] = false;
        merges.push(Merge {
            into: a,
            from: b,
            height: best,
        });

        let (na, da) = rescan(node, &alive, a);
        nn[a] = na;
        nnd[a] = da;
        for c in 0..m {
            if !alive[c] || c == a {
                continue;
            }
            if nn[c] == a || nn[c] == b {
                // `c`'s neighbour either died or changed shape; its cached distance can only be
                // trusted downwards, so it has to be found again.
                let (nc, dc) = rescan(node, &alive, c);
                nn[c] = nc;
                nnd[c] = dc;
            } else {
                let d = dist(&node[a], &node[c]);
                if d < nnd[c] {
                    nnd[c] = d;
                    nn[c] = a;
                }
            }
        }
    }
    merges
}

/// Union-find root with path compression.
pub(crate) fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut root = x;
    while parent[root] != root {
        root = parent[root];
    }
    let mut cur = x;
    while parent[cur] != root {
        let next = parent[cur];
        parent[cur] = root;
        cur = next;
    }
    root
}

/// Apply the first `t` merges and return contiguous `0..(m − t)` labels. The prefix is a valid
/// horizontal cut without sorting, because Anderberg agglomerates in globally-minimal order.
pub(crate) fn labels_at<R: Real>(m: usize, merges: &[Merge<R>], t: usize) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..m).collect();
    for mg in merges.iter().take(t) {
        let ra = uf_find(&mut parent, mg.into);
        let rb = uf_find(&mut parent, mg.from);
        if ra != rb {
            parent[rb] = ra;
        }
    }
    let mut label_of = vec![usize::MAX; m];
    let mut next = 0;
    let mut labels = vec![0usize; m];
    for (i, lab) in labels.iter_mut().enumerate() {
        let r = uf_find(&mut parent, i);
        if label_of[r] == usize::MAX {
            label_of[r] = next;
            next += 1;
        }
        *lab = label_of[r];
    }
    labels
}

/// Result of an agglomerative run over features.
pub struct Agglomerative {
    /// Cluster label per input feature (contiguous `0..k`).
    pub labels: Vec<usize>,
}

/// Agglomeratively cluster `features` into `k` clusters. `k` is clamped to `[1, features.len()]`.
pub fn agglomerative<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    linkage: Linkage,
    k: usize,
) -> Agglomerative {
    let m = features.len();
    if m == 0 {
        return Agglomerative { labels: Vec::new() };
    }
    let k = k.max(1).min(m);
    let merges = dendrogram(features, linkage);
    Agglomerative {
        labels: labels_at(m, &merges, m - k),
    }
}

/// Agglomerative clustering with automatic `k`: score every horizontal cut in `[k_min, k_max]` by
/// the Calinski-Harabasz index over the summary and keep the best.
///
/// Same rationale as [`ward_hac_auto`](super::ward::ward_hac_auto) — the height-jump ("elbow")
/// heuristic hallucinates structure in noise, and centroid/median linkage make it worse still by
/// admitting inversions, which give the height sequence jumps that mean nothing at all.
pub fn agglomerative_auto<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    linkage: Linkage,
    k_min: usize,
    k_max: usize,
) -> Agglomerative {
    let m = features.len();
    if m == 0 {
        return Agglomerative { labels: Vec::new() };
    }
    let k_lo = k_min.max(1).min(m);
    let k_hi = k_max.max(k_lo).min(m);
    let merges = dendrogram(features, linkage);
    let mut best = labels_at(m, &merges, m - k_lo);
    let mut best_score = f64::NEG_INFINITY;
    for k in k_lo.max(2)..=k_hi {
        let labels = labels_at(m, &merges, m - k);
        let score = crate::validity::calinski_harabasz(features, &labels, k);
        if score > best_score {
            best_score = score;
            best = labels;
        }
    }
    Agglomerative { labels: best }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{AverageIntercluster, CFDistance, CentroidEuclidean, VarianceIncrease};
    use crate::feature::Spherical;

    const ALL: [Linkage; 5] = [
        Linkage::Average,
        Linkage::Weighted,
        Linkage::Centroid,
        Linkage::Median,
        Linkage::Ward,
    ];

    fn leaf(points: &[[f64; 2]]) -> Spherical<f64> {
        let mut cf = Spherical::new(2);
        for p in points {
            cf.push(p, 1.0);
        }
        cf
    }

    /// A fixture with unequal leaf masses and non-zero leaf scatter — the only regime in which the
    /// U and W families, and the `V` term, can be told apart.
    fn fixture() -> Vec<Spherical<f64>> {
        vec![
            leaf(&[[0.0, 0.0], [0.4, 0.2], [0.1, -0.3]]),
            leaf(&[[1.0, 0.5]]),
            leaf(&[[3.0, 3.0], [3.6, 2.7]]),
            leaf(&[[5.0, -1.0], [4.2, -1.5], [4.8, -0.4], [5.5, -1.1]]),
            leaf(&[[-2.0, 4.0], [-2.3, 3.1]]),
            leaf(&[[7.0, 7.0]]),
        ]
    }

    // ───────── the independent reference: textbook Lance-Williams on a full matrix ─────────

    /// Initial leaf-level distance matrix, in the convention every linkage in the table shares.
    fn seed_matrix(f: &[Spherical<f64>], linkage: Linkage) -> Vec<Vec<f64>> {
        let m = f.len();
        let mut d = vec![vec![0.0; m]; m];
        for i in 0..m {
            for j in 0..m {
                if i == j {
                    continue;
                }
                let d2 = sq_euclidean(f[i].mean(), f[j].mean());
                let (si, sj) = (f[i].ssd() / f[i].weight(), f[j].ssd() / f[j].weight());
                let (ni, nj) = (f[i].weight(), f[j].weight());
                d[i][j] = match linkage {
                    Linkage::Average | Linkage::Weighted => d2 + si + sj,
                    Linkage::Centroid | Linkage::Median => d2,
                    Linkage::Ward => 2.0 * ni * nj / (ni + nj) * d2,
                };
            }
        }
        d
    }

    /// `O(m³)` agglomeration driven by the Lance-Williams recurrences as they are stated in the
    /// literature — an implementation that shares no algebra with the accumulator under test.
    fn lance_williams(f: &[Spherical<f64>], linkage: Linkage) -> (Vec<(usize, usize)>, Vec<f64>) {
        let m = f.len();
        let mut d = seed_matrix(f, linkage);
        let mut size: Vec<f64> = f.iter().map(|c| c.weight()).collect();
        let mut alive = vec![true; m];
        let mut steps = Vec::new();
        let mut heights = Vec::new();

        for _ in 1..m {
            let (mut a, mut b, mut best) = (usize::MAX, usize::MAX, f64::INFINITY);
            for i in 0..m {
                for j in 0..m {
                    if alive[i] && alive[j] && i != j && d[i][j] < best {
                        best = d[i][j];
                        a = i;
                        b = j;
                    }
                }
            }
            steps.push((a, b));
            heights.push(best);
            let (na, nb) = (size[a], size[b]);
            let dab = d[a][b];
            for c in 0..m {
                if !alive[c] || c == a || c == b {
                    continue;
                }
                let (dac, dbc, nc) = (d[a][c], d[b][c], size[c]);
                let new = match linkage {
                    Linkage::Average => (na * dac + nb * dbc) / (na + nb),
                    Linkage::Weighted => 0.5 * dac + 0.5 * dbc,
                    Linkage::Centroid => {
                        (na * dac + nb * dbc) / (na + nb) - na * nb * dab / ((na + nb) * (na + nb))
                    }
                    Linkage::Median => 0.5 * dac + 0.5 * dbc - 0.25 * dab,
                    Linkage::Ward => {
                        ((na + nc) * dac + (nb + nc) * dbc - nc * dab) / (na + nb + nc)
                    }
                };
                d[a][c] = new;
                d[c][a] = new;
            }
            alive[b] = false;
            size[a] = na + nb;
        }
        (steps, heights)
    }

    #[test]
    fn every_linkage_matches_the_textbook_lance_williams_recurrence() {
        let f = fixture();
        for linkage in ALL {
            let got = dendrogram(&f, linkage);
            let (steps, heights) = lance_williams(&f, linkage);
            assert_eq!(got.len(), steps.len(), "{linkage:?}");
            for (i, mg) in got.iter().enumerate() {
                let pair = {
                    let mut p = [mg.into, mg.from];
                    p.sort_unstable();
                    p
                };
                let want = {
                    let mut p = [steps[i].0, steps[i].1];
                    p.sort_unstable();
                    p
                };
                assert_eq!(pair, want, "{linkage:?} step {i}");
                assert!(
                    (mg.height - heights[i]).abs() < 1e-9 * heights[i].abs().max(1.0),
                    "{linkage:?} step {i}: {} vs {}",
                    mg.height,
                    heights[i]
                );
            }
        }
    }

    #[test]
    fn upgma_is_the_mean_squared_distance_over_all_cross_pairs() {
        let a = [[0.0, 0.0], [1.0, 2.0], [-0.5, 0.7]];
        let b = [[4.0, 1.0], [3.2, -0.4]];
        let mut want = 0.0;
        for p in &a {
            for q in &b {
                want += sq_euclidean(p, q);
            }
        }
        want /= (a.len() * b.len()) as f64;
        let got = Linkage::Average.value(&Node::leaf(&leaf(&a)), &Node::leaf(&leaf(&b)));
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn the_u_family_matches_its_cf_distance_counterpart() {
        // Table 4.1: UPGMA <-> D2², UPGMC <-> D0², Ward <-> 2·D4², all on squared distances.
        let f = fixture();
        for (i, j) in [(0, 3), (1, 4), (2, 5), (3, 4)] {
            let (na, nb) = (Node::leaf(&f[i]), Node::leaf(&f[j]));
            let pairs = [
                (
                    Linkage::Average,
                    AverageIntercluster.between(&f[i], &f[j]),
                    1.0,
                ),
                (
                    Linkage::Centroid,
                    CentroidEuclidean.between(&f[i], &f[j]),
                    1.0,
                ),
                (Linkage::Ward, VarianceIncrease.between(&f[i], &f[j]), 2.0),
            ];
            for (linkage, cf_value, factor) in pairs {
                let got = linkage.value(&na, &nb);
                let want = factor * cf_value;
                assert!(
                    (got - want).abs() < 1e-12 * want.abs().max(1.0),
                    "{linkage:?}: {got} vs {want}"
                );
            }
        }
        // The correspondence must survive merging, not just hold on leaves.
        let mut merged = Node::leaf(&f[0]);
        merged.absorb(&Node::leaf(&f[1]), Linkage::Average);
        let mut cf = f[0].clone();
        cf.merge(&f[1]);
        assert!(
            (Linkage::Average.value(&merged, &Node::leaf(&f[3]))
                - AverageIntercluster.between(&cf, &f[3]))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn the_factor_two_is_what_makes_all_five_agree_on_single_point_leaves() {
        let a = Node::leaf(&leaf(&[[1.0, -2.0]]));
        let b = Node::leaf(&leaf(&[[4.0, 2.0]]));
        let want = 25.0; // 3² + 4²
        for linkage in ALL {
            let got = linkage.value(&a, &b);
            assert!((got - want).abs() < 1e-12, "{linkage:?}: {got} vs {want}");
        }
        // Dropping the factor would halve Ward alone, which is the failure the table guards.
        assert!(
            (Linkage::Ward.value(&a, &b)
                - 2.0 * VarianceIncrease.between(&leaf(&[[1.0, -2.0]]), &leaf(&[[4.0, 2.0]]),))
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn the_ward_arm_reproduces_the_nn_chain_dendrogram() {
        let f = fixture();
        for k in 1..=f.len() {
            let mine = agglomerative(&f, Linkage::Ward, k).labels;
            let chain = super::super::ward::ward_hac(&f, k).labels;
            assert_eq!(mine, chain, "k = {k}");
        }
    }

    #[test]
    fn the_w_family_ignores_mass_where_the_u_family_follows_it() {
        // One heavy leaf and one light one, far apart; a third leaf sits near the heavy one.
        // UPGMC's merged centroid is pulled to the heavy child, WPGMC's sits at the midpoint.
        let heavy = leaf(&[[0.0, 0.0]; 19]);
        let light = leaf(&[[10.0, 0.0]]);
        let probe = leaf(&[[1.0, 0.0]]);
        let (mut u, mut w) = (Node::leaf(&heavy), Node::leaf(&heavy));
        u.absorb(&Node::leaf(&light), Linkage::Centroid);
        w.absorb(&Node::leaf(&light), Linkage::Median);
        assert!((u.mean[0] - 0.5).abs() < 1e-12, "{}", u.mean[0]);
        assert!((w.mean[0] - 5.0).abs() < 1e-12, "{}", w.mean[0]);
        let p = Node::leaf(&probe);
        assert!(Linkage::Centroid.value(&u, &p) < Linkage::Median.value(&w, &p));
    }

    #[test]
    fn centroid_and_median_invert_where_average_weighted_and_ward_cannot() {
        // Three near-equilateral points: merging two of them puts their centroid closer to the
        // third than the original pair distance, which is the classic centroid inversion.
        let f = vec![
            leaf(&[[0.0, 0.0]]),
            leaf(&[[1.0, 0.0]]),
            leaf(&[[0.5, 0.8]]),
        ];
        let monotone = |linkage: Linkage| {
            let h: Vec<f64> = dendrogram(&f, linkage).iter().map(|m| m.height).collect();
            h.windows(2).all(|w| w[1] >= w[0] - 1e-12)
        };
        assert!(monotone(Linkage::Average));
        assert!(monotone(Linkage::Weighted));
        assert!(monotone(Linkage::Ward));
        assert!(!monotone(Linkage::Centroid));
        assert!(!monotone(Linkage::Median));
    }

    #[test]
    fn every_linkage_recovers_well_separated_groups() {
        let mut f = Vec::new();
        for (cx, cy) in [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)] {
            for t in 0..5 {
                let o = t as f64 * 0.2;
                f.push(leaf(&[[cx + o, cy - o], [cx - o, cy + o]]));
            }
        }
        for linkage in ALL {
            let labels = agglomerative(&f, linkage, 3).labels;
            for g in 0..3 {
                let first = labels[g * 5];
                assert!(
                    labels[g * 5..g * 5 + 5].iter().all(|&l| l == first),
                    "{linkage:?} split group {g}: {labels:?}"
                );
            }
            let mut sorted = labels.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 3, "{linkage:?}");
        }
    }

    #[test]
    fn auto_k_finds_the_three_groups_without_being_told() {
        let mut f = Vec::new();
        for (cx, cy) in [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0)] {
            for t in 0..5 {
                let o = t as f64 * 0.2;
                f.push(leaf(&[[cx + o, cy - o], [cx - o, cy + o]]));
            }
        }
        for linkage in ALL {
            let labels = agglomerative_auto(&f, linkage, 2, 8).labels;
            let k = labels.iter().max().unwrap() + 1;
            assert_eq!(k, 3, "{linkage:?} chose k = {k}");
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        let empty: Vec<Spherical<f64>> = Vec::new();
        assert!(agglomerative(&empty, Linkage::Average, 3).labels.is_empty());
        assert!(
            agglomerative_auto(&empty, Linkage::Median, 2, 5)
                .labels
                .is_empty()
        );
        let one = vec![leaf(&[[1.0, 1.0]])];
        for linkage in ALL {
            assert_eq!(agglomerative(&one, linkage, 4).labels, vec![0]);
        }
        // Coincident leaves: every distance is zero and every merge is still well defined.
        let same = vec![leaf(&[[2.0, 2.0]]); 4];
        for linkage in ALL {
            let labels = agglomerative(&same, linkage, 2).labels;
            assert_eq!(labels.len(), 4);
            assert_eq!(labels.iter().max().unwrap() + 1, 2);
        }
    }

    #[test]
    fn the_spread_recurrence_stays_exact_far_from_the_origin() {
        // The cancelling form `Σα‖μ‖² − ‖m‖²` loses every digit here; the recurrence must not.
        let shift = 1e7;
        let a = leaf(&[[shift, 0.0], [shift + 2.0, 0.0]]);
        let b = leaf(&[[shift + 6.0, 0.0], [shift + 8.0, 0.0]]);
        let mut n = Node::leaf(&a);
        n.absorb(&Node::leaf(&b), Linkage::Average);
        let mut cf = a.clone();
        cf.merge(&b);
        let want = cf.ssd() / cf.weight(); // deviations ∓4, ∓2 about shift + 4 → 40/4
        assert!((n.spread - want).abs() < 1e-9, "{} vs {want}", n.spread);
        assert!((n.spread - 10.0).abs() < 1e-9, "{}", n.spread);
    }
}
