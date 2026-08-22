//! k-prototypes clustering of **mixed numeric + categorical** data (Huang, 1997/1998).
//!
//! Each cluster is summarised by a *mixed clustering feature* [`MixedCf`]: the numerically stable
//! `(n, μ, S)` of its numeric attributes (a reused [`Diagonal`] CF) plus one category-count histogram
//! per categorical attribute. Both halves are exact, mergeable monoids, so a cluster centre is itself
//! a `MixedCf` and its prototype is `(numeric mean, per-attribute mode)`. The k-prototypes distance
//! between a point and a prototype is
//!
//! ```text
//! d = Σ_j∈num (x_j − μ_j)²  +  γ · Σ_j∈cat [x_j ≠ mode_j]
//! ```
//!
//! where `γ` trades numeric scale against categorical mismatch (Huang's heuristic: `γ ≈ ½·mean σ`).
//! Numeric-only data reduces to k-means and categorical-only to k-modes; the head is exposed for the
//! genuinely *mixed* case.

use crate::clustering::kmeans::weighted_pick;
use crate::clustering::rng::SplitMix64;
use crate::feature::{ClusterFeature, Diagonal};
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// A mixed clustering feature: numeric `(n, μ, S)` plus per-attribute categorical counts.
#[derive(Clone)]
pub struct MixedCf<R: Real> {
    num: Diagonal<R>,
    /// `cat[j][c]` = total weight of category `c` in categorical attribute `j`.
    cat: Vec<Vec<R>>,
    /// Cached arg-max of each `cat[j]` (the per-attribute mode); ties keep the lower code.
    mode: Vec<usize>,
}

impl<R: Real> MixedCf<R> {
    /// Empty feature for `n_numeric` numeric attributes and one histogram per entry of
    /// `cardinalities` (the number of distinct codes in each categorical attribute).
    pub fn new(n_numeric: usize, cardinalities: &[usize]) -> Self {
        Self {
            num: Diagonal::new(n_numeric),
            cat: cardinalities.iter().map(|&c| vec![R::zero(); c]).collect(),
            mode: vec![0; cardinalities.len()],
        }
    }

    /// Aggregated weight (point count).
    pub fn weight(&self) -> R {
        self.num.weight()
    }

    /// Number of numeric attributes.
    pub fn n_numeric(&self) -> usize {
        self.num.dim()
    }

    /// Number of categorical attributes.
    pub fn n_categorical(&self) -> usize {
        self.cat.len()
    }

    /// Numeric mean `μ`.
    pub fn numeric_mean(&self) -> &[R] {
        self.num.mean()
    }

    /// Numeric within-feature scatter `S` (the trace of the numeric scatter matrix).
    pub fn numeric_ssd(&self) -> R {
        self.num.ssd()
    }

    /// Per-attribute mode (the categorical centroid).
    pub fn mode(&self) -> &[usize] {
        &self.mode
    }

    /// Cardinality (histogram length) of each categorical attribute.
    pub fn cardinalities(&self) -> Vec<usize> {
        self.cat.iter().map(|h| h.len()).collect()
    }

    /// Weight of category `code` in attribute `j`.
    fn count(&self, j: usize, code: usize) -> R {
        self.cat[j][code]
    }

    /// Add a mixed point: `num` (length `n_numeric`) and category codes `cat` (length
    /// `n_categorical`, each in range for its attribute). The mode cache is kept current.
    pub fn push(&mut self, num: &[R], cat: &[usize], w: R) {
        self.num.push(num, w);
        for (j, &code) in cat.iter().enumerate() {
            let hist = &mut self.cat[j];
            hist[code] = hist[code] + w;
            if hist[code] > hist[self.mode[j]] {
                self.mode[j] = code;
            }
        }
    }

    /// Merge another feature of the same schema (exact; the mode is recomputed).
    pub fn merge(&mut self, other: &Self) {
        self.num.merge(&other.num);
        for (j, (a, b)) in self.cat.iter_mut().zip(&other.cat).enumerate() {
            for (x, &y) in a.iter_mut().zip(b) {
                *x = *x + y;
            }
            self.mode[j] = argmax(a);
        }
    }
}

fn argmax<R: Real>(hist: &[R]) -> usize {
    let mut best = 0;
    let mut bv = hist.first().copied().unwrap_or(R::zero());
    for (i, &v) in hist.iter().enumerate().skip(1) {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// Mismatch count between a point's category codes and a prototype's modes.
fn cat_mismatch(cat: &[usize], mode: &[usize]) -> usize {
    cat.iter().zip(mode).filter(|(a, b)| a != b).count()
}

/// k-prototypes distance from a mixed point to a prototype `(c_num, c_mode)`.
fn point_dist<R: Real>(num: &[R], cat: &[usize], c_num: &[R], c_mode: &[usize], gamma: R) -> R {
    sq_euclidean(num, c_num) + gamma * R::from_usize(cat_mismatch(cat, c_mode)).unwrap()
}

/// Distance from a weighted micro-cluster to a prototype: the numeric term is the micro's mass times
/// its centroid's squared distance, the categorical term is `γ ×` the number of the micro's points
/// whose category differs from the prototype mode (summed over attributes).
fn micro_dist<R: Real>(m: &MixedCf<R>, c_num: &[R], c_mode: &[usize], gamma: R) -> R {
    let w = m.weight();
    let mut cat_cost = R::zero();
    for (j, &mode) in c_mode.iter().enumerate() {
        cat_cost = cat_cost + (w - m.count(j, mode));
    }
    w * sq_euclidean(m.numeric_mean(), c_num) + gamma * cat_cost
}

/// Single-pass leader summarisation into at most `max_leaders` mixed micro-clusters: each point joins
/// its nearest leader within `threshold` (k-prototypes distance), otherwise starts a new leader. Once
/// the cap is reached every further point joins its nearest leader regardless of `threshold` — bounded
/// memory with graceful accuracy degradation (raise `max_leaders` for finer summaries).
#[allow(clippy::too_many_arguments)]
pub fn summarize_mixed<R: Real>(
    num: &[R],
    cat: &[usize],
    n: usize,
    n_num: usize,
    cards: &[usize],
    gamma: R,
    threshold: R,
    max_leaders: usize,
) -> Vec<MixedCf<R>> {
    let n_cat = cards.len();
    let mut leaders: Vec<MixedCf<R>> = Vec::new();
    for i in 0..n {
        let xn = &num[i * n_num..(i + 1) * n_num];
        let xc = &cat[i * n_cat..(i + 1) * n_cat];
        let mut best = usize::MAX;
        let mut bd = R::infinity();
        for (li, l) in leaders.iter().enumerate() {
            let d = point_dist(xn, xc, l.numeric_mean(), l.mode(), gamma);
            if d < bd {
                bd = d;
                best = li;
            }
        }
        if best != usize::MAX && (bd <= threshold || leaders.len() >= max_leaders) {
            leaders[best].push(xn, xc, R::one());
        } else {
            let mut l = MixedCf::new(n_num, cards);
            l.push(xn, xc, R::one());
            leaders.push(l);
        }
    }
    leaders
}

/// Index of the micro-cluster nearest to a mixed point (k-prototypes distance to its prototype).
pub fn nearest_micro<R: Real>(micros: &[MixedCf<R>], num: &[R], cat: &[usize], gamma: R) -> usize {
    let mut best = 0;
    let mut bd = point_dist(num, cat, micros[0].numeric_mean(), micros[0].mode(), gamma);
    for (i, m) in micros.iter().enumerate().skip(1) {
        let d = point_dist(num, cat, m.numeric_mean(), m.mode(), gamma);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

/// k-prototypes++ seeding over micro-clusters: pick `k` micro indices, the first by weight and the
/// rest by `weight · D²` where `D²` is the mixed distance to the nearest already-chosen prototype.
fn kpp_init<R: Real>(
    micros: &[MixedCf<R>],
    k: usize,
    gamma: R,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let n = micros.len();
    let w: Vec<f64> = micros
        .iter()
        .map(|m| m.weight().to_f64().unwrap_or(0.0))
        .collect();
    let dist = |a: usize, b: usize| -> f64 {
        point_dist(
            micros[a].numeric_mean(),
            micros[a].mode(),
            micros[b].numeric_mean(),
            micros[b].mode(),
            gamma,
        )
        .to_f64()
        .unwrap_or(0.0)
    };
    let mut chosen = Vec::with_capacity(k);
    chosen.push(weighted_pick(&w, rng));
    let mut d2: Vec<f64> = (0..n).map(|i| dist(i, chosen[0])).collect();
    while chosen.len() < k {
        let probs: Vec<f64> = (0..n).map(|i| w[i] * d2[i]).collect();
        let next = weighted_pick(&probs, rng);
        for (i, di) in d2.iter_mut().enumerate() {
            let nd = dist(i, next);
            if nd < *di {
                *di = nd;
            }
        }
        chosen.push(next);
    }
    chosen
}

/// Cluster mixed micro-clusters into `k` groups by Lloyd-style k-prototypes: assign each micro to its
/// nearest prototype `(numeric mean, per-attribute mode)`, then rebuild each prototype as the merge of
/// its members. `n_init` restarts are tried and the one with the lowest objective is kept. Returns one
/// cluster label per micro-cluster.
pub fn kprototypes<R: Real>(
    micros: &[MixedCf<R>],
    k: usize,
    gamma: R,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> Vec<usize> {
    assert!(!micros.is_empty(), "need at least one micro-cluster");
    let n = micros.len();
    let k = k.min(n).max(1);
    let n_num = micros[0].n_numeric();
    let cards: Vec<usize> = micros[0].cat.iter().map(|h| h.len()).collect();

    let mut rng = SplitMix64::new(seed);
    let mut best: Option<(R, Vec<usize>)> = None;
    for _ in 0..n_init.max(1) {
        let mut centers: Vec<MixedCf<R>> = kpp_init(micros, k, gamma, &mut rng)
            .into_iter()
            .map(|s| micros[s].clone())
            .collect();
        let mut labels = vec![usize::MAX; n];
        for _ in 0..max_iter.max(1) {
            let proto: Vec<(Vec<R>, Vec<usize>)> = centers
                .iter()
                .map(|c| (c.numeric_mean().to_vec(), c.mode().to_vec()))
                .collect();
            let mut changed = false;
            for (i, m) in micros.iter().enumerate() {
                let mut best_c = 0;
                let mut bd = micro_dist(m, &proto[0].0, &proto[0].1, gamma);
                for (c, p) in proto.iter().enumerate().skip(1) {
                    let d = micro_dist(m, &p.0, &p.1, gamma);
                    if d < bd {
                        bd = d;
                        best_c = c;
                    }
                }
                if labels[i] != best_c {
                    labels[i] = best_c;
                    changed = true;
                }
            }
            let mut acc: Vec<MixedCf<R>> = (0..k).map(|_| MixedCf::new(n_num, &cards)).collect();
            for (i, m) in micros.iter().enumerate() {
                acc[labels[i]].merge(m);
            }
            for (c, a) in acc.into_iter().enumerate() {
                if a.weight() > R::zero() {
                    centers[c] = a;
                }
            }
            if !changed {
                break;
            }
        }
        let proto: Vec<(Vec<R>, Vec<usize>)> = centers
            .iter()
            .map(|c| (c.numeric_mean().to_vec(), c.mode().to_vec()))
            .collect();
        let mut inertia = R::zero();
        for (i, m) in micros.iter().enumerate() {
            let p = &proto[labels[i]];
            inertia = inertia + m.numeric_ssd() + micro_dist(m, &p.0, &p.1, gamma);
        }
        match &best {
            Some((bi, _)) if inertia >= *bi => {}
            _ => best = Some((inertia, labels)),
        }
    }
    best.expect("at least one init").1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::ari;

    /// Build micro-clusters one-point-per-row from parallel numeric + categorical arrays.
    fn micros(
        num: &[f64],
        cat: &[usize],
        n: usize,
        n_num: usize,
        cards: &[usize],
    ) -> Vec<MixedCf<f64>> {
        let n_cat = cards.len();
        (0..n)
            .map(|i| {
                let mut m = MixedCf::new(n_num, cards);
                m.push(
                    &num[i * n_num..(i + 1) * n_num],
                    &cat[i * n_cat..(i + 1) * n_cat],
                    1.0,
                );
                m
            })
            .collect()
    }

    #[test]
    fn mixed_recovers_numeric_blobs() {
        // Two numeric blobs, categorical attribute irrelevant: k-prototypes recovers the blobs.
        let mut rng = SplitMix64::new(1);
        let (mut num, mut cat, mut truth) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..200 {
            let far = i % 2;
            num.push(far as f64 * 10.0 + rng.gauss() * 0.5);
            num.push(rng.gauss() * 0.5);
            cat.push(rng.next_u64() as usize % 3); // noise category
            truth.push(far);
        }
        let m = micros(&num, &cat, 200, 2, &[3]);
        let lab = kprototypes(&m, 2, 0.5, 100, 4, 7);
        assert!(ari(&lab, &truth) > 0.95, "ARI = {}", ari(&lab, &truth));
    }

    #[test]
    fn categorical_breaks_numeric_tie() {
        // All points numerically coincident; only the categorical attribute distinguishes the two
        // groups. With γ > 0 k-prototypes must split on the category.
        let (mut num, mut cat, mut truth) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..100 {
            num.push(0.0);
            cat.push(i % 2);
            truth.push(i % 2);
        }
        let m = micros(&num, &cat, 100, 1, &[2]);
        let lab = kprototypes(&m, 2, 1.0, 100, 4, 3);
        assert!(ari(&lab, &truth) > 0.99, "ARI = {}", ari(&lab, &truth));
    }

    #[test]
    fn mode_and_merge_are_exact() {
        let mut a = MixedCf::<f64>::new(1, &[3]);
        a.push(&[1.0], &[2], 1.0);
        a.push(&[3.0], &[2], 1.0);
        a.push(&[2.0], &[0], 1.0);
        assert_eq!(a.mode(), &[2]); // category 2 appears twice
        assert!((a.numeric_mean()[0] - 2.0).abs() < 1e-12);
        let mut b = MixedCf::<f64>::new(1, &[3]);
        b.push(&[0.0], &[0], 1.0);
        b.push(&[0.0], &[0], 1.0);
        a.merge(&b);
        assert_eq!(a.weight() as i64, 5);
        assert_eq!(a.mode(), &[0]); // now category 0 appears three times
    }

    #[test]
    fn accessors_and_nearest_micro() {
        // Two one-point micros: (num 0, cat 0) and (num 10, cat 1). A query routes to the closer one.
        let m = micros(&[0.0, 10.0], &[0, 1], 2, 1, &[2]);
        assert_eq!(m[0].n_categorical(), 1);
        assert_eq!(m[0].cardinalities(), vec![2]);
        assert_eq!(nearest_micro(&m, &[0.1], &[0], 1.0), 0);
        assert_eq!(nearest_micro(&m, &[9.5], &[1], 1.0), 1);
    }

    #[test]
    fn summarize_caps_leaders() {
        // threshold 0 ⇒ distinct points would each be a leader, but the cap bounds the count.
        let (mut num, mut cat) = (Vec::new(), Vec::new());
        for i in 0..500 {
            num.push(i as f64);
            cat.push(i % 4);
        }
        let m = summarize_mixed(&num, &cat, 500, 1, &[4], 0.5, 0.0, 16);
        assert!(m.len() <= 16);
        let total: f64 = m.iter().map(|c| c.weight()).sum();
        assert_eq!(total as i64, 500); // mass conserved
    }

    /// The objective `kprototypes` minimises, rebuilt from a labelling: each cluster's prototype is
    /// the merge of its members, and the cost is the within-micro scatter plus the micro-to-prototype
    /// distance. The function returns only labels, so this is the only way to observe what it chose.
    fn objective(micros: &[MixedCf<f64>], labels: &[usize], k: usize, gamma: f64) -> f64 {
        let n_num = micros[0].n_numeric();
        let cards: Vec<usize> = micros[0].cat.iter().map(|h| h.len()).collect();
        let mut acc: Vec<MixedCf<f64>> = (0..k).map(|_| MixedCf::new(n_num, &cards)).collect();
        for (i, m) in micros.iter().enumerate() {
            acc[labels[i]].merge(m);
        }
        micros
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let p = &acc[labels[i]];
                m.numeric_ssd() + micro_dist(m, p.numeric_mean(), p.mode(), gamma)
            })
            .sum()
    }

    fn mixed_fixture() -> Vec<MixedCf<f64>> {
        // Three groups: two numeric modes crossed with a categorical split, so neither the numeric
        // nor the categorical part alone recovers the partition.
        let mut rng = SplitMix64::new(19);
        let mut num = Vec::new();
        let mut cat = Vec::new();
        for (c, (mx, my, a)) in [(0.0, 0.0, 0usize), (6.0, 0.5, 1), (0.5, 6.0, 0)]
            .into_iter()
            .enumerate()
        {
            let _ = c;
            for _ in 0..20 {
                num.push(mx + 0.6 * rng.gauss());
                num.push(my + 0.6 * rng.gauss());
                cat.push(a);
            }
        }
        micros(&num, &cat, 60, 2, &[2])
    }

    #[test]
    fn the_returned_labelling_is_a_lloyd_fixed_point() {
        // Every micro-cluster must already sit with its nearest prototype: that is what the
        // assignment loop converges to, and a comparison that stops updating -- or a prototype
        // rebuild that skips a cluster -- leaves micro-clusters stranded beside a nearer one.
        let ms = mixed_fixture();
        let (k, gamma) = (3usize, 1.0);
        let labels = kprototypes(&ms, k, gamma, 100, 4, 11);
        assert_eq!(labels.len(), ms.len());

        let n_num = ms[0].n_numeric();
        let cards: Vec<usize> = ms[0].cat.iter().map(|h| h.len()).collect();
        let mut acc: Vec<MixedCf<f64>> = (0..k).map(|_| MixedCf::new(n_num, &cards)).collect();
        for (i, m) in ms.iter().enumerate() {
            acc[labels[i]].merge(m);
        }
        assert!(
            acc.iter().filter(|a| a.weight() > 0.0).count() >= 2,
            "the fixture collapsed to one cluster, so nothing is tested"
        );
        for (i, m) in ms.iter().enumerate() {
            let own = micro_dist(
                m,
                acc[labels[i]].numeric_mean(),
                acc[labels[i]].mode(),
                gamma,
            );
            for (c, a) in acc.iter().enumerate() {
                if a.weight() <= 0.0 {
                    continue;
                }
                let d = micro_dist(m, a.numeric_mean(), a.mode(), gamma);
                assert!(
                    own <= d + 1e-9,
                    "micro {i} sits in {} at {own} but {c} is at {d}",
                    labels[i]
                );
            }
        }
    }

    #[test]
    fn more_restarts_never_return_a_worse_partition() {
        // Restarts share one RNG stream, so `n_init = m` runs exactly the first `m` inits of
        // `n_init = m + 1` and must keep the best of them. A broken objective, or a restart
        // comparison that keeps the *later* candidate, shows up as a cost that goes back up.
        let ms = mixed_fixture();
        let (k, gamma) = (3usize, 1.0);
        let mut prev = f64::INFINITY;
        let mut distinct = 0;
        for n_init in 1..=8 {
            let labels = kprototypes(&ms, k, gamma, 100, n_init, 5);
            let cost = objective(&ms, &labels, k, gamma);
            assert!(
                cost <= prev + 1e-9,
                "n_init = {n_init} cost {cost} is worse than {prev}"
            );
            if cost < prev - 1e-9 {
                distinct += 1;
            }
            prev = cost;
        }
        assert!(
            distinct > 0,
            "every restart found the same cost, so the selection rule is untested"
        );
    }

    #[test]
    fn kpp_init_spreads_one_prototype_per_far_group() {
        // D²-weighted sampling over the mixed distance: with the groups far apart in the numeric
        // part, seeding two prototypes in one group is vanishingly unlikely unless the D² update or
        // the sampling weight is broken.
        let mut num = Vec::new();
        let mut cat = Vec::new();
        for (gx, gy) in [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)] {
            for j in 0..8 {
                num.push(gx + j as f64 * 0.05);
                num.push(gy);
                cat.push(0usize);
            }
        }
        let ms = micros(&num, &cat, 24, 2, &[1]);
        for seed in 0..24u64 {
            let mut rng = SplitMix64::new(seed);
            let chosen = kpp_init(&ms, 3, 1.0, &mut rng);
            assert_eq!(chosen.len(), 3);
            let groups: Vec<usize> = chosen.iter().map(|&i| i / 8).collect();
            let mut seen = groups.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(
                seen.len(),
                3,
                "seed {seed} seeded twice in one group: {groups:?}"
            );
        }
    }

    /// A four-point mixed micro with distinct numeric spread and two categorical attributes of
    /// different cardinality, so no two terms of the distance can be swapped without moving it.
    fn mixed_micro() -> MixedCf<f64> {
        let mut m = MixedCf::<f64>::new(2, &[3, 2]);
        m.push(&[0.0, 0.0], &[0, 0], 1.0);
        m.push(&[2.0, 0.0], &[0, 1], 1.0);
        m.push(&[0.0, 2.0], &[1, 0], 1.0);
        m.push(&[2.0, 2.0], &[2, 0], 1.0);
        m
    }

    #[test]
    fn the_mixed_distances_match_their_closed_forms() {
        // μ = [1,1], w = 4; attribute 0 has counts [2,1,1] and attribute 1 has [3,1].
        let m = mixed_micro();
        assert_eq!(m.n_numeric(), 2);
        assert_eq!(m.n_categorical(), 2);
        assert_eq!(m.weight(), 4.0);
        assert_eq!(m.numeric_mean(), &[1.0, 1.0]);
        assert_eq!(m.mode(), &[0, 0]);
        assert_eq!(m.numeric_ssd(), 8.0);

        // micro_dist = w·‖μ − c‖² + γ·Σ_j (w − count_j(mode_j))
        //            = 4·2 + 3·((4 − 1) + (4 − 1)) = 8 + 18 = 26.
        let got = micro_dist(&m, &[0.0, 0.0], &[1, 1], 3.0);
        assert_eq!(got, 26.0, "micro_dist");

        // point_dist = ‖x − c‖² + γ·#mismatches, swept over all three mismatch counts so that a
        // constant mismatch (0, 1) and an inverted comparison each land somewhere else.
        for (cat, mismatches) in [([1usize, 1usize], 0), ([0, 1], 1), ([0, 0], 2)] {
            let got = point_dist(&[2.0, 0.0], &cat, &[0.0, 0.0], &[1, 1], 3.0);
            assert_eq!(got, 4.0 + 3.0 * mismatches as f64, "cat {cat:?}");
        }
    }

    #[test]
    fn a_tied_category_keeps_the_first_mode() {
        // `push` refreshes the mode only on a strict improvement, so the code that reached the
        // count first keeps it; `argmax`, which `merge` uses instead, must agree.
        let mut m = MixedCf::<f64>::new(1, &[3]);
        m.push(&[0.0], &[2], 1.0);
        m.push(&[0.0], &[0], 1.0);
        assert_eq!(m.mode(), &[2], "a tie moved the mode off the incumbent");
        assert_eq!(argmax(&[2.0, 2.0, 1.0]), 0);
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    }

    #[test]
    fn nearest_micro_breaks_an_exact_tie_towards_the_first() {
        // The query sits exactly halfway between two micros that share a category, so only the
        // comparison decides.
        let ms = micros(&[0.0, 2.0], &[0, 0], 2, 1, &[1]);
        let a = point_dist(&[1.0], &[0], ms[0].numeric_mean(), ms[0].mode(), 1.0);
        let b = point_dist(&[1.0], &[0], ms[1].numeric_mean(), ms[1].mode(), 1.0);
        assert_eq!(a.to_bits(), b.to_bits(), "the fixture is not an exact tie");
        assert_eq!(nearest_micro(&ms, &[1.0], &[0], 1.0), 0);
    }

    /// Independent re-derivation of [`summarize_mixed`]: the row slices are cut by chunking the
    /// flat arrays rather than by index arithmetic, and the nearest leader is chosen from a
    /// materialized distance row.
    fn reference_summarize(
        num: &[f64],
        cat: &[usize],
        n_num: usize,
        cards: &[usize],
        gamma: f64,
        threshold: f64,
        max_leaders: usize,
    ) -> Vec<MixedCf<f64>> {
        let mut leaders: Vec<MixedCf<f64>> = Vec::new();
        for (xn, xc) in num.chunks_exact(n_num).zip(cat.chunks_exact(cards.len())) {
            let row: Vec<f64> = leaders
                .iter()
                .map(|l| point_dist(xn, xc, l.numeric_mean(), l.mode(), gamma))
                .collect();
            let nearest =
                row.iter()
                    .enumerate()
                    .fold(None::<(usize, f64)>, |acc, (i, &d)| match acc {
                        Some((_, bd)) if bd <= d => acc,
                        _ => Some((i, d)),
                    });
            match nearest {
                Some((li, bd)) if bd <= threshold || leaders.len() >= max_leaders => {
                    leaders[li].push(xn, xc, 1.0)
                }
                _ => {
                    let mut l = MixedCf::new(n_num, cards);
                    l.push(xn, xc, 1.0);
                    leaders.push(l);
                }
            }
        }
        leaders
    }

    /// Rows wider than one numeric and one categorical column, so that reading row `i` with `i / w`
    /// or `(i + 1) * w` picks up a different point instead of the same one.
    fn wide_rows() -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let mut rng = SplitMix64::new(23);
        let (mut num, mut cat) = (Vec::new(), Vec::new());
        for i in 0..40 {
            let g = i % 4;
            num.push(g as f64 * 3.0 + 0.4 * rng.gauss());
            num.push(g as f64 * -2.0 + 0.4 * rng.gauss());
            num.push(0.4 * rng.gauss());
            cat.push(g % 2);
            cat.push(i % 3);
        }
        (num, cat, vec![2, 3])
    }

    #[test]
    fn summarize_mixed_matches_an_independent_leader_pass() {
        let (num, cat, cards) = wide_rows();
        for (threshold, cap) in [(1.0, 64usize), (4.0, 64), (1.0, 5)] {
            let got = summarize_mixed(&num, &cat, 40, 3, &cards, 0.7, threshold, cap);
            let want = reference_summarize(&num, &cat, 3, &cards, 0.7, threshold, cap);
            assert_eq!(
                got.len(),
                want.len(),
                "threshold {threshold}, cap {cap}: leader count"
            );
            assert!(
                got.len() > 1,
                "threshold {threshold}, cap {cap}: one leader absorbed everything"
            );
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                assert_eq!(a.weight(), b.weight(), "leader {i} weight");
                assert_eq!(a.mode(), b.mode(), "leader {i} mode");
                for (d, (x, y)) in a.numeric_mean().iter().zip(b.numeric_mean()).enumerate() {
                    assert!((x - y).abs() < 1e-12, "leader {i}[{d}]: {x} vs {y}");
                }
            }
        }
    }

    #[test]
    fn summarize_mixed_starts_a_new_leader_beyond_the_threshold() {
        // Two points 100 apart with the cap far away: the second cannot join the first. The same
        // pair under a cap of one must join it instead — the two halves of the admission rule,
        // measured separately so neither can stand in for the other.
        let num = [0.0, 0.0, 100.0, 0.0];
        let cat = [0usize, 0, 0, 0];
        let spread = summarize_mixed(&num, &cat, 2, 2, &[1, 1], 1.0, 1.0, 16);
        assert_eq!(
            spread.len(),
            2,
            "the far point joined a leader it is not near"
        );

        let capped = summarize_mixed(&num, &cat, 2, 2, &[1, 1], 1.0, 1.0, 1);
        assert_eq!(capped.len(), 1, "the cap did not force the far point in");
        assert_eq!(capped[0].weight(), 2.0);
    }

    #[test]
    fn summarize_mixed_breaks_an_exact_leader_tie_towards_the_first() {
        // The third point is the same distance from both leaders to the bit, and inside the
        // threshold, so which leader absorbs it is decided purely by the scan's comparison. Sending
        // it to the later leader silently moves mass between micro-clusters, which every downstream
        // fit then inherits.
        let num = [0.0f64, 2.0, 1.0];
        let cat = [0usize, 0, 0];
        let gamma = 1.0f64;
        let left = point_dist(&num[2..3], &cat[2..3], &num[0..1], &cat[0..1], gamma);
        let right = point_dist(&num[2..3], &cat[2..3], &num[1..2], &cat[1..2], gamma);
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "the fixture is not an exact tie ({left} vs {right}), so it cannot see the comparison"
        );

        let leaders = summarize_mixed(&num, &cat, 3, 1, &[1], gamma, left, 16);
        assert_eq!(leaders.len(), 2, "the fixture did not open two leaders");
        assert_eq!(leaders[0].weight(), 2.0, "the tie went to the later leader");
        assert_eq!(leaders[1].weight(), 1.0);
        assert_eq!(leaders[0].numeric_mean(), &[0.5]);
    }

    /// Independent re-derivation of the [`kprototypes`] Lloyd loop and restart selection, sharing
    /// only the seeding (`kpp_init`, pinned by its own test) so the RNG streams line up. The
    /// assignment materializes a distance row and folds it to an argmin, the prototypes are rebuilt
    /// from per-cluster member lists, and the objective is a separate pass.
    fn reference_kprototypes(
        ms: &[MixedCf<f64>],
        k: usize,
        gamma: f64,
        max_iter: usize,
        n_init: usize,
        seed: u64,
    ) -> Vec<usize> {
        let n = ms.len();
        let k = k.min(n).max(1);
        let n_num = ms[0].n_numeric();
        let cards: Vec<usize> = ms[0].cardinalities();
        let mut rng = SplitMix64::new(seed);
        let mut best: Option<(f64, Vec<usize>)> = None;

        for _ in 0..n_init.max(1) {
            let mut centers: Vec<MixedCf<f64>> = kpp_init(ms, k, gamma, &mut rng)
                .into_iter()
                .map(|s| ms[s].clone())
                .collect();
            let mut labels = vec![usize::MAX; n];
            for _ in 0..max_iter.max(1) {
                let mut changed = false;
                for (i, m) in ms.iter().enumerate() {
                    let row: Vec<f64> = centers
                        .iter()
                        .map(|c| micro_dist(m, c.numeric_mean(), c.mode(), gamma))
                        .collect();
                    let pick = row
                        .iter()
                        .enumerate()
                        .fold((0usize, f64::INFINITY), |(bi, bd), (c, &d)| {
                            if d < bd { (c, d) } else { (bi, bd) }
                        })
                        .0;
                    if labels[i] != pick {
                        labels[i] = pick;
                        changed = true;
                    }
                }
                for (c, center) in centers.iter_mut().enumerate() {
                    let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
                    if members.is_empty() {
                        continue;
                    }
                    let mut a = MixedCf::new(n_num, &cards);
                    for i in members {
                        a.merge(&ms[i]);
                    }
                    *center = a;
                }
                if !changed {
                    break;
                }
            }
            let inertia: f64 = ms
                .iter()
                .zip(&labels)
                .map(|(m, &c)| {
                    m.numeric_ssd()
                        + micro_dist(m, centers[c].numeric_mean(), centers[c].mode(), gamma)
                })
                .sum();
            if best.as_ref().is_none_or(|(bi, _)| inertia < *bi) {
                best = Some((inertia, labels));
            }
        }
        best.expect("at least one init").1
    }

    #[test]
    fn kprototypes_matches_an_independent_lloyd_run() {
        let ms = mixed_fixture();
        for (k, gamma, n_init, seed) in
            [(3usize, 1.0, 4usize, 11u64), (2, 0.5, 6, 5), (4, 2.0, 3, 2)]
        {
            let got = kprototypes(&ms, k, gamma, 100, n_init, seed);
            let want = reference_kprototypes(&ms, k, gamma, 100, n_init, seed);
            let mut seen = want.clone();
            seen.sort_unstable();
            seen.dedup();
            assert!(
                seen.len() > 1,
                "k {k}, seed {seed}: the reference collapsed to one cluster"
            );
            assert_eq!(got, want, "k {k}, γ {gamma}, n_init {n_init}, seed {seed}");
        }
    }

    #[test]
    fn kprototypes_breaks_an_exact_prototype_tie_towards_the_first() {
        // Three collinear micros one unit apart, sharing the one category so the categorical term
        // cancels: whenever the seeding takes the outer two, the middle micro sits at distance 1
        // from both, bit for bit, and only the comparison decides which cluster it joins.
        let ms = micros(&[0.0, 1.0, 2.0], &[0, 0, 0], 3, 1, &[1]);
        let (k, gamma) = (2usize, 1.0);
        let mut ties = 0;
        for seed in 0..16u64 {
            let mut rng = SplitMix64::new(seed);
            let row: Vec<u64> = kpp_init(&ms, k, gamma, &mut rng)
                .into_iter()
                .map(|s| micro_dist(&ms[1], ms[s].numeric_mean(), ms[s].mode(), gamma).to_bits())
                .collect();
            if row[0] == row[1] {
                ties += 1;
            }
            assert_eq!(
                kprototypes(&ms, k, gamma, 100, 1, seed),
                reference_kprototypes(&ms, k, gamma, 100, 1, seed),
                "seed {seed}"
            );
        }
        assert!(
            ties > 0,
            "no seeding put the middle micro between the prototypes; the tie is untested"
        );
    }

    #[test]
    fn kprototypes_matches_the_reference_when_a_cluster_goes_empty() {
        // Asking for more clusters than the fixture has distinct sites leaves prototypes without
        // members: once every remaining seeding candidate has `D² = 0` the draw falls back to a
        // uniform pick and duplicates a prototype, and the duplicate loses every exact tie.
        //
        // This pins the labelling across that path, not the emptied prototype's *value* -- nothing
        // public reads it. Keeping it (rather than overwriting it with the empty accumulator, which
        // would place it at the origin with mode 0) is observable only if a later pass runs *and*
        // the collapsed prototype outranks a real one for some micro; a micro sitting exactly on its
        // own prototype cannot be outranked, which is the state every fixture tried here lands in.
        let num = [
            0.0, 0.3, 0.6, 10.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, -10.0,
        ];
        let cat = [0usize; 11];
        let ms = micros(&num, &cat, 11, 1, &[1]);
        let mut emptied = 0;
        for k in 4..=8usize {
            for seed in 0..12u64 {
                let got = kprototypes(&ms, k, 1.0, 100, 1, seed);
                let mut used = got.clone();
                used.sort_unstable();
                used.dedup();
                if used.len() < k {
                    emptied += 1;
                }
                assert_eq!(
                    got,
                    reference_kprototypes(&ms, k, 1.0, 100, 1, seed),
                    "k {k}, seed {seed}"
                );
            }
        }
        assert!(
            emptied > 0,
            "every cluster kept a member; the empty-cluster branch is untested"
        );
    }
}
