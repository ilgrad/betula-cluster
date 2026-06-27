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
}
