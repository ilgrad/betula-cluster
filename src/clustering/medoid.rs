//! Medoid-silhouette clustering with an automatically chosen `k` (DynMSC) over CF microclusters.
//!
//! Lenssen & Schubert, *Medoid silhouette clustering with automatic cluster number selection*
//! (Inf. Syst. 120, 2024). The silhouette is normally a *score*: cluster at several `k`, then rank.
//! Their observation is that the **medoid** silhouette — `s = 1 − d(x, m_own) / d(x, m_nearest
//! other)`, distances taken to `k` medoids rather than to all `N` points — is cheap enough to be the
//! objective a swap-based optimiser maximises directly, and that sweeping `k` downward while reusing
//! the medoid set turns "choose `k`" into one run instead of `k_max` independent ones.
//!
//! The objective here is exactly [`crate::validity::medoid_silhouette`], not a proxy for it: a leaf
//! is at mean squared distance `‖μ_i − μ_j‖² + S_i/n_i` from the medoid *point* `μ_j`, which is
//! exact for the points the leaf stands for and keeps a leaf off zero even when it is itself the
//! medoid. That the crate's published metric and this head's objective are the same function is a
//! property worth having: the head cannot win on a number the metric would score differently.
//!
//! Cost is `O(k_max · iter · m² · d)` on the `m ≪ N` microclusters — a swap pass has to price every
//! candidate against every leaf, which is the same `O(m²)` shape as the density head's MST and is
//! bounded by the leaf budget rather than by `N`. The `m × m` distances are recomputed per candidate
//! rather than cached, keeping memory at `O(m·d)`.

use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Result of a [`dyn_msc`] run.
pub struct MedoidClustering {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
    /// Number of clusters, chosen by the sweep rather than given.
    pub k: usize,
    /// Medoid silhouette at the chosen `k`; `1.0` is the ceiling.
    pub score: f64,
    /// The medoid of each cluster, as an index into `features`.
    pub medoids: Vec<usize>,
}

/// The two smallest of three, in order. `f64::INFINITY` stands in for "no such medoid".
fn two_smallest(a: f64, b: f64, c: f64) -> (f64, f64) {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    if c <= lo {
        (c, lo)
    } else if c <= hi {
        (lo, c)
    } else {
        (lo, hi)
    }
}

/// A leaf's contribution to the loss, `d_own / d_other`. The silhouette is `1 −` this, so the loss
/// is what the swap search minimises. A zero `d_other` is the coincident-cluster degeneracy that
/// [`crate::validity::medoid_silhouette`] scores at zero, i.e. loss one.
fn ratio(own: f64, other: f64) -> f64 {
    if other > 0.0 { own / other } else { 1.0 }
}

/// Everything the search needs about the leaves, in the working precision of the topology math.
struct Leaves {
    mu: Vec<Vec<f64>>,
    w: Vec<f64>,
    /// `S_i / n_i`, the mean squared distance from the leaf's points to its own mean.
    spread: Vec<f64>,
    total: f64,
}

impl Leaves {
    fn of<R: Real, C: ClusterFeature<R>>(features: &[C]) -> Self {
        let mu = features
            .iter()
            .map(|f| f.mean().iter().map(|v| v.to_f64().unwrap_or(0.0)).collect())
            .collect();
        let w: Vec<f64> = features
            .iter()
            .map(|f| f.weight().to_f64().unwrap_or(0.0))
            .collect();
        // A leaf that carries no mass has no scatter to report, and `S/n` on it is `0/0`. Guarding
        // here rather than at the four places that read `spread` is the difference between one
        // invariant and four defensive branches — a NaN loose in this module reaches
        // `partial_cmp(..).unwrap()` in the seeding and takes the process with it.
        let spread = features
            .iter()
            .map(|f| {
                if f.weight() > R::zero() {
                    (f.ssd() / f.weight()).to_f64().unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .collect();
        let total = w.iter().sum();
        Self {
            mu,
            w,
            spread,
            total,
        }
    }

    fn len(&self) -> usize {
        self.mu.len()
    }

    /// Mean squared distance from leaf `i`'s points to the point `μ_j`. Asymmetric by construction:
    /// only `i` contributes its own scatter, because only `i`'s points are being measured.
    fn dist(&self, i: usize, j: usize) -> f64 {
        sq_euclidean(&self.mu[i], &self.mu[j]) + self.spread[i]
    }
}

/// The three nearest medoid *slots* to each leaf, ascending, as `(distance, slot)`.
///
/// Three, not two, although the objective only reads two: evaluating the removal of a leaf's nearest
/// or second-nearest medoid needs the one behind it, and any medoid outside the three is farther
/// than all of them and can never enter the new pair.
fn three_nearest(lv: &Leaves, medoids: &[usize]) -> Vec<[(f64, usize); 3]> {
    let far = (f64::INFINITY, usize::MAX);
    (0..lv.len())
        .map(|i| {
            let mut best = [far; 3];
            for (slot, &m) in medoids.iter().enumerate() {
                let d = lv.dist(i, m);
                if d < best[0].0 {
                    best[2] = best[1];
                    best[1] = best[0];
                    best[0] = (d, slot);
                } else if d < best[1].0 {
                    best[2] = best[1];
                    best[1] = (d, slot);
                } else if d < best[2].0 {
                    best[2] = (d, slot);
                }
            }
            best
        })
        .collect()
}

fn loss_of(lv: &Leaves, near: &[[(f64, usize); 3]]) -> f64 {
    (0..lv.len())
        .map(|i| lv.w[i] * ratio(near[i][0].0, near[i][1].0))
        .sum()
}

/// Greedy weighted k-means++ over the leaves, returning medoid **indices** rather than centroids.
///
/// The exact CF potential of a leaf at squared distance `d2` from its nearest chosen medoid:
/// `S_i + n_i·D²_i`, written in the mean-scatter form the [`Leaves`] view carries. Named because the
/// two terms are two different claims — the distance to the seeding set, and the scatter a leaf keeps
/// even when it sits on one — and a sampling weight that quietly loses either is not detectable in
/// the clustering the search converges to.
fn potential(w: f64, d2: f64, spread: f64) -> f64 {
    w * d2 + w * spread
}

/// Greedy weighted k-means++ over the leaves, returning medoid **indices** rather than centroids.
///
/// The sampling weight is the exact CF [`potential`], the same one the k-means head seeds on: a leaf
/// sitting on a chosen medoid still carries its own scatter and stays a candidate.
fn seed_medoids(lv: &Leaves, k: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let m = lv.len();
    let mut medoids = Vec::with_capacity(k);
    let mut d2: Vec<f64> = vec![f64::INFINITY; m];
    let mut probs: Vec<f64> = lv.w.clone();
    let first = crate::clustering::kmeans::weighted_pick(&probs, rng);
    medoids.push(first);
    while medoids.len() < k {
        let last = *medoids.last().unwrap();
        for i in 0..m {
            let d = sq_euclidean(&lv.mu[i], &lv.mu[last]);
            if d < d2[i] {
                d2[i] = d;
            }
            probs[i] = potential(lv.w[i], d2[i], lv.spread[i]);
        }
        let pick = crate::clustering::kmeans::weighted_pick(&probs, rng);
        if medoids.contains(&pick) {
            // Sampling landed on a medoid already held; take the leaf with the largest potential
            // instead, so a degenerate distribution still terminates.
            match (0..m)
                .filter(|i| !medoids.contains(i))
                .max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap())
            {
                Some(alt) => medoids.push(alt),
                None => break,
            }
        } else {
            medoids.push(pick);
        }
    }
    medoids
}

/// One FastMSC swap pass: the `(slot, candidate)` exchange that lowers the loss most, or `None`.
///
/// Pricing every `(slot, candidate)` pair separately would be `O(k·m)` per candidate. It is `O(m)`
/// instead because removing a medoid that is not among a leaf's two nearest cannot change that
/// leaf's pair — so one shared total covers every such slot, and only two corrections per leaf are
/// left to accumulate.
fn best_swap(
    lv: &Leaves,
    medoids: &[usize],
    near: &[[(f64, usize); 3]],
    current: f64,
) -> Option<(usize, usize, f64)> {
    let m = lv.len();
    let k = medoids.len();
    let mut best: Option<(usize, usize, f64)> = None;
    let mut dx = vec![0.0f64; m];
    for x in 0..m {
        if medoids.contains(&x) {
            continue;
        }
        for (i, d) in dx.iter_mut().enumerate() {
            *d = lv.dist(i, x);
        }
        let mut shared = 0.0;
        let mut correction = vec![0.0f64; k];
        for i in 0..m {
            let [(d1, s1), (d2, s2), (d3, _)] = near[i];
            let (a, b) = two_smallest(d1, d2, dx[i]);
            let generic = lv.w[i] * ratio(a, b);
            shared += generic;
            if s1 != usize::MAX {
                let (a, b) = two_smallest(d2, d3, dx[i]);
                correction[s1] += lv.w[i] * ratio(a, b) - generic;
            }
            if s2 != usize::MAX {
                let (a, b) = two_smallest(d1, d3, dx[i]);
                correction[s2] += lv.w[i] * ratio(a, b) - generic;
            }
        }
        for (slot, corr) in correction.iter().enumerate() {
            let loss = shared + corr;
            if loss < current - 1e-12 && best.is_none_or(|(_, _, b)| loss < b) {
                best = Some((slot, x, loss));
            }
        }
    }
    best
}

/// The medoid whose removal costs the least, with the loss that remains after removing it.
fn cheapest_removal(lv: &Leaves, k: usize, near: &[[(f64, usize); 3]]) -> (usize, f64) {
    let mut loss = vec![0.0f64; k];
    for (i, &[(d1, s1), (d2, s2), (d3, _)]) in near.iter().enumerate() {
        let keep = lv.w[i] * ratio(d1, d2);
        for l in loss.iter_mut() {
            *l += keep;
        }
        if s1 != usize::MAX {
            loss[s1] += lv.w[i] * ratio(d2, d3) - keep;
        }
        if s2 != usize::MAX {
            loss[s2] += lv.w[i] * ratio(d1, d3) - keep;
        }
    }
    let slot = (0..k)
        .min_by(|&a, &b| loss[a].partial_cmp(&loss[b]).unwrap())
        .unwrap_or(0);
    (slot, loss[slot])
}

fn labels_of(near: &[[(f64, usize); 3]]) -> Vec<usize> {
    near.iter()
        .map(|n| if n[0].1 == usize::MAX { 0 } else { n[0].1 })
        .collect()
}

/// Cluster `features` by maximising the medoid silhouette, choosing `k` from `2..=k_max`.
///
/// DynMSC: optimise at `k_max`, then repeatedly drop the medoid whose removal costs least and
/// re-optimise, keeping the `k` that scored best. Sweeping downward with the medoid set carried over
/// is what makes this one run rather than `k_max − 1` independent ones, and it is why the `k` it
/// returns is comparable across the sweep — every level starts from the level above's solution.
///
/// `max_iter` bounds the swap passes at each `k`; `0` means the default. A `k_max` of `0` or `1`, or
/// fewer than two features, returns a single cluster with score `0.0`, the value the metric gives
/// where it is undefined.
pub fn dyn_msc<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> MedoidClustering {
    let lv = Leaves::of(features);
    let m = lv.len();
    let k_max = k_max.min(m);
    if m == 0 {
        return MedoidClustering {
            labels: Vec::new(),
            k: 0,
            score: 0.0,
            medoids: Vec::new(),
        };
    }
    if k_max < 2 {
        return MedoidClustering {
            labels: vec![0; m],
            k: 1,
            score: 0.0,
            medoids: vec![0],
        };
    }
    let max_iter = if max_iter == 0 { 100 } else { max_iter };

    let mut rng = SplitMix64::new(seed);
    let mut medoids = seed_medoids(&lv, k_max, &mut rng);
    let mut best: Option<MedoidClustering> = None;

    while medoids.len() >= 2 {
        let mut near = three_nearest(&lv, &medoids);
        let mut loss = loss_of(&lv, &near);
        for _ in 0..max_iter {
            match best_swap(&lv, &medoids, &near, loss) {
                Some((slot, x, new_loss)) => {
                    medoids[slot] = x;
                    near = three_nearest(&lv, &medoids);
                    loss = new_loss;
                }
                None => break,
            }
        }
        let score = if lv.total > 0.0 {
            1.0 - loss / lv.total
        } else {
            0.0
        };
        if best.as_ref().is_none_or(|b| score > b.score) {
            best = Some(MedoidClustering {
                labels: labels_of(&near),
                k: medoids.len(),
                score,
                medoids: medoids.clone(),
            });
        }
        if medoids.len() == 2 {
            break;
        }
        let (drop, _) = cheapest_removal(&lv, medoids.len(), &near);
        medoids.remove(drop);
    }

    best.unwrap_or(MedoidClustering {
        labels: vec![0; m],
        k: 1,
        score: 0.0,
        medoids: vec![0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::testutil::{ari, blobs, grid_micros};
    use crate::feature::Spherical;

    fn fixture(seed: u64, spread: f64) -> (Vec<Spherical<f64>>, Vec<usize>, Vec<usize>) {
        let mut rng = SplitMix64::new(seed);
        let centres = [[0.0, 0.0], [12.0, 0.0], [6.0, 11.0], [18.0, 11.0]];
        let (pts, truth) = blobs(&mut rng, 150, &centres, spread);
        let (micros, assign) = grid_micros(&pts, 0.6);
        (micros, assign, truth)
    }

    /// Leaves at fixed positions with unit weight and no scatter, for the pieces whose behaviour is
    /// a tie-breaking rule rather than a clustering.
    fn placed(mu: &[[f64; 2]]) -> Leaves {
        Leaves {
            mu: mu.iter().map(|p| p.to_vec()).collect(),
            w: vec![1.0; mu.len()],
            spread: vec![0.0; mu.len()],
            total: mu.len() as f64,
        }
    }

    #[test]
    fn a_vanishing_second_distance_scores_the_worst_ratio_rather_than_dividing_by_zero() {
        // Two medoids on the same point: the silhouette is undefined and the crate's published
        // metric scores that degeneracy at zero, so the loss it is one minus must be one.
        assert_eq!(ratio(0.5, 0.0), 1.0);
        assert_eq!(ratio(0.0, 0.0), 1.0);
        assert_eq!(ratio(1.0, 4.0), 0.25);
    }

    #[test]
    fn the_nearest_three_break_ties_towards_the_earlier_medoid() {
        // Four medoids at exactly the same distance and one further out. Which of the tied three
        // land in which slot is not cosmetic: the swap search prices a leaf by slots one and two and
        // corrects by slot three, so a rule that lets a later tie displace an earlier one silently
        // reshuffles the corrections.
        let lv = placed(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, 2.0],
        ]);
        let near = three_nearest(&lv, &[1, 2, 3, 4, 5]);
        assert_eq!(
            near[0],
            [(1.0, 0), (1.0, 1), (1.0, 2)],
            "ties must fill the slots in medoid order and the fourth tie must not displace them"
        );
    }

    #[test]
    fn the_seeding_weight_is_the_scatter_plus_the_distance_and_not_either_alone() {
        // `S_i + n_i·D²_i`. A leaf sitting on a chosen medoid (`d2 = 0`) still carries its scatter
        // and stays a candidate; a leaf far away with no scatter is a candidate on distance alone.
        assert_eq!(potential(2.0, 3.0, 5.0), 16.0);
        assert_eq!(potential(3.0, 0.0, 4.0), 12.0);
        assert_eq!(potential(3.0, 4.0, 0.0), 12.0);
        assert_eq!(potential(0.0, 4.0, 5.0), 0.0);
    }

    #[test]
    fn the_seeding_returns_exactly_k_distinct_leaves_even_where_the_potential_is_degenerate() {
        // Every leaf identical: the sampling repeatedly lands on a medoid already held, so the
        // fallback is the only thing producing progress and the loop is the only thing stopping it.
        let lv = placed(&[[0.0, 0.0]; 6]);
        let mut rng = SplitMix64::new(5);
        let seeded = seed_medoids(&lv, 4, &mut rng);
        assert_eq!(seeded.len(), 4, "asked for four medoids, got {seeded:?}");
        let mut sorted = seeded.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 4, "a medoid was chosen twice: {seeded:?}");
    }

    #[test]
    fn the_swap_search_returns_the_best_improving_exchange_and_the_same_one_a_full_search_finds() {
        let (micros, _, _) = fixture(4, 0.7);
        let lv = Leaves::of(&micros);
        let mut rng = SplitMix64::new(3);
        let mut medoids = seed_medoids(&lv, 5, &mut rng);
        let mut compared = 0;
        for _ in 0..40 {
            let near = three_nearest(&lv, &medoids);
            let loss = loss_of(&lv, &near);
            // Candidate-outer, slot-inner, matching `best_swap`'s own order, so that ties resolve
            // the same way and the comparison is of the choice and not only of its price.
            let mut brute: Option<(usize, usize, f64)> = None;
            for x in 0..lv.len() {
                if medoids.contains(&x) {
                    continue;
                }
                for slot in 0..medoids.len() {
                    let mut trial = medoids.clone();
                    trial[slot] = x;
                    let l = loss_of(&lv, &three_nearest(&lv, &trial));
                    if l < loss - 1e-12 && brute.is_none_or(|(_, _, b)| l < b) {
                        brute = Some((slot, x, l));
                    }
                }
            }
            match (best_swap(&lv, &medoids, &near, loss), brute) {
                (None, None) => break,
                (Some((s, x, l)), Some((bs, bx, bl))) => {
                    assert_eq!(
                        (s, x),
                        (bs, bx),
                        "chose ({s}, {x}), full search says ({bs}, {bx})"
                    );
                    assert!(
                        (l - bl).abs() < 1e-9,
                        "priced it {l}, full search says {bl}"
                    );
                    medoids[s] = x;
                    compared += 1;
                }
                (a, b) => panic!("the incremental and full searches disagree: {a:?} vs {b:?}"),
            }
        }
        assert!(
            compared >= 2,
            "the fixture converged too fast to compare anything"
        );
    }

    #[test]
    fn a_swap_that_only_ties_the_current_loss_is_not_an_improvement() {
        // Every leaf has an exact duplicate, so at any medoid set there is a swap to the twin that
        // leaves the loss untouched. An improvement test that admits equality would take it, and
        // the search would cycle between twins instead of stopping.
        let lv = placed(&[
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [0.0, 10.0],
        ]);
        let medoids = vec![0, 2, 4];
        let near = three_nearest(&lv, &medoids);
        let loss = loss_of(&lv, &near);
        assert!(
            best_swap(&lv, &medoids, &near, loss).is_none(),
            "a zero-delta swap to a duplicate leaf was reported as an improvement"
        );
    }

    #[test]
    fn a_k_max_of_two_is_a_sweep_and_not_the_single_cluster_shortcut() {
        let (micros, _, _) = fixture(6, 0.7);
        let res = dyn_msc(&micros, 2, 100, 6);
        assert_eq!(
            res.k, 2,
            "k_max = 2 must still be optimised, not short-circuited"
        );
        assert!(
            res.score > 0.0,
            "score {} is the undefined-metric value",
            res.score
        );
    }

    #[test]
    fn the_iteration_cap_is_taken_literally_and_only_zero_means_the_default() {
        let (micros, _, _) = fixture(8, 0.9);
        let one = dyn_msc(&micros, 6, 1, 8);
        let many = dyn_msc(&micros, 6, 100, 8);
        let zero = dyn_msc(&micros, 6, 0, 8);
        assert!(
            one.score < many.score,
            "one swap pass scored {} against {} for a hundred — the cap is being ignored",
            one.score,
            many.score
        );
        assert_eq!(
            zero.score, many.score,
            "max_iter = 0 must mean the default, which is a hundred"
        );
    }

    #[test]
    fn a_massless_summary_scores_zero_rather_than_dividing_by_its_own_total_weight() {
        let empty: Vec<Spherical<f64>> = (0..4).map(|_| Spherical::new(2)).collect();
        let res = dyn_msc(&empty, 3, 10, 1);
        assert_eq!(
            res.score, 0.0,
            "score {} is not the undefined value",
            res.score
        );
        assert!(res.score.is_finite());
    }

    #[test]
    fn a_tie_across_the_sweep_keeps_the_larger_k_it_was_found_at_first() {
        // Identical leaves make every k score the same. The sweep runs downward from `k_max`, so
        // keeping the first of the tie means keeping `k_max`; admitting equality would walk the
        // answer all the way down to two without a single score to justify it.
        let lv: Vec<Spherical<f64>> = (0..6)
            .map(|_| {
                let mut f = Spherical::new(2);
                f.push(&[1.0, 1.0], 1.0);
                f
            })
            .collect();
        let res = dyn_msc(&lv, 4, 50, 2);
        assert_eq!(res.k, 4, "a tied sweep must keep the k it reached first");
    }

    #[test]
    fn the_head_optimises_the_metric_the_crate_publishes() {
        // The score a run reports has to be the score `validity::medoid_silhouette` would give the
        // labelling it returns. If the two ever drift apart the head is winning on its own scale.
        for seed in [1u64, 5, 9] {
            let (micros, _, _) = fixture(seed, 0.8);
            let got = dyn_msc(&micros, 8, 100, seed);
            let scored = crate::validity::medoid_silhouette(&micros, &got.labels, got.k);
            assert!(
                (got.score - scored).abs() < 1e-9,
                "seed {seed}: head {} vs metric {scored}",
                got.score
            );
        }
    }

    #[test]
    fn the_incremental_swap_price_matches_a_full_recomputation() {
        // `best_swap` prices `k` exchanges per candidate in one `O(m)` pass by sharing the term that
        // does not depend on which medoid leaves. The shortcut is only worth having if it is exact.
        let (micros, _, _) = fixture(3, 0.9);
        let lv = Leaves::of(&micros);
        let mut rng = SplitMix64::new(3);
        let medoids = seed_medoids(&lv, 5, &mut rng);
        let near = three_nearest(&lv, &medoids);
        let current = loss_of(&lv, &near);

        let (slot, x, quoted) = best_swap(&lv, &medoids, &near, current)
            .expect("a fresh k-means++ seeding leaves an improving swap");
        let mut swapped = medoids.clone();
        swapped[slot] = x;
        let actual = loss_of(&lv, &three_nearest(&lv, &swapped));
        assert!(
            (quoted - actual).abs() < 1e-9,
            "quoted {quoted}, recomputed {actual}"
        );
        assert!(quoted < current, "a swap was taken that does not improve");
    }

    #[test]
    fn the_removal_price_matches_a_full_recomputation() {
        let (micros, _, _) = fixture(3, 0.9);
        let lv = Leaves::of(&micros);
        let mut rng = SplitMix64::new(11);
        let medoids = seed_medoids(&lv, 6, &mut rng);
        let near = three_nearest(&lv, &medoids);

        let (drop, quoted) = cheapest_removal(&lv, medoids.len(), &near);
        let mut kept = medoids.clone();
        kept.remove(drop);
        let actual = loss_of(&lv, &three_nearest(&lv, &kept));
        assert!(
            (quoted - actual).abs() < 1e-9,
            "quoted {quoted}, recomputed {actual}"
        );
        // And it must be the cheapest, not merely a priced one.
        for slot in 0..medoids.len() {
            let mut alt = medoids.clone();
            alt.remove(slot);
            let other = loss_of(&lv, &three_nearest(&lv, &alt));
            assert!(actual <= other + 1e-9, "slot {slot} was cheaper to remove");
        }
    }

    #[test]
    fn the_sweep_finds_the_number_of_blobs_without_being_told_it() {
        for seed in [1u64, 5, 9] {
            let (micros, assign, truth) = fixture(seed, 0.8);
            let got = dyn_msc(&micros, 10, 100, seed);
            assert_eq!(got.k, 4, "seed {seed} chose k = {}", got.k);
            let labels: Vec<usize> = assign.iter().map(|&i| got.labels[i]).collect();
            assert!(
                ari(&labels, &truth) > 0.99,
                "seed {seed}: ARI = {}",
                ari(&labels, &truth)
            );
        }
    }

    #[test]
    fn every_medoid_is_a_leaf_of_its_own_cluster() {
        let (micros, _, _) = fixture(7, 0.8);
        let got = dyn_msc(&micros, 6, 100, 7);
        assert_eq!(got.medoids.len(), got.k);
        for (slot, &m) in got.medoids.iter().enumerate() {
            assert_eq!(
                got.labels[m], slot,
                "medoid {m} sits outside its own cluster"
            );
        }
    }

    #[test]
    fn the_degenerate_inputs_answer_rather_than_panic() {
        let empty: Vec<Spherical<f64>> = Vec::new();
        let got = dyn_msc(&empty, 5, 100, 0);
        assert_eq!(got.k, 0);
        assert!(got.labels.is_empty());

        let (micros, _, _) = fixture(2, 0.8);
        for k_max in [0usize, 1] {
            let got = dyn_msc(&micros, k_max, 100, 0);
            assert_eq!(got.k, 1);
            assert_eq!(got.score, 0.0);
            assert!(got.labels.iter().all(|&l| l == 0));
        }
        // More clusters asked for than leaves available: `k_max` is clamped, not honoured.
        let two = &micros[..2];
        let got = dyn_msc(two, 50, 100, 0);
        assert!(got.k <= 2, "k = {} on two leaves", got.k);
    }

    /// The measurement behind the DynMSC section of `bench/RESULTS.md`. Run it with
    /// `cargo test --lib measure_dyn_msc_against_the_ch_selector -- --ignored --nocapture`.
    ///
    /// The crate already chooses `k` automatically, by Calinski–Harabasz over the Ward cuts, so the
    /// question is not "does a selector work" but "does this one pick better than the one shipped".
    /// Both are given the same leaves and the same `2..=12` range.
    #[test]
    #[ignore]
    fn measure_dyn_msc_against_the_ch_selector() {
        println!("shape  k_true spread | DynMSC k  ari | Ward+CH k  ari");
        for &(shape, stretch) in &[("round", 1.0f64), ("elongated", 4.0)] {
            for k_true in [3usize, 4, 6, 8] {
                for &spread in &[1.6f64, 2.4, 3.2] {
                    let mut dyn_k = Vec::new();
                    let mut dyn_a = Vec::new();
                    let mut ch_k = Vec::new();
                    let mut ch_a = Vec::new();
                    for seed in [1u64, 5, 9, 13, 21] {
                        let mut rng = SplitMix64::new(seed);
                        // Centres on a circle wide enough that `k_true` is unambiguous at spread 0.6 and
                        // genuinely contested at 1.6.
                        let centres: Vec<[f64; 2]> = (0..k_true)
                            .map(|c| {
                                let t = std::f64::consts::TAU * c as f64 / k_true as f64;
                                [9.0 * t.cos(), 9.0 * t.sin()]
                            })
                            .collect();
                        let (mut pts, truth) = blobs(&mut rng, 150, &centres, spread);
                        // Stretching one axis leaves the truth untouched and breaks the sphericity that
                        // the variance-ratio criterion is built on.
                        for p in pts.iter_mut() {
                            p[0] *= stretch;
                        }
                        let (micros, assign) = grid_micros(&pts, 0.6);

                        let d = dyn_msc(&micros, 12, 100, seed);
                        let dl: Vec<usize> = assign.iter().map(|&i| d.labels[i]).collect();
                        dyn_k.push(d.k as f64);
                        dyn_a.push(ari(&dl, &truth));

                        let w = crate::clustering::ward_hac_auto(&micros, 2, 12);
                        let wk = w
                            .labels
                            .iter()
                            .collect::<std::collections::HashSet<_>>()
                            .len();
                        let wl: Vec<usize> = assign.iter().map(|&i| w.labels[i]).collect();
                        ch_k.push(wk as f64);
                        ch_a.push(ari(&wl, &truth));
                    }
                    let med = |v: &mut Vec<f64>| {
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        v[v.len() / 2]
                    };
                    println!(
                        "{shape:9} {k_true:6} {spread:6} | {:6.0} {:6.3} | {:8.0} {:6.3}",
                        med(&mut dyn_k),
                        med(&mut dyn_a),
                        med(&mut ch_k),
                        med(&mut ch_a)
                    );
                }
            }
        }
    }

    /// Does the swap search actually cost `O(m²)`? Run with
    /// `cargo test --release --lib measure_dyn_msc_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn measure_dyn_msc_cost() {
        println!("leaves   seconds   ratio vs previous");
        let mut prev: Option<(f64, f64)> = None;
        for &cell in &[1.2f64, 0.85, 0.6, 0.42] {
            let mut rng = SplitMix64::new(4);
            let centres = [[0.0, 0.0], [12.0, 0.0], [6.0, 11.0], [18.0, 11.0]];
            let (pts, _) = blobs(&mut rng, 1200, &centres, 1.4);
            let (micros, _) = grid_micros(&pts, cell);
            let m = micros.len() as f64;
            let t = std::time::Instant::now();
            let got = dyn_msc(&micros, 8, 100, 4);
            let el = t.elapsed().as_secs_f64();
            let note = match prev {
                Some((pm, pt)) => format!("m x{:.2}, time x{:.2}", m / pm, el / pt),
                None => String::from("-"),
            };
            prev = Some((m, el));
            println!("{:6.0}  {el:8.2}   {note}   (k = {})", m, got.k);
        }
    }
}
