//! X-means: k-means that decides its own `k` by testing every centre for a split.
//!
//! Pelleg & Moore, *X-means: Extending K-means with Efficient Estimation of the Number of Clusters*
//! (ICML 2000). The algorithm alternates two steps: **Improve-Params** runs k-means to convergence,
//! and **Improve-Structure** asks of each centre separately whether the leaves it owns are better
//! described by two centres than by one, deciding with a BIC computed on that subset alone. Accepted
//! splits raise `k`; a round that accepts none is the stopping rule.
//!
//! This is a different algorithm from [`super::kmeans::kmeans_auto`], not a reskin of it. The sweep
//! there fits a full k-means at every `k` and takes the best score, which is `O(k_max²)` Lloyd passes
//! over the whole leaf set; the split test is `O(k)` two-centre problems on subsets that shrink as
//! the recursion deepens. The practical consequence is reach rather than only speed: a sweep has to
//! be bounded by a `k_max` chosen in advance, while a splitter walks up to whatever `k` the data
//! supports and stops on its own.
//!
//! Everything the score reads is exact on the summary. The Pelleg-Moore log-likelihood is a function
//! of the per-cluster point counts, `d`, `k` and the within-cluster sum of squares only, and
//! `Σ_{x∈cl} w‖x − c‖² = Σ_{l⊆cl} (S_l + w_l‖μ_l − c‖²)` holds exactly for a whole-leaf partition. So
//! the *value* is the value the same partition would score on the raw points; what the summary
//! restricts is the *search*, to partitions that do not cut a leaf — the restriction every head here
//! already accepts. Both halves of that identity have to be carried: the `S_l` term is constant in
//! `k`, but it is inside a logarithm, and a score built on the between-leaf part alone sends `σ̂²` to
//! zero as `k` reaches the leaf count and then prefers one cluster per leaf on any input.
//!
//! Cost is dominated by the split tests, each a 2-means over one cluster's leaves. Subsets are
//! materialised by cloning their features, which is `O(m·d)` per round in total and lets the split
//! reuse the shipped k-means rather than growing a second implementation of Lloyd.

use crate::clustering::kmeans::{KMeans, kmeans, lloyd_from};
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Restarts for the two-centre problem inside a split test. Matching the sweep's `n_init` keeps the
/// comparison between the two heads about the search strategy rather than about seeding effort.
const SPLIT_N_INIT: usize = 4;

/// Pelleg-Moore X-means BIC, **maximised** — a classification log-likelihood minus half the penalty,
/// which is the opposite sign to [`super::gmm::bic`] and its callers.
///
/// `nk` is the point weight of each cluster and `sse` the within-cluster sum of squares **of the
/// points**, `Σ_l (S_l + w_l‖μ_l − c‖²)`. The within-leaf scatter `Σ_l S_l` is constant in `k` but it
/// does not cancel: it sits inside `ln σ̂²`, and dropping it drives the variance estimate to zero as
/// `k` approaches the leaf count, at which point the score prefers one cluster per leaf however many
/// clusters the data has.
pub(crate) fn pelleg_moore_bic<R: Real>(nk: &[R], sse: R, dim: usize) -> R {
    let k = nk.len();
    let nr = nk.iter().fold(R::zero(), |a, &b| a + b);
    let dr = R::from_usize(dim).unwrap();
    let half = R::from_f64(0.5).unwrap();
    let two_pi = R::from_f64(std::f64::consts::TAU).unwrap();
    let tiny = R::from_f64(1e-12).unwrap();

    let var = (sse / (nr - R::from_usize(k).unwrap()).max(R::one()) / dr).max(tiny);
    let log_2pi_var = (two_pi * var).ln();
    let mut loglik = R::zero();
    for &n_k in nk {
        if n_k > R::zero() {
            loglik = loglik + n_k * n_k.ln()
                - n_k * nr.ln()
                - half * n_k * dr * log_2pi_var
                - half * (n_k - R::one()) * dr;
        }
    }
    let params = R::from_usize(k * (dim + 1)).unwrap();
    loglik - half * params * nr.ln()
}

/// Per-cluster point weight and the within-cluster sum of squares of the underlying **points**,
/// `Σ_l (S_l + w_l‖μ_l − c‖²)` — the CF identity, exact for any whole-leaf partition.
fn cluster_stats<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    labels: &[usize],
    centers: &[Vec<R>],
) -> (Vec<R>, R) {
    let mut nk = vec![R::zero(); centers.len()];
    let mut sse = R::zero();
    for (f, &c) in features.iter().zip(labels) {
        nk[c] = nk[c] + f.weight();
        sse = sse + f.ssd() + f.weight() * sq_euclidean(f.mean(), &centers[c]);
    }
    (nk, sse)
}

/// The one-centre model of a leaf set: its weight-mean, which is the centre that minimises the SSE.
fn weighted_mean<R: Real, C: ClusterFeature<R>>(features: &[C], dim: usize) -> Vec<R> {
    let mut acc = vec![R::zero(); dim];
    let mut total = R::zero();
    for f in features {
        total = total + f.weight();
        for (a, &x) in acc.iter_mut().zip(f.mean()) {
            *a = *a + f.weight() * x;
        }
    }
    if total > R::zero() {
        for a in &mut acc {
            *a = *a / total;
        }
    }
    acc
}

/// BIC of a whole model over the full leaf set.
fn model_bic<R: Real, C: ClusterFeature<R>>(features: &[C], km: &KMeans<R>, dim: usize) -> R {
    let (nk, sse) = cluster_stats(features, &km.labels, &km.centers);
    pelleg_moore_bic(&nk, sse, dim)
}

/// Cluster `features`, choosing `k` by recursive splitting. `k_max` is an **upper bound**, not a
/// target: the result is the best-BIC model among those visited, and a run that stops early stopped
/// because no centre wanted to split.
///
/// `k_min` is where the recursion starts, and it is not cosmetic. A greedy splitter has no way back
/// from a refused split, so at `k_min = 1` the whole answer rides on one BIC comparison over the
/// entire leaf set — the comparison the accept threshold makes hardest, since a cloud of many
/// well-separated groups is itself close to isotropic and a 2-way cut of an isotropic cloud captures
/// only `0.6366/d` of its scatter against the `2·ln2/d` a split has to buy. Refuse there and the head
/// answers 1 on data with any number of groups. Starting at 2 puts the first *decision* one level
/// down, where each half is already a truncated cloud; from there the recursion has been measured to
/// run to the true `k`. Pelleg & Moore, ELKI and pyclustering all default to 2 for this reason.
///
/// Panics on an empty feature set, as [`kmeans`] does.
pub fn xmeans<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> KMeans<R> {
    let m = features.len();
    assert!(m > 0, "need at least one feature");
    let dim = features[0].dim();
    let cap = k_max.min(m).max(1);
    let start = k_min.max(1).min(cap);

    // `k = 1` needs no restarts: the single centre is the weight-mean whatever the seeding does.
    let n_init = if start == 1 { 1 } else { SPLIT_N_INIT };
    let mut current = kmeans(features, start, max_iter, n_init, seed);
    let mut best_bic = model_bic(features, &current, dim);
    let (mut best_labels, mut best_centers, mut best_inertia) = (
        current.labels.clone(),
        current.centers.clone(),
        current.inertia,
    );

    // Each round that continues adds at least one centre, so `cap` rounds is a bound by
    // construction. Written as a `for` rather than as `while accepted > 0` so that no mutation of
    // the accept test can make the walk re-entrant.
    for round in 0..cap {
        let n_now = current.centers.len();
        let mut next_centers: Vec<Vec<R>> = Vec::with_capacity(cap);
        let mut accepted = 0usize;
        for j in 0..n_now {
            let sub: Vec<C> = (0..m)
                .filter(|&i| current.labels[i] == j)
                .map(|i| features[i].clone())
                .collect();
            let parent = weighted_mean(&sub, dim);
            // A cluster of one leaf has nothing to split. The second clause is the cap: splitting
            // `j` would leave `next_centers.len() + (n_now - j) + 1` centres once the untouched
            // clusters after it are re-emitted, so the split is not tested at all rather than tested
            // and thrown away.
            if sub.len() < 2 || next_centers.len() + (n_now - j) >= cap {
                next_centers.push(parent);
                continue;
            }
            // k-means++ on the subset rather than the paper's "parent ± a random vector": the
            // crate's seeding is strictly better informed, and the split test does not depend on
            // how the two candidate centres were reached.
            let two = kmeans(
                &sub,
                2,
                max_iter,
                SPLIT_N_INIT,
                seed ^ ((round as u64) << 32) ^ j as u64,
            );
            let (nk2, sse2) = cluster_stats(&sub, &two.labels, &two.centers);
            // A child holding no weight makes `n_k·ln n_k` and the pooled variance meaningless; the
            // split is refused rather than scored.
            if nk2.iter().any(|&w| w <= R::zero()) {
                next_centers.push(parent);
                continue;
            }
            let one = vec![0usize; sub.len()];
            let (nk1, sse1) = cluster_stats(&sub, &one, std::slice::from_ref(&parent));
            if pelleg_moore_bic(&nk2, sse2, dim) > pelleg_moore_bic(&nk1, sse1, dim) {
                next_centers.extend(two.centers);
                accepted += 1;
            } else {
                next_centers.push(parent);
            }
        }
        if accepted == 0 {
            break;
        }
        // Improve-Params refines the centres the split test produced. Re-seeding here would throw
        // them away and turn the head back into a sweep.
        current = lloyd_from(features, next_centers, max_iter);
        let bic = model_bic(features, &current, dim);
        if bic > best_bic {
            best_bic = bic;
            best_labels.clone_from(&current.labels);
            best_centers.clone_from(&current.centers);
            best_inertia = current.inertia;
        }
    }

    KMeans {
        labels: best_labels,
        centers: best_centers,
        inertia: best_inertia,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::kmeans::kmeans_auto;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blob_leaves, blobs, grid_micros};
    use crate::feature::Spherical;

    /// Re-derivation of the Pelleg-Moore score from the paper's own variable names, kept apart from
    /// the shipped one so that a mutation of either has to be argued against the other. `nr` is the
    /// total point *weight*, which is the value the extraction now recovers from `nk` rather than
    /// being handed by the caller — the one thing the move could have got wrong.
    fn reference_bic(nk: &[f64], sse: f64, d: usize) -> f64 {
        let k = nk.len() as f64;
        let nr: f64 = nk.iter().sum();
        let var = (sse / (nr - k).max(1.0) / d as f64).max(1e-12);
        let l: f64 = nk
            .iter()
            .filter(|&&n| n > 0.0)
            .map(|&n| {
                n * (n / nr).ln()
                    - 0.5 * n * d as f64 * (std::f64::consts::TAU * var).ln()
                    - 0.5 * (n - 1.0) * d as f64
            })
            .sum();
        l - 0.5 * (k * (d + 1) as f64) * nr.ln()
    }

    #[test]
    fn the_score_agrees_with_an_independent_re_derivation() {
        for (nk, sse, d) in [
            (vec![4.0, 6.0], 20.0, 2),
            (vec![10.0], 3.5, 1),
            (vec![1.0, 1.0, 1.0, 7.0], 0.75, 5),
            (vec![50.0, 0.0, 25.0], 900.0, 3),
            // `nr - k` goes non-positive here, which is what the `.max(1)` floor exists for.
            (vec![0.5, 0.5], 1.0, 2),
            // A partition with no spread at all: the variance floor, not a division by zero.
            (vec![3.0, 3.0], 0.0, 4),
        ] {
            let got = pelleg_moore_bic(&nk, sse, d);
            let want = reference_bic(&nk, sse, d);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "nk = {nk:?}, sse = {sse}, d = {d}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn scaling_the_data_shifts_every_score_by_the_same_amount() {
        // Under x -> λx the spread grows as λ² and the score falls by `d·nr·ln λ` — a constant in
        // `k`, which is why every split decision the head makes is scale-invariant even though no
        // individual score is. A `d` or an `nr` misplaced in the variance term breaks the *equality*
        // between the two shifts below while leaving each score finite and plausible.
        let (sse, d, lam) = (12.0f64, 3usize, 4.0f64);
        let shift =
            |nk: &[f64]| pelleg_moore_bic(nk, sse * lam * lam, d) - pelleg_moore_bic(nk, sse, d);
        let two = shift(&[6.0, 4.0]);
        let three = shift(&[5.0, 3.0, 2.0]);
        let want = -(d as f64) * 10.0 * lam.ln();
        assert!((two - three).abs() < 1e-9, "{two} vs {three}");
        assert!((two - want).abs() < 1e-9, "{two} vs {want}");
    }

    #[test]
    fn a_zero_weight_leaf_set_has_a_mean_and_not_a_division_by_zero() {
        // `Spherical::new` before any `push` weighs nothing. `weighted_mean` is the only place a
        // whole cluster's weight lands in a denominator, and the split test's own zero-weight guard
        // is downstream of it, so this has to be safe on its own.
        let empty: Vec<Spherical<f64>> = vec![Spherical::new(3), Spherical::new(3)];
        let mu = weighted_mean(&empty, 3);
        assert_eq!(mu.len(), 3);
        assert!(mu.iter().all(|v| v.is_finite() && *v == 0.0), "{mu:?}");
    }

    #[test]
    fn neither_head_prefers_one_cluster_per_leaf() {
        // `Σ_l S_l` is constant in `k`, but it sits inside `ln σ̂²`. Score only the between-leaf part
        // and `σ̂²` reaches its floor exactly when every leaf is its own cluster, so the BIC diverges
        // there and both heads answer `k = n_leaves` on any input whose cap lets them look that far.
        // `AUTO_K_MAX = 20` kept every shipped path and every earlier test short of it.
        let (micros, truth) = blob_leaves(6, 10, 40, 0);
        let n = micros.len();
        assert_eq!(n, 24, "the fixture is four leaves per blob");

        let swept = kmeans_auto(&micros, 1, n, 100, 0);
        assert_eq!(swept.centers.len(), 6, "the sweep ran away toward {n}");
        assert!(ari(&swept.labels, &truth) > 0.99);

        let split = xmeans(&micros, 2, n, 100, 0);
        assert_eq!(split.centers.len(), 6, "x-means ran away toward {n}");
        assert!(ari(&split.labels, &truth) > 0.99);
    }

    #[test]
    fn the_split_test_recovers_the_cluster_count_it_was_not_told() {
        for (n_true, d) in [(10usize, 5usize), (10, 10), (30, 10), (30, 32), (30, 64)] {
            for seed in [0u64, 1, 2] {
                let (micros, truth) = blob_leaves(n_true, d, 40, seed);
                let km = xmeans(&micros, 2, micros.len(), 100, seed);
                assert_eq!(
                    km.centers.len(),
                    n_true,
                    "d = {d}, true k = {n_true}, seed {seed}"
                );
                assert!(
                    ari(&km.labels, &truth) > 0.99,
                    "d = {d}, true k = {n_true}, seed {seed}"
                );
            }
        }
    }

    #[test]
    fn the_first_split_is_where_a_greedy_head_can_lose_everything() {
        // Sixty well-separated blobs. Their *centres* are themselves a draw from one isotropic
        // Gaussian, so at `k = 1` the head is asked the question its accept threshold answers worst:
        // a 2-way cut of a round cloud captures `0.6366/d` against the `2·ln2/d` it must buy. Refuse
        // and the recursion is over — there is no way back from a refused split, so the head answers
        // 1 on data with sixty groups. Measured at `k_min = 1`: correct at 10, 20 and 30 blobs, then
        // collapsing to 1 in five of twenty (n_true, seed) cells and in all four seeds at 60.
        //
        // Starting at 2 moves the first *decision* one level down, where each half is a truncated
        // cloud rather than a round one, and the recursion runs to the true `k` in all of them.
        for seed in [0u64, 1, 2] {
            let (micros, truth) = blob_leaves(60, 10, 40, seed);
            let km = xmeans(&micros, 2, micros.len(), 100, seed);
            assert_eq!(km.centers.len(), 60, "seed {seed}");
            assert!(ari(&km.labels, &truth) > 0.99, "seed {seed}");
        }
    }

    #[test]
    fn an_isotropic_layout_is_the_case_greedy_splitting_refuses() {
        // The accept threshold, pinned from the side where it bites hardest, and the exact price of
        // where the recursion starts. A balanced binary split costs `n·ln 2` in mixture weight and
        // buys `½·n·d·ln(S_1/S_2)`, so it is accepted only when it captures `B/S_1 > 1 − 2^(−2/d)`
        // of the region's scatter — **half of it** at `d = 2`. A cut through a cloud that is round
        // at every scale captures about `0.6366/d`, always less than the `1.386/d` the rule asks
        // for, so a square grid of nine equal blobs is refused at the very first split.
        let centers: Vec<[f64; 2]> = (0..9)
            .map(|i| [8.0 * (i % 3) as f64, 8.0 * (i / 3) as f64])
            .collect();
        let mut rng = SplitMix64::new(3);
        let (pts, _truth) = blobs(&mut rng, 60, &centers, 0.3);
        let (micros, _) = grid_micros(&pts, 0.5);
        assert_eq!(xmeans(&micros, 1, 20, 100, 3).centers.len(), 1);
        // And what that refusal actually costs: *everything*, but only at `k_min = 1`. Start one
        // level down and the same threshold on the same leaves accepts every split from there to
        // the truth — so the failure above is the single top-level comparison, not the rule.
        assert_eq!(xmeans(&micros, 2, 20, 100, 3).centers.len(), 9);
        // The sweep, on the same leaves, is not fooled either — so the fixture does have nine
        // groups in it and the refusal at `k_min = 1` is the split test's, not the data's.
        assert_eq!(kmeans_auto(&micros, 1, 20, 100, 3).centers.len(), 9);
    }

    #[test]
    fn k_max_is_a_cap_the_head_cannot_exceed() {
        let (micros, _truth) = blob_leaves(9, 10, 40, 3);
        for cap in 1..=9 {
            let got = xmeans(&micros, 2, cap, 100, 3).centers.len();
            assert!(got <= cap, "cap {cap}: the head returned {got}");
            assert!(got >= 1, "cap {cap}: the head returned nothing");
        }
        // The cap binds rather than decorates: nine real groups are cut to four when asked for four.
        assert_eq!(xmeans(&micros, 2, 4, 100, 3).centers.len(), 4);
        assert_eq!(xmeans(&micros, 2, 1, 100, 3).centers.len(), 1);
    }

    #[test]
    fn a_single_group_is_left_whole() {
        // The stopping rule, from the other side: on one Gaussian cloud every split must be refused,
        // which is the only thing between the entropy term and an unbounded recursion.
        let mut rng = SplitMix64::new(19);
        let (pts, _truth) = blobs(&mut rng, 400, &[[0.0, 0.0]], 1.0);
        let (micros, _) = grid_micros(&pts, 0.4);
        assert_eq!(xmeans(&micros, 1, 10, 100, 19).centers.len(), 1);
    }

    #[test]
    fn the_recursion_reaches_a_k_the_capped_sweep_cannot() {
        // The reason this head exists. `AUTO_K_MAX = 20` bounds the sweep because the sweep costs
        // `O(k_max²)` full k-means over every leaf; x-means stops on its own, so at `n_clusters = 0`
        // `model.rs` bounds it only by the leaf count. Thirty separated groups is therefore a `k` the
        // sweep cannot reach through the shipped path however good its score is.
        let (micros, truth) = blob_leaves(30, 10, 40, 11);

        let swept = kmeans_auto(&micros, 1, 20, 100, 11);
        assert!(
            swept.centers.len() > 15,
            "the sweep answered {} well inside its cap, so the cap is not what bounds it here and \
             the comparison below would be about the score rather than about the reach",
            swept.centers.len()
        );

        let split = xmeans(&micros, 2, micros.len(), 100, 11);
        assert_eq!(split.centers.len(), 30, "x-means did not reach the true k");

        let (a_split, a_swept) = (ari(&split.labels, &truth), ari(&swept.labels, &truth));
        assert!(
            a_split > a_swept,
            "ARI {a_split:.4} (k = 30) did not beat the capped sweep's {a_swept:.4} (k = {})",
            swept.centers.len()
        );
    }

    #[test]
    fn the_head_reads_the_data_and_not_the_frame() {
        // x-means inherits k-means' symmetry group — translate, rotate, uniform scale, swap axes —
        // because the split test is built from a Euclidean sum of squares and a score whose only
        // scale dependence is the `k`-independent shift pinned above. `tests/equivariance.rs` runs
        // the head on the shared 2-D fixture at a fixed `k`; this covers the case that one cannot,
        // the automatic `k` at a dimension where the recursion runs deep.
        let (base, _truth) = blob_leaves(8, 10, 40, 5);
        let d = 10;
        let map = |f: &dyn Fn(&[f64]) -> Vec<f64>| -> Vec<usize> {
            let moved: Vec<Spherical<f64>> = base
                .iter()
                .map(|leaf| {
                    let mut out = Spherical::new(d);
                    // A leaf is a summary, so re-summarise its mean at the leaf's own weight; the
                    // scatter transforms with it and the head reads both.
                    out.push(&f(leaf.mean()), leaf.weight());
                    out
                })
                .collect();
            xmeans(&moved, 2, moved.len(), 100, 5).labels
        };
        let identity = map(&|x| x.to_vec());
        for (name, t) in [
            (
                "translation",
                &(|x: &[f64]| x.iter().map(|v| v + 137.5).collect()) as &dyn Fn(&[f64]) -> Vec<f64>,
            ),
            ("uniform scaling", &|x: &[f64]| {
                x.iter().map(|v| v * 37.5).collect()
            }),
            ("an axis swap", &|x: &[f64]| {
                let mut v = x.to_vec();
                v.swap(0, d - 1);
                v
            }),
            ("a rotation", &|x: &[f64]| {
                // Givens rotation in the (0, 1) plane: enough to break any model that reads an axis.
                let (s, c) = (0.7f64).sin_cos();
                let mut v = x.to_vec();
                v[0] = c * x[0] - s * x[1];
                v[1] = s * x[0] + c * x[1];
                v
            }),
        ] {
            assert_eq!(
                ari(&identity, &map(t)),
                1.0,
                "x-means is not invariant under {name}"
            );
        }
    }

    /// Head-to-head on the two axes that decide whether the head earns its place: how close the
    /// selected `k` lands, and what it costs to get there. Reported, not asserted — a measurement
    /// whose numbers belong in `bench/RESULTS.md`, run with
    /// `cargo test --release -- --ignored --nocapture xmeans`.
    #[test]
    #[ignore = "measurement, not an assertion"]
    fn recursive_versus_sweep_on_the_same_leaves() {
        println!("  d  k*   sweep |dk|   sweep s   xmeans |dk|  xmeans s   cap");
        for d in [2usize, 5, 10, 32, 64] {
            for n_true in [10usize, 30] {
                let (mut ds, mut dx, mut ts, mut tx, mut kx_last) = (0i64, 0i64, 0f64, 0f64, 0);
                for seed in [0u64, 1, 2] {
                    let (micros, _truth) = blob_leaves(n_true, d, 40, seed);
                    let hi = 20.min(micros.len()); // AUTO_K_MAX, the sweep's shipped cap

                    let t0 = std::time::Instant::now();
                    let ks = kmeans_auto(&micros, 1, hi, 100, seed).centers.len();
                    ts += t0.elapsed().as_secs_f64();

                    let t1 = std::time::Instant::now();
                    let kx = xmeans(&micros, 2, micros.len(), 100, seed).centers.len();
                    tx += t1.elapsed().as_secs_f64();

                    ds += (ks as i64 - n_true as i64).abs();
                    dx += (kx as i64 - n_true as i64).abs();
                    kx_last = kx;
                }
                println!(
                    "{d:>3} {n_true:>3}  {:>10.1} {:>9.4}  {:>11.1} {:>9.4}   {kx_last:>3}",
                    ds as f64 / 3.0,
                    ts / 3.0,
                    dx as f64 / 3.0,
                    tx / 3.0
                );
            }
        }
    }
}
