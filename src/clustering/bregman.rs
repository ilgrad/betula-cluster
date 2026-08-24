//! Phase-3 heads over a [`BregmanCf`] summary: k-means and agglomerative clustering.
//!
//! Both are the Euclidean algorithms with `‖·‖²` replaced by `d_φ`, and both are exact on the
//! summary for the same reason the Euclidean ones are — the bias–variance identity
//! `Σ_{x∈i} d_φ(x, c) = S_i + n_i·d_φ(μ_i, c)` holds for every `φ`, so a leaf can be scored against
//! a candidate centre without revisiting a point.
//!
//! Two things do **not** carry over, and both are load-bearing:
//!
//! - **No triangle inequality, so no Hamerly bounds.** `d_φ` is not a metric — it is asymmetric and
//!   fails the triangle inequality — so the bound-based acceleration in
//!   [`kmeans`](super::kmeans) has no analogue here. The Lloyd loop is the plain one.
//! - **No nearest-neighbour chain.** Bregman-Ward is not reducible for `d ≥ 2` outside squared
//!   Euclidean, measured; the agglomerative head runs Anderberg. See
//!   `docs/adr/002-bregman-ward-anderberg.md` for the numbers that decided it.
//!
//! The argument order is not a free choice either. The objective is minimised over `c` by the
//! weighted **arithmetic** mean exactly when the divergence is written `d_φ(x, c)` — point first,
//! centre second (Banerjee et al. 2005). Reversing it minimises a different functional and the mean
//! update would no longer be the M-step.

use crate::bregman::{BregmanCf, BregmanDivergence};
use crate::clustering::agglomerative::{Agglomerative, Merge, anderberg, labels_at};
use crate::clustering::kmeans::KMeans;
use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::types::Real;

/// Cluster `features` into `k` groups under their own divergence.
///
/// `n_init` seeded restarts, lowest reported inertia wins. The inertia is the true Bregman
/// information of the underlying points about their centres, `Σ_i [S_i + n_i·d_φ(μ_i, c)]`, not the
/// centroid-only part — the same convention [`kmeans`](super::kmeans::kmeans) uses.
pub fn bregman_kmeans<R: Real, B: BregmanDivergence<R>>(
    features: &[BregmanCf<R, B>],
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> KMeans<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let dim = features[0].dim();
    let means: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let weights: Vec<R> = features.iter().map(ClusterFeature::weight).collect();
    let info: Vec<R> = features.iter().map(ClusterFeature::ssd).collect();

    let mut rng = SplitMix64::new(seed ^ 0x0B_7E9_A11);
    let mut best: Option<KMeans<R>> = None;
    for _ in 0..n_init.max(1) {
        let init = plus_plus::<R, B>(&means, &weights, &info, k, &mut rng);
        let res = lloyd::<R, B>(&means, &weights, &info, init, max_iter, dim);
        match &best {
            Some(b) if res.inertia >= b.inertia => {}
            _ => best = Some(res),
        }
    }
    best.expect("at least one init")
}

/// Bregman k-means++, seeded on the **exact** leaf potential.
///
/// The sampling weight is `S_i + n_i·d_φ(μ_i, nearest centre)` — the whole cost the leaf contributes,
/// not just its centroid's share. A leaf with large internal information is a place the summary is
/// coarse, and a seed there is worth more than one at an equally distant point leaf.
fn plus_plus<R: Real, B: BregmanDivergence<R>>(
    means: &[Vec<R>],
    weights: &[R],
    info: &[R],
    k: usize,
    rng: &mut SplitMix64,
) -> Vec<Vec<R>> {
    let div = B::default();
    let m = means.len();
    let mut centers: Vec<Vec<R>> = Vec::with_capacity(k);
    let first = (rng.next_u64() as usize) % m;
    centers.push(means[first].clone());

    let mut closest: Vec<R> = means.iter().map(|mu| div.vector(mu, &centers[0])).collect();

    while centers.len() < k {
        let potential: Vec<R> = (0..m)
            .map(|i| (info[i] + weights[i] * closest[i]).max(R::zero()))
            .collect();
        let total: R = potential.iter().copied().sum();
        let pick = if total > R::zero() {
            let target = R::from_f64(rng.next_f64()).unwrap() * total;
            let mut acc = R::zero();
            let mut chosen = m - 1;
            for (i, &p) in potential.iter().enumerate() {
                acc = acc + p;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            (rng.next_u64() as usize) % m
        };
        centers.push(means[pick].clone());
        let last = centers.len() - 1;
        for (i, mu) in means.iter().enumerate() {
            let d = div.vector(mu, &centers[last]);
            if d < closest[i] {
                closest[i] = d;
            }
        }
    }
    centers
}

fn lloyd<R: Real, B: BregmanDivergence<R>>(
    means: &[Vec<R>],
    weights: &[R],
    info: &[R],
    mut centers: Vec<Vec<R>>,
    max_iter: usize,
    dim: usize,
) -> KMeans<R> {
    let div = B::default();
    let m = means.len();
    let k = centers.len();
    let mut labels = vec![0usize; m];

    for _ in 0..max_iter.max(1) {
        let mut moved = false;
        for (i, mu) in means.iter().enumerate() {
            let mut best = 0;
            let mut best_d = R::infinity();
            for (c, centre) in centers.iter().enumerate() {
                let d = div.vector(mu, centre);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                moved = true;
            }
        }

        // M-step: the weighted arithmetic mean is the Bregman centroid for every φ. An emptied
        // centre keeps its position rather than being reseeded, so a run is a deterministic
        // function of its seeding and `k` is an upper bound on the clusters actually returned.
        let mut sums = vec![vec![R::zero(); dim]; k];
        let mut mass = vec![R::zero(); k];
        for (i, mu) in means.iter().enumerate() {
            let c = labels[i];
            mass[c] = mass[c] + weights[i];
            for (s, &x) in sums[c].iter_mut().zip(mu) {
                *s = *s + weights[i] * x;
            }
        }
        for (c, centre) in centers.iter_mut().enumerate() {
            if mass[c] > R::zero() {
                for (t, &s) in centre.iter_mut().zip(&sums[c]) {
                    *t = s / mass[c];
                }
            }
        }
        if !moved {
            break;
        }
    }

    let inertia = (0..m)
        .map(|i| info[i] + weights[i] * div.vector(&means[i], &centers[labels[i]]))
        .sum();
    KMeans {
        labels,
        centers,
        inertia,
    }
}

/// A live cluster during agglomeration. `D4_φ` is a pure centroid measure — both stored
/// informations cancel out of `S_AB − S_A − S_B` — so the driver never needs to carry one.
struct Node<R: Real> {
    mass: R,
    mean: Vec<R>,
}

impl<R: Real> Node<R> {
    fn ward<B: BregmanDivergence<R>>(&self, other: &Self) -> R {
        let total = self.mass + other.mass;
        if self.mass <= R::zero() || other.mass <= R::zero() {
            return R::zero();
        }
        let div = B::default();
        let factor = other.mass / total;
        self.mean
            .iter()
            .zip(&other.mean)
            .map(|(&ma, &mb)| {
                let merged = ma + factor * (mb - ma);
                self.mass * div.divergence(ma, merged) + other.mass * div.divergence(mb, merged)
            })
            .sum()
    }
}

/// Agglomerate `features` into `k` clusters under Bregman-Ward, by Anderberg.
///
/// Anderberg rather than the nearest-neighbour chain is a correctness requirement, not a preference:
/// at `d = 20` and `m = 12` a chain already builds a different dendrogram in ~1 % of Itakura–Saito
/// and exponential instances, the rate grows with `m`, and when it fires the partition is destroyed
/// rather than perturbed. `k` is clamped to `[1, features.len()]`.
pub fn bregman_agglomerative<R: Real, B: BregmanDivergence<R>>(
    features: &[BregmanCf<R, B>],
    k: usize,
) -> Agglomerative {
    let m = features.len();
    if m == 0 {
        return Agglomerative { labels: Vec::new() };
    }
    let k = k.max(1).min(m);
    let merges = dendrogram::<R, B>(features);
    Agglomerative {
        labels: labels_at(m, &merges, m - k),
    }
}

fn dendrogram<R: Real, B: BregmanDivergence<R>>(features: &[BregmanCf<R, B>]) -> Vec<Merge<R>> {
    let mut node: Vec<Node<R>> = features
        .iter()
        .map(|cf| Node {
            mass: cf.weight(),
            mean: cf.mean().to_vec(),
        })
        .collect();
    anderberg(
        &mut node,
        |a, b| a.ward::<B>(b),
        |nodes, a, b| {
            let total = nodes[a].mass + nodes[b].mass;
            if total > R::zero() {
                let factor = nodes[b].mass / total;
                for j in 0..nodes[a].mean.len() {
                    let (ma, mb) = (nodes[a].mean[j], nodes[b].mean[j]);
                    nodes[a].mean[j] = ma + factor * (mb - ma);
                }
            }
            nodes[a].mass = total;
        },
    )
}

/// Lloyd iterations spent on the k-means warm start, independent of the EM budget so that
/// raising `max_iter` refines the same fit instead of starting from a different one. Matches the
/// Gaussian heads' constant.
const WARM_START_ITERS: usize = 50;

/// A fitted soft Bregman mixture.
///
/// The bijection is Banerjee et al.'s: every regular exponential family has a unique Bregman
/// divergence with `p(x | θ_k) = exp(−d_φ(x, μ_k))·b_φ(x)`, and `b_φ` does not depend on `k`. So
/// soft Bregman clustering *is* EM for that family, and this is the diagonal-GMM head's structure
/// with `d_φ` where the Mahalanobis term was.
pub struct BregmanMixture<R: Real> {
    /// Hard label (argmax responsibility) per input feature.
    pub labels: Vec<usize>,
    /// Soft responsibilities `[feature][component]`.
    pub resp: Vec<Vec<R>>,
    /// Mixture weights `π_k`.
    pub weights: Vec<R>,
    /// Component means `μ_k` — the Bregman centroids, hence the plain weighted arithmetic means.
    pub means: Vec<Vec<R>>,
    /// The evidence lower bound at convergence, up to the `Σ_x log b_φ(x)` term that no choice of
    /// model can change. See [`bregman_em`] for why this is a bound and not the likelihood.
    pub elbo: R,
}

/// Fit a `k`-component soft Bregman mixture to `features`, warm-started from [`bregman_kmeans`].
///
/// `beta` is the **inverse dispersion**: the model is `p(x | k) ∝ exp(−β·d_φ(x, μ_k))·b_φ(x)`, and
/// `β = 1` is Banerjee et al.'s soft Bregman clustering exactly. It is a parameter rather than a
/// fitted quantity because fitting it needs the family's log-partition function, which has no form
/// generic in `φ`; shared across components it is a constant in the M-step and shifts the bound by
/// a constant, so nothing here depends on knowing it.
///
/// **It is also the knob that decides whether the head can separate anything at all.** Without it
/// the mixture runs at a fixed temperature, and separation is measured in *nats of divergence*, not
/// in coordinates. Itakura–Saito is scale-invariant — `d(ax, ay) = d(x, y)` — so three groups whose
/// centres look far apart at `[6, 12, 4]`, `[18, 11, 10]` and `[19, 19, 18]` are only ≈ 0.33 nats
/// apart, `exp(−0.33) = 0.72`, and a `β = 1` mixture correctly reports that they overlap almost
/// completely and collapses all three means onto the global mean. That is the model being honest,
/// not the optimiser failing. Raise `β` until the responsibilities are as sharp as the application
/// needs, or use a divergence whose scale matches the question.
///
/// **What is exact and what is bounded.** The E-step needs `E_{x∈leaf}[·]`, and the leaf carries
/// only `(n, μ, S_φ)`. Two facts settle what that buys:
///
/// - `E_{x∈leaf}[d_φ(x, μ_k)] = S_i/n_i + d_φ(μ_i, μ_k)` — the bias–variance identity, **exact** for
///   every `φ`. So the expected complete-data log-likelihood is computable from the summary with no
///   approximation of the within-leaf shape at all.
/// - The *observed*-data log-likelihood needs `E[log Σ_k …]`, which is not linear and is not
///   recoverable from three moments. Tying the responsibilities within a leaf is the only
///   approximation made, and it makes this exact **variational** EM: `elbo` is a true lower bound on
///   the point-level log-likelihood, and it increases monotonically.
///
/// One consequence is worth stating because it is testable and surprising: `S_i/n_i` is the same for
/// every `k`, so it cancels in the E-step's normalisation. **Given the centres, a leaf's internal
/// spread does not move its responsibilities at all** — only the value of the bound. The fitted
/// result still depends on `S`, because the k-means++ warm start samples on the leaf potential
/// `S_i + n_i·d_φ`, which is where a coarse leaf earns its extra seeding weight; the independence is
/// a property of the E-step, not of the whole fit.
pub fn bregman_em<R: Real, B: BregmanDivergence<R>>(
    features: &[BregmanCf<R, B>],
    k: usize,
    beta: R,
    max_iter: usize,
    seed: u64,
) -> BregmanMixture<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(
        beta > R::zero() && beta.is_finite(),
        "beta must be positive"
    );
    assert!(features.len() >= k, "need at least k features");
    let div = B::default();
    let dim = features[0].dim();
    let m = features.len();
    let means: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let mass: Vec<R> = features.iter().map(ClusterFeature::weight).collect();
    let info: Vec<R> = features.iter().map(ClusterFeature::ssd).collect();
    let total: R = mass.iter().copied().sum();

    let warm = bregman_kmeans(features, k, WARM_START_ITERS, 1, seed);
    let mut centres = warm.centers;
    let mut pi = vec![R::one() / R::from_usize(k).unwrap(); k];
    let mut resp = vec![vec![R::zero(); k]; m];
    let mut elbo = R::neg_infinity();
    // Keeps a component that loses all its mass from producing NaN weights; it simply stops
    // competing rather than poisoning the normalisation.
    let floor = R::from_f64(1e-300).unwrap();

    for _ in 0..max_iter.max(1) {
        let mut bound = R::zero();
        for i in 0..m {
            let logs: Vec<R> = (0..k)
                .map(|c| pi[c].max(floor).ln() - beta * div.vector(&means[i], &centres[c]))
                .collect();
            let hi = logs.iter().copied().fold(R::neg_infinity(), R::max);
            let sum: R = logs.iter().map(|&l| (l - hi).exp()).sum();
            let lse = hi + sum.ln();
            for (c, r) in resp[i].iter_mut().enumerate() {
                *r = (logs[c] - lse).exp();
            }
            // The bound the leaf contributes: its mass times the tied log-mixture value, less the
            // internal information every component pays equally.
            bound = bound + mass[i] * lse - beta * info[i];
        }

        let mut nk = vec![R::zero(); k];
        let mut sums = vec![vec![R::zero(); dim]; k];
        for i in 0..m {
            for c in 0..k {
                let w = mass[i] * resp[i][c];
                nk[c] = nk[c] + w;
                for (s, &x) in sums[c].iter_mut().zip(&means[i]) {
                    *s = *s + w * x;
                }
            }
        }
        for c in 0..k {
            pi[c] = if total > R::zero() {
                nk[c] / total
            } else {
                R::zero()
            };
            if nk[c] > R::zero() {
                for (t, &s) in centres[c].iter_mut().zip(&sums[c]) {
                    *t = s / nk[c];
                }
            }
        }

        let improved = bound - elbo;
        elbo = bound;
        if improved.abs() <= R::from_f64(1e-10).unwrap() * bound.abs().max(R::one()) {
            break;
        }
    }

    let labels = resp
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .fold((0usize, R::neg_infinity()), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
                .0
        })
        .collect();

    BregmanMixture {
        labels,
        resp,
        weights: pi,
        means: centres,
        elbo,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bregman::{ItakuraSaito, KullbackLeibler, SquaredEuclidean};
    use crate::clustering::agglomerative::{Linkage, agglomerative};
    use crate::feature::Spherical;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        fn span(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next()
        }
    }

    /// `m` well-separated groups of point-leaves, so the right answer is unambiguous.
    fn clustered<B: BregmanDivergence<f64>>(
        rng: &mut Lcg,
        groups: usize,
        per: usize,
        dim: usize,
        spread: f64,
    ) -> (Vec<BregmanCf<f64, B>>, Vec<usize>) {
        let mut cfs = Vec::new();
        let mut truth = Vec::new();
        for g in 0..groups {
            let centre: Vec<f64> = (0..dim).map(|_| rng.span(2.0, 20.0)).collect();
            for _ in 0..per {
                let mut cf = BregmanCf::<f64, B>::new(dim);
                let p: Vec<f64> = centre
                    .iter()
                    .map(|&c| (c * (1.0 + spread * rng.span(-1.0, 1.0))).max(0.05))
                    .collect();
                cf.push(&p, 1.0);
                cfs.push(cf);
                truth.push(g);
            }
        }
        (cfs, truth)
    }

    fn grouped_correctly(labels: &[usize], truth: &[usize]) -> bool {
        let k = truth.iter().max().unwrap() + 1;
        let mut of = vec![usize::MAX; k];
        for (&l, &t) in labels.iter().zip(truth) {
            if of[t] == usize::MAX {
                of[t] = l;
            } else if of[t] != l {
                return false;
            }
        }
        let mut seen = of.clone();
        seen.sort_unstable();
        seen.dedup();
        seen.len() == k
    }

    #[test]
    fn bregman_kmeans_recovers_well_separated_groups_under_a_non_euclidean_divergence() {
        let mut rng = Lcg(21);
        let (cfs, truth) = clustered::<ItakuraSaito>(&mut rng, 4, 25, 3, 0.02);
        let out = bregman_kmeans(&cfs, 4, 100, 5, 7);
        assert!(grouped_correctly(&out.labels, &truth), "{:?}", out.labels);
        assert!(out.inertia >= 0.0 && out.inertia.is_finite());
    }

    #[test]
    fn bregman_agglomerative_recovers_the_same_groups() {
        let mut rng = Lcg(22);
        let (cfs, truth) = clustered::<KullbackLeibler>(&mut rng, 4, 20, 3, 0.02);
        let out = bregman_agglomerative(&cfs, 4);
        assert!(grouped_correctly(&out.labels, &truth), "{:?}", out.labels);
    }

    #[test]
    fn the_reported_inertia_is_the_true_information_about_the_centres() {
        // The bias-variance identity is what lets a leaf be scored without its points; if the
        // reported inertia drifted from the brute-force sum, the head would be optimising a
        // quantity it does not report.
        let mut rng = Lcg(23);
        let dim = 4;
        let mut cfs = Vec::new();
        let mut pts = Vec::new();
        for _ in 0..40 {
            let mut cf = BregmanCf::<f64, ItakuraSaito>::new(dim);
            let mut owned = Vec::new();
            for _ in 0..6 {
                let p: Vec<f64> = (0..dim).map(|_| rng.span(0.5, 9.0)).collect();
                cf.push(&p, 1.0);
                owned.push(p);
            }
            cfs.push(cf);
            pts.push(owned);
        }
        let out = bregman_kmeans(&cfs, 3, 100, 3, 11);
        let div = ItakuraSaito;
        let brute: f64 = pts
            .iter()
            .zip(&out.labels)
            .map(|(group, &c)| {
                group
                    .iter()
                    .map(|p| div.vector(p, &out.centers[c]))
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (out.inertia - brute).abs() <= 1e-9 * brute,
            "{} vs {brute}",
            out.inertia
        );
    }

    #[test]
    fn at_squared_euclidean_both_heads_reproduce_the_euclidean_ones() {
        // The regression that matters: generalising must not have changed the Euclidean answer.
        // Ward here is 2*D4 against Bregman-Ward's D4, which ranks identically, so the dendrograms
        // and hence the labels must agree exactly.
        let mut rng = Lcg(24);
        let (cfs, _) = clustered::<SquaredEuclidean>(&mut rng, 5, 18, 4, 0.05);
        let spherical: Vec<Spherical<f64>> = cfs
            .iter()
            .map(|cf| Spherical::from_moments(cf.weight(), cf.mean().to_vec(), cf.ssd()))
            .collect();

        for k in [2, 3, 5, 7] {
            let mine = bregman_agglomerative(&cfs, k);
            let theirs = agglomerative(&spherical, Linkage::Ward, k);
            assert_eq!(mine.labels, theirs.labels, "k = {k}");
        }
    }

    #[test]
    fn the_seeding_prefers_a_coarse_leaf_over_an_equally_distant_point_leaf() {
        // Two candidates the same distance from the only centre, one carrying real internal
        // information. The potential must rank the informative one higher, which is the whole
        // point of seeding on S_i + n_i*d rather than on the centroid distance alone.
        let dim = 2;
        let mut coarse = BregmanCf::<f64, ItakuraSaito>::new(dim);
        for p in [[4.0, 6.0], [6.0, 4.0], [5.0, 5.0]] {
            coarse.push(&p, 1.0);
        }
        let mut tight = BregmanCf::<f64, ItakuraSaito>::new(dim);
        for _ in 0..3 {
            tight.push(coarse.mean(), 1.0);
        }
        assert!(coarse.ssd() > 0.0);
        assert_eq!(tight.ssd(), 0.0);
        assert_eq!(coarse.mean(), tight.mean());

        let div = ItakuraSaito;
        let centre = vec![20.0, 20.0];
        let d = div.vector(coarse.mean(), &centre);
        let coarse_potential = coarse.ssd() + coarse.weight() * d;
        let tight_potential = tight.ssd() + tight.weight() * d;
        assert!(
            coarse_potential > tight_potential,
            "{coarse_potential} vs {tight_potential}"
        );
    }

    #[test]
    fn bregman_em_recovers_the_groups_and_its_bound_only_goes_up() {
        let mut rng = Lcg(26);
        let (cfs, truth) = clustered::<SquaredEuclidean>(&mut rng, 3, 30, 3, 0.02);
        let fit = bregman_em(&cfs, 3, 1.0, 200, 5);
        assert!(grouped_correctly(&fit.labels, &truth), "{:?}", fit.labels);
        assert!(fit.elbo.is_finite());
        assert!((fit.weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        for r in &fit.resp {
            assert!((r.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }

        // Variational EM never decreases its bound, so a longer run cannot score worse — and the
        // warm start has its own budget, so raising `max_iter` refines the same fit rather than
        // starting a different one.
        let mut prev = f64::NEG_INFINITY;
        for iters in [1, 2, 3, 5, 10, 40] {
            let e = bregman_em(&cfs, 3, 1.0, iters, 5).elbo;
            assert!(
                e >= prev - 1e-9 * e.abs().max(1.0),
                "{e} after {iters} iterations < {prev}"
            );
            prev = e;
        }
    }

    #[test]
    fn beta_is_what_lets_a_scale_invariant_divergence_separate_anything() {
        // Itakura-Saito is scale-invariant, so groups that look far apart in coordinates can be a
        // fraction of a nat apart in divergence. At beta = 1 the mixture correctly reports that they
        // overlap and collapses; raising beta sharpens the same geometry until it separates. This is
        // the documented limitation, pinned so it cannot be mistaken for a defect later.
        let mut rng = Lcg(29);
        let (cfs, truth) = clustered::<ItakuraSaito>(&mut rng, 3, 30, 3, 0.02);

        let flat = bregman_em(&cfs, 3, 1.0, 200, 5);
        assert!(
            !grouped_correctly(&flat.labels, &truth),
            "beta = 1 was expected to under-separate this fixture"
        );
        let spread = flat.resp[0].iter().cloned().fold(0.0f64, f64::max);
        assert!(
            spread < 0.9,
            "responsibilities were already sharp: {spread}"
        );

        let sharp = bregman_em(&cfs, 3, 200.0, 200, 5);
        assert!(
            grouped_correctly(&sharp.labels, &truth),
            "{:?}",
            sharp.labels
        );
        let peak = sharp.resp[0].iter().cloned().fold(0.0f64, f64::max);
        assert!(peak > 0.99, "raising beta did not sharpen: {peak}");
    }

    #[test]
    fn given_the_centres_a_leaf_s_spread_cannot_move_its_responsibilities() {
        // E[d_φ(x, μ_k)] = S_i/n_i + d_φ(μ_i, μ_k), and S_i/n_i is the same for every k, so it
        // cancels in the E-step's normalisation. Stated as the identity that makes it true: the
        // *difference* of expected divergences between any two centres is a function of the mean
        // alone. (The fitted result does still depend on S -- the k-means++ warm start samples on
        // the leaf potential -- so this is asserted where it holds, on the E-step.)
        let mut rng = Lcg(27);
        let dim = 3;
        let mut tight = BregmanCf::<f64, KullbackLeibler>::new(dim);
        let centre: Vec<f64> = (0..dim).map(|_| rng.span(2.0, 9.0)).collect();
        tight.push(&centre, 4.0);

        let mut coarse = BregmanCf::<f64, KullbackLeibler>::new(dim);
        let lo: Vec<f64> = centre.iter().map(|&x| x * 0.7).collect();
        let hi: Vec<f64> = centre.iter().map(|&x| x * 1.3).collect();
        coarse.push(&lo, 2.0);
        coarse.push(&hi, 2.0);

        assert!((coarse.weight() - tight.weight()).abs() < 1e-12);
        for (a, b) in coarse.mean().iter().zip(tight.mean()) {
            assert!((a - b).abs() <= 1e-12 * b.abs(), "{a} vs {b}");
        }
        assert!(coarse.ssd() > 0.0 && tight.ssd() == 0.0);

        for _ in 0..50 {
            let c1: Vec<f64> = (0..dim).map(|_| rng.span(1.0, 12.0)).collect();
            let c2: Vec<f64> = (0..dim).map(|_| rng.span(1.0, 12.0)).collect();
            let gap_tight =
                (tight.information_about(&c1) - tight.information_about(&c2)) / tight.weight();
            let gap_coarse =
                (coarse.information_about(&c1) - coarse.information_about(&c2)) / coarse.weight();
            assert!(
                (gap_tight - gap_coarse).abs() <= 1e-9 * gap_tight.abs().max(1.0),
                "{gap_tight} vs {gap_coarse}"
            );
        }

        // And the spread is not free: it costs the bound, since every component pays it.
        let mut rng = Lcg(28);
        let (leaves, _) = clustered::<KullbackLeibler>(&mut rng, 3, 12, 2, 0.02);
        let widened: Vec<BregmanCf<f64, KullbackLeibler>> = leaves
            .iter()
            .map(|cf| {
                let mut wide = BregmanCf::<f64, KullbackLeibler>::new(cf.dim());
                let lo: Vec<f64> = cf.mean().iter().map(|&x| x * 0.7).collect();
                let hi: Vec<f64> = cf.mean().iter().map(|&x| x * 1.3).collect();
                wide.push(&lo, cf.weight() / 2.0);
                wide.push(&hi, cf.weight() / 2.0);
                wide
            })
            .collect();
        let a = bregman_em(&leaves, 3, 1.0, 100, 3);
        let b = bregman_em(&widened, 3, 1.0, 100, 3);
        assert!(b.elbo < a.elbo, "{} vs {}", b.elbo, a.elbo);
    }

    #[test]
    fn a_degenerate_request_does_not_panic() {
        let mut rng = Lcg(25);
        let (cfs, _) = clustered::<KullbackLeibler>(&mut rng, 2, 3, 2, 0.01);
        assert_eq!(bregman_agglomerative(&cfs, 0).labels.len(), cfs.len());
        assert_eq!(bregman_agglomerative(&cfs, 10_000).labels.len(), cfs.len());
        let empty: Vec<BregmanCf<f64, KullbackLeibler>> = Vec::new();
        assert!(bregman_agglomerative(&empty, 3).labels.is_empty());
        let one = bregman_kmeans(&cfs, 1, 10, 1, 0);
        assert!(one.labels.iter().all(|&l| l == 0));
    }
}
