//! Weighted k-means on leaf clustering features.
//!
//! Each feature is a weighted point at its mean `μ_i` with weight `n_i`. Initialisation is
//! weighted k-means++; iterations are **Hamerly-accelerated exact Lloyd** (triangle-inequality
//! bounds skip redundant distance computations without changing the output). The reported inertia
//! is the true SSE of the underlying points, including within-feature spread:
//! `Σ_i [S_i + n_i‖μ_i − c‖²]`. [`xmeans`] picks `k` automatically by BIC.

use crate::clustering::rng::SplitMix64;
use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// Result of a k-means run over features.
pub struct KMeans<R: Real> {
    /// Cluster index per input feature.
    pub labels: Vec<usize>,
    /// Cluster centres.
    pub centers: Vec<Vec<R>>,
    /// Total within-cluster sum of squares (includes within-feature spread).
    pub inertia: R,
}

/// Cluster `features` into `k` groups. Runs `n_init` k-means++ restarts and keeps the lowest
/// inertia; each restart runs up to `max_iter` Lloyd iterations.
pub fn kmeans<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> KMeans<R> {
    assert!(k >= 1, "k must be >= 1");
    assert!(features.len() >= k, "need at least k features");
    let dim = features[0].dim();
    let means: Vec<Vec<R>> = features.iter().map(|f| f.mean().to_vec()).collect();
    let weights: Vec<R> = features.iter().map(|f| f.weight()).collect();
    let ssd: Vec<R> = features.iter().map(|f| f.ssd()).collect();

    let mut rng = SplitMix64::new(seed);
    let mut best: Option<KMeans<R>> = None;
    for _ in 0..n_init.max(1) {
        let init = kmeans_plus_plus(&means, &weights, k, &mut rng);
        let res = lloyd_hamerly(&means, &weights, &ssd, init, max_iter, dim);
        match &best {
            Some(b) if res.inertia >= b.inertia => {}
            _ => best = Some(res),
        }
    }
    best.expect("at least one init")
}

/// Why a constrained run could not produce a valid labelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintError {
    /// A must-link group is itself cannot-linked: the constraints contradict and no labelling of any
    /// `k` can satisfy them.
    Contradiction,
    /// The greedy assignment reached a dead end at this `k` (e.g. more mutually cannot-linked groups
    /// than clusters). COP-KMeans is greedy, so this can also fire on instances that *are* satisfiable
    /// under a different order — raise `k` or relax constraints.
    Infeasible,
}

fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while parent[r] != r {
        r = parent[r];
    }
    let mut c = x;
    while parent[c] != r {
        let next = parent[c];
        parent[c] = r;
        c = next;
    }
    r
}

/// COP-KMeans (Wagstaff et al., ICML 2001) over weighted clustering features. `must` / `cannot` are
/// index pairs into `features`. Must-link is transitively closed into groups ("chunklets") assigned
/// as a unit; each greedy assignment step places a chunklet in its nearest centre that violates no
/// cannot-link with chunklets already there. Returns one cluster label per feature, or a typed error
/// when the constraints cannot be met. `n_init` restarts (different k-means++ seeds) are tried and the
/// feasible run with the lowest true SSE (`Σ_i [S_i + n_i‖μ_i − c‖²]`) is kept.
#[allow(clippy::too_many_arguments)]
pub fn cop_kmeans<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k: usize,
    must: &[(usize, usize)],
    cannot: &[(usize, usize)],
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> Result<Vec<usize>, ConstraintError> {
    assert!(!features.is_empty(), "need at least one feature");
    let n = features.len();
    let dim = features[0].dim();

    // 1) Must-link transitive closure → chunklets, compacted to dense ids 0..g.
    let mut parent: Vec<usize> = (0..n).collect();
    for &(a, b) in must {
        if a < n && b < n {
            let (ra, rb) = (uf_find(&mut parent, a), uf_find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
    }
    let mut root = vec![0usize; n];
    for (i, r) in root.iter_mut().enumerate() {
        *r = uf_find(&mut parent, i);
    }
    let mut remap = vec![usize::MAX; n];
    let mut g = 0;
    for &r in &root {
        if remap[r] == usize::MAX {
            remap[r] = g;
            g += 1;
        }
    }
    let chunk_of: Vec<usize> = root.iter().map(|&r| remap[r]).collect();

    // 2) Chunklet weighted centroid + total weight (the weighted mean of its member features).
    let mut cw = vec![R::zero(); g];
    let mut csum = vec![vec![R::zero(); dim]; g];
    for (i, f) in features.iter().enumerate() {
        let c = chunk_of[i];
        let w = f.weight();
        cw[c] = cw[c] + w;
        for (s, &v) in csum[c].iter_mut().zip(f.mean()) {
            *s = *s + w * v;
        }
    }
    let cmean: Vec<Vec<R>> = (0..g)
        .map(|c| {
            if cw[c] > R::zero() {
                csum[c].iter().map(|&s| s / cw[c]).collect()
            } else {
                vec![R::zero(); dim]
            }
        })
        .collect();

    // 3) Cannot-link lifted to chunklets; a within-chunklet cannot-link contradicts a must-link.
    let mut cl_adj: Vec<Vec<usize>> = vec![Vec::new(); g];
    for &(a, b) in cannot {
        if a >= n || b >= n {
            continue;
        }
        let (ca, cb) = (chunk_of[a], chunk_of[b]);
        if ca == cb {
            return Err(ConstraintError::Contradiction);
        }
        cl_adj[ca].push(cb);
        cl_adj[cb].push(ca);
    }
    for adj in &mut cl_adj {
        adj.sort_unstable();
        adj.dedup();
    }

    let k = k.min(g).max(1);
    // Assign heaviest chunklets first (most data to place well); id tiebreak keeps it deterministic.
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by(|&i, &j| {
        cw[j]
            .partial_cmp(&cw[i])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });

    let mut rng = SplitMix64::new(seed);
    let mut best: Option<(R, Vec<usize>)> = None; // (inertia, chunk → centre)
    for _ in 0..n_init.max(1) {
        let mut centers = kmeans_plus_plus(&cmean, &cw, k, &mut rng);
        let mut assign = vec![usize::MAX; g];
        let mut feasible = true;
        for _ in 0..max_iter.max(1) {
            let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
            let mut next = vec![usize::MAX; g];
            let mut placed_all = true;
            for &ch in &order {
                let mut cand: Vec<(R, usize)> = (0..k)
                    .map(|c| (sq_euclidean(&cmean[ch], &centers[c]), c))
                    .collect();
                cand.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.cmp(&b.1))
                });
                let pick = cand.into_iter().find_map(|(_, c)| {
                    let conflict = members[c]
                        .iter()
                        .any(|&m| cl_adj[ch].binary_search(&m).is_ok());
                    (!conflict).then_some(c)
                });
                match pick {
                    Some(c) => {
                        next[ch] = c;
                        members[c].push(ch);
                    }
                    None => {
                        placed_all = false;
                        break;
                    }
                }
            }
            if !placed_all {
                feasible = false;
                break;
            }
            let changed = next != assign;
            assign = next;
            if !changed {
                break;
            }
            update_centers(&mut centers, &cmean, &cw, &assign, dim);
        }
        if !feasible {
            continue;
        }
        let mut inertia = R::zero();
        for (i, f) in features.iter().enumerate() {
            let c = assign[chunk_of[i]];
            inertia = inertia + f.ssd() + f.weight() * sq_euclidean(f.mean(), &centers[c]);
        }
        match &best {
            Some((bi, _)) if inertia >= *bi => {}
            _ => best = Some((inertia, assign)),
        }
    }

    let (_, assign) = best.ok_or(ConstraintError::Infeasible)?;
    Ok(chunk_of.iter().map(|&c| assign[c]).collect())
}

/// **Greedy** weighted k-means++ (Arthur–Vassilvitskii; scikit-learn's default since 1.0). Each new
/// centre samples `n_trials = 2 + ⌊ln k⌋` candidates ∝ weight·D² and keeps the one that most reduces
/// the total potential `Σ_i w_i · min_c ‖x_i − c‖²` — strictly lower-variance, lower-inertia seeds than
/// single-candidate sampling, at ~`ln k`× the (already negligible) init cost over the `M ≪ N` leaves.
fn kmeans_plus_plus<R: Real>(
    means: &[Vec<R>],
    weights: &[R],
    k: usize,
    rng: &mut SplitMix64,
) -> Vec<Vec<R>> {
    let to_f = |r: R| r.to_f64().unwrap_or(0.0);
    let mut centers = Vec::with_capacity(k);
    let w0: Vec<f64> = weights.iter().map(|&w| to_f(w)).collect();
    centers.push(means[weighted_pick(&w0, rng)].clone());

    // d2[i] = squared distance of point i to the nearest chosen centre.
    let mut d2: Vec<f64> = means
        .iter()
        .map(|m| to_f(sq_euclidean(m, &centers[0])))
        .collect();
    let n_trials = 2 + (k as f64).ln().floor().max(0.0) as usize;
    while centers.len() < k {
        let probs: Vec<f64> = w0.iter().zip(&d2).map(|(&w, &d)| w * d).collect();
        // Sample `n_trials` candidates ∝ weight·D²; keep the one giving the lowest resulting potential.
        let mut best = usize::MAX;
        let mut best_pot = f64::INFINITY;
        for _ in 0..n_trials {
            let cand = weighted_pick(&probs, rng);
            let mut pot = 0.0;
            for (i, m) in means.iter().enumerate() {
                pot += w0[i] * to_f(sq_euclidean(m, &means[cand])).min(d2[i]);
            }
            if pot < best_pot {
                best_pot = pot;
                best = cand;
            }
        }
        for (di, m) in d2.iter_mut().zip(means) {
            let nd = to_f(sq_euclidean(m, &means[best]));
            if nd < *di {
                *di = nd;
            }
        }
        centers.push(means[best].clone());
    }
    centers
}

pub(crate) fn weighted_pick(probs: &[f64], rng: &mut SplitMix64) -> usize {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 {
        return (rng.next_u64() as usize) % probs.len();
    }
    let mut r = rng.next_f64() * total;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

/// Brute-force exact Lloyd — kept as the reference implementation that [`lloyd_hamerly`] is tested
/// against (the accelerated version must produce identical output).
#[cfg(test)]
fn lloyd<R: Real>(
    means: &[Vec<R>],
    weights: &[R],
    ssd: &[R],
    mut centers: Vec<Vec<R>>,
    max_iter: usize,
    dim: usize,
) -> KMeans<R> {
    let n = means.len();
    let k = centers.len();
    let mut labels = vec![usize::MAX; n];

    for _ in 0..max_iter {
        let mut changed = false;
        for (i, m) in means.iter().enumerate() {
            let mut best = 0;
            let mut bd = sq_euclidean(m, &centers[0]);
            for (c, center) in centers.iter().enumerate().skip(1) {
                let d = sq_euclidean(m, center);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut sums = vec![vec![R::zero(); dim]; k];
        let mut wsum = vec![R::zero(); k];
        for (i, m) in means.iter().enumerate() {
            let l = labels[i];
            wsum[l] = wsum[l] + weights[i];
            for (s, &v) in sums[l].iter_mut().zip(m) {
                *s = *s + weights[i] * v;
            }
        }
        for (c, ws) in wsum.iter().enumerate() {
            if *ws > R::zero() {
                for d in 0..dim {
                    centers[c][d] = sums[c][d] / *ws;
                }
            }
        }
    }

    let mut inertia = R::zero();
    for (i, m) in means.iter().enumerate() {
        inertia = inertia + ssd[i] + weights[i] * sq_euclidean(m, &centers[labels[i]]);
    }
    KMeans {
        labels,
        centers,
        inertia,
    }
}

/// Nearest and second-nearest centre to `m`; returns `(index, sq-dist nearest, sq-dist 2nd)`.
fn nearest_two<R: Real>(m: &[R], centers: &[Vec<R>]) -> (usize, R, R) {
    let mut best = 0;
    let mut d1 = sq_euclidean(m, &centers[0]);
    let mut d2 = R::infinity();
    for (c, center) in centers.iter().enumerate().skip(1) {
        let d = sq_euclidean(m, center);
        if d < d1 {
            d2 = d1;
            d1 = d;
            best = c;
        } else if d < d2 {
            d2 = d;
        }
    }
    (best, d1, d2)
}

/// Weighted centroid update: `centre_c = (Σ_{i∈c} w_i μ_i) / Σ_{i∈c} w_i`.
fn update_centers<R: Real>(
    centers: &mut [Vec<R>],
    means: &[Vec<R>],
    weights: &[R],
    labels: &[usize],
    dim: usize,
) {
    let k = centers.len();
    let mut sums = vec![vec![R::zero(); dim]; k];
    let mut wsum = vec![R::zero(); k];
    for (i, m) in means.iter().enumerate() {
        let l = labels[i];
        wsum[l] = wsum[l] + weights[i];
        for (s, &v) in sums[l].iter_mut().zip(m) {
            *s = *s + weights[i] * v;
        }
    }
    for (c, ws) in wsum.iter().enumerate() {
        if *ws > R::zero() {
            for d in 0..dim {
                centers[c][d] = sums[c][d] / *ws;
            }
        }
    }

    // Reseed an empty cluster from the worst-served leaf, as the spherical head does. A cluster that
    // loses its last member is otherwise stranded: nothing moves its centre again, it keeps losing the
    // assignment race, and the run silently returns fewer than `k` clusters while reporting success.
    // Measured on 20-newsgroups TF-IDF at `n_clusters=20`: **14 non-empty clusters** and ARI 0.017,
    // against 20 and 0.085 once the stranded centres are restarted. Reseeding on the leaf currently
    // furthest from its own centre is the choice that buys the most inertia per relocation.
    if wsum.iter().any(|w| *w <= R::zero()) {
        let mut served: Vec<R> = means
            .iter()
            .enumerate()
            .map(|(i, m)| weights[i] * sq_euclidean(m, &centers[labels[i]]))
            .collect();
        for c in 0..k {
            if wsum[c] > R::zero() {
                continue;
            }
            let worst = argmax(&served);
            centers[c].copy_from_slice(&means[worst]);
            served[worst] = R::neg_infinity(); // don't reseed two clusters onto the same leaf
        }
    }
}

fn argmax<R: Real>(v: &[R]) -> usize {
    let mut best = 0;
    for (i, x) in v.iter().enumerate().skip(1) {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Hamerly-accelerated **exact** Lloyd: per-point upper/lower distance bounds skip the full centre
/// scan whenever an assignment provably cannot change (triangle inequality). The output is
/// identical to brute Lloyd from the same initialisation — only faster.
fn lloyd_hamerly<R: Real>(
    means: &[Vec<R>],
    weights: &[R],
    ssd: &[R],
    mut centers: Vec<Vec<R>>,
    max_iter: usize,
    dim: usize,
) -> KMeans<R> {
    let n = means.len();
    let k = centers.len();
    let mut labels = vec![0usize; n];
    let mut upper = vec![R::zero(); n]; // upper bound on distance to the assigned centre
    let mut lower = vec![R::zero(); n]; // lower bound on distance to the closest *other* centre
    for (i, m) in means.iter().enumerate() {
        let (a, d1, d2) = nearest_two(m, &centers);
        labels[i] = a;
        upper[i] = d1.sqrt();
        lower[i] = d2.sqrt();
    }

    for _ in 0..max_iter {
        let mut next = centers.clone();
        update_centers(&mut next, means, weights, &labels, dim);
        let drift: Vec<R> = (0..k)
            .map(|c| sq_euclidean(&centers[c], &next[c]).sqrt())
            .collect();
        let max_drift = drift.iter().copied().fold(R::zero(), R::max);
        centers = next;

        let mut changed = false;
        for i in 0..n {
            upper[i] = upper[i] + drift[labels[i]];
            lower[i] = lower[i] - max_drift;
            if upper[i] <= lower[i] {
                continue; // assignment provably unchanged
            }
            upper[i] = sq_euclidean(&means[i], &centers[labels[i]]).sqrt(); // tighten then recheck
            if upper[i] <= lower[i] {
                continue;
            }
            let (a, d1, d2) = nearest_two(&means[i], &centers);
            if a != labels[i] {
                labels[i] = a;
                changed = true;
            }
            upper[i] = d1.sqrt();
            lower[i] = d2.sqrt();
        }
        if !changed {
            break;
        }
    }

    let mut inertia = R::zero();
    for (i, m) in means.iter().enumerate() {
        inertia = inertia + ssd[i] + weights[i] * sq_euclidean(m, &centers[labels[i]]);
    }
    KMeans {
        labels,
        centers,
        inertia,
    }
}

/// X-means: choose `k` automatically in `[k_min, k_max]` by repeatedly running k-means and keeping
/// the model with the best BIC (lower is better). BIC over the leaf features treats each as a
/// weighted point of a spherical Gaussian mixture; `p = k·(d+1)` free parameters.
pub fn xmeans<R: Real, C: ClusterFeature<R>>(
    features: &[C],
    k_min: usize,
    k_max: usize,
    max_iter: usize,
    seed: u64,
) -> KMeans<R> {
    let m = features.len();
    let d = features[0].dim();
    let hi = k_max.min(m).max(1);
    let lo = k_min.max(1).min(hi);
    let nr = features
        .iter()
        .map(|f| f.weight())
        .fold(R::zero(), |a, b| a + b);
    let dr = R::from_usize(d).unwrap();
    let half = R::from_f64(0.5).unwrap();
    let two_pi = R::from_f64(std::f64::consts::TAU).unwrap();
    let tiny = R::from_f64(1e-12).unwrap();

    let mut best: Option<KMeans<R>> = None;
    let mut best_bic = R::neg_infinity();
    for k in lo..=hi {
        let km = kmeans(features, k, max_iter, 4, seed);
        // Cluster weights and pure between-feature SSE (the within-feature spread is fixed in k).
        let mut nk = vec![R::zero(); k];
        let mut sse = R::zero();
        for (i, f) in features.iter().enumerate() {
            let c = km.labels[i];
            nk[c] = nk[c] + f.weight();
            sse = sse + f.weight() * sq_euclidean(f.mean(), &km.centers[c]);
        }
        // Pelleg–Moore X-means BIC (maximise): the `Σ n_k ln n_k` entropy term penalises splitting,
        // which a plain inertia-based score lacks.
        let var = (sse / (nr - R::from_usize(k).unwrap()).max(R::one()) / dr).max(tiny);
        let log_2pi_var = (two_pi * var).ln();
        let mut loglik = R::zero();
        for &n_k in &nk {
            if n_k > R::zero() {
                loglik = loglik + n_k * n_k.ln()
                    - n_k * nr.ln()
                    - half * n_k * dr * log_2pi_var
                    - half * (n_k - R::one()) * dr;
            }
        }
        let params = R::from_usize(k * (d + 1)).unwrap();
        let bic = loglik - half * params * nr.ln();
        if bic > best_bic {
            best_bic = bic;
            best = Some(km);
        }
    }
    best.expect("at least one k")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs, grid_micros};
    use crate::feature::{ClusterFeature, Spherical};

    #[test]
    fn kmeans_recovers_separated_blobs() {
        let mut rng = SplitMix64::new(42);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.6);
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let km = kmeans(&micros, 4, 100, 4, 7);
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| km.labels[m]).collect();
        let score = ari(&labels, &truth);
        assert!(score > 0.95, "ARI = {score}");
    }

    #[test]
    fn hamerly_equals_brute_lloyd() {
        let mut rng = SplitMix64::new(123);
        let centers = [[0.0, 0.0], [6.0, 1.0], [2.0, 8.0], [9.0, 9.0], [4.0, 4.0]];
        let (pts, _) = blobs(&mut rng, 200, &centers, 1.5);
        let (micros, _) = grid_micros(&pts, 0.4);
        let means: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let weights: Vec<f64> = micros.iter().map(|f| f.weight()).collect();
        let ssd: Vec<f64> = micros.iter().map(|f| f.ssd()).collect();
        let mut r = SplitMix64::new(7);
        let init = kmeans_plus_plus(&means, &weights, 5, &mut r);
        let brute = lloyd(&means, &weights, &ssd, init.clone(), 100, 2);
        let fast = lloyd_hamerly(&means, &weights, &ssd, init, 100, 2);
        assert_eq!(
            brute.labels, fast.labels,
            "Hamerly diverged from brute Lloyd"
        );
        assert!((brute.inertia - fast.inertia).abs() < 1e-9);
    }

    #[test]
    fn xmeans_recovers_cluster_count() {
        let mut rng = SplitMix64::new(31);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.6);
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let km = xmeans(&micros, 1, 8, 100, 7);
        assert_eq!(km.centers.len(), 4, "selected k = {}", km.centers.len());
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| km.labels[m]).collect();
        assert!(ari(&labels, &truth) > 0.95);
    }

    fn feats(means: &[[f64; 2]]) -> Vec<crate::feature::Spherical<f64>> {
        means
            .iter()
            .map(|m| {
                let mut f = crate::feature::Spherical::new(2);
                f.push(m, 1.0);
                f
            })
            .collect()
    }

    #[test]
    fn cop_kmeans_unconstrained_recovers_blobs() {
        let mut rng = SplitMix64::new(42);
        let centers = [[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]];
        let (pts, truth) = blobs(&mut rng, 400, &centers, 0.6);
        let (micros, point_to_micro) = grid_micros(&pts, 0.5);
        let lab = cop_kmeans(&micros, 4, &[], &[], 100, 4, 7).expect("feasible");
        let labels: Vec<usize> = point_to_micro.iter().map(|&m| lab[m]).collect();
        assert!(
            ari(&labels, &truth) > 0.95,
            "ARI = {}",
            ari(&labels, &truth)
        );
    }

    #[test]
    fn cop_kmeans_must_link_groups_features() {
        // Two tight pairs far apart; must-link one feature from each pair forces them to share a
        // cluster even though geometry puts them in different ones.
        let f = feats(&[[0.0, 0.0], [0.2, 0.0], [10.0, 0.0], [10.2, 0.0]]);
        let lab = cop_kmeans(&f, 2, &[(0, 2)], &[], 100, 4, 1).expect("feasible");
        assert_eq!(lab[0], lab[2], "must-link not honoured");
    }

    #[test]
    fn cop_kmeans_cannot_link_separates_features() {
        // Two near-coincident features that k-means would merge; cannot-link forces them apart.
        let f = feats(&[[0.0, 0.0], [0.2, 0.0], [10.0, 0.0]]);
        let plain = cop_kmeans(&f, 2, &[], &[], 100, 4, 1).expect("feasible");
        assert_eq!(
            plain[0], plain[1],
            "without constraints the close pair merges"
        );
        let lab = cop_kmeans(&f, 2, &[], &[(0, 1)], 100, 4, 1).expect("feasible");
        assert_ne!(lab[0], lab[1], "cannot-link not honoured");
    }

    #[test]
    fn cop_kmeans_contradiction_is_reported() {
        let f = feats(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        let err = cop_kmeans(&f, 2, &[(0, 1)], &[(0, 1)], 100, 4, 1).unwrap_err();
        assert_eq!(err, ConstraintError::Contradiction);
    }

    #[test]
    fn cop_kmeans_infeasible_when_too_few_clusters() {
        // Three mutually cannot-linked features need three clusters; k = 2 cannot satisfy them.
        let f = feats(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        let err = cop_kmeans(&f, 2, &[], &[(0, 1), (0, 2), (1, 2)], 100, 4, 1).unwrap_err();
        assert_eq!(err, ConstraintError::Infeasible);
    }

    #[test]
    fn cop_kmeans_ignores_out_of_range_pairs() {
        // The core tolerates out-of-range indices (the Python layer validates row indices); they are
        // skipped, so a single feasible labelling is still produced.
        let f = feats(&[[0.0, 0.0], [0.2, 0.0], [10.0, 0.0]]);
        let lab = cop_kmeans(&f, 2, &[(0, 99)], &[(1, 99)], 100, 4, 1).expect("feasible");
        assert_eq!(lab.len(), 3);
    }

    /// Independent re-derivation of the Pelleg–Moore BIC that [`xmeans`] maximises. The score itself
    /// is never returned, so the only end-to-end test asserts the selected `k` on four separated
    /// blobs — a target so easy that most of the score can be corrupted without moving it.
    fn reference_xmeans_bic(features: &[Spherical<f64>], km: &KMeans<f64>, k: usize) -> f64 {
        let d = features[0].dim();
        let nr: f64 = features.iter().map(|f| f.weight()).sum();
        let mut nk = vec![0.0; k];
        let mut sse = 0.0;
        for (i, f) in features.iter().enumerate() {
            let c = km.labels[i];
            nk[c] += f.weight();
            sse += f.weight() * sq_euclidean(f.mean(), &km.centers[c]);
        }
        let var = (sse / (nr - k as f64).max(1.0) / d as f64).max(1e-12);
        let log_2pi_var = (std::f64::consts::TAU * var).ln();
        let loglik: f64 = nk
            .iter()
            .filter(|&&n_k| n_k > 0.0)
            .map(|&n_k| {
                n_k * n_k.ln()
                    - n_k * nr.ln()
                    - 0.5 * n_k * d as f64 * log_2pi_var
                    - 0.5 * (n_k - 1.0) * d as f64
            })
            .sum();
        loglik - 0.5 * (k * (d + 1)) as f64 * nr.ln()
    }

    #[test]
    fn xmeans_selects_the_argmax_of_an_independently_scored_bic() {
        // Three fixtures with different true k, plus a single homogeneous cloud where the entropy
        // and parameter penalties are the only thing stopping the score from splitting for ever.
        let fixtures: [(&[[f64; 2]], f64); 4] = [
            (&[[0.0, 0.0], [9.0, 0.0], [0.0, 9.0], [9.0, 9.0]], 0.6),
            (&[[0.0, 0.0], [7.0, 1.0]], 0.8),
            (&[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]], 1.0),
            (&[[0.0, 0.0]], 1.5),
        ];
        for (f, (centers, spread)) in fixtures.iter().enumerate() {
            for seed in [7u64, 23, 91] {
                let mut rng = SplitMix64::new(seed);
                let (pts, _) = blobs(&mut rng, 120, centers, *spread);
                let (micros, _) = grid_micros(&pts, 0.5);
                let (lo, hi) = (1usize, 6usize);

                let mut want = lo;
                let mut best = f64::NEG_INFINITY;
                for k in lo..=hi {
                    let km = kmeans(&micros, k, 100, 4, seed);
                    let bic = reference_xmeans_bic(&micros, &km, k);
                    if bic > best {
                        best = bic;
                        want = k;
                    }
                }
                let got = xmeans(&micros, lo, hi, 100, seed).centers.len();
                assert_eq!(got, want, "fixture {f}, seed {seed}");
            }
        }
    }

    #[test]
    fn update_centers_takes_the_weighted_centroid() {
        let mut centers: Vec<Vec<f64>> = vec![vec![0.0; 2], vec![0.0; 2]];
        let means = vec![vec![0.0, 0.0], vec![4.0, 2.0], vec![10.0, 10.0]];
        let weights = vec![1.0, 3.0, 2.0];
        update_centers(&mut centers, &means, &weights, &[0, 0, 1], 2);
        // (1·[0,0] + 3·[4,2]) / 4 = [3, 1.5]; the singleton keeps its own mean.
        assert_eq!(centers[0], vec![3.0, 1.5]);
        assert_eq!(centers[1], vec![10.0, 10.0]);
    }

    #[test]
    fn update_centers_reseeds_every_stranded_cluster_on_a_distinct_leaf() {
        let mut centers: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![99.0, 0.0], vec![-99.0, 0.0]];
        let means = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![7.0, 0.0]];
        let weights = vec![1.0, 1.0, 1.0];
        update_centers(&mut centers, &means, &weights, &[0, 0, 0], 2);
        // centre 0 becomes the centroid 8/3; served = [(8/3)², (1-8/3)², (7-8/3)²] = [7.11, 2.78,
        // 18.78], so the worst-served leaf is index 2 and the next is index 0. Reseeding both empty
        // clusters onto index 2 would leave the run with a duplicate centre and one cluster short.
        assert!((centers[0][0] - 8.0 / 3.0).abs() < 1e-12);
        assert_eq!(centers[1], vec![7.0, 0.0]);
        assert_eq!(centers[2], vec![0.0, 0.0]);
    }

    #[test]
    fn weighted_pick_is_proportional_and_never_leaves_the_slice() {
        let mut rng = SplitMix64::new(5);
        let mut hits = [0usize; 3];
        for _ in 0..6000 {
            hits[weighted_pick(&[1.0, 0.0, 3.0], &mut rng)] += 1;
        }
        assert_eq!(hits[1], 0, "a zero-probability entry was drawn");
        let ratio = hits[2] as f64 / hits[0] as f64;
        assert!((2.6..3.4).contains(&ratio), "0:2 ratio {ratio}, want 3");

        // Degenerate total falls back to a uniform draw, which must still be an index.
        let mut seen = [false; 3];
        for _ in 0..300 {
            let i = weighted_pick(&[0.0, 0.0, 0.0], &mut rng);
            assert!(i < 3, "index {i} out of range");
            seen[i] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "uniform fallback never reached some index"
        );
    }

    #[test]
    fn kmeans_plus_plus_spreads_one_seed_per_far_group() {
        // D²-proportional sampling is the whole point of k-means++: with groups 100 units apart the
        // probability of seeding two centres in one group is negligible, so a corrupted D² update or
        // a corrupted candidate score shows up as a duplicated group.
        let groups = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]];
        let means: Vec<Vec<f64>> = groups
            .iter()
            .flat_map(|g| (0..8).map(move |j| vec![g[0] + j as f64 * 0.1, g[1]]))
            .collect();
        let weights = vec![1.0; means.len()];
        for seed in 0..24u64 {
            let mut rng = SplitMix64::new(seed);
            let centers = kmeans_plus_plus(&means, &weights, 3, &mut rng);
            let mut hit = [false; 3];
            for c in &centers {
                let g = groups
                    .iter()
                    .position(|g| sq_euclidean(c, &[g[0], g[1]]) < 100.0)
                    .expect("centre landed outside every group");
                assert!(!hit[g], "seed {seed} put two centres in group {g}");
                hit[g] = true;
            }
        }
    }

    #[test]
    fn nearest_two_keeps_the_first_of_tied_nearest_centres() {
        // Ties matter: `<` vs `<=` on the nearest test silently changes which cluster a leaf joins.
        let (best, d1, d2) = nearest_two(&[0.0], &[vec![-1.0], vec![5.0], vec![1.0]]);
        assert_eq!(best, 0);
        assert_eq!((d1, d2), (1.0, 1.0));
        let (best, d1, d2) = nearest_two(&[0.0], &[vec![4.0], vec![1.0], vec![-2.0]]);
        assert_eq!(best, 1);
        assert_eq!((d1, d2), (1.0, 4.0));
    }

    #[test]
    fn argmax_keeps_the_first_of_equal_scores() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[3.0, 3.0, 1.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0]), 0);
    }

    #[test]
    fn more_restarts_never_return_a_worse_partition() {
        // One rng stream feeds the restarts in order, so `n_init = m + 1` sees exactly the inits
        // `n_init = m` saw plus one more: keeping the best of them can only lower the inertia.
        let mut rng = SplitMix64::new(31);
        let centers = [[0.0, 0.0], [3.0, 0.4], [1.2, 3.1], [4.5, 3.6]];
        let (pts, _truth) = blobs(&mut rng, 60, &centers, 1.1);
        let (micros, _assign) = grid_micros(&pts, 0.55);
        let mut prev = f64::INFINITY;
        let mut improvements = 0;
        for n_init in 1..=10 {
            let got = kmeans(&micros, 4, 100, n_init, 5).inertia;
            assert!(got <= prev + 1e-9, "n_init = {n_init}: {got} > {prev}");
            if got < prev - 1e-9 {
                improvements += 1;
            }
            prev = got;
        }
        assert!(
            improvements > 1,
            "every restart found the same optimum; the fixture cannot see the choice"
        );
    }

    #[test]
    fn a_constraint_index_equal_to_the_feature_count_is_out_of_range() {
        // `n` itself is the first invalid id, and it is the only one a bound that admits it can
        // reach: the union-find has exactly `n` slots.
        let f = feats(&[[0.0, 0.0], [0.2, 0.0], [10.0, 0.0]]);
        let lab = cop_kmeans(&f, 2, &[(0, 3), (3, 1)], &[(2, 3)], 100, 4, 1).expect("feasible");
        assert_eq!(lab[0], lab[1], "the near pair was split");
        assert_ne!(lab[0], lab[2], "the far feature joined the near pair");
    }

    /// Independent re-derivation of the SSE [`cop_kmeans`] minimises across restarts,
    /// `Σ_i [S_i + w_i ‖μ_i − c_{a(i)}‖²]`, evaluated at the mass-weighted centroid of each label.
    fn reference_sse(features: &[Spherical<f64>], labels: &[usize]) -> f64 {
        let k = labels.iter().max().map_or(0, |&m| m + 1);
        let dim = features[0].dim();
        let mut wsum = vec![0.0; k];
        let mut csum = vec![vec![0.0; dim]; k];
        for (f, &l) in features.iter().zip(labels) {
            wsum[l] += f.weight();
            for (s, &m) in csum[l].iter_mut().zip(f.mean()) {
                *s += f.weight() * m;
            }
        }
        for (c, &w) in csum.iter_mut().zip(&wsum) {
            if w > 0.0 {
                for s in c.iter_mut() {
                    *s /= w;
                }
            }
        }
        features
            .iter()
            .zip(labels)
            .map(|(f, &l)| f.ssd() + f.weight() * sq_euclidean(f.mean(), &csum[l]))
            .sum()
    }

    /// One constrained assignment step of COP-KMeans, re-derived from Wagstaff et al.: heaviest
    /// chunklet first, then the nearest centre that no already-placed cannot-link partner occupies.
    /// With no must-links each feature is its own chunklet, so it indexes `features` directly.
    fn constrained_step(
        features: &[Spherical<f64>],
        cannot: &[(usize, usize)],
        centers: &[Vec<f64>],
    ) -> Vec<usize> {
        let n = features.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            features[j]
                .weight()
                .partial_cmp(&features[i].weight())
                .unwrap()
                .then(i.cmp(&j))
        });
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); centers.len()];
        let mut assign = vec![usize::MAX; n];
        for &i in &order {
            let mut cand: Vec<(f64, usize)> = centers
                .iter()
                .enumerate()
                .map(|(c, ctr)| (sq_euclidean(features[i].mean(), ctr), c))
                .collect();
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
            let c = cand
                .into_iter()
                .map(|(_, c)| c)
                .find(|&c| {
                    !members[c]
                        .iter()
                        .any(|&m| cannot.contains(&(i, m)) || cannot.contains(&(m, i)))
                })
                .expect("a feasible centre");
            assign[i] = c;
            members[c].push(i);
        }
        assign
    }

    /// Mass-weighted centroid per label, in label order.
    fn centroids(features: &[Spherical<f64>], labels: &[usize], k: usize) -> Vec<Vec<f64>> {
        let dim = features[0].dim();
        let mut wsum = vec![0.0; k];
        let mut csum = vec![vec![0.0; dim]; k];
        for (f, &l) in features.iter().zip(labels) {
            wsum[l] += f.weight();
            for (s, &m) in csum[l].iter_mut().zip(f.mean()) {
                *s += f.weight() * m;
            }
        }
        for (c, &w) in csum.iter_mut().zip(&wsum) {
            if w > 0.0 {
                for s in c.iter_mut() {
                    *s /= w;
                }
            }
        }
        csum
    }

    #[test]
    fn cop_kmeans_returns_a_constrained_fixed_point() {
        // Stopping one step early still satisfies every constraint, so only a fixed-point test
        // sees it: re-running the assignment step from the returned centres must change nothing.
        let mut rng = SplitMix64::new(19);
        let centers = [[0.0, 0.0], [2.6, 0.5], [1.0, 2.8], [3.8, 3.2]];
        let (pts, _truth) = blobs(&mut rng, 50, &centers, 1.0);
        let (micros, _assign) = grid_micros(&pts, 0.6);
        let cannot = [(0usize, 1usize), (2, 3)];
        let lab = cop_kmeans(&micros, 4, &[], &cannot, 100, 6, 3).expect("feasible");
        for &(a, b) in &cannot {
            assert_ne!(lab[a], lab[b], "cannot-link ({a}, {b}) was violated");
        }
        let ctrs = centroids(&micros, &lab, 4);
        assert_eq!(
            constrained_step(&micros, &cannot, &ctrs),
            lab,
            "the returned labelling is not a fixed point of the assignment step"
        );
    }

    #[test]
    fn cop_kmeans_keeps_the_restart_with_the_lowest_independently_scored_sse() {
        let mut rng = SplitMix64::new(23);
        let centers = [[0.0, 0.0], [2.4, 0.6], [1.1, 2.7]];
        let (pts, _truth) = blobs(&mut rng, 50, &centers, 1.0);
        let (micros, _assign) = grid_micros(&pts, 0.6);
        let cannot = [(0usize, 1usize)];
        let mut prev = f64::INFINITY;
        let mut improvements = 0;
        for n_init in 1..=12 {
            let lab = cop_kmeans(&micros, 3, &[], &cannot, 100, n_init, 11).expect("feasible");
            let sse = reference_sse(&micros, &lab);
            assert!(sse <= prev + 1e-9, "n_init = {n_init}: {sse} > {prev}");
            if sse < prev - 1e-9 {
                improvements += 1;
            }
            prev = sse;
        }
        assert!(
            improvements > 1,
            "every restart scored the same; the fixture cannot see the choice"
        );
    }

    /// Greedy k-means++ re-derived from Arthur–Vassilvitskii plus scikit-learn's greedy variant:
    /// the first centre is drawn ∝ weight, then each further centre is the best of `2 + ⌊ln k⌋`
    /// candidates drawn ∝ weight·D², scored by the potential `Σ_i w_i · min(‖x_i − cand‖², D²_i)`
    /// it would leave behind. It shares [`weighted_pick`], so it consumes the same rng stream.
    fn reference_kpp(
        means: &[Vec<f64>],
        weights: &[f64],
        k: usize,
        rng: &mut SplitMix64,
    ) -> Vec<Vec<f64>> {
        let mut centers = vec![means[weighted_pick(weights, rng)].clone()];
        let mut d2: Vec<f64> = means.iter().map(|m| sq_euclidean(m, &centers[0])).collect();
        let n_trials = 2 + (k as f64).ln().floor().max(0.0) as usize;
        while centers.len() < k {
            let probs: Vec<f64> = weights.iter().zip(&d2).map(|(&w, &d)| w * d).collect();
            let mut best = usize::MAX;
            let mut best_pot = f64::INFINITY;
            for _ in 0..n_trials {
                let cand = weighted_pick(&probs, rng);
                let pot: f64 = means
                    .iter()
                    .zip(weights)
                    .zip(&d2)
                    .map(|((m, &w), &d)| w * sq_euclidean(m, &means[cand]).min(d))
                    .sum();
                if pot < best_pot {
                    best_pot = pot;
                    best = cand;
                }
            }
            for (di, m) in d2.iter_mut().zip(means) {
                *di = di.min(sq_euclidean(m, &means[best]));
            }
            centers.push(means[best].clone());
        }
        centers
    }

    #[test]
    fn kmeans_plus_plus_matches_the_greedy_reference_draw_for_draw() {
        // The seeds it returns are the only thing downstream sees, and on separated data every
        // sane sampler lands on the same ones — so the fixture is deliberately unseparated, and
        // the assertion is on the exact sequence rather than on the partition it leads to.
        let mut rng = SplitMix64::new(77);
        let centers = [[0.0, 0.0], [1.8, 0.7], [0.9, 2.0], [3.0, 2.4]];
        let (pts, _truth) = blobs(&mut rng, 40, &centers, 1.2);
        let (micros, _assign) = grid_micros(&pts, 0.5);
        let means: Vec<Vec<f64>> = micros.iter().map(|f| f.mean().to_vec()).collect();
        let weights: Vec<f64> = micros.iter().map(|f| f.weight()).collect();
        for k in [2usize, 4, 7] {
            let mut a = SplitMix64::new(2024);
            let mut b = SplitMix64::new(2024);
            assert_eq!(
                kmeans_plus_plus(&means, &weights, k, &mut a),
                reference_kpp(&means, &weights, k, &mut b),
                "k = {k}"
            );
            assert_eq!(a.next_u64(), b.next_u64(), "k = {k}: rng streams diverged");
        }
    }
}
