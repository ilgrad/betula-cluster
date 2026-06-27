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

    let mut d2: Vec<f64> = means
        .iter()
        .map(|m| to_f(sq_euclidean(m, &centers[0])))
        .collect();
    while centers.len() < k {
        let probs: Vec<f64> = w0.iter().zip(&d2).map(|(&w, &d)| w * d).collect();
        let next = means[weighted_pick(&probs, rng)].clone();
        for (di, m) in d2.iter_mut().zip(means) {
            let nd = to_f(sq_euclidean(m, &next));
            if nd < *di {
                *di = nd;
            }
        }
        centers.push(next);
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
    use crate::feature::ClusterFeature;

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
}
