//! Density-based clustering over an evolving data stream with noise: [`DenStream`] (Cao et al., SDM
//! 2006) and [`DbStream`] (Hahsler & Bolaños, TKDE 2016).
//!
//! Cao, Ester, Qian & Zhou, *SDM 2006*, specialised to BETULA stable CFs. Two pools of fading
//! micro-clusters are kept: **potential** (`p`, dense enough to seed a cluster) and **outlier**
//! (`o`, a buffer). Each micro-cluster's weight decays as `2^(-λ·Δt)` in stream time (one tick per
//! point); a micro-cluster is a CF feature, so decay is exact (`decay` scales weight *and* scatter
//! together). A useful consequence: **decay does not move the centroid and does not change the RMS
//! radius** `√(S/w)` — only the weight — so the merge/radius test is decay-invariant and only the
//! weight comparisons (promotion, pruning) need the fading factor.
//!
//! Online (per point `x` at tick `t`): merge into the nearest `p` micro-cluster if the result stays
//! within radius `ε`; else into the nearest `o` micro-cluster (promoting it to `p` once its weight
//! reaches `β·μ`); else open a new `o` micro-cluster. Every `Tp` ticks, prune faded micro-clusters.
//! Offline ([`DenStream::cluster`]): connected components of the `p` micro-clusters within `2ε`,
//! keeping components of total weight ≥ `μ` as clusters (others → noise `-1`).
//!
//! Working precision for the weight/decay arithmetic is `f64`; geometry stays in the tree's `R`.

use crate::feature::ClusterFeature;
use crate::kernels::sq_euclidean;
use crate::types::Real;
use std::collections::HashMap;
use std::marker::PhantomData;

/// A fading micro-cluster: a CF plus the tick of its last update and of its creation.
struct Micro<C> {
    cf: C,
    last_t: f64,
    t0: f64,
}

/// Union–find with path halving + union by rank (offline connected components of `p` micro-clusters).
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

/// A DenStream clusterer over CF feature `C`.
pub struct DenStream<R: Real, C: ClusterFeature<R>> {
    dim: usize,
    eps2: f64,    // ε² — radius² and (via 4ε²) the offline connection threshold
    lambda: f64,  // decay rate λ > 0
    beta_mu: f64, // β·μ — o→p promotion weight and p-micro prune floor (> 1)
    mu: f64,      // μ — minimum cluster weight (offline)
    tp: f64,      // prune interval in ticks
    t: f64,       // current stream time (points seen)
    p: Vec<Micro<C>>,
    o: Vec<Micro<C>>,
    labels: Vec<i64>, // offline cluster label per p-micro-cluster (set by `cluster`)
    _r: PhantomData<R>,
}

/// Fading factor `2^(-λ·(t − last_t))` for a micro-cluster last updated at `last_t`.
fn fade(lambda: f64, t: f64, last_t: f64) -> f64 {
    (-lambda * (t - last_t)).exp2()
}

impl<R: Real, C: ClusterFeature<R>> DenStream<R, C> {
    /// New clusterer. `eps` is the micro-cluster radius, `lambda` the decay rate, `beta` the outlier
    /// fraction and `mu` the minimum weight; `beta*mu` (the promotion threshold) must exceed 1.
    pub fn new(
        dim: usize,
        eps: f64,
        lambda: f64,
        beta: f64,
        mu: f64,
    ) -> Result<Self, &'static str> {
        if dim == 0 {
            return Err("dim must be > 0");
        }
        if eps.is_nan() || eps <= 0.0 {
            return Err("eps must be > 0");
        }
        if lambda.is_nan() || lambda <= 0.0 {
            return Err("lambda must be > 0");
        }
        let beta_mu = beta * mu;
        if beta_mu.is_nan() || beta_mu <= 1.0 {
            return Err("beta * mu must be > 1");
        }
        // Recommended prune interval (paper §4): Tp = ⌈(1/λ)·log2(βμ / (βμ − 1))⌉.
        let tp = ((1.0 / lambda) * (beta_mu / (beta_mu - 1.0)).log2())
            .ceil()
            .max(1.0);
        Ok(Self {
            dim,
            eps2: eps * eps,
            lambda,
            beta_mu,
            mu,
            tp,
            t: 0.0,
            p: Vec::new(),
            o: Vec::new(),
            labels: Vec::new(),
            _r: PhantomData,
        })
    }

    /// Point dimensionality fixed at construction.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of potential (cluster-eligible) micro-clusters.
    pub fn potential_count(&self) -> usize {
        self.p.len()
    }

    /// Offline cluster label per potential micro-cluster (empty until [`DenStream::cluster`]).
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }

    /// Number of distinct (non-noise) clusters from the last [`DenStream::cluster`].
    pub fn n_clusters(&self) -> usize {
        let mut v: Vec<i64> = self.labels.iter().copied().filter(|&l| l >= 0).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Index of the nearest micro-cluster in `pool` to `x` by centroid distance (`None` if empty).
    fn nearest(pool: &[Micro<C>], x: &[R]) -> Option<usize> {
        let mut best: Option<(usize, R)> = None;
        for (i, m) in pool.iter().enumerate() {
            let d = sq_euclidean(m.cf.mean(), x);
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Try to fold `x` into `pool[i]`: fade it to the current tick, add `x`, and accept only if the
    /// RMS radius stays within `ε`. On success the micro-cluster is updated in place. Returns success.
    fn try_merge(pool: &mut [Micro<C>], i: usize, x: &[R], t: f64, lambda: f64, eps2: f64) -> bool {
        let mut cf = pool[i].cf.clone();
        cf.decay(R::from_f64(fade(lambda, t, pool[i].last_t)).unwrap());
        cf.push(x, R::one());
        let w = cf.weight().to_f64().unwrap();
        let ssd = cf.ssd().to_f64().unwrap();
        if w > 0.0 && ssd / w <= eps2 {
            pool[i].cf = cf;
            pool[i].last_t = t;
            true
        } else {
            false
        }
    }

    /// Absorb one point.
    pub fn insert(&mut self, x: &[R]) {
        debug_assert!(x.len() >= self.dim);
        self.labels.clear(); // new data invalidates the offline clustering
        let (t, lambda, eps2) = (self.t, self.lambda, self.eps2);

        if let Some(i) = Self::nearest(&self.p, x) {
            if Self::try_merge(&mut self.p, i, x, t, lambda, eps2) {
                self.tick();
                return;
            }
        }
        if let Some(i) = Self::nearest(&self.o, x) {
            if Self::try_merge(&mut self.o, i, x, t, lambda, eps2) {
                // The merged o-micro is current (faded to `t`); promote once it is dense enough.
                if self.o[i].cf.weight().to_f64().unwrap() >= self.beta_mu {
                    let m = self.o.remove(i);
                    self.p.push(m);
                }
                self.tick();
                return;
            }
        }
        let mut cf = C::new(self.dim);
        cf.push(x, R::one());
        self.o.push(Micro {
            cf,
            last_t: t,
            t0: t,
        });
        self.tick();
    }

    /// Advance the clock and prune every `Tp` ticks.
    fn tick(&mut self) {
        self.t += 1.0;
        let period = self.tp as u64;
        if period > 0 && (self.t as u64).is_multiple_of(period) {
            self.prune();
        }
    }

    /// Drop faded micro-clusters: potential ones below `β·μ`, outliers below the time-decaying
    /// threshold `ξ(t,t0) = (2^(-λ(t−t0+Tp)) − 1)/(2^(-λTp) − 1)` (paper §4).
    fn prune(&mut self) {
        let (t, lambda, tp, beta_mu) = (self.t, self.lambda, self.tp, self.beta_mu);
        self.p
            .retain(|m| m.cf.weight().to_f64().unwrap() * fade(lambda, t, m.last_t) >= beta_mu);
        let denom = (-lambda * tp).exp2() - 1.0;
        self.o.retain(|m| {
            let w = m.cf.weight().to_f64().unwrap() * fade(lambda, t, m.last_t);
            let xi = ((-lambda * (t - m.t0 + tp)).exp2() - 1.0) / denom;
            w >= xi
        });
    }

    /// Offline step: label potential micro-clusters by connected components within `2ε`, keeping
    /// components whose total (faded) weight is ≥ `μ` (others become noise `-1`).
    pub fn cluster(&mut self) {
        let np = self.p.len();
        let r2 = 4.0 * self.eps2;
        let mut uf = UnionFind::new(np);
        for i in 0..np {
            for j in (i + 1)..np {
                let d = sq_euclidean(self.p[i].cf.mean(), self.p[j].cf.mean())
                    .to_f64()
                    .unwrap();
                if d <= r2 {
                    uf.union(i, j);
                }
            }
        }
        let (t, lambda) = (self.t, self.lambda);
        let mut roots = vec![0usize; np];
        let mut comp_weight: HashMap<usize, f64> = HashMap::new();
        for (i, root) in roots.iter_mut().enumerate() {
            *root = uf.find(i);
            let w = self.p[i].cf.weight().to_f64().unwrap() * fade(lambda, t, self.p[i].last_t);
            *comp_weight.entry(*root).or_insert(0.0) += w;
        }
        let mut label_of: HashMap<usize, i64> = HashMap::new();
        let mut next = 0i64;
        self.labels = (0..np)
            .map(|i| {
                if comp_weight[&roots[i]] >= self.mu {
                    *label_of.entry(roots[i]).or_insert_with(|| {
                        let l = next;
                        next += 1;
                        l
                    })
                } else {
                    -1
                }
            })
            .collect();
    }

    /// Cluster label of `x`: the label of its nearest potential micro-cluster if `x` is within `ε`
    /// of it, else `-1` (noise). Requires a prior [`DenStream::cluster`].
    pub fn predict(&self, x: &[R]) -> i64 {
        if self.labels.len() != self.p.len() {
            return -1;
        }
        match Self::nearest(&self.p, x) {
            Some(i) => {
                let d = sq_euclidean(self.p[i].cf.mean(), x).to_f64().unwrap();
                if d <= self.eps2 {
                    self.labels[i]
                } else {
                    -1
                }
            }
            None => -1,
        }
    }

    /// Per-potential-micro-cluster `(centers_flat, weights, radii, dim)` in `f64` (weights faded to
    /// the current tick; centroids and radii are decay-invariant).
    pub fn potential_stats(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let (t, lambda) = (self.t, self.lambda);
        let mut centers = Vec::with_capacity(self.p.len() * self.dim);
        let mut weights = Vec::with_capacity(self.p.len());
        let mut radii = Vec::with_capacity(self.p.len());
        for m in &self.p {
            for &v in m.cf.mean() {
                centers.push(v.to_f64().unwrap());
            }
            let raw_w = m.cf.weight().to_f64().unwrap();
            weights.push(raw_w * fade(lambda, t, m.last_t));
            let ssd = m.cf.ssd().to_f64().unwrap();
            radii.push(if raw_w > 0.0 {
                (ssd / raw_w).sqrt()
            } else {
                0.0
            });
        }
        (centers, weights, radii, self.dim)
    }
}

// ───────────────────────────────── DBSTREAM (shared density) ─────────────────────────────────────

/// A DBSTREAM micro-cluster: a fading CF with a stable id (so shared-density pair keys survive
/// cleanup, which reorders the backing `Vec`).
struct DbMicro<C> {
    cf: C,
    last_t: f64,
    id: u64,
}

/// Shared density between a pair of micro-clusters: the faded count of points seen within `r` of
/// *both*, plus the tick it was last updated.
struct Shared {
    value: f64,
    last_t: f64,
}

/// **DBSTREAM** (Hahsler & Bolaños, TKDE 2016) over BETULA stable CFs.
///
/// Unlike [`DenStream`], which connects micro-clusters that are merely *close* (within `2ε`),
/// DBSTREAM connects them by **shared density** — the mass of points that fall within radius `r` of
/// *both* — so two dense regions separated by an empty gap are kept apart even when their centres are
/// close, and a single arbitrarily-shaped cluster is recovered as a chain of overlapping
/// micro-clusters. Online (per point `x`): every micro-cluster whose centre is within `r` absorbs `x`
/// (fixed-radius multi-assignment, density-style); if none does, a new one is seeded. Each *pair* of
/// co-absorbing micro-clusters has its shared density incremented. Weights and shared densities fade
/// as `2^(-λ·Δt)`; weak ones are pruned every `t_gap` ticks. Offline ([`DbStream::cluster`]):
/// connected components of the *strong* (faded weight ≥ `min_weight`) micro-clusters, joining a pair
/// when its shared density is at least `α` of their average weight.
pub struct DbStream<R: Real, C: ClusterFeature<R>> {
    dim: usize,
    radius2: f64,     // r² — fixed-radius neighbourhood on centroids (squared euclidean)
    lambda: f64,      // decay rate λ > 0
    alpha: f64,       // shared-density bridge threshold (× min_weight) α ∈ (0, 1]
    min_weight: f64,  // faded weight for a micro-cluster to seed/extend a cluster (offline)
    t_gap: f64,       // cleanup interval in ticks
    clean_floor: f64, // faded weight below which a micro-cluster / shared density is pruned
    t: f64,
    next_id: u64,
    micros: Vec<DbMicro<C>>,
    shared: HashMap<(u64, u64), Shared>,
    labels: Vec<i64>,
    _r: PhantomData<R>,
}

impl<R: Real, C: ClusterFeature<R>> DbStream<R, C> {
    /// New clusterer. `r` is the micro-cluster radius, `lambda` the decay rate, `alpha ∈ (0, 1]` the
    /// shared-density bridge threshold (a pair links when their overlap mass ≥ `alpha · min_weight`),
    /// and `min_weight` the faded weight a micro-cluster needs to form a cluster offline.
    pub fn new(
        dim: usize,
        r: f64,
        lambda: f64,
        alpha: f64,
        min_weight: f64,
    ) -> Result<Self, &'static str> {
        if dim == 0 {
            return Err("dim must be > 0");
        }
        if r.is_nan() || r <= 0.0 {
            return Err("r must be > 0");
        }
        if lambda.is_nan() || lambda <= 0.0 {
            return Err("lambda must be > 0");
        }
        if alpha.is_nan() || alpha <= 0.0 || alpha > 1.0 {
            return Err("alpha must be in (0, 1]");
        }
        if min_weight.is_nan() || min_weight <= 0.0 {
            return Err("min_weight must be > 0");
        }
        // Clean every ~1/λ ticks; then a point's weight has faded to ≈ 2^(-1) = 0.5 — the floor below
        // which a micro-cluster holds less than half of one recent point and is treated as noise.
        let t_gap = (1.0 / lambda).ceil().max(1.0);
        let clean_floor = (-lambda * t_gap).exp2();
        Ok(Self {
            dim,
            radius2: r * r,
            lambda,
            alpha,
            min_weight,
            t_gap,
            clean_floor,
            t: 0.0,
            next_id: 0,
            micros: Vec::new(),
            shared: HashMap::new(),
            labels: Vec::new(),
            _r: PhantomData,
        })
    }

    /// Point dimensionality fixed at construction.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of live micro-clusters.
    pub fn micro_count(&self) -> usize {
        self.micros.len()
    }

    /// Offline cluster label per micro-cluster (empty until [`DbStream::cluster`]).
    pub fn labels(&self) -> &[i64] {
        &self.labels
    }

    /// Number of distinct (non-noise) clusters from the last [`DbStream::cluster`].
    pub fn n_clusters(&self) -> usize {
        let mut v: Vec<i64> = self.labels.iter().copied().filter(|&l| l >= 0).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    }

    /// Faded weight of a micro-cluster at the current tick (centroid/radius are decay-invariant).
    fn faded_weight(&self, m: &DbMicro<C>) -> f64 {
        m.cf.weight().to_f64().unwrap() * fade(self.lambda, self.t, m.last_t)
    }

    /// Absorb one point: every micro-cluster within `r` of it absorbs `x` (the centroid is
    /// decay-invariant, so the neighbour test needs no fading); each co-absorbing pair's shared
    /// density is bumped. If no micro-cluster is within `r`, a new one is seeded.
    pub fn insert(&mut self, x: &[R]) {
        debug_assert!(x.len() >= self.dim);
        self.labels.clear();
        let (t, lambda, r2) = (self.t, self.lambda, self.radius2);
        let neigh: Vec<usize> = (0..self.micros.len())
            .filter(|&i| sq_euclidean(self.micros[i].cf.mean(), x).to_f64().unwrap() <= r2)
            .collect();
        if neigh.is_empty() {
            let mut cf = C::new(self.dim);
            cf.push(x, R::one());
            self.micros.push(DbMicro {
                cf,
                last_t: t,
                id: self.next_id,
            });
            self.next_id += 1;
        } else {
            for &i in &neigh {
                let f = fade(lambda, t, self.micros[i].last_t);
                self.micros[i].cf.decay(R::from_f64(f).unwrap());
                self.micros[i].cf.push(x, R::one());
                self.micros[i].last_t = t;
            }
            for (a, &i) in neigh.iter().enumerate() {
                for &j in &neigh[a + 1..] {
                    let key = pair(self.micros[i].id, self.micros[j].id);
                    let s = self.shared.entry(key).or_insert(Shared {
                        value: 0.0,
                        last_t: t,
                    });
                    s.value = s.value * fade(lambda, t, s.last_t) + 1.0;
                    s.last_t = t;
                }
            }
        }
        self.tick();
    }

    fn tick(&mut self) {
        self.t += 1.0;
        let period = self.t_gap as u64;
        if period > 0 && (self.t as u64).is_multiple_of(period) {
            self.cleanup();
        }
    }

    /// Drop micro-clusters whose faded weight has fallen below the noise floor, then any shared
    /// density that has faded out or now references a dropped micro-cluster.
    fn cleanup(&mut self) {
        let (t, lambda, floor) = (self.t, self.lambda, self.clean_floor);
        self.micros
            .retain(|m| m.cf.weight().to_f64().unwrap() * fade(lambda, t, m.last_t) >= floor);
        let live: std::collections::HashSet<u64> = self.micros.iter().map(|m| m.id).collect();
        self.shared.retain(|(a, b), s| {
            live.contains(a) && live.contains(b) && s.value * fade(lambda, t, s.last_t) >= floor
        });
    }

    /// Offline step: connected components of the strong (faded weight ≥ `min_weight`) micro-clusters,
    /// joining a pair when the faded mass in their overlap is at least `α · min_weight` — a density
    /// bridge of at least a fraction `α` of a cluster-seed's worth of points. Weak micro-clusters are
    /// labelled noise (`-1`). The shared density and `min_weight` fade together, so the test is
    /// decay-invariant.
    pub fn cluster(&mut self) {
        let n = self.micros.len();
        let strong: Vec<bool> = (0..n)
            .map(|i| self.faded_weight(&self.micros[i]) >= self.min_weight)
            .collect();
        let id_to_idx: HashMap<u64, usize> = self
            .micros
            .iter()
            .enumerate()
            .map(|(i, m)| (m.id, i))
            .collect();
        let mut uf = UnionFind::new(n);
        let (t, lambda) = (self.t, self.lambda);
        let bridge = self.alpha * self.min_weight;
        for ((a, b), s) in &self.shared {
            let (Some(&i), Some(&j)) = (id_to_idx.get(a), id_to_idx.get(b)) else {
                continue;
            };
            if !strong[i] || !strong[j] {
                continue;
            }
            if s.value * fade(lambda, t, s.last_t) >= bridge {
                uf.union(i, j);
            }
        }
        let mut label_of: HashMap<usize, i64> = HashMap::new();
        let mut next = 0i64;
        self.labels = (0..n)
            .map(|i| {
                if !strong[i] {
                    return -1;
                }
                let root = uf.find(i);
                *label_of.entry(root).or_insert_with(|| {
                    let l = next;
                    next += 1;
                    l
                })
            })
            .collect();
    }

    /// Index of the nearest micro-cluster to `x` (`None` if there are none).
    fn nearest(&self, x: &[R]) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        for (i, m) in self.micros.iter().enumerate() {
            let d = sq_euclidean(m.cf.mean(), x).to_f64().unwrap();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((i, d));
            }
        }
        best.map(|(i, _)| i)
    }

    /// Cluster label of `x`: the label of its nearest micro-cluster if `x` is within `r` of it, else
    /// `-1` (noise). Requires a prior [`DbStream::cluster`].
    pub fn predict(&self, x: &[R]) -> i64 {
        if self.labels.len() != self.micros.len() {
            return -1;
        }
        match self.nearest(x) {
            Some(i)
                if sq_euclidean(self.micros[i].cf.mean(), x).to_f64().unwrap() <= self.radius2 =>
            {
                self.labels[i]
            }
            _ => -1,
        }
    }

    /// Per-micro-cluster `(centers_flat, weights, radii, dim)` in `f64` (weights faded to the current
    /// tick; centroids and radii are decay-invariant).
    pub fn micro_stats(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let mut centers = Vec::with_capacity(self.micros.len() * self.dim);
        let mut weights = Vec::with_capacity(self.micros.len());
        let mut radii = Vec::with_capacity(self.micros.len());
        for m in &self.micros {
            for &v in m.cf.mean() {
                centers.push(v.to_f64().unwrap());
            }
            let raw_w = m.cf.weight().to_f64().unwrap();
            weights.push(raw_w * fade(self.lambda, self.t, m.last_t));
            let ssd = m.cf.ssd().to_f64().unwrap();
            radii.push(if raw_w > 0.0 {
                (ssd / raw_w).sqrt()
            } else {
                0.0
            });
        }
        (centers, weights, radii, self.dim)
    }
}

/// Order a micro-cluster id pair as `(lo, hi)` so shared-density keys are symmetric.
fn pair(a: u64, b: u64) -> (u64, u64) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::clustering::testutil::{ari, blobs};
    use crate::feature::Spherical;

    fn ds(eps: f64, lambda: f64) -> DenStream<f64, Spherical<f64>> {
        DenStream::new(2, eps, lambda, 0.5, 4.0).unwrap() // beta*mu = 2
    }

    #[test]
    fn denstream_recovers_two_blobs() {
        let mut rng = SplitMix64::new(7);
        let (pts, truth) = blobs(&mut rng, 250, &[[0.0, 0.0], [9.0, 0.0]], 0.5);
        let mut d = ds(1.5, 0.001); // slow fade so a short stream is ~static
        for p in &pts {
            d.insert(p);
        }
        d.cluster();
        assert_eq!(d.n_clusters(), 2);
        let labels: Vec<i64> = pts.iter().map(|p| d.predict(p)).collect();
        let assigned: Vec<usize> = labels.iter().map(|&l| l.max(0) as usize).collect();
        assert!(ari(&assigned, &truth) > 0.9, "ARI too low");
    }

    #[test]
    fn denstream_fades_and_prunes_stale_clusters() {
        let mut rng = SplitMix64::new(1);
        let (a, _) = blobs(&mut rng, 60, &[[0.0, 0.0]], 0.3);
        let (b, _) = blobs(&mut rng, 1200, &[[20.0, 0.0]], 0.3);
        let mut d = ds(1.5, 0.1); // fade fast enough to drop A, slow enough to keep streamed B
        for p in &a {
            d.insert(p);
        }
        for p in &b {
            d.insert(p);
        }
        d.cluster();
        // The old region A has faded out; a point there is noise, one in B is a real cluster.
        assert_eq!(d.predict(&[0.0, 0.0]), -1);
        assert!(d.predict(&[20.0, 0.0]) >= 0);
    }

    #[test]
    fn denstream_keeps_sparse_points_as_outliers() {
        let mut rng = SplitMix64::new(3);
        let (dense, _) = blobs(&mut rng, 300, &[[0.0, 0.0]], 0.4);
        let mut d = ds(1.0, 0.001);
        for p in &dense {
            d.insert(p);
        }
        d.insert(&[50.0, 50.0]); // a lone far point → stays an outlier micro, never promoted
        d.cluster();
        assert_eq!(d.n_clusters(), 1);
        assert_eq!(d.predict(&[50.0, 50.0]), -1);
    }

    #[test]
    fn denstream_promotes_outlier_once_dense() {
        let mut d = ds(1.0, 0.001);
        assert_eq!(d.potential_count(), 0);
        for _ in 0..5 {
            d.insert(&[1.0, 1.0]); // βμ = 2 → promotes after enough mass at one spot
        }
        assert_eq!(d.potential_count(), 1);
    }

    #[test]
    fn denstream_empty_and_single() {
        let mut d = ds(1.0, 0.1);
        d.cluster();
        assert_eq!(d.n_clusters(), 0);
        assert_eq!(d.predict(&[0.0, 0.0]), -1);
        d.insert(&[0.0, 0.0]); // one point → an outlier micro, not yet potential
        assert_eq!(d.potential_count(), 0);
        d.cluster();
        assert_eq!(d.predict(&[0.0, 0.0]), -1);
    }

    #[test]
    fn denstream_validates_params() {
        assert!(DenStream::<f64, Spherical<f64>>::new(0, 1.0, 0.1, 0.5, 4.0).is_err()); // dim
        assert!(DenStream::<f64, Spherical<f64>>::new(2, 0.0, 0.1, 0.5, 4.0).is_err()); // eps
        assert!(DenStream::<f64, Spherical<f64>>::new(2, 1.0, 0.0, 0.5, 4.0).is_err()); // lambda
        assert!(DenStream::<f64, Spherical<f64>>::new(2, 1.0, 0.1, 0.1, 4.0).is_err());
        // beta*mu ≤ 1
    }

    fn db(r: f64, lambda: f64) -> DbStream<f64, Spherical<f64>> {
        DbStream::new(2, r, lambda, 0.1, 2.0).unwrap()
    }

    #[test]
    fn dbstream_recovers_two_blobs() {
        let mut rng = SplitMix64::new(7);
        let (pts, truth) = blobs(&mut rng, 250, &[[0.0, 0.0], [9.0, 0.0]], 0.5);
        let mut d = db(1.5, 0.0005);
        for p in &pts {
            d.insert(p);
        }
        d.cluster();
        assert_eq!(d.n_clusters(), 2);
        let assigned: Vec<usize> = pts.iter().map(|p| d.predict(p).max(0) as usize).collect();
        assert!(ari(&assigned, &truth) > 0.9, "ARI too low");
    }

    #[test]
    fn dbstream_chains_overlapping_micros_into_one_cluster() {
        // One dense, wide blob is split into several overlapping micro-clusters at this radius; their
        // shared density bridges them back into a single density-connected cluster.
        let mut rng = SplitMix64::new(11);
        let (pts, _) = blobs(&mut rng, 500, &[[0.0, 0.0]], 1.2);
        let mut d = db(1.0, 0.0005);
        for p in &pts {
            d.insert(p);
        }
        d.cluster();
        assert!(
            d.micro_count() > 1,
            "blob should span several micro-clusters"
        );
        assert_eq!(d.n_clusters(), 1);
        assert!(d.predict(&[0.0, 0.0]) >= 0);
    }

    #[test]
    fn dbstream_keeps_disconnected_close_blobs_separate() {
        // Two tight blobs whose centres are within 2r (a distance-only rule would merge them) but with
        // an empty gap between → zero shared density → DBSTREAM keeps them as two clusters.
        let mut rng = SplitMix64::new(4);
        let (mut a, _) = blobs(&mut rng, 200, &[[0.0, 0.0]], 0.25);
        let (b, _) = blobs(&mut rng, 200, &[[2.6, 0.0]], 0.25);
        a.extend(b);
        let mut d = db(1.5, 0.0005); // 2r = 3.0 > 2.6 centre gap, yet no points bridge the blobs
        for p in &a {
            d.insert(p);
        }
        d.cluster();
        assert_eq!(d.n_clusters(), 2);
    }

    #[test]
    fn dbstream_fades_stale_region() {
        let mut rng = SplitMix64::new(1);
        let (old, _) = blobs(&mut rng, 60, &[[0.0, 0.0]], 0.3);
        let (recent, _) = blobs(&mut rng, 1500, &[[20.0, 0.0]], 0.3);
        let mut d = db(1.5, 0.05); // fast enough to fade the old region out
        for p in &old {
            d.insert(p);
        }
        for p in &recent {
            d.insert(p);
        }
        d.cluster();
        assert_eq!(d.predict(&[0.0, 0.0]), -1);
        assert!(d.predict(&[20.0, 0.0]) >= 0);
    }

    #[test]
    fn dbstream_empty_and_single() {
        let mut d = db(1.0, 0.01);
        d.cluster();
        assert_eq!(d.n_clusters(), 0);
        assert_eq!(d.predict(&[0.0, 0.0]), -1);
        d.insert(&[0.0, 0.0]); // one point → a single weak micro, below min_weight
        d.cluster();
        assert_eq!(d.predict(&[0.0, 0.0]), -1);
    }

    #[test]
    fn dbstream_accessors_report_state() {
        let mut rng = SplitMix64::new(2);
        let (pts, _) = blobs(&mut rng, 200, &[[0.0, 0.0]], 0.4);
        let mut d = db(1.5, 0.0005);
        assert_eq!(d.dim(), 2);
        for p in &pts {
            d.insert(p);
        }
        d.cluster();
        assert_eq!(d.labels().len(), d.micro_count());
        let (centers, weights, radii, dim) = d.micro_stats();
        assert_eq!(dim, 2);
        assert_eq!(weights.len(), d.micro_count());
        assert_eq!(radii.len(), d.micro_count());
        assert_eq!(centers.len(), d.micro_count() * 2);
        assert!(weights.iter().all(|&w| w > 0.0));
        assert_eq!(d.predict(&[100.0, 100.0]), -1); // far point → noise
    }

    #[test]
    fn dbstream_validates_params() {
        assert!(DbStream::<f64, Spherical<f64>>::new(0, 1.0, 0.1, 0.3, 2.0).is_err()); // dim
        assert!(DbStream::<f64, Spherical<f64>>::new(2, 0.0, 0.1, 0.3, 2.0).is_err()); // r
        assert!(DbStream::<f64, Spherical<f64>>::new(2, 1.0, 0.0, 0.3, 2.0).is_err()); // lambda
        assert!(DbStream::<f64, Spherical<f64>>::new(2, 1.0, 0.1, 0.0, 2.0).is_err()); // alpha low
        assert!(DbStream::<f64, Spherical<f64>>::new(2, 1.0, 0.1, 1.5, 2.0).is_err()); // alpha high
        assert!(DbStream::<f64, Spherical<f64>>::new(2, 1.0, 0.1, 0.3, 0.0).is_err());
        // min_weight
    }
}
