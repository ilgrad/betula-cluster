//! k-prototypes clustering of **mixed numeric + categorical + directional** data (Huang, 1997/1998,
//! extended by a third block).
//!
//! Each cluster is summarised by a *mixed clustering feature* [`MixedCf`], which carries one exact
//! mergeable monoid per block:
//!
//! | block | what it stores | its prototype | its dissimilarity to a point |
//! |---|---|---|---|
//! | numeric | `(n, μ, S)` — a reused [`Diagonal`] CF | the mean `μ` | `Σ_j (x_j − μ_j)²` |
//! | categorical | one category-count histogram per attribute | the per-attribute mode | `Σ_j [x_j ≠ mode_j]` |
//! | directional | the resultant `R = Σ w_i u_i` over L2-normalised rows | `R/‖R‖` | `‖u − c‖² = 2 − 2 uᵀc` |
//!
//! So a cluster centre is itself a `MixedCf`, and the distance from a point to a prototype is
//!
//! ```text
//! d = Σ_j∈num (x_j − μ_j)²  +  γ_cat · Σ_j∈cat [x_j ≠ mode_j]  +  γ_dir · (2 − 2 uᵀc)
//! ```
//!
//! **The two block weights are the same open problem, twice.** `γ_cat` is Huang's, with his heuristic
//! `γ ≈ ½·mean σ`; `γ_dir` has no literature at all, and this crate's default — the mean numeric
//! *variance*, so one unit of `‖u − c‖²` costs one numeric variance — is a scale-matching convention,
//! not a result. Both are exposed. Read [`BlockWeights`] before trusting either.
//!
//! **Why a third block and not three more numeric columns.** A direction has no mean: averaging unit
//! vectors leaves the sphere, and two rows a full turn apart average to zero rather than to a point
//! between them. The resultant `R` is the right summary — it is what the von Mises–Fisher head uses —
//! and because `‖u‖ = 1` the block's cost over a whole micro-cluster is `2n_i − 2⟨R_i, c⟩`: affine in
//! the summary, with **no scatter term**, exactly as in [`crate::clustering::hyperbolic`].
//!
//! Numeric-only data reduces to k-means and categorical-only to k-modes; the head is exposed for the
//! genuinely *mixed* case.

use crate::clustering::kmeans::weighted_pick;
use crate::clustering::rng::SplitMix64;
use crate::feature::{ClusterFeature, Diagonal};
use crate::kernels::sq_euclidean;
use crate::types::Real;

/// The shape of a mixed row: how many numeric attributes, how many codes each categorical attribute
/// has, and how many coordinates the directional block spans (`0` = no directional block).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedSchema {
    /// Number of numeric attributes.
    pub numeric: usize,
    /// Number of distinct codes in each categorical attribute; the length is the attribute count.
    pub cardinalities: Vec<usize>,
    /// Number of directional coordinates. `0` disables the block; `1` is meaningless (a "direction"
    /// in one dimension is a sign) and is rejected by the callers that build rows.
    pub directional: usize,
}

/// How the non-numeric blocks are priced against the numeric one.
///
/// Both are heuristics and neither is a result. `categorical` is Huang's `γ`, whose published
/// default is half the mean numeric standard deviation; `directional` has no published default at
/// all, and the mean numeric variance is what this crate uses so that one unit of `‖u − c‖²` — which
/// runs over `[0, 4]` whatever the data — costs one numeric variance. A block weight is a modelling
/// choice about how much a category mismatch or a turned direction is *worth*, and no amount of
/// dispersion matching answers that; treat both as parameters to sweep, not as settings to accept.
#[derive(Clone, Copy, Debug)]
pub struct BlockWeights<R> {
    /// `γ_cat`, multiplying the per-attribute mismatch count.
    pub categorical: R,
    /// `γ_dir`, multiplying `‖u − c‖² = 2 − 2 uᵀc`.
    pub directional: R,
}

/// A batch of mixed rows in the three parallel layouts [`summarize_mixed`] consumes: `numeric` is
/// `n × schema.numeric`, `categorical` is `n × schema.cardinalities.len()` integer codes, and
/// `directional` is `n × schema.directional` **already L2-normalised per row**.
#[derive(Clone, Copy)]
pub struct MixedRows<'a, R> {
    /// Row-major numeric values.
    pub numeric: &'a [R],
    /// Row-major integer category codes, each in range for its attribute.
    pub categorical: &'a [usize],
    /// Row-major unit vectors.
    pub directional: &'a [R],
    /// Number of rows.
    pub n: usize,
}

/// A mixed clustering feature: numeric `(n, μ, S)`, per-attribute categorical counts, and the
/// directional resultant.
#[derive(Clone)]
pub struct MixedCf<R: Real> {
    num: Diagonal<R>,
    /// `cat[j][c]` = total weight of category `c` in categorical attribute `j`.
    cat: Vec<Vec<R>>,
    /// Cached arg-max of each `cat[j]` (the per-attribute mode); ties keep the lower code.
    mode: Vec<usize>,
    /// `Σ_i w_i u_i` over the block's unit vectors — the resultant, not a mean.
    dir: Vec<R>,
    /// Cached `dir/‖dir‖` (zero when the resultant has cancelled), for the same reason `mode` is
    /// cached: the leader pass evaluates every point against every leader, so materialising a
    /// prototype inside that loop is `O(n · leaders)` allocations.
    unit: Vec<R>,
}

impl<R: Real> MixedCf<R> {
    /// Empty feature for `schema`.
    pub fn new(schema: &MixedSchema) -> Self {
        Self {
            num: Diagonal::new(schema.numeric),
            cat: schema
                .cardinalities
                .iter()
                .map(|&c| vec![R::zero(); c])
                .collect(),
            mode: vec![0; schema.cardinalities.len()],
            dir: vec![R::zero(); schema.directional],
            unit: vec![R::zero(); schema.directional],
        }
    }

    /// The schema this feature was built for.
    pub fn schema(&self) -> MixedSchema {
        MixedSchema {
            numeric: self.num.dim(),
            cardinalities: self.cardinalities(),
            directional: self.dir.len(),
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

    /// Number of directional coordinates.
    pub fn n_directional(&self) -> usize {
        self.dir.len()
    }

    /// The directional resultant `R = Σ w_i u_i`. Its length relative to the weight is the block's
    /// concentration; its direction is the prototype.
    pub fn resultant(&self) -> &[R] {
        &self.dir
    }

    /// The directional prototype `R/‖R‖`. A resultant that has cancelled to (near) zero has no
    /// direction, and the zero vector is returned — which makes the block's term equal for every
    /// prototype, so it stops voting instead of voting arbitrarily.
    pub fn direction(&self) -> &[R] {
        &self.unit
    }

    /// Recompute the cached unit direction. Called from every mutator; there are two.
    fn refresh_direction(&mut self) {
        let norm = self.dir.iter().fold(R::zero(), |a, &v| a + v * v).sqrt();
        let cancelled = norm <= R::from_f64(1e-12).unwrap();
        for (u, &r) in self.unit.iter_mut().zip(&self.dir) {
            *u = if cancelled { R::zero() } else { r / norm };
        }
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

    /// Add a mixed point: `num` (length `numeric`), category codes `cat` (length
    /// `cardinalities.len()`, each in range for its attribute) and the unit vector `dir` (length
    /// `directional`). The mode and direction caches are kept current.
    ///
    /// `dir` is taken as given. Normalising here would be a per-consumer guard for something the
    /// boundary already has to do once per row — see [`MixedRows::directional`].
    pub fn push(&mut self, num: &[R], cat: &[usize], dir: &[R], w: R) {
        self.num.push(num, w);
        for (j, &code) in cat.iter().enumerate() {
            let hist = &mut self.cat[j];
            hist[code] = hist[code] + w;
            if hist[code] > hist[self.mode[j]] {
                self.mode[j] = code;
            }
        }
        for (r, &u) in self.dir.iter_mut().zip(dir) {
            *r = *r + w * u;
        }
        self.refresh_direction();
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
        for (r, &o) in self.dir.iter_mut().zip(&other.dir) {
            *r = *r + o;
        }
        self.refresh_direction();
    }
}

/// A prototype: the three blocks' centres, held together so a distance call cannot be handed the
/// numeric centre of one cluster and the mode of another. Borrowed, not owned — the leader pass
/// builds one per (point, leader) pair, and three `Vec`s there would be `O(n · leaders)` allocations.
#[derive(Clone, Copy)]
struct Prototype<'a, R> {
    numeric: &'a [R],
    mode: &'a [usize],
    direction: &'a [R],
}

impl<'a, R: Real> Prototype<'a, R> {
    fn of(cf: &'a MixedCf<R>) -> Self {
        Self {
            numeric: cf.numeric_mean(),
            mode: cf.mode(),
            direction: cf.direction(),
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

/// `⟨a, b⟩` over the directional block.
fn dot<R: Real>(a: &[R], b: &[R]) -> R {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x * y)
        .fold(R::zero(), |p, q| p + q)
}

/// k-prototypes distance from a mixed point to a prototype.
fn point_dist<R: Real>(
    num: &[R],
    cat: &[usize],
    dir: &[R],
    p: &Prototype<R>,
    w: BlockWeights<R>,
) -> R {
    let two = R::one() + R::one();
    let angular = if p.direction.is_empty() {
        R::zero()
    } else {
        two - two * dot(dir, p.direction)
    };
    sq_euclidean(num, p.numeric)
        + w.categorical * R::from_usize(cat_mismatch(cat, p.mode)).unwrap()
        + w.directional * angular
}

/// Distance from a weighted micro-cluster to a prototype.
///
/// Every block is summed over the micro's points, not evaluated at its centre: the numeric term is
/// the mass times the centroid's squared distance (König–Huygens leaves the within-micro scatter as a
/// separate constant), the categorical term counts the micro's points whose category differs from the
/// mode, and the directional term is `2n_i − 2⟨R_i, c⟩` — exact, because `‖u‖ = 1` makes the block's
/// cost affine in the resultant with no scatter term at all.
fn micro_dist<R: Real>(m: &MixedCf<R>, p: &Prototype<R>, w: BlockWeights<R>) -> R {
    let two = R::one() + R::one();
    let mass = m.weight();
    let mut cat_cost = R::zero();
    for (j, &mode) in p.mode.iter().enumerate() {
        cat_cost = cat_cost + (mass - m.count(j, mode));
    }
    let angular = if p.direction.is_empty() {
        R::zero()
    } else {
        two * mass - two * dot(m.resultant(), p.direction)
    };
    mass * sq_euclidean(m.numeric_mean(), p.numeric)
        + w.categorical * cat_cost
        + w.directional * angular
}

/// Single-pass leader summarisation into at most `max_leaders` mixed micro-clusters: each point joins
/// its nearest leader within `threshold` (k-prototypes distance), otherwise starts a new leader. Once
/// the cap is reached every further point joins its nearest leader regardless of `threshold` — bounded
/// memory with graceful accuracy degradation (raise `max_leaders` for finer summaries).
pub fn summarize_mixed<R: Real>(
    rows: MixedRows<'_, R>,
    schema: &MixedSchema,
    weights: BlockWeights<R>,
    threshold: R,
    max_leaders: usize,
) -> Vec<MixedCf<R>> {
    let n_cat = schema.cardinalities.len();
    let mut leaders: Vec<MixedCf<R>> = Vec::new();
    for i in 0..rows.n {
        let xn = &rows.numeric[i * schema.numeric..(i + 1) * schema.numeric];
        let xc = &rows.categorical[i * n_cat..(i + 1) * n_cat];
        let xd = &rows.directional[i * schema.directional..(i + 1) * schema.directional];
        let mut best = usize::MAX;
        let mut bd = R::infinity();
        for (li, l) in leaders.iter().enumerate() {
            let d = point_dist(xn, xc, xd, &Prototype::of(l), weights);
            if d < bd {
                bd = d;
                best = li;
            }
        }
        if best != usize::MAX && (bd <= threshold || leaders.len() >= max_leaders) {
            leaders[best].push(xn, xc, xd, R::one());
        } else {
            let mut l = MixedCf::new(schema);
            l.push(xn, xc, xd, R::one());
            leaders.push(l);
        }
    }
    leaders
}

/// Index of the micro-cluster nearest to a mixed point (k-prototypes distance to its prototype).
pub fn nearest_micro<R: Real>(
    micros: &[MixedCf<R>],
    num: &[R],
    cat: &[usize],
    dir: &[R],
    weights: BlockWeights<R>,
) -> usize {
    let mut best = 0;
    let mut bd = point_dist(num, cat, dir, &Prototype::of(&micros[0]), weights);
    for (i, m) in micros.iter().enumerate().skip(1) {
        let d = point_dist(num, cat, dir, &Prototype::of(m), weights);
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
    weights: BlockWeights<R>,
    rng: &mut SplitMix64,
) -> Vec<usize> {
    let n = micros.len();
    let w: Vec<f64> = micros
        .iter()
        .map(|m| m.weight().to_f64().unwrap_or(0.0))
        .collect();
    let protos: Vec<Prototype<R>> = micros.iter().map(Prototype::of).collect();
    let dist = |a: usize, b: usize| -> f64 {
        point_dist(
            micros[a].numeric_mean(),
            micros[a].mode(),
            protos[a].direction,
            &protos[b],
            weights,
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
    weights: BlockWeights<R>,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> Vec<usize> {
    assert!(!micros.is_empty(), "need at least one micro-cluster");
    let n = micros.len();
    let k = k.min(n).max(1);
    let schema = micros[0].schema();

    let mut rng = SplitMix64::new(seed);
    let mut best: Option<(R, Vec<usize>)> = None;
    for _ in 0..n_init.max(1) {
        let mut centers: Vec<MixedCf<R>> = kpp_init(micros, k, weights, &mut rng)
            .into_iter()
            .map(|s| micros[s].clone())
            .collect();
        let mut labels = vec![usize::MAX; n];
        for _ in 0..max_iter.max(1) {
            let proto: Vec<Prototype<R>> = centers.iter().map(Prototype::of).collect();
            let mut changed = false;
            for (i, m) in micros.iter().enumerate() {
                let mut best_c = 0;
                let mut bd = micro_dist(m, &proto[0], weights);
                for (c, p) in proto.iter().enumerate().skip(1) {
                    let d = micro_dist(m, p, weights);
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
            let mut acc: Vec<MixedCf<R>> = (0..k).map(|_| MixedCf::new(&schema)).collect();
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
        let proto: Vec<Prototype<R>> = centers.iter().map(Prototype::of).collect();
        let mut inertia = R::zero();
        for (i, m) in micros.iter().enumerate() {
            inertia = inertia + m.numeric_ssd() + micro_dist(m, &proto[labels[i]], weights);
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
    use std::f64::consts::PI;

    /// A two-block schema — numeric + categorical, directional block absent.
    fn flat(n_num: usize, cards: &[usize]) -> MixedSchema {
        MixedSchema {
            numeric: n_num,
            cardinalities: cards.to_vec(),
            directional: 0,
        }
    }

    /// Block weights with the directional term switched off.
    fn gam(categorical: f64) -> BlockWeights<f64> {
        BlockWeights {
            categorical,
            directional: 0.0,
        }
    }

    /// A prototype assembled from loose parts, for the closed-form checks that name a centre no
    /// fixture ever merges.
    fn proto<'a>(
        numeric: &'a [f64],
        mode: &'a [usize],
        direction: &'a [f64],
    ) -> Prototype<'a, f64> {
        Prototype {
            numeric,
            mode,
            direction,
        }
    }

    /// Build micro-clusters one-point-per-row from parallel numeric + categorical arrays.
    fn micros(
        num: &[f64],
        cat: &[usize],
        n: usize,
        n_num: usize,
        cards: &[usize],
    ) -> Vec<MixedCf<f64>> {
        let schema = flat(n_num, cards);
        let n_cat = cards.len();
        (0..n)
            .map(|i| {
                let mut m = MixedCf::new(&schema);
                m.push(
                    &num[i * n_num..(i + 1) * n_num],
                    &cat[i * n_cat..(i + 1) * n_cat],
                    &[],
                    1.0,
                );
                m
            })
            .collect()
    }

    /// Two-block rows: no directional coordinates.
    fn rows<'a>(num: &'a [f64], cat: &'a [usize], n: usize) -> MixedRows<'a, f64> {
        MixedRows {
            numeric: num,
            categorical: cat,
            directional: &[],
            n,
        }
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
        let lab = kprototypes(&m, 2, gam(0.5), 100, 4, 7);
        assert!(ari(&lab, &truth) > 0.95, "ARI = {}", ari(&lab, &truth));
    }

    #[test]
    fn categorical_breaks_numeric_tie() {
        // All points numerically coincident; only the categorical attribute distinguishes the two
        // groups. With γ_cat > 0 k-prototypes must split on the category.
        let (mut num, mut cat, mut truth) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..100 {
            num.push(0.0);
            cat.push(i % 2);
            truth.push(i % 2);
        }
        let m = micros(&num, &cat, 100, 1, &[2]);
        let lab = kprototypes(&m, 2, gam(1.0), 100, 4, 3);
        assert!(ari(&lab, &truth) > 0.99, "ARI = {}", ari(&lab, &truth));
    }

    #[test]
    fn mode_and_merge_are_exact() {
        let s = flat(1, &[3]);
        let mut a = MixedCf::<f64>::new(&s);
        a.push(&[1.0], &[2], &[], 1.0);
        a.push(&[3.0], &[2], &[], 1.0);
        a.push(&[2.0], &[0], &[], 1.0);
        assert_eq!(a.mode(), &[2]); // category 2 appears twice
        assert!((a.numeric_mean()[0] - 2.0).abs() < 1e-12);
        let mut b = MixedCf::<f64>::new(&s);
        b.push(&[0.0], &[0], &[], 1.0);
        b.push(&[0.0], &[0], &[], 1.0);
        a.merge(&b);
        assert_eq!(a.weight() as i64, 5);
        assert_eq!(a.mode(), &[0]); // now category 0 appears three times
    }

    #[test]
    fn accessors_and_nearest_micro() {
        // Two one-point micros: (num 0, cat 0) and (num 10, cat 1). A query routes to the closer one.
        let m = micros(&[0.0, 10.0], &[0, 1], 2, 1, &[2]);
        assert_eq!(m[0].n_categorical(), 1);
        assert_eq!(m[0].n_directional(), 0);
        assert_eq!(m[0].cardinalities(), vec![2]);
        assert_eq!(m[0].schema(), flat(1, &[2]));
        assert_eq!(nearest_micro(&m, &[0.1], &[0], &[], gam(1.0)), 0);
        assert_eq!(nearest_micro(&m, &[9.5], &[1], &[], gam(1.0)), 1);
    }

    #[test]
    fn summarize_caps_leaders() {
        // threshold 0 ⇒ distinct points would each be a leader, but the cap bounds the count.
        let (mut num, mut cat) = (Vec::new(), Vec::new());
        for i in 0..500 {
            num.push(i as f64);
            cat.push(i % 4);
        }
        let m = summarize_mixed(rows(&num, &cat, 500), &flat(1, &[4]), gam(0.5), 0.0, 16);
        assert!(m.len() <= 16);
        let total: f64 = m.iter().map(|c| c.weight()).sum();
        assert_eq!(total as i64, 500); // mass conserved
    }

    /// The objective `kprototypes` minimises, rebuilt from a labelling: each cluster's prototype is
    /// the merge of its members, and the cost is the within-micro scatter plus the micro-to-prototype
    /// distance. The function returns only labels, so this is the only way to observe what it chose.
    fn objective(ms: &[MixedCf<f64>], labels: &[usize], k: usize, w: BlockWeights<f64>) -> f64 {
        let schema = ms[0].schema();
        let mut acc: Vec<MixedCf<f64>> = (0..k).map(|_| MixedCf::new(&schema)).collect();
        for (i, m) in ms.iter().enumerate() {
            acc[labels[i]].merge(m);
        }
        ms.iter()
            .enumerate()
            .map(|(i, m)| m.numeric_ssd() + micro_dist(m, &Prototype::of(&acc[labels[i]]), w))
            .sum()
    }

    fn mixed_fixture() -> Vec<MixedCf<f64>> {
        // Three groups: two numeric modes crossed with a categorical split, so neither the numeric
        // nor the categorical part alone recovers the partition.
        let mut rng = SplitMix64::new(19);
        let mut num = Vec::new();
        let mut cat = Vec::new();
        for (mx, my, a) in [(0.0, 0.0, 0usize), (6.0, 0.5, 1), (0.5, 6.0, 0)] {
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
        let (k, w) = (3usize, gam(1.0));
        let labels = kprototypes(&ms, k, w, 100, 4, 11);
        assert_eq!(labels.len(), ms.len());

        let schema = ms[0].schema();
        let mut acc: Vec<MixedCf<f64>> = (0..k).map(|_| MixedCf::new(&schema)).collect();
        for (i, m) in ms.iter().enumerate() {
            acc[labels[i]].merge(m);
        }
        assert!(
            acc.iter().filter(|a| a.weight() > 0.0).count() >= 2,
            "the fixture collapsed to one cluster, so nothing is tested"
        );
        for (i, m) in ms.iter().enumerate() {
            let own = micro_dist(m, &Prototype::of(&acc[labels[i]]), w);
            for (c, a) in acc.iter().enumerate() {
                if a.weight() <= 0.0 {
                    continue;
                }
                let d = micro_dist(m, &Prototype::of(a), w);
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
        let (k, w) = (3usize, gam(1.0));
        let mut prev = f64::INFINITY;
        let mut distinct = 0;
        for n_init in 1..=8 {
            let labels = kprototypes(&ms, k, w, 100, n_init, 5);
            let cost = objective(&ms, &labels, k, w);
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
            let chosen = kpp_init(&ms, 3, gam(1.0), &mut rng);
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

    #[test]
    fn kpp_init_never_reseeds_a_site_it_already_holds() {
        // The sampling weight is a *product*: a micro sitting exactly on an already-chosen
        // prototype has D² = 0, so its probability is zero however heavy it is. Any rule that
        // merely *adds* the two loses that guarantee, and the mass alone carries the draw. The
        // spread-out fixture above cannot tell the two apart, because there every micro's weight
        // is 1 against a D² of 10⁴; this one puts almost all the mass on the coincident site —
        // twelve micros of weight 100 at the origin against one of weight 0.01 three units away.
        let schema = flat(1, &[1]);
        let mut ms: Vec<MixedCf<f64>> = (0..12)
            .map(|_| {
                let mut m = MixedCf::new(&schema);
                m.push(&[0.0], &[0], &[], 100.0);
                m
            })
            .collect();
        let mut far = MixedCf::new(&schema);
        far.push(&[3.0], &[0], &[], 0.01);
        ms.push(far);

        for seed in 0..16u64 {
            let mut rng = SplitMix64::new(seed);
            let chosen = kpp_init(&ms, 2, gam(1.0), &mut rng);
            assert_eq!(chosen.len(), 2);
            assert!(
                chosen.contains(&12),
                "seed {seed} seeded the coincident site twice: {chosen:?}"
            );
        }
    }

    /// A four-point mixed micro with distinct numeric spread and two categorical attributes of
    /// different cardinality, so no two terms of the distance can be swapped without moving it.
    fn mixed_micro() -> MixedCf<f64> {
        let mut m = MixedCf::<f64>::new(&flat(2, &[3, 2]));
        m.push(&[0.0, 0.0], &[0, 0], &[], 1.0);
        m.push(&[2.0, 0.0], &[0, 1], &[], 1.0);
        m.push(&[0.0, 2.0], &[1, 0], &[], 1.0);
        m.push(&[2.0, 2.0], &[2, 0], &[], 1.0);
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

        // micro_dist = w·‖μ − c‖² + γ_cat·Σ_j (w − count_j(mode_j))
        //            = 4·2 + 3·((4 − 1) + (4 − 1)) = 8 + 18 = 26.
        let got = micro_dist(&m, &proto(&[0.0, 0.0], &[1, 1], &[]), gam(3.0));
        assert_eq!(got, 26.0, "micro_dist");

        // point_dist = ‖x − c‖² + γ_cat·#mismatches, swept over all three mismatch counts so that a
        // constant mismatch (0, 1) and an inverted comparison each land somewhere else.
        for (cat, mismatches) in [([1usize, 1usize], 0), ([0, 1], 1), ([0, 0], 2)] {
            let got = point_dist(
                &[2.0, 0.0],
                &cat,
                &[],
                &proto(&[0.0, 0.0], &[1, 1], &[]),
                gam(3.0),
            );
            assert_eq!(got, 4.0 + 3.0 * mismatches as f64, "cat {cat:?}");
        }
    }

    #[test]
    fn a_tied_category_keeps_the_first_mode() {
        // `push` refreshes the mode only on a strict improvement, so the code that reached the
        // count first keeps it; `argmax`, which `merge` uses instead, must agree.
        let mut m = MixedCf::<f64>::new(&flat(1, &[3]));
        m.push(&[0.0], &[2], &[], 1.0);
        m.push(&[0.0], &[0], &[], 1.0);
        assert_eq!(m.mode(), &[2], "a tie moved the mode off the incumbent");
        assert_eq!(argmax(&[2.0, 2.0, 1.0]), 0);
        assert_eq!(argmax(&[1.0, 3.0, 3.0]), 1);
    }

    #[test]
    fn nearest_micro_breaks_an_exact_tie_towards_the_first() {
        // The query sits exactly halfway between two micros that share a category, so only the
        // comparison decides.
        let ms = micros(&[0.0, 2.0], &[0, 0], 2, 1, &[1]);
        let a = point_dist(&[1.0], &[0], &[], &Prototype::of(&ms[0]), gam(1.0));
        let b = point_dist(&[1.0], &[0], &[], &Prototype::of(&ms[1]), gam(1.0));
        assert_eq!(a.to_bits(), b.to_bits(), "the fixture is not an exact tie");
        assert_eq!(nearest_micro(&ms, &[1.0], &[0], &[], gam(1.0)), 0);
    }

    /// Independent re-derivation of [`summarize_mixed`]: the row slices are materialized by chunking
    /// the flat arrays rather than by index arithmetic, and the nearest leader is chosen from a
    /// materialized distance row.
    fn reference_summarize(
        rows: MixedRows<'_, f64>,
        schema: &MixedSchema,
        weights: BlockWeights<f64>,
        threshold: f64,
        max_leaders: usize,
    ) -> Vec<MixedCf<f64>> {
        let cut = |v: &[f64], w: usize| -> Vec<Vec<f64>> {
            if w == 0 {
                vec![Vec::new(); rows.n]
            } else {
                v.chunks_exact(w).map(<[f64]>::to_vec).collect()
            }
        };
        let nums = cut(rows.numeric, schema.numeric);
        let dirs = cut(rows.directional, schema.directional);
        let cats: Vec<Vec<usize>> = if schema.cardinalities.is_empty() {
            vec![Vec::new(); rows.n]
        } else {
            rows.categorical
                .chunks_exact(schema.cardinalities.len())
                .map(<[usize]>::to_vec)
                .collect()
        };

        let mut leaders: Vec<MixedCf<f64>> = Vec::new();
        for i in 0..rows.n {
            let row: Vec<f64> = leaders
                .iter()
                .map(|l| point_dist(&nums[i], &cats[i], &dirs[i], &Prototype::of(l), weights))
                .collect();
            let nearest =
                row.iter()
                    .enumerate()
                    .fold(None::<(usize, f64)>, |acc, (j, &d)| match acc {
                        Some((_, bd)) if bd <= d => acc,
                        _ => Some((j, d)),
                    });
            match nearest {
                Some((li, bd)) if bd <= threshold || leaders.len() >= max_leaders => {
                    leaders[li].push(&nums[i], &cats[i], &dirs[i], 1.0)
                }
                _ => {
                    let mut l = MixedCf::new(schema);
                    l.push(&nums[i], &cats[i], &dirs[i], 1.0);
                    leaders.push(l);
                }
            }
        }
        leaders
    }

    /// Rows wider than one column in every block, so that reading row `i` with `i / w` or
    /// `(i + 1) * w` picks up a different point instead of the same one — and so that the three
    /// blocks' row strides differ from one another.
    fn wide_rows() -> (Vec<f64>, Vec<usize>, Vec<f64>, MixedSchema) {
        let mut rng = SplitMix64::new(23);
        let (mut num, mut cat, mut dir) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..40 {
            let g = i % 4;
            num.push(g as f64 * 3.0 + 0.4 * rng.gauss());
            num.push(g as f64 * -2.0 + 0.4 * rng.gauss());
            num.push(0.4 * rng.gauss());
            cat.push(g % 2);
            cat.push(i % 3);
            let a = 0.5 * PI * g as f64 + 0.1 * rng.gauss();
            dir.push(a.cos());
            dir.push(a.sin());
        }
        (
            num,
            cat,
            dir,
            MixedSchema {
                numeric: 3,
                cardinalities: vec![2, 3],
                directional: 2,
            },
        )
    }

    #[test]
    fn summarize_mixed_matches_an_independent_leader_pass() {
        let (num, cat, dir, schema) = wide_rows();
        let r = MixedRows {
            numeric: &num,
            categorical: &cat,
            directional: &dir,
            n: 40,
        };
        let w = BlockWeights {
            categorical: 0.7,
            directional: 0.9,
        };
        for (threshold, cap) in [(1.0, 64usize), (4.0, 64), (1.0, 5)] {
            let got = summarize_mixed(r, &schema, w, threshold, cap);
            let want = reference_summarize(r, &schema, w, threshold, cap);
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
                for (d, (x, y)) in a.resultant().iter().zip(b.resultant()).enumerate() {
                    assert!(
                        (x - y).abs() < 1e-12,
                        "leader {i} resultant[{d}]: {x} vs {y}"
                    );
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
        let schema = flat(2, &[1, 1]);
        let spread = summarize_mixed(rows(&num, &cat, 2), &schema, gam(1.0), 1.0, 16);
        assert_eq!(
            spread.len(),
            2,
            "the far point joined a leader it is not near"
        );

        let capped = summarize_mixed(rows(&num, &cat, 2), &schema, gam(1.0), 1.0, 1);
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
        let w = gam(1.0);
        let left = point_dist(
            &num[2..3],
            &cat[2..3],
            &[],
            &proto(&num[0..1], &cat[0..1], &[]),
            w,
        );
        let right = point_dist(
            &num[2..3],
            &cat[2..3],
            &[],
            &proto(&num[1..2], &cat[1..2], &[]),
            w,
        );
        assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "the fixture is not an exact tie ({left} vs {right}), so it cannot see the comparison"
        );

        let leaders = summarize_mixed(rows(&num, &cat, 3), &flat(1, &[1]), w, left, 16);
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
        w: BlockWeights<f64>,
        max_iter: usize,
        n_init: usize,
        seed: u64,
    ) -> Vec<usize> {
        let n = ms.len();
        let k = k.min(n).max(1);
        let schema = ms[0].schema();
        let mut rng = SplitMix64::new(seed);
        let mut best: Option<(f64, Vec<usize>)> = None;

        for _ in 0..n_init.max(1) {
            let mut centers: Vec<MixedCf<f64>> = kpp_init(ms, k, w, &mut rng)
                .into_iter()
                .map(|s| ms[s].clone())
                .collect();
            let mut labels = vec![usize::MAX; n];
            for _ in 0..max_iter.max(1) {
                let mut changed = false;
                for (i, m) in ms.iter().enumerate() {
                    let row: Vec<f64> = centers
                        .iter()
                        .map(|c| micro_dist(m, &Prototype::of(c), w))
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
                    let mut a = MixedCf::new(&schema);
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
                .map(|(m, &c)| m.numeric_ssd() + micro_dist(m, &Prototype::of(&centers[c]), w))
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
            let got = kprototypes(&ms, k, gam(gamma), 100, n_init, seed);
            let want = reference_kprototypes(&ms, k, gam(gamma), 100, n_init, seed);
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
        let (k, w) = (2usize, gam(1.0));
        let mut ties = 0;
        for seed in 0..16u64 {
            let mut rng = SplitMix64::new(seed);
            let row: Vec<u64> = kpp_init(&ms, k, w, &mut rng)
                .into_iter()
                .map(|s| micro_dist(&ms[1], &Prototype::of(&ms[s]), w).to_bits())
                .collect();
            if row[0] == row[1] {
                ties += 1;
            }
            assert_eq!(
                kprototypes(&ms, k, w, 100, 1, seed),
                reference_kprototypes(&ms, k, w, 100, 1, seed),
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
                let got = kprototypes(&ms, k, gam(1.0), 100, 1, seed);
                let mut used = got.clone();
                used.sort_unstable();
                used.dedup();
                if used.len() < k {
                    emptied += 1;
                }
                assert_eq!(
                    got,
                    reference_kprototypes(&ms, k, gam(1.0), 100, 1, seed),
                    "k {k}, seed {seed}"
                );
            }
        }
        assert!(
            emptied > 0,
            "every cluster kept a member; the empty-cluster branch is untested"
        );
    }

    // ---- the directional block ------------------------------------------------------------

    /// A three-block schema: two numeric attributes, one binary categorical, a 2-D direction.
    fn three_block_schema() -> MixedSchema {
        MixedSchema {
            numeric: 2,
            cardinalities: vec![2],
            directional: 2,
        }
    }

    /// Four groups laid out so that **no two blocks suffice**: the numeric block splits
    /// `{g0, g1}` from `{g2, g3}`, the directional block is the only thing separating `g0` from
    /// `g1`, and the categorical block is the only thing separating `g2` from `g3`.
    fn three_block_fixture() -> (Vec<MixedCf<f64>>, Vec<usize>) {
        let schema = three_block_schema();
        let mut rng = SplitMix64::new(31);
        let (mut ms, mut truth) = (Vec::new(), Vec::new());
        for (g, (site, angle, code)) in [
            (0.0, 0.0, 0usize),
            (0.0, PI, 0),
            (6.0, 0.0, 0),
            (6.0, 0.0, 1),
        ]
        .into_iter()
        .enumerate()
        {
            for _ in 0..25 {
                let a = angle + 0.15 * rng.gauss();
                let mut m = MixedCf::new(&schema);
                m.push(
                    &[site + 0.35 * rng.gauss(), 0.35 * rng.gauss()],
                    &[code],
                    &[a.cos(), a.sin()],
                    1.0,
                );
                ms.push(m);
                truth.push(g);
            }
        }
        (ms, truth)
    }

    #[test]
    fn every_block_is_load_bearing() {
        // The point of a third block: switching either non-numeric weight to zero must cost real
        // ARI on a fixture whose partition needs all three. A directional term that is dropped,
        // double-counted into the numeric one, or read as a mean instead of a resultant collapses
        // this to the two-block answer.
        let (ms, truth) = three_block_fixture();
        let full = BlockWeights {
            categorical: 4.0,
            directional: 4.0,
        };
        let a = ari(&kprototypes(&ms, 4, full, 100, 8, 5), &truth);
        let no_dir = ari(
            &kprototypes(
                &ms,
                4,
                BlockWeights {
                    directional: 0.0,
                    ..full
                },
                100,
                8,
                5,
            ),
            &truth,
        );
        let no_cat = ari(
            &kprototypes(
                &ms,
                4,
                BlockWeights {
                    categorical: 0.0,
                    ..full
                },
                100,
                8,
                5,
            ),
            &truth,
        );
        assert!(a > 0.9, "all three blocks: ARI = {a}");
        assert!(a > no_dir + 0.1, "γ_dir = 0 cost nothing: {a} vs {no_dir}");
        assert!(a > no_cat + 0.1, "γ_cat = 0 cost nothing: {a} vs {no_cat}");
    }

    #[test]
    fn the_resultant_is_the_exact_sum_and_the_direction_its_unit() {
        // The block's monoid is the resultant, not a mean: merging two features must add their
        // resultants, and `direction()` must be that sum normalised — never the mean of the two
        // directions, which is a different vector whenever the weights differ.
        let schema = MixedSchema {
            numeric: 1,
            cardinalities: vec![],
            directional: 2,
        };
        let mut a = MixedCf::<f64>::new(&schema);
        for _ in 0..3 {
            a.push(&[0.0], &[], &[1.0, 0.0], 1.0);
        }
        let mut b = MixedCf::<f64>::new(&schema);
        b.push(&[0.0], &[], &[0.0, 1.0], 1.0);
        a.merge(&b);
        assert_eq!(a.resultant(), &[3.0, 1.0]);
        let d = a.direction();
        let norm = (10.0f64).sqrt();
        assert!((d[0] - 3.0 / norm).abs() < 1e-12 && (d[1] - 1.0 / norm).abs() < 1e-12);
        // The mean of the two unit directions would be (1,1)/√2 — a full 26.6° away.
        assert!(
            (d[0] - d[1]).abs() > 0.5,
            "direction() averaged the members instead of normalising their sum"
        );
    }

    #[test]
    fn a_cancelled_resultant_stops_voting() {
        // Two opposite unit vectors sum to zero: there is no direction to report, and returning any
        // unit vector would make the block vote arbitrarily. The zero vector makes its term equal
        // for every prototype instead.
        let schema = MixedSchema {
            numeric: 1,
            cardinalities: vec![],
            directional: 2,
        };
        let mut m = MixedCf::<f64>::new(&schema);
        m.push(&[0.0], &[], &[1.0, 0.0], 1.0);
        m.push(&[0.0], &[], &[-1.0, 0.0], 1.0);
        assert_eq!(m.direction(), vec![0.0, 0.0]);
        let w = BlockWeights {
            categorical: 0.0,
            directional: 1.0,
        };
        let p = Prototype::of(&m);
        let east = point_dist(&[0.0], &[], &[1.0, 0.0], &p, w);
        let north = point_dist(&[0.0], &[], &[0.0, 1.0], &p, w);
        assert_eq!(east, north, "a cancelled resultant still picked a side");
        assert_eq!(east, 2.0);
    }

    #[test]
    fn the_directional_block_reads_only_the_resultant() {
        // `‖u − c‖² = 2 − 2uᵀc` is affine in `u`, so a micro-cluster's directional cost is
        // `2n − 2⟨R, c⟩` with no scatter term at all. Two micros with the same resultant but very
        // different spread must therefore cost the same — the property that lets a leaf carry the
        // block in `(n, R)` and nothing more.
        let schema = MixedSchema {
            numeric: 1,
            cardinalities: vec![],
            directional: 2,
        };
        let w = BlockWeights {
            categorical: 0.0,
            directional: 1.0,
        };
        // Four unit vectors at 0°, 0°, +90°, −90° sum to (2, 0); so do four at ±60°. Same count,
        // same resultant, different second moment — the only pair of statistics an affine cost can
        // and cannot see.
        let build = |angles: [f64; 4]| {
            let mut m = MixedCf::<f64>::new(&schema);
            for a in angles {
                m.push(&[0.0], &[], &[a.cos(), a.sin()], 1.0);
            }
            m
        };
        let (wa, na) = (
            [0.0, 0.0, PI / 2.0, -PI / 2.0],
            [PI / 3.0, -PI / 3.0, PI / 3.0, -PI / 3.0],
        );
        let (wide, narrow) = (build(wa), build(na));
        assert_eq!(wide.weight(), narrow.weight());
        for (x, y) in wide.resultant().iter().zip(narrow.resultant()) {
            assert!((x - y).abs() < 1e-12, "the fixture resultants differ");
        }
        // The second moment `Σ (uᵀe_x)²` is 2 against 1 — the sets really are differently spread,
        // in exactly the statistic an affine cost cannot see.
        let moment = |a: [f64; 4]| a.iter().map(|x| x.cos() * x.cos()).sum::<f64>();
        assert!(
            (moment(wa) - moment(na)).abs() > 0.5,
            "the two fixtures share their second moment, so nothing distinguishes them"
        );

        let p = proto(&[0.0], &[], &[0.6, 0.8]);
        let (a, b) = (micro_dist(&wide, &p, w), micro_dist(&narrow, &p, w));
        assert!((a - 5.6).abs() < 1e-12, "2n − 2⟨R, c⟩ = 8 − 2.4, got {a}");
        assert!(
            (a - b).abs() < 1e-12,
            "the directional cost read the spread"
        );
    }

    #[test]
    fn the_micro_cost_is_konig_huygens_over_all_three_blocks() {
        // The head sums every block over the micro's *points*, never evaluates it at the centre:
        // `Σ_i d(x_i, c) = S + micro_dist(m, c)`, where `S` is the numeric scatter and the other two
        // blocks contribute no scatter at all. This is the identity that makes the CF summary exact,
        // and it fails the moment a block is evaluated at the micro's own centroid instead.
        let schema = three_block_schema();
        let mut rng = SplitMix64::new(101);
        let w = BlockWeights {
            categorical: 1.7,
            directional: 2.3,
        };
        let pts: Vec<(Vec<f64>, Vec<usize>, Vec<f64>)> = (0..30)
            .map(|i| {
                let a = 2.0 * PI * rng.next_u64() as f64 / u64::MAX as f64;
                (
                    vec![rng.gauss(), 2.0 + rng.gauss()],
                    vec![i % 2],
                    vec![a.cos(), a.sin()],
                )
            })
            .collect();
        let mut m = MixedCf::new(&schema);
        for (n, c, d) in &pts {
            m.push(n, c, d, 1.0);
        }
        let p = proto(&[1.0, -1.0], &[1], &[0.0, 1.0]);
        let summed: f64 = pts.iter().map(|(n, c, d)| point_dist(n, c, d, &p, w)).sum();
        let via_cf = m.numeric_ssd() + micro_dist(&m, &p, w);
        assert!(
            (summed - via_cf).abs() < 1e-9,
            "points sum to {summed}, the CF says {via_cf}"
        );
    }

    #[test]
    fn the_three_block_distance_matches_its_closed_form() {
        // One micro of two points, hand-computed: μ = [1, 0], w = 2, attribute counts [1, 1],
        // R = (1, 1). Against c = ([0,0], mode 1, (0,1)) with γ_cat = 3, γ_dir = 5:
        //   numeric      2·‖[1,0]‖²                   = 2
        //   categorical  3·(2 − count(1)) = 3·(2 − 1) = 3
        //   directional  5·(2·2 − 2·⟨(1,1),(0,1)⟩)    = 5·2 = 10
        let schema = MixedSchema {
            numeric: 1,
            cardinalities: vec![2],
            directional: 2,
        };
        let mut m = MixedCf::<f64>::new(&schema);
        m.push(&[0.0], &[0], &[1.0, 0.0], 1.0);
        m.push(&[2.0], &[1], &[0.0, 1.0], 1.0);
        assert_eq!(m.resultant(), &[1.0, 1.0]);
        assert_eq!(m.n_directional(), 2);
        assert_eq!(m.schema(), schema);
        let w = BlockWeights {
            categorical: 3.0,
            directional: 5.0,
        };
        let p = proto(&[0.0], &[1], &[0.0, 1.0]);
        assert!((micro_dist(&m, &p, w) - 15.0).abs() < 1e-12);
        // The same prototype against one point: ‖2 − 0‖² + 3·0 + 5·(2 − 2·1) = 4.
        assert!((point_dist(&[2.0], &[1], &[0.0, 1.0], &p, w) - 4.0).abs() < 1e-12);
    }
}
