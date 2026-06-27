//! Arena CF-tree (BIRCH/BETULA Phase 1).
//!
//! Streams points into a height-balanced tree of clustering features: descend to the nearest
//! leaf, absorb into the nearest entry if the absorption criterion stays within `threshold`,
//! otherwise start a new entry; split overflowing nodes and propagate upward.
//!
//! Node CFs are kept exact: an insert folds the new point into every node on the leaf→root path
//! incrementally (`O(d)` per level, an exact CF push), and a split recomputes only the two nodes it
//! repartitions from their children. This keeps the "double update on split" bug class (present in
//! earlier impls) unrepresentable without an `O(branching)` recompute on every level of every insert.
//!
//! When the leaf count exceeds `max_leaves` the tree rebuilds with a grown threshold (BIRCH
//! reducibility), reinserting the existing leaf features via [`CFTree::insert_cf`].

use crate::distance::CFDistance;
use crate::feature::ClusterFeature;
use crate::types::Real;

#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
struct Node<C> {
    cf: C,
    /// Leaf: indices into `entries`. Internal: indices into `nodes`.
    children: Vec<usize>,
    leaf: bool,
    parent: Option<usize>,
}

/// A CF-tree parameterised by feature model `C`, routing distance `D`, and absorption `A`.
#[cfg_attr(feature = "persistence", derive(serde::Serialize, serde::Deserialize))]
pub struct CFTree<R: Real, C: ClusterFeature<R>, D: CFDistance<R, C>, A: CFDistance<R, C>> {
    nodes: Vec<Node<C>>,
    entries: Vec<C>,
    root: usize,
    dim: usize,
    branching: usize,
    leaf_cap: usize,
    threshold: R,
    max_leaves: usize,
    rebuilds: usize,
    dist: D,
    abs: A,
    /// Huber/winsorization radius (in per-dimension std units): an inserted point's coordinates are
    /// clamped to within `k·σ` of its target microcluster before folding in, so outliers cannot
    /// stretch the centroid or radius. `None` = off (plain, non-robust updates).
    huber_k: Option<R>,
}

/// A microcluster must hold at least this many points before its scale is trusted enough to clip
/// against (avoids winsorizing wildly against a 1–2-point estimate during warm-up).
const ROBUST_MIN_WEIGHT: f64 = 5.0;

impl<R: Real, C: ClusterFeature<R>, D: CFDistance<R, C>, A: CFDistance<R, C>> CFTree<R, C, D, A> {
    /// New empty tree. `branching` = max children per internal node, `leaf_cap` = max entries
    /// per leaf, `threshold` = absorption limit (units of `abs`, squared for euclidean),
    /// `max_leaves` = entry count that triggers a rebuild with a grown threshold.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: usize,
        branching: usize,
        leaf_cap: usize,
        threshold: R,
        max_leaves: usize,
        dist: D,
        abs: A,
    ) -> Self {
        let root = Node {
            cf: C::new(dim),
            children: Vec::new(),
            leaf: true,
            parent: None,
        };
        Self {
            nodes: vec![root],
            entries: Vec::new(),
            root: 0,
            dim,
            branching,
            leaf_cap,
            threshold,
            max_leaves,
            rebuilds: 0,
            dist,
            abs,
            huber_k: None,
        }
    }

    /// Enable robust (Huber/winsorized) point insertion: each point is clamped to within `k`
    /// per-dimension standard deviations of its target microcluster before being folded in. `None`
    /// disables it. Affects only point inserts; rebuild reinserts of existing CFs are unaffected.
    pub fn set_huber_k(&mut self, k: Option<R>) {
        self.huber_k = k;
    }

    /// The leaf micro-clusters (used as input to global clustering).
    pub fn leaf_features(&self) -> &[C] {
        &self.entries
    }

    /// Number of leaf entries (micro-clusters).
    pub fn num_leaves(&self) -> usize {
        self.entries.len()
    }

    /// Exponentially decay every feature in the tree by `factor ∈ (0, 1]` — recent data dominates
    /// without distorting cluster shapes (EWMA / concept-drift streaming).
    pub fn decay(&mut self, factor: R) {
        for node in &mut self.nodes {
            node.cf.decay(factor);
        }
        for e in &mut self.entries {
            e.decay(factor);
        }
    }

    /// Build a tree from `n` row-major points in `flat` using `shards` parallel workers: each worker
    /// builds an independent sub-tree over a contiguous slice, then their leaf CFs are merged into a
    /// final tree. Phase-1 insertion is otherwise serial (each insert depends on the tree state), so
    /// this is the main lever for large `N`. The result is a *valid* summary of all points with
    /// exact moments (CF is a commutative monoid), but its leaf structure — and hence the labels —
    /// differs from the sequential build, exactly as a different BIRCH insertion order would. Use
    /// the sequential path when bit-exact reproducibility matters.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    pub fn build_parallel(
        dim: usize,
        branching: usize,
        leaf_cap: usize,
        threshold: R,
        max_leaves: usize,
        dist: D,
        abs: A,
        flat: &[R],
        n: usize,
        shards: usize,
    ) -> Self
    where
        D: Clone,
        A: Clone,
    {
        use rayon::prelude::*;
        let shards = shards.max(1).min(n.max(1));
        let chunk = n.div_ceil(shards);
        // Each shard summarises its slice to `max_leaves / shards` leaves — the same points-per-leaf
        // granularity as the sequential build (`N/shards ÷ max_leaves/shards = N/max_leaves`), so the
        // merge handles only ~`max_leaves` CFs total instead of `shards · max_leaves` (which would
        // make the sequential merge dominate and erase the parallel gain).
        let sub_max = (max_leaves / shards).max(leaf_cap.max(branching));
        let subtrees: Vec<Self> = (0..shards)
            .into_par_iter()
            .map(|s| {
                let lo = s * chunk;
                let hi = ((s + 1) * chunk).min(n);
                let mut t = Self::new(
                    dim,
                    branching,
                    leaf_cap,
                    threshold,
                    sub_max,
                    dist.clone(),
                    abs.clone(),
                );
                for i in lo..hi {
                    t.insert(&flat[i * dim..(i + 1) * dim]);
                }
                t
            })
            .collect();
        let mut tree = Self::new(dim, branching, leaf_cap, threshold, max_leaves, dist, abs);
        for sub in &subtrees {
            for cf in sub.leaf_features() {
                tree.insert_cf(cf.clone());
            }
        }
        tree
    }

    /// Root summary (covers all inserted points).
    pub fn summary(&self) -> &C {
        &self.nodes[self.root].cf
    }

    /// Number of times the tree has been rebuilt (threshold-grown) under the leaf bound.
    pub fn rebuilds(&self) -> usize {
        self.rebuilds
    }

    /// Current absorption threshold (grows as the tree rebuilds under the leaf bound).
    pub fn threshold(&self) -> R {
        self.threshold
    }

    /// Index of the leaf entry nearest to `x` (assigns a point to a micro-cluster).
    pub fn nearest_entry(&self, x: &[R]) -> usize {
        let leaf = self.descend(x);
        let ch = &self.nodes[leaf].children;
        let mut best = ch[0];
        let mut bd = self.dist.point(&self.entries[best], x);
        for &e in &ch[1..] {
            let d = self.dist.point(&self.entries[e], x);
            if d < bd {
                bd = d;
                best = e;
            }
        }
        best
    }

    /// Insert an existing feature (used when rebuilding with a larger threshold).
    pub fn insert_cf(&mut self, cf: C) {
        let leaf = self.descend_cf(&cf);
        let mut cur = Some(leaf);
        while let Some(n) = cur {
            self.nodes[n].cf.merge(&cf);
            cur = self.nodes[n].parent;
        }
        if !self.try_absorb_cf(leaf, &cf) {
            let eid = self.entries.len();
            self.entries.push(cf);
            self.nodes[leaf].children.push(eid);
        }
        self.split_up(leaf);
    }

    fn descend_cf(&self, cf: &C) -> usize {
        let mut cur = self.root;
        while !self.nodes[cur].leaf {
            let ch = &self.nodes[cur].children;
            let mut best = ch[0];
            let mut bd = self.dist.between(&self.nodes[best].cf, cf);
            for &c in &ch[1..] {
                let d = self.dist.between(&self.nodes[c].cf, cf);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            cur = best;
        }
        cur
    }

    fn try_absorb_cf(&mut self, leaf: usize, cf: &C) -> bool {
        let n = self.nodes[leaf].children.len();
        if n == 0 {
            return false;
        }
        let mut best = self.nodes[leaf].children[0];
        let mut bd = self.dist.between(&self.entries[best], cf);
        for i in 1..n {
            let e = self.nodes[leaf].children[i];
            let d = self.dist.between(&self.entries[e], cf);
            if d < bd {
                bd = d;
                best = e;
            }
        }
        if self.abs.between(&self.entries[best], cf) <= self.threshold {
            self.entries[best].merge(cf);
            true
        } else {
            false
        }
    }

    /// Rebuild a smaller tree by reinserting the leaf entries under a raised threshold (BIRCH
    /// reducibility). The threshold rises monotonically to the within-leaf nearest-sibling estimate —
    /// never lowered, never force-grown. A single rebuild already shrinks the tree (entries closer
    /// than the typical sibling gap merge on reinsertion), so the leaf count settles just under
    /// `max_leaves`; the old multiplicative bump compounded across the hundreds of rebuilds a large
    /// stream triggers and collapsed the tree far below `max_leaves`.
    fn rebuild(&mut self) {
        let estimate = self.estimate_threshold();
        if estimate > self.threshold {
            self.threshold = estimate;
        }

        let entries = self.collect_entries_dfs();
        self.entries.clear();
        self.nodes.clear();
        self.nodes.push(Node {
            cf: C::new(self.dim),
            children: Vec::new(),
            leaf: true,
            parent: None,
        });
        self.root = 0;
        // Reinsert in reverse DFS-leaf order. BIRCH tree shape is insertion-order dependent; the
        // reference (ELKI/betulars) reinserts back-to-front, which packs nodes more evenly and keeps
        // descend paths short — a faster *and* better-shaped tree than forward reinsertion.
        for e in entries.into_iter().rev() {
            self.insert_cf(e);
        }
        self.rebuilds += 1;
    }

    /// Mean nearest-neighbour distance between entries that *share a leaf node* — the within-leaf
    /// absorption granularity used by ELKI/BETULA.
    ///
    /// Cost is `O(Σ_leaf child_count²)` ≈ `O(m·capacity)` (a leaf holds ≤ `capacity` entries), an
    /// order of magnitude below the global all-pairs scan it replaces, yet it tracks the same scale:
    /// the threshold gates absorption *within* a leaf, so the typical nearest-sibling gap there is
    /// exactly the quantity it should reflect — and unlike a sampled global scan it cannot
    /// systematically over-estimate the NN and collapse the tree below `max_leaves`. The nearest
    /// sibling is located under the routing measure (`dist`) and its value taken under the absorption
    /// measure (`abs`), mirroring how insertion routes then absorbs. Distances are averaged in linear
    /// space and squared back (ELKI convention); `1 + 4ε` guards `sqrt(d)² < d` rounding at the gate.
    fn estimate_threshold(&self) -> R {
        let mut sum = R::zero();
        let mut count = 0usize;
        for node in &self.nodes {
            if !node.leaf || node.children.len() < 2 {
                continue;
            }
            let ch = &node.children;
            for (i, &ei) in ch.iter().enumerate() {
                let entry = &self.entries[ei];
                let mut best = R::infinity();
                let mut best_e = ei;
                for (j, &ej) in ch.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let d = self.dist.between(entry, &self.entries[ej]);
                    if d < best {
                        best = d;
                        best_e = ej;
                    }
                }
                sum = sum + self.abs.between(entry, &self.entries[best_e]).sqrt();
                count += 1;
            }
        }
        if count == 0 {
            return self.threshold;
        }
        let mean = sum / R::from_usize(count).unwrap();
        mean * mean * (R::one() + R::from_f64(4.0).unwrap() * R::epsilon())
    }

    fn collect_entries_dfs(&self) -> Vec<C> {
        let mut out = Vec::with_capacity(self.entries.len());
        self.collect_from(self.root, &mut out);
        out
    }

    fn collect_from(&self, id: usize, out: &mut Vec<C>) {
        let node = &self.nodes[id];
        if node.leaf {
            for &e in &node.children {
                out.push(self.entries[e].clone());
            }
        } else {
            for &c in &node.children {
                self.collect_from(c, out);
            }
        }
    }

    /// Insert a point.
    pub fn insert(&mut self, x: &[R]) {
        debug_assert!(x.len() >= self.dim);
        if let Some(k) = self.huber_k {
            self.insert_robust(x, k);
            return;
        }
        let leaf = self.descend(x);
        // Fold the point into every node on the leaf→root path incrementally: a CF push is exact and
        // associative, so each ancestor's CF stays equal to the merge of its subtree — `O(d)` per
        // level, versus recomputing each node from all of its children (`O(branching·d)` per level)
        // on every insert.
        let mut cur = Some(leaf);
        while let Some(n) = cur {
            self.nodes[n].cf.push(x, R::one());
            cur = self.nodes[n].parent;
        }
        if !self.try_absorb(leaf, x) {
            let mut e = C::new(self.dim);
            e.push(x, R::one());
            let eid = self.entries.len();
            self.entries.push(e);
            self.nodes[leaf].children.push(eid);
        }
        self.split_up(leaf);
        if self.entries.len() > self.max_leaves {
            self.rebuild();
        }
    }

    /// Robust point insert: winsorize `x` to within `k·σ` of its nearest mature microcluster before
    /// folding it in, so a single outlier cannot stretch a centroid or radius. The clip is applied
    /// once, up front, then the *same* clipped point flows into the ancestor CFs and the leaf entry —
    /// the CF-is-a-monoid invariant (every node = merge of its subtree) is preserved exactly. Falls
    /// back to the raw point when the target leaf is empty or its nearest entry is too small to give a
    /// trustworthy scale (warm-up).
    fn insert_robust(&mut self, x: &[R], k: R) {
        let leaf = self.descend(x);
        let min_w = R::from_f64(ROBUST_MIN_WEIGHT).unwrap();
        let clipped = self
            .nearest_in_leaf(leaf, x)
            .filter(|&e| self.entries[e].weight() >= min_w)
            .map(|e| self.clip_point(x, e, k));
        let xc: &[R] = clipped.as_deref().unwrap_or(x);

        let mut cur = Some(leaf);
        while let Some(n) = cur {
            self.nodes[n].cf.push(xc, R::one());
            cur = self.nodes[n].parent;
        }
        if !self.try_absorb(leaf, xc) {
            let mut e = C::new(self.dim);
            e.push(xc, R::one());
            let eid = self.entries.len();
            self.entries.push(e);
            self.nodes[leaf].children.push(eid);
        }
        self.split_up(leaf);
        if self.entries.len() > self.max_leaves {
            self.rebuild();
        }
    }

    /// Nearest leaf entry to `x` within `leaf`, or `None` when the leaf has no entries yet.
    fn nearest_in_leaf(&self, leaf: usize, x: &[R]) -> Option<usize> {
        let (&first, rest) = self.nodes[leaf].children.split_first()?;
        let mut best = first;
        let mut bd = self.dist.point(&self.entries[best], x);
        for &e in rest {
            let d = self.dist.point(&self.entries[e], x);
            if d < bd {
                bd = d;
                best = e;
            }
        }
        Some(best)
    }

    /// Winsorize `x` against entry `e`: clamp each coordinate to `[μ_j − k·σ_j, μ_j + k·σ_j]` where
    /// `σ_j = √variance(e, j)`. Dimensions with zero variance (degenerate / single-point clusters)
    /// have no scale to clip against and are passed through unchanged.
    fn clip_point(&self, x: &[R], e: usize, k: R) -> Vec<R> {
        let cf = &self.entries[e];
        let mu = cf.mean();
        (0..self.dim)
            .map(|j| {
                let sd = cf.variance(j).sqrt();
                if sd > R::zero() {
                    x[j].max(mu[j] - k * sd).min(mu[j] + k * sd)
                } else {
                    x[j]
                }
            })
            .collect()
    }

    fn descend(&self, x: &[R]) -> usize {
        let mut cur = self.root;
        while !self.nodes[cur].leaf {
            cur = self.nearest_child(cur, x);
        }
        cur
    }

    fn nearest_child(&self, node: usize, x: &[R]) -> usize {
        let ch = &self.nodes[node].children;
        let mut best = ch[0];
        let mut bestd = self.dist.point(&self.nodes[best].cf, x);
        for &c in &ch[1..] {
            let d = self.dist.point(&self.nodes[c].cf, x);
            if d < bestd {
                bestd = d;
                best = c;
            }
        }
        best
    }

    fn try_absorb(&mut self, leaf: usize, x: &[R]) -> bool {
        let n = self.nodes[leaf].children.len();
        if n == 0 {
            return false;
        }
        // Index into `children` per step (each access copies a `usize`) so no `Vec` is cloned per
        // insert and the borrow of `nodes` never overlaps the `entries` mutation below.
        let mut best = self.nodes[leaf].children[0];
        let mut bestd = self.dist.point(&self.entries[best], x);
        for i in 1..n {
            let e = self.nodes[leaf].children[i];
            let d = self.dist.point(&self.entries[e], x);
            if d < bestd {
                bestd = d;
                best = e;
            }
        }
        if self.abs.point(&self.entries[best], x) <= self.threshold {
            self.entries[best].push(x, R::one());
            true
        } else {
            false
        }
    }

    /// Walk `leaf`→root splitting any overflowing node, recomputing only the two nodes a split
    /// repartitions (and a freshly created root). The caller has already folded the new data into
    /// every ancestor's CF incrementally, so non-splitting levels need no work — the walk stops at
    /// the first node that fits (a split is the only thing that grows a parent). This keeps the
    /// "double update on split" bug class unrepresentable without an `O(branching)` recompute per
    /// level of every insert.
    fn split_up(&mut self, leaf: usize) {
        let mut node = leaf;
        loop {
            let cap = if self.nodes[node].leaf {
                self.leaf_cap
            } else {
                self.branching
            };
            if self.nodes[node].children.len() <= cap {
                break; // fits → no new child propagates to the parent → ancestors are unaffected
            }
            let sibling = self.split(node);
            self.recompute_cf(node);
            self.recompute_cf(sibling);
            match self.nodes[node].parent {
                Some(p) => {
                    self.nodes[p].children.push(sibling);
                    node = p;
                }
                None => {
                    let nr = self.nodes.len();
                    self.nodes.push(Node {
                        cf: C::new(self.dim),
                        children: vec![node, sibling],
                        leaf: false,
                        parent: None,
                    });
                    self.nodes[node].parent = Some(nr);
                    self.nodes[sibling].parent = Some(nr);
                    self.root = nr;
                    self.recompute_cf(nr);
                    break;
                }
            }
        }
    }

    /// Split `node`'s children into two groups (farthest-pair seeds), keeping group one in `node`
    /// and returning a new sibling holding group two.
    fn split(&mut self, node: usize) -> usize {
        let leaf = self.nodes[node].leaf;
        let children = self.nodes[node].children.clone();
        let k = children.len();
        // Snapshot child CFs so the read-only seed/assign loops don't hold a borrow of `self`
        // across the structural mutation below (k is small: at most cap + 1).
        let cfs: Vec<C> = children
            .iter()
            .map(|&c| {
                if leaf {
                    self.entries[c].clone()
                } else {
                    self.nodes[c].cf.clone()
                }
            })
            .collect();

        // farthest pair of children = the two seeds
        let (mut s1, mut s2, mut maxd) = (0usize, 1usize, R::zero());
        for i in 0..k {
            for j in (i + 1)..k {
                let d = self.dist.between(&cfs[i], &cfs[j]);
                if d > maxd {
                    maxd = d;
                    s1 = i;
                    s2 = j;
                }
            }
        }

        let (mut g1, mut g2) = (Vec::new(), Vec::new());
        for (i, &c) in children.iter().enumerate() {
            let d1 = self.dist.between(&cfs[i], &cfs[s1]);
            let d2 = self.dist.between(&cfs[i], &cfs[s2]);
            if d1 < d2 || (d1 == d2 && g1.len() <= g2.len()) {
                g1.push(c);
            } else {
                g2.push(c);
            }
        }

        let parent = self.nodes[node].parent;
        let sibling = self.nodes.len();
        self.nodes.push(Node {
            cf: C::new(self.dim),
            children: g2.clone(),
            leaf,
            parent,
        });
        self.nodes[node].children = g1;
        if !leaf {
            for &c in &g2 {
                self.nodes[c].parent = Some(sibling);
            }
        }
        sibling
    }

    fn recompute_cf(&mut self, id: usize) {
        let children = self.nodes[id].children.clone();
        let leaf = self.nodes[id].leaf;
        let mut cf = C::new(self.dim);
        for c in children {
            if leaf {
                cf.merge(&self.entries[c]);
            } else {
                cf.merge(&self.nodes[c].cf);
            }
        }
        self.nodes[id].cf = cf;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::CentroidEuclidean;
    use crate::feature::{Diagonal, Full, Spherical};

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    /// Every node's CF must equal the merge of its children (recompute-from-children invariant),
    /// and the root must summarise exactly the inserted points.
    fn verify<C: ClusterFeature<f64>, D: CFDistance<f64, C>, A: CFDistance<f64, C>>(
        tree: &CFTree<f64, C, D, A>,
        n_points: usize,
    ) {
        for id in 0..tree.nodes.len() {
            let node = &tree.nodes[id];
            if node.children.is_empty() {
                continue;
            }
            let mut cf = C::new(tree.dim);
            for &c in &node.children {
                if node.leaf {
                    cf.merge(&tree.entries[c]);
                } else {
                    cf.merge(&tree.nodes[c].cf);
                }
            }
            assert!(close(cf.weight(), node.cf.weight()), "weight node {id}");
            for d in 0..tree.dim {
                assert!(
                    close(cf.mean()[d], node.cf.mean()[d]),
                    "mean node {id} dim {d}"
                );
            }
            assert!(close(cf.ssd(), node.cf.ssd()), "ssd node {id}");
        }
        assert!(
            close(tree.summary().weight(), n_points as f64),
            "total weight"
        );
    }

    fn pseudo(n: usize, dim: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| (((i * 1103515245 + j * 12345 + 7) % 1009) as f64) / 100.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn invariant_holds_threshold_zero_forces_many_splits() {
        // threshold 0 -> nothing absorbs -> every point is its own entry -> heavy splitting.
        let mut tree: CFTree<f64, Spherical<f64>, _, _> = CFTree::new(
            2,
            4,
            4,
            0.0,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        let pts = pseudo(200, 2);
        for p in &pts {
            tree.insert(p);
        }
        verify(&tree, pts.len());
        assert!(tree.num_leaves() > 1);
    }

    #[test]
    fn invariant_holds_with_absorption() {
        let mut tree: CFTree<f64, Spherical<f64>, _, _> = CFTree::new(
            2,
            8,
            8,
            0.25,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        let pts = pseudo(500, 2);
        for p in &pts {
            tree.insert(p);
        }
        verify(&tree, pts.len());
    }

    #[test]
    fn invariant_holds_full_feature_high_dim() {
        // Full feature + dim>=4 exercises the cross-product merge during node CF recompute.
        let mut tree: CFTree<f64, Full<f64>, _, _> = CFTree::new(
            5,
            3,
            3,
            0.0,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        let pts = pseudo(120, 5);
        for p in &pts {
            tree.insert(p);
        }
        verify(&tree, pts.len());
    }

    #[test]
    fn high_threshold_absorbs_into_few_leaves() {
        let mut tree: CFTree<f64, Spherical<f64>, _, _> = CFTree::new(
            2,
            8,
            8,
            1e9,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for p in &pseudo(300, 2) {
            tree.insert(p);
        }
        assert_eq!(tree.num_leaves(), 1); // everything absorbs into one entry
        assert!(close(tree.summary().weight(), 300.0));
    }

    #[test]
    fn rebuild_bounds_leaf_count_and_keeps_invariant() {
        // threshold 0 -> every point its own entry -> exceeds max_leaves -> forces rebuilds.
        let mut tree: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(2, 8, 8, 0.0, 30, CentroidEuclidean, CentroidEuclidean);
        let pts = pseudo(400, 2);
        for p in &pts {
            tree.insert(p);
        }
        assert!(tree.rebuilds() > 0, "expected at least one rebuild");
        assert!(tree.threshold > 0.0, "threshold must have grown");
        assert!(tree.num_leaves() < pts.len(), "tree must compress");
        verify(&tree, pts.len());
    }

    #[test]
    fn estimate_threshold_tracks_within_leaf_nn() {
        // The rebuild threshold is the mean nearest-sibling gap among entries that share a leaf node.
        // For unit-spaced points the true nearest-sibling (squared) distance is 1.0 everywhere, so the
        // estimate must land at that scale — never systematically above it, which would coarsen the
        // tree below `max_leaves` on rebuild. Many entries across many leaf nodes here ⇒ this exercises
        // the per-leaf scan, not a single fused leaf.
        let mut tree: CFTree<f64, Spherical<f64>, _, _> = CFTree::new(
            1,
            16,
            16,
            0.0,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        let pts: Vec<Vec<f64>> = (0..4200).map(|i| vec![i as f64]).collect();
        for p in &pts {
            tree.insert(p);
        }
        assert!(
            tree.num_leaves() > 4096,
            "threshold 0 ⇒ no absorption ⇒ 4200 distinct entries spread over many leaf nodes"
        );
        let est = tree.estimate_threshold();
        assert!(
            (0.5..=1.5).contains(&est),
            "within-leaf threshold {est} drifted from the unit nearest-sibling scale (≈1.0)"
        );
    }

    #[test]
    fn estimate_threshold_falls_back_with_no_within_leaf_pair() {
        // `leaf_cap = 1` ⇒ every leaf node holds a single entry ⇒ there is no sibling pair to
        // measure. The estimate must fall back to the current threshold, not divide by a zero count.
        let mut tree: CFTree<f64, Spherical<f64>, _, _> = CFTree::new(
            1,
            2,
            1,
            5.0,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for p in [[0.0], [10.0], [20.0]] {
            tree.insert(&p);
        }
        assert_eq!(tree.estimate_threshold(), tree.threshold());
    }

    #[test]
    fn decay_scales_tree_mass() {
        let mut tree: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(2, 8, 8, 0.5, 200, CentroidEuclidean, CentroidEuclidean);
        let pts = pseudo(200, 2);
        for p in &pts {
            tree.insert(p);
        }
        let w0 = tree.summary().weight();
        assert!(tree.threshold() >= 0.5); // grows from the initial 0.5 across rebuilds
        tree.decay(0.5);
        assert!((tree.summary().weight() - 0.5 * w0).abs() < 1e-6);
    }

    #[test]
    fn nearest_entry_returns_closest_in_leaf() {
        // leaf_cap large + threshold 0 ⇒ several distinct entries share one leaf; the scan must pick
        // the nearest (exercises the `entries[1..]` comparison loop).
        let mut tree: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(2, 64, 64, 0.0, 200, CentroidEuclidean, CentroidEuclidean);
        for p in [[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]] {
            tree.insert(&p);
        }
        let near = tree.nearest_entry(&[9.5, 0.1]);
        assert!((tree.leaf_features()[near].mean()[0] - 10.0).abs() < 1e-9);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_build_is_exact_and_bounded() {
        // Shard+merge summarizes every point (exact total weight) and respects the leaf bound.
        let pts = pseudo(3000, 3);
        let n = pts.len();
        let flat: Vec<f64> = pts.iter().flatten().copied().collect();
        let par = CFTree::<f64, Spherical<f64>, _, _>::build_parallel(
            3,
            16,
            16,
            1.0,
            200,
            CentroidEuclidean,
            CentroidEuclidean,
            &flat,
            n,
            8,
        );
        assert!(
            close(par.summary().weight(), n as f64),
            "exact total weight"
        );
        assert!(par.num_leaves() >= 1 && par.num_leaves() <= 200);
    }

    fn dist2(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    /// A single mature entry (everything absorbs into it: huge threshold) plus one far outlier. With
    /// winsorization on, the entry centroid stays near the clean cluster mean; with it off, the raw
    /// outlier drags it far away.
    #[test]
    fn robust_winsorization_caps_outlier_pull() {
        let tight: Vec<[f64; 2]> = (0..50)
            .map(|i| [(i % 10) as f64 * 0.1, 5.0 + (i % 10) as f64 * 0.1])
            .collect();
        let outlier = [100.0, 100.0];
        let build = |huber: Option<f64>, with_outlier: bool| {
            let mut t: CFTree<f64, Diagonal<f64>, _, _> = CFTree::new(
                2,
                64,
                64,
                1e9,
                usize::MAX,
                CentroidEuclidean,
                CentroidEuclidean,
            );
            t.set_huber_k(huber);
            for p in &tight {
                t.insert(p);
            }
            if with_outlier {
                t.insert(&outlier);
            }
            assert_eq!(t.num_leaves(), 1, "huge threshold ⇒ one entry");
            t.leaf_features()[0].mean().to_vec()
        };
        let clean = build(None, false);
        let off = build(None, true);
        let on = build(Some(2.0), true);
        let drift_off = dist2(&off, &clean);
        let drift_on = dist2(&on, &clean);
        assert!(drift_off > 1.0, "raw outlier must move the centroid a lot");
        assert!(
            drift_on < 0.05,
            "clipped outlier barely moves it: {drift_on}"
        );
        assert!(drift_on < drift_off);
    }

    /// During warm-up the nearest entry is below `ROBUST_MIN_WEIGHT`, so no scale is trusted and the
    /// point is folded in unclipped — the outlier shows up at full magnitude.
    #[test]
    fn robust_falls_back_during_warmup() {
        let mut t: CFTree<f64, Diagonal<f64>, _, _> = CFTree::new(
            2,
            64,
            64,
            1e9,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        t.set_huber_k(2.0.into());
        t.insert(&[0.0, 0.0]);
        t.insert(&[100.0, 100.0]);
        assert_eq!(t.num_leaves(), 1);
        assert!(
            close(t.leaf_features()[0].mean()[0], 50.0),
            "no warm-up clip"
        );
    }

    /// A dimension with zero variance offers no scale to clip against and must pass through unchanged,
    /// while a dimension with spread is still winsorized.
    #[test]
    fn robust_passes_through_zero_variance_dim() {
        let mut t: CFTree<f64, Diagonal<f64>, _, _> = CFTree::new(
            2,
            64,
            64,
            1e9,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        t.set_huber_k(2.0.into());
        for i in 0..50 {
            t.insert(&[(i % 10) as f64 * 0.1, 5.0]); // dim 1 constant ⇒ variance 0
        }
        t.insert(&[100.0, 100.0]);
        let mean = t.leaf_features()[0].mean();
        assert!(mean[0] < 1.0, "dim 0 has spread ⇒ clipped: {}", mean[0]);
        assert!(
            mean[1] > 6.0,
            "dim 1 zero-variance ⇒ outlier passes: {}",
            mean[1]
        );
    }

    /// Toggling robust mode back off reproduces the plain insertion path exactly.
    #[test]
    fn robust_off_matches_plain_tree() {
        let pts = pseudo(400, 3);
        let mut plain: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(3, 8, 8, 0.5, 200, CentroidEuclidean, CentroidEuclidean);
        let mut toggled: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(3, 8, 8, 0.5, 200, CentroidEuclidean, CentroidEuclidean);
        toggled.set_huber_k(2.0.into());
        toggled.set_huber_k(None);
        for p in &pts {
            plain.insert(p);
            toggled.insert(p);
        }
        assert_eq!(plain.num_leaves(), toggled.num_leaves());
        for (a, b) in plain.leaf_features().iter().zip(toggled.leaf_features()) {
            assert!(close(a.mean()[0], b.mean()[0]));
            assert!(close(a.weight(), b.weight()));
        }
    }

    /// The CF-is-a-monoid invariant (each node = merge of its subtree, exact total weight) must hold
    /// under robust inserts just as it does for plain inserts.
    #[test]
    fn robust_insert_preserves_cf_invariant() {
        let mut t: CFTree<f64, Diagonal<f64>, _, _> = CFTree::new(
            2,
            8,
            8,
            0.25,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        t.set_huber_k(3.0.into());
        let pts = pseudo(500, 2);
        for p in &pts {
            t.insert(p);
        }
        verify(&t, pts.len());
    }

    /// Robust inserts must coexist with the leaf-bound rebuild: a small `max_leaves` forces rebuilds,
    /// the invariant still holds, and the threshold grows monotonically.
    #[test]
    fn robust_insert_rebuilds_under_leaf_bound() {
        let mut t: CFTree<f64, Diagonal<f64>, _, _> =
            CFTree::new(2, 8, 8, 0.05, 40, CentroidEuclidean, CentroidEuclidean);
        t.set_huber_k(2.0.into());
        let pts = pseudo(2000, 2);
        for p in &pts {
            t.insert(p);
        }
        assert!(t.rebuilds() > 0, "small max_leaves must trigger a rebuild");
        assert!(t.num_leaves() <= 40);
        assert!(t.threshold() >= 0.05);
        verify(&t, pts.len());
    }
}
