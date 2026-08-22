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
//! reducibility), reinserting the existing leaf features via [`CFTree::insert_cf`]. The grown
//! threshold is the order statistic of the within-leaf nearest-sibling distances that the leaf
//! budget asks for, so the rebuilt tree lands just under `max_leaves` instead of overshooting it.

use crate::distance::CFDistance;
use crate::feature::ClusterFeature;
use crate::types::Real;
use core::cmp::Ordering;

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
    /// Entries merged by compaction since the last rebalance.
    #[cfg_attr(feature = "persistence", serde(default))]
    merged_since_rebalance: usize,
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
            merged_since_rebalance: 0,
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

    /// Bring the entry count back under `max_leaves` by merging the closest sibling pairs, and raise
    /// the absorption threshold to the widest gap that took (BIRCH reducibility).
    ///
    /// Two departures from the textbook rebuild, both measured:
    ///
    /// *The count is reduced in place, not by reinserting every entry.* Merging two entries **inside
    /// their own leaf node** leaves every node CF in the tree exactly unchanged — a node's CF is the
    /// merge of its subtree, and merging two of its children does not change that multiset union. Mass
    /// is conserved per node, so no ancestor needs touching and no leaf can be emptied. That splits
    /// the two jobs BIRCH's rebuild conflates: *reducing the count*, which is all the leaf bound asks
    /// for, and *rebalancing the node structure*, which costs a descent per entry. Cost is one
    /// `O(Σ_leaf child_count²)` sibling scan plus an `O(m log m)` sort, against `O(m · depth ·
    /// branching)`. Compaction cannot reach pairs that landed in different leaves, so [`Self::reinsert`]
    /// remains the fallback — taken only when merging every available sibling pair still leaves the
    /// tree over budget.
    ///
    /// *The number of merges is chosen, not predicted.* Growing the threshold first and merging
    /// whatever falls under it makes the resulting count a guess, and under concentration of measure
    /// that guess has no safe value: on 3000-dimensional TF-IDF the achievable leaf count jumps from
    /// 7755 to 12 between thresholds 1.0 and 1.3, so *every* threshold-first policy either fails to
    /// reduce or collapses the tree — measured at 3 leaves against a 2000 budget. Merging the `k`
    /// closest pairs and reading the threshold off the last one inverts that: `k` is exact, and the
    /// cliff cannot be stepped over because merging is capped at one pair per entry per rebuild.
    /// The 10% margin below `max_leaves` is what keeps the next insert from rebuilding immediately.
    fn rebuild(&mut self) {
        let target = (self.max_leaves - self.max_leaves / 10).max(1);
        let want = self.entries.len().saturating_sub(target);
        let mut pairs = self.sibling_pairs();
        pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let mut alive = vec![true; self.entries.len()];
        let mut merged = 0usize;
        let mut widest = R::zero();
        for (gap, ei, ej) in pairs {
            if merged == want {
                break;
            }
            if !alive[ei] || !alive[ej] {
                continue;
            }
            let absorbed = self.entries[ej].clone();
            self.entries[ei].merge(&absorbed);
            alive[ej] = false;
            merged += 1;
            widest = gap;
        }
        if merged > 0 {
            // Absorption is gated on `<= threshold`, so the widest gap merged here must itself pass;
            // `1 + 4ε` covers the `sqrt(d)² < d` rounding the linear-space scan introduces.
            let grown = widest * (R::one() + R::from_f64(4.0).unwrap() * R::epsilon());
            if grown > self.threshold {
                self.threshold = grown;
            }
            self.drop_merged(&alive);
            self.merged_since_rebalance += merged;
            if self.merged_since_rebalance >= self.max_leaves {
                self.merged_since_rebalance = 0;
                self.reinsert();
            }
        }
        self.rebuilds += 1;
    }

    /// Every leaf entry paired with its nearest sibling *inside its own leaf node*: `(gap, entry,
    /// sibling)`, with the sibling chosen under the routing measure (`dist`) and the gap measured
    /// under the absorption measure (`abs`) — mirroring how insertion routes, then absorbs. Each
    /// entry contributes at most one pair, which is what caps a rebuild at one merge per entry.
    fn sibling_pairs(&self) -> Vec<(R, usize, usize)> {
        let mut out = Vec::with_capacity(self.entries.len());
        for node in &self.nodes {
            if !node.leaf || node.children.len() < 2 {
                continue;
            }
            let ch = &node.children;
            for (i, &ei) in ch.iter().enumerate() {
                let entry = &self.entries[ei];
                let mut best = ei;
                let mut bd = R::infinity();
                for (j, &ej) in ch.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let d = self.dist.between(entry, &self.entries[ej]);
                    if d < bd {
                        bd = d;
                        best = ej;
                    }
                }
                out.push((self.abs.between(entry, &self.entries[best]), ei, best));
            }
        }
        out
    }

    /// Compact the entry arena after a merge pass, dropping every entry marked dead and rewriting the
    /// leaf child lists to the new indices.
    fn drop_merged(&mut self, alive: &[bool]) {
        let mut remap = vec![0usize; self.entries.len()];
        let mut kept = Vec::with_capacity(self.entries.len());
        for (old, e) in std::mem::take(&mut self.entries).into_iter().enumerate() {
            if alive[old] {
                remap[old] = kept.len();
                kept.push(e);
            }
        }
        self.entries = kept;
        for node in &mut self.nodes {
            if node.leaf {
                node.children.retain(|&e| alive[e]);
                for c in &mut node.children {
                    *c = remap[*c];
                }
            }
        }
    }

    /// Rebalance: route every leaf entry through a fresh tree, merging nothing.
    ///
    /// Compaction merges strictly within a leaf, so it can shrink a leaf that mixes two clusters but
    /// never split it — insertion order decides which entries share a node, and nothing afterwards
    /// revisits that decision. Rebalancing does: each entry is routed against the tree as it now
    /// stands, so leaves re-partition around the geometry the data actually has. Absorption stays off
    /// throughout, which keeps the two jobs separate — the entry count is compaction's to set, and a
    /// rebalance that also merged would be free to walk off the concentration cliff compaction was
    /// built to avoid (measured: reinserting *with* absorption collapses a d = 50 blob mixture to 9
    /// leaves against a 500 budget).
    fn reinsert(&mut self) {
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
            self.place_cf(e);
        }
    }

    /// Route a feature to its leaf and keep it as its own entry — [`Self::insert_cf`] without the
    /// absorption step.
    fn place_cf(&mut self, cf: C) {
        let leaf = self.descend_cf(&cf);
        let mut cur = Some(leaf);
        while let Some(n) = cur {
            self.nodes[n].cf.merge(&cf);
            cur = self.nodes[n].parent;
        }
        let eid = self.entries.len();
        self.entries.push(cf);
        self.nodes[leaf].children.push(eid);
        self.split_up(leaf);
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
    fn absorption_boundary_is_inclusive() {
        // A point at squared distance EXACTLY `threshold` from an entry must be absorbed — the gate is
        // `<= threshold`, not `<`. `num_leaves` can't tell (both fit one leaf node under leaf_cap), so
        // assert the entry count: absorbed -> 1 entry; a spurious `<` split -> 2.
        let mut tree: CFTree<f64, Spherical<f64>, _, _> = CFTree::new(
            2,
            4,
            8,
            4.0,
            usize::MAX,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        tree.insert(&[0.0, 0.0]); // seeds one entry at the origin
        tree.insert(&[2.0, 0.0]); // ‖(2,0)‖² = 4.0 == threshold -> must absorb, not seed a new entry
        assert_eq!(
            tree.entries.len(),
            1,
            "a point at distance² == threshold must be absorbed"
        );
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
    fn rebuild_lands_near_the_leaf_budget() {
        // `max_leaves` is a resolution budget: the summary handed to the global clustering is only as
        // fine as the leaves it actually keeps. A rebuild that overshoots throws away resolution the
        // caller asked and paid for, and the old policy — grow the threshold to the *mean* sibling gap,
        // then merge whatever falls under it — routinely spent a third of the budget that way.
        for budget in [200usize, 600] {
            let mut tree: CFTree<f64, Spherical<f64>, _, _> =
                CFTree::new(6, 16, 16, 0.0, budget, CentroidEuclidean, CentroidEuclidean);
            let pts = pseudo(20_000, 6);
            for p in &pts {
                tree.insert(p);
            }
            let used = tree.num_leaves() as f64 / budget as f64;
            assert!(
                (0.8..=1.0).contains(&used),
                "budget {budget}: {} leaves is {used:.2} of it",
                tree.num_leaves()
            );
            verify(&tree, pts.len());
        }
    }

    #[test]
    fn rebuild_cannot_step_over_a_concentration_cliff() {
        // In high dimension pairwise distances concentrate, so the leaf count is a near-discontinuous
        // function of the threshold: below the bulk of the distance distribution nothing merges, above
        // it everything does. Any policy that picks a threshold and then merges whatever falls under it
        // is therefore one step away from collapsing the tree — measured at 3 leaves against a 2000
        // budget on TF-IDF. Choosing the number of merges instead caps a rebuild at one merge per
        // entry, which makes stepping over the cliff unrepresentable rather than unlikely.
        //
        // The data reproduces the geometry that causes it: unit-norm vectors with 8 of 256 dimensions
        // populated, as an L2-normalized bag of words is. Disjoint supports sit at squared distance
        // exactly 2, pairs sharing `k` terms at `2 - k/4`, so the distribution is a spike at 2 with a
        // thin left tail — and a threshold anywhere past the spike absorbs everything.
        let dim = 256;
        let pts: Vec<Vec<f64>> = (0..6000)
            .map(|i| {
                let mut v = vec![0.0; dim];
                let mut h = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
                for _ in 0..8 {
                    h ^= h >> 33;
                    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
                    v[(h >> 32) as usize % dim] = 1.0 / (8.0f64).sqrt();
                }
                v
            })
            .collect();
        let mut tree: CFTree<f64, Spherical<f64>, _, _> =
            CFTree::new(dim, 16, 16, 0.0, 300, CentroidEuclidean, CentroidEuclidean);
        for p in &pts {
            tree.insert(p);
        }
        assert!(
            tree.num_leaves() >= 150,
            "collapsed to {} leaves against a 300 budget",
            tree.num_leaves()
        );
        verify(&tree, pts.len());
    }

    #[test]
    fn sibling_pairs_track_the_within_leaf_nn() {
        // A rebuild merges the closest sibling pairs and reads the grown threshold off the widest gap
        // it took, so every gap the scan reports must sit at the true nearest-sibling scale. For
        // unit-spaced points that (squared) distance is 1.0 everywhere; a gap systematically above it
        // would coarsen the tree below `max_leaves` on rebuild. Many entries across many leaf nodes
        // here ⇒ this exercises the per-leaf scan, not a single fused leaf.
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
        let pairs = tree.sibling_pairs();
        assert_eq!(
            pairs.len(),
            tree.num_leaves(),
            "every entry contributes exactly one pair"
        );
        for (gap, ei, ej) in pairs {
            assert_ne!(ei, ej, "an entry must not pair with itself");
            assert!(
                (0.5..=1.5).contains(&gap),
                "sibling gap {gap} drifted from the unit nearest-sibling scale (≈1.0)"
            );
        }
    }

    #[test]
    fn no_sibling_pairs_when_every_leaf_holds_one_entry() {
        // `leaf_cap = 1` ⇒ every leaf node holds a single entry ⇒ there is no sibling pair to merge.
        // The scan must come back empty rather than pairing an entry with itself; a rebuild here has
        // nothing to compact and falls through to a reinsertion.
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
        assert!(tree.sibling_pairs().is_empty());
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
    /// Build a one-leaf tree holding exactly the given points as separate entries, so the private
    /// descent/absorb/clip paths can be driven with known geometry.
    fn tree_with_entries(
        pts: &[&[f64]],
        threshold: f64,
    ) -> CFTree<f64, Spherical<f64>, CentroidEuclidean, CentroidEuclidean> {
        let mut t = CFTree::new(
            pts[0].len(),
            8,
            16,
            threshold,
            10_000,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for p in pts {
            t.insert(p);
        }
        t
    }

    #[test]
    fn clip_point_clamps_each_coordinate_to_k_sigma_and_passes_through_flat_dimensions() {
        // One entry from two points: mean [3, 5], per-dimension variance 1 in x (spherical pools
        // both dims: ssd = (1² + 0²)·2/2 = 2 over 2 dims ⇒ variance 0.5 each), so sd = √0.5.
        let t = tree_with_entries(&[&[2.0, 5.0], &[4.0, 5.0]], 1e9);
        let sd = 0.5_f64.sqrt();
        let clipped = t.clip_point(&[100.0, -100.0], 0, 2.0);
        assert!(
            close(clipped[0], 3.0 + 2.0 * sd),
            "upper clamp: {clipped:?}"
        );
        assert!(
            close(clipped[1], 5.0 - 2.0 * sd),
            "lower clamp: {clipped:?}"
        );
        // Inside the band the point is untouched.
        let inside = t.clip_point(&[3.1, 5.1], 0, 2.0);
        assert!(close(inside[0], 3.1) && close(inside[1], 5.1), "{inside:?}");

        // A zero-variance entry has no band at all and must pass the coordinate through unchanged
        // rather than collapsing it onto the mean.
        let flat = tree_with_entries(&[&[7.0, 7.0]], 1e9);
        let out = flat.clip_point(&[99.0, -99.0], 0, 2.0);
        assert!(close(out[0], 99.0) && close(out[1], -99.0), "{out:?}");
    }

    #[test]
    fn descent_and_absorption_pick_the_nearest_candidate_not_the_first() {
        // Three well-separated entries in one leaf, ordered so that the nearest to the probe is the
        // *last* one inserted: a scan that returns index 0, or one that never updates its best, gets
        // a different entry.
        let mut t = tree_with_entries(&[&[0.0], &[50.0], &[100.0]], 1e-9);
        assert_eq!(t.nodes[t.root].children.len(), 3, "expected three entries");
        assert_eq!(t.descend(&[99.0]), t.root, "single leaf tree");

        // Absorption below threshold must land on entry 2, not entry 0.
        let before: Vec<f64> = (0..3).map(|e| t.entries[e].weight()).collect();
        t.threshold = 10.0;
        t.insert(&[99.0]);
        let after: Vec<f64> = (0..3).map(|e| t.entries[e].weight()).collect();
        assert_eq!(
            (
                after[0] - before[0],
                after[1] - before[1],
                after[2] - before[2]
            ),
            (0.0, 0.0, 1.0),
            "absorbed into the wrong entry: {before:?} -> {after:?}"
        );

        // With many entries the tree grows internal nodes; the descent must still reach the leaf
        // whose subtree is nearest, so a far probe lands beside the far cluster.
        let mut wide = CFTree::<f64, Spherical<f64>, _, _>::new(
            1,
            2,
            2,
            1e-9,
            10_000,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for i in 0..16 {
            wide.insert(&[i as f64 * 10.0]);
        }
        // Descent must agree with an independent walk that takes the nearest child at every level,
        // and must reach more than one leaf across the probes -- a `descend` that always returns the
        // same node satisfies "the leaf is non-empty" without doing any work.
        fn reference_descend(
            t: &CFTree<f64, Spherical<f64>, CentroidEuclidean, CentroidEuclidean>,
            x: &[f64],
        ) -> usize {
            let mut cur = t.root;
            while !t.nodes[cur].leaf {
                cur = *t.nodes[cur]
                    .children
                    .iter()
                    .min_by(|&&a, &&b| {
                        t.dist
                            .point(&t.nodes[a].cf, x)
                            .total_cmp(&t.dist.point(&t.nodes[b].cf, x))
                    })
                    .expect("internal node without children");
            }
            cur
        }
        let mut reached = std::collections::BTreeSet::new();
        for probe in [-5.0, 21.0, 73.0, 118.0, 151.0] {
            let got = wide.descend(&[probe]);
            assert_eq!(got, reference_descend(&wide, &[probe]), "probe {probe}");
            assert!(wide.nodes[got].leaf, "descend stopped at an internal node");
            reached.insert(got);
        }
        assert!(reached.len() > 1, "every probe landed in the same leaf");

        let leaf = wide.descend(&[151.0]);
        assert!(!wide.nodes[leaf].children.is_empty());
        // Exact tie: two children at 0 and 8, probed at 4. Squared distances are 16 either way, so
        // the scan must keep the first candidate rather than sliding onto the last equal one.
        let mut tied = CFTree::<f64, Spherical<f64>, _, _>::new(
            1,
            2,
            1,
            1e-9,
            10_000,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        tied.insert(&[0.0]);
        tied.insert(&[8.0]);
        let kids = tied.nodes[tied.root].children.clone();
        assert_eq!(kids.len(), 2, "expected a split into two children");
        // Neither group may come out of a split empty: an empty leaf is invisible to `verify`,
        // which skips childless nodes, and the tree silently loses a branch of its fan-out.
        for &c in &kids {
            assert!(
                !tied.nodes[c].children.is_empty(),
                "split left node {c} empty: {:?}",
                kids
            );
        }
        assert_eq!(
            tied.nearest_child(tied.root, &[4.0]),
            kids[0],
            "an exact tie must keep the first child"
        );

        let near = wide.nearest_child(wide.root, &[151.0]);
        let far = wide.nearest_child(wide.root, &[-5.0]);
        assert_ne!(near, far, "every probe descends into the same child");
    }

    /// Return each leaf's entry set, sorted, so a split's partition can be asserted directly.
    fn leaf_groups(
        t: &CFTree<f64, Spherical<f64>, CentroidEuclidean, CentroidEuclidean>,
    ) -> Vec<Vec<usize>> {
        let mut g: Vec<Vec<usize>> = (0..t.nodes.len())
            .filter(|&i| t.nodes[i].leaf && !t.nodes[i].children.is_empty())
            .map(|i| {
                let mut c = t.nodes[i].children.clone();
                c.sort_unstable();
                c
            })
            .collect();
        g.sort();
        g
    }

    #[test]
    fn a_child_equidistant_from_both_seeds_joins_the_smaller_group() {
        // Entries at 0, 10 and 5 with a leaf capacity of 2. The seeds are the farthest pair (0, 10);
        // the midpoint at 5 is exactly 25 from each, so only the size rule can place it. Both groups
        // hold one entry when it is weighed, so it must join the first.
        let mut t = CFTree::<f64, Spherical<f64>, _, _>::new(
            1,
            8,
            2,
            1e-9,
            10_000,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for x in [0.0, 10.0, 5.0] {
            t.insert(&[x]);
        }
        verify(&t, 3);
        assert_eq!(
            leaf_groups(&t),
            vec![vec![0, 2], vec![1]],
            "tied child went to the wrong group"
        );
    }

    #[test]
    fn the_seed_pair_is_the_first_farthest_pair_encountered() {
        // Unit square: both diagonals are at squared distance 2, so the farthest pair is tied.
        // Scanning with `>` keeps the first diagonal (corners 0 and 2) as the seeds, and the two
        // side corners then split by group size into {0,3} and {1,2}. Sliding onto the second
        // diagonal instead (`>=`) partitions as {0,1} / {2,3}, so the expected value pins the seed
        // scan and the size rule together.
        let mut t = CFTree::<f64, Spherical<f64>, _, _>::new(
            2,
            8,
            3,
            1e-9,
            10_000,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for p in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]] {
            t.insert(&p);
        }
        verify(&t, 4);
        assert_eq!(
            leaf_groups(&t),
            vec![vec![0, 3], vec![1, 2]],
            "the split ran along the wrong diagonal"
        );
    }

    #[test]
    fn a_split_breaks_ties_toward_the_smaller_group() {
        // Four children on a line at 0, 1, 9, 10 with a branching factor of 2 forces a split. The
        // seeds are the farthest pair (0 and 10); the tie rule only shows up when a child is
        // equidistant from both seeds, which the midpoint at 5 supplies.
        let mut t = CFTree::<f64, Spherical<f64>, _, _>::new(
            1,
            2,
            1,
            1e-9,
            10_000,
            CentroidEuclidean,
            CentroidEuclidean,
        );
        for x in [0.0, 10.0, 5.0, 1.0, 9.0] {
            t.insert(&[x]);
        }
        verify(&t, 5);
        assert!(t.nodes.len() > 1, "tree never split");
        // Every entry survives the split exactly once, and no group is left empty.
        assert_eq!(t.entries.len(), 5);
        for id in 0..t.nodes.len() {
            if !t.nodes[id].children.is_empty() {
                assert!(
                    !t.nodes[id].children.is_empty(),
                    "node {id} was left with no children"
                );
            }
        }
        let leaves: Vec<usize> = (0..t.nodes.len())
            .filter(|&i| t.nodes[i].leaf && !t.nodes[i].children.is_empty())
            .collect();
        let total: usize = leaves.iter().map(|&l| t.nodes[l].children.len()).sum();
        assert_eq!(total, 5, "split lost or duplicated entries");
    }
}

#[cfg(test)]
mod prop_tests {
    //! Property-based "tree CF = Σ points": folding every leaf microcluster reconstructs the
    //! whole-dataset feature, across random dimensions, point sets and absorption thresholds — the
    //! invariant DESIGN.md advertises for the CF-tree.
    #![allow(clippy::needless_range_loop)] // per-dimension mean comparison reads clearest with an index
    use super::*;
    use crate::distance::CentroidEuclidean;
    use crate::feature::{ClusterFeature, Full};
    use proptest::prelude::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-7 * a.abs().max(b.abs()).max(1.0)
    }

    /// `(dim, points)` with every point sharing `dim ∈ [2, 6]`.
    fn dim_points(min_pts: usize, max_pts: usize) -> impl Strategy<Value = (usize, Vec<Vec<f64>>)> {
        (2usize..=6).prop_flat_map(move |d| {
            prop::collection::vec(
                prop::collection::vec(-50.0f64..50.0, d..=d),
                min_pts..=max_pts,
            )
            .prop_map(move |pts| (d, pts))
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        #[test]
        fn tree_leaves_fold_to_sum_of_points((d, pts) in dim_points(1, 80), thr in 0.0f64..5.0) {
            let mut tree: CFTree<f64, Full<f64>, _, _> =
                CFTree::new(d, 4, 4, thr, usize::MAX, CentroidEuclidean, CentroidEuclidean);
            for p in &pts {
                tree.insert(p);
            }
            let mut folded: Full<f64> = Full::new(d);
            for leaf in tree.leaf_features() {
                folded.merge(leaf);
            }
            let mut total: Full<f64> = Full::new(d);
            for p in &pts {
                total.push(p, 1.0);
            }
            prop_assert!(close(folded.weight(), total.weight()), "weight");
            for i in 0..d {
                prop_assert!(close(folded.mean()[i], total.mean()[i]), "mean {i}");
            }
            prop_assert!(close(folded.ssd(), total.ssd()), "ssd");
        }

        #[test]
        fn tree_leaves_fold_to_sum_of_points_under_rebuild((d, pts) in dim_points(20, 200)) {
            // threshold 0 + small max_leaves forces rebuilds (threshold-grow + reinsert of leaf CFs);
            // folding the post-rebuild leaves must still reconstruct the whole-dataset feature.
            let mut tree: CFTree<f64, Full<f64>, _, _> =
                CFTree::new(d, 4, 4, 0.0, 12, CentroidEuclidean, CentroidEuclidean);
            for p in &pts {
                tree.insert(p);
            }
            prop_assert!(tree.rebuilds() > 0);
            let mut folded: Full<f64> = Full::new(d);
            for leaf in tree.leaf_features() {
                folded.merge(leaf);
            }
            let mut total: Full<f64> = Full::new(d);
            for p in &pts {
                total.push(p, 1.0);
            }
            prop_assert!(close(folded.weight(), total.weight()), "weight");
            for i in 0..d {
                prop_assert!(close(folded.mean()[i], total.mean()[i]), "mean {i}");
            }
            prop_assert!(close(folded.ssd(), total.ssd()), "ssd");
        }
    }
}
