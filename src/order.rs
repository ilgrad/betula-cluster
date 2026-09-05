//! A canonical insertion order, so a CF-tree summarises a *multiset* rather than a *sequence*.
//!
//! Every BIRCH-family tree has the same defect, and it is structural rather than a bug: the
//! prototype set is a greedy `T`-net built in arrival order. Row `i` is absorbed when it falls
//! within the threshold of an entry that rows `0..i` created, and starts a new entry otherwise, so
//! which regions get a prototype — and how many — is a function of the order the rows arrived in.
//! Two permutations of the same data give two different trees, and at real compression the gap
//! between them is wider than the gap between two random seeds. [`crate::tree::CFTree::refit_leaves`]
//! moves each prototype onto the centroid of its cell but never changes which cells exist, which is
//! exactly why it improved quality and left the order gap alone.
//!
//! The cure is to stop letting the caller choose the order. If the rows are sorted by a key computed
//! from the data itself, the build is a pure function of the multiset and any permutation produces
//! **bit-identical** labels.
//!
//! # Choosing the key
//!
//! Two properties are required and a third is wanted.
//!
//! 1. *A total order.* A key alone is not enough: distinct rows that share a key are left in
//!    whatever order the caller passed, and the result stops being canonical. Sorting by squared
//!    norm fails exactly this way on integer-valued data, where distinct rows collide constantly.
//!    Ties are therefore broken on the full row.
//! 2. *Order independence.* The key may be computed from the data, but only through statistics that
//!    do not depend on the row order — here a per-column min/max, and a projection matrix drawn from
//!    a fixed constant rather than from the caller's `seed`. Using the caller's seed would tie the
//!    tree's shape to a knob that exists to vary the *head*, and destroy the one workflow this
//!    enables: averaging several head seeds over one fixed summary.
//! 3. *Spatial locality*, so that nearby rows arrive together and the net is built by sweeping the
//!    space rather than hopping around it. This is what a Morton (Z-order) code over a handful of
//!    random projections buys, and it is why the projection is worth its one GEMM: sorting raw
//!    coordinates lexicographically is a valid canonical order but a poor one, because in high
//!    dimension the leading coordinates are arbitrary (on MNIST they are corner pixels that are zero
//!    for every image).
//!
//! # What it does and does not buy
//!
//! The published sweep is `bench/insertion_order.py` — `digits` / `covtype-20k` / `mnist-10k`, three
//! leaf budgets, `kmeans` / `gmm` / `ward`, eight permutations per arm, 27 canonical cells:
//!
//! - **Order invariance is exact**, not approximate. In every one of the 27 cells the ARI spread is
//!   `0.0000`, the pairwise ARI is `1.0000` as both mean and minimum, and the realised leaf count is
//!   constant — the label arrays are equal element for element. The benchmark asserts this rather
//!   than reporting it.
//! - **The build cost moves both ways and the mechanism is two opposing effects**, decomposed in
//!   `benches/canonical_order.rs`: a coherent stream re-descends the same subtree, which is
//!   cache-friendly, but it also fills the leaf budget with fine leaves in one region and then has
//!   to *rebuild* when the next region arrives. Rebuild counts go 67 -> 60 at `d = 20` and 3 -> 30 at
//!   `d = 784, max_leaves = 8000`, where the second effect wins and the insert costs 1.56x.
//! - **Quality is a wash.** Against the arrival order's *median* draw: mean +0.0136, median
//!   **-0.0017**, non-negative in 10 of 27, and inside the order arm's own `[min, max]` in 21 of 27.
//!   This removes the lottery; it does not improve the expected result, and no honest reading of the
//!   numbers says otherwise. The positive mean is two cells where arrival order was collapsing a
//!   `gmm` head.
//!
//! The *scheme* was chosen on a separate 16-cell study (`local/scratch/canonical_choice.py`) that
//! scored four candidate keys against the same yardstick: this one was non-negative in 12/16 cells
//! against `lex`'s 8/16, and low-discrepancy walks over the sorted order (van der Corput,
//! round-robin stride) came in at 5/16 and 7/16, so they do not earn their extra constant.

use crate::clustering::rng::SplitMix64;
use crate::types::Real;

/// Projections mixed into the code. `PROJECTIONS * BITS` is exactly 64, so a code is one `u64`.
const PROJECTIONS: usize = 8;
/// Quantisation levels per projection, as a bit count.
const BITS: u32 = 8;
/// Fixed, so the order is a function of the data alone. Not the caller's `seed` — see the module
/// docs for why tying the two together would be the wrong coupling.
const PROJECTION_SEED: u64 = 0x0BE7_014A_C0DE_0117;

/// Rows per shard in a canonical build. Chosen from the measured knee, not from taste: at
/// `n = 200 000, max_leaves = 2000` the shard sweep reads 1.00 / 1.90 / 2.66 / **3.42** / 3.71 /
/// 3.91× for 1 / 2 / 4 / 8 / 16 / 32 shards on a `kmeans` fit, so eight shards — 25 000 rows each —
/// buy 87 % of the available speed-up, and the rest costs a four-fold finer partition.
const ROWS_PER_SHARD: usize = 25_000;
/// Ceiling on that count. Past the knee the curve is flat, and every extra shard is one more
/// sub-summary the serial merge has to absorb.
const MAX_SHARDS: usize = 64;

/// How many shards a canonical build splits into — a function of `n` alone.
///
/// The shard count *is* part of the answer: two counts hold different point sets, build different
/// sub-summaries, and the merge cannot repair that. So a build that promises invariance cannot take
/// its shard count from the thread count, or the promise holds only while nobody re-tunes `n_jobs`.
/// Measured, the gap is not academic — at real compression, labels from `n_jobs = 1` and `n_jobs = 8`
/// agree at pairwise ARI 0.46 on average and 0.098 at worst, which is as far apart as two row orders
/// were before any of this.
///
/// Deriving it from `n` costs the parallelism above the returned count and buys a summary that does
/// not move when the machine does. Small inputs return 1, so they keep the plain sequential build.
pub fn canonical_shards(n: usize) -> usize {
    (n / ROWS_PER_SHARD).clamp(1, MAX_SHARDS)
}

/// The permutation that puts `n` row-major rows of `flat` into canonical order.
///
/// `O(n · dim)` for the projections plus `O(n log n)` for the sort, against the build's
/// `O(n · dim · depth)` distance evaluations — a few percent of a fit. Returns row indices, never a
/// reordered copy of the data: at 10 M × 784 a copy is 29 GB, which is the duplicate the zero-copy
/// ingest exists to avoid.
pub fn canonical_permutation<R: Real>(flat: &[R], n: usize, dim: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..n as u32).collect();
    if n <= 1 || dim == 0 {
        return idx;
    }
    let codes = morton_codes(flat, n, dim);
    idx.sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        codes[a].cmp(&codes[b]).then_with(|| {
            // Only reached on a code collision, which the quantisation makes common enough to
            // matter and rare enough not to cost: without this the order inside a collision is the
            // caller's, and the whole guarantee evaporates.
            let (ra, rb) = (&flat[a * dim..(a + 1) * dim], &flat[b * dim..(b + 1) * dim]);
            ra.iter()
                .zip(rb)
                .find_map(|(x, y)| match x.partial_cmp(y) {
                    Some(std::cmp::Ordering::Equal) | None => None,
                    Some(ord) => Some(ord),
                })
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    idx
}

/// The canonical permutation for CSR rows — the sparse twin of [`canonical_permutation`].
///
/// The key is the same construction reading the non-zeros only, so it costs `O(nnz · PROJECTIONS)`
/// rather than `O(n · dim · PROJECTIONS)`: an implicit zero contributes nothing to a projection, so
/// a sparse row and its dense expansion produce the same code by construction, and a row with no
/// non-zeros lands where the zero vector belongs rather than in a special case.
///
/// # Errors
///
/// The tie-break walks two rows in column order, so it needs each row's indices ascending. CSR from
/// `scipy` normally satisfies this and `validate_csr` does not check it, so this is the one place
/// that must: with unsorted indices the merge-walk would compare the wrong columns and silently
/// return a *nearly* canonical order, which is worse than no order at all — the guarantee would hold
/// on most inputs and fail on some, which is the shape of bug that survives a test suite.
pub fn canonical_permutation_csr<R: Real>(
    data: &[R],
    indices: &[i64],
    indptr: &[i64],
    dim: usize,
) -> Result<Vec<u32>, &'static str> {
    let n = indptr.len().saturating_sub(1);
    let mut idx: Vec<u32> = (0..n as u32).collect();
    if n <= 1 || dim == 0 {
        return Ok(idx);
    }
    let row = |i: usize| {
        let (lo, hi) = (indptr[i] as usize, indptr[i + 1] as usize);
        (&indices[lo..hi], &data[lo..hi])
    };
    for i in 0..n {
        let (cols, _) = row(i);
        if cols.windows(2).any(|w| w[1] <= w[0]) {
            return Err(
                "canonical_order needs CSR indices sorted and unique within each row; call                  X.sort_indices() (or pass X.sorted_indices()) before fitting",
            );
        }
    }

    let codes = morton_codes_csr(data, indices, indptr, dim, n);
    idx.sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        codes[a].cmp(&codes[b]).then_with(|| {
            // Dense-lexicographic order over two sparse rows: walk both column lists together and
            // stop at the first column where they differ, treating a missing column as an explicit
            // zero. Comparing the stored values pairwise instead would order `[(0, 1.0)]` against
            // `[(5, 1.0)]` by value and call them equal.
            let ((ca, va), (cb, vb)) = (row(a), row(b));
            let (mut p, mut q) = (0usize, 0usize);
            while p < ca.len() || q < cb.len() {
                let (x, y) = match (ca.get(p), cb.get(q)) {
                    (Some(&i), Some(&j)) if i == j => {
                        let r = (va[p], vb[q]);
                        p += 1;
                        q += 1;
                        r
                    }
                    (Some(&i), Some(&j)) if i < j => {
                        let r = (va[p], R::zero());
                        p += 1;
                        r
                    }
                    (Some(_), Some(_)) => {
                        let r = (R::zero(), vb[q]);
                        q += 1;
                        r
                    }
                    (Some(_), None) => {
                        let r = (va[p], R::zero());
                        p += 1;
                        r
                    }
                    (None, Some(_)) => {
                        let r = (R::zero(), vb[q]);
                        q += 1;
                        r
                    }
                    (None, None) => unreachable!("the loop guard excludes both being exhausted"),
                };
                match x.partial_cmp(&y) {
                    Some(std::cmp::Ordering::Equal) | None => {}
                    Some(ord) => return ord,
                }
            }
            std::cmp::Ordering::Equal
        })
    });
    Ok(idx)
}

/// [`morton_codes`] reading CSR rows. Shares the projection draw and the quantisation, so a CSR
/// matrix and its dense expansion sort the same way.
fn morton_codes_csr<R: Real>(
    data: &[R],
    indices: &[i64],
    indptr: &[i64],
    dim: usize,
    n: usize,
) -> Vec<u64> {
    let proj = projections::<R>(dim);
    let mut z = vec![0.0f64; n * PROJECTIONS];
    for i in 0..n {
        let (lo, hi) = (indptr[i] as usize, indptr[i + 1] as usize);
        for k in lo..hi {
            let (c, v) = (indices[k] as usize, data[k].to_f64().unwrap_or(0.0));
            for j in 0..PROJECTIONS {
                z[i * PROJECTIONS + j] += v * proj[j * dim + c].to_f64().unwrap_or(0.0);
            }
        }
    }
    quantise(z, n)
}

/// The fixed projection matrix, projection-major (`PROJECTIONS × dim`).
///
/// Projection-major, not row-major, so each projection is a contiguous `dim`-length slice and the
/// dense pass can be [`crate::kernels::dot`] — the same SIMD kernel the distance path uses. As a
/// row-major `dim × PROJECTIONS` rank-1 accumulate it did not vectorise and cost 7–21 % of the fit
/// (`benches/canonical_order.rs`). The row is re-read once per projection rather than once in total,
/// which is free: it is in L1 by the second pass at every `dim` this sees.
///
/// The draw is deliberately *not* transposed with the layout — the `k`-th gaussian still lands at
/// `(d, j)` exactly as it did row-major, so every projection vector is bit-identical to the version
/// this replaced and the published order-invariance sweep still describes the shipped key. The
/// `1/sqrt(dim)` scale only keeps the values readable; each projection is ranged independently in
/// [`quantise`], so it cancels.
fn projections<R: Real>(dim: usize) -> Vec<R> {
    let mut rng = SplitMix64::new(PROJECTION_SEED);
    let scale = 1.0 / (dim as f64).sqrt();
    let mut proj: Vec<R> = vec![R::zero(); dim * PROJECTIONS];
    for d in 0..dim {
        for j in 0..PROJECTIONS {
            proj[j * dim + d] = R::from_f64(rng.gauss() * scale).unwrap_or_else(R::zero);
        }
    }
    proj
}

/// Range each projection over its own observed span, quantise to [`BITS`] levels, and interleave the
/// bits so the most significant bit of every projection leads.
///
/// Interleaving is what gives the code locality at every scale: a prefix of the code is a coarse cell
/// of the whole space, not a fine slice of one axis. Ranging per projection rather than globally is
/// what makes the code scale-free — and it is also why the `1/sqrt(dim)` in [`projections`] does not
/// matter.
fn quantise(z: Vec<f64>, n: usize) -> Vec<u64> {
    let levels = ((1u64 << BITS) - 1) as f64;
    let mut codes = vec![0u64; n];
    for j in 0..PROJECTIONS {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..n {
            let v = z[i * PROJECTIONS + j];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        // A constant projection contributes nothing rather than a division by zero.
        let span = if hi > lo { hi - lo } else { 1.0 };
        for i in 0..n {
            let q = ((z[i * PROJECTIONS + j] - lo) / span * levels).round();
            let q = (q.clamp(0.0, levels) as u64) & ((1 << BITS) - 1);
            for b in 0..BITS {
                let bit = (q >> b) & 1;
                codes[i] |= bit << (b as usize * PROJECTIONS + j);
            }
        }
    }
    codes
}

/// One `u64` Morton code per dense row.
fn morton_codes<R: Real>(flat: &[R], n: usize, dim: usize) -> Vec<u64> {
    let proj = projections::<R>(dim);
    let mut z = vec![0.0f64; n * PROJECTIONS];
    for i in 0..n {
        let row = &flat[i * dim..(i + 1) * dim];
        for j in 0..PROJECTIONS {
            z[i * PROJECTIONS + j] = crate::kernels::dot(row, &proj[j * dim..(j + 1) * dim])
                .to_f64()
                .unwrap_or(0.0);
        }
    }
    quantise(z, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut rng = SplitMix64::new(seed);
        (0..n * dim).map(|_| rng.gauss()).collect()
    }

    fn permute(flat: &[f64], dim: usize, perm: &[usize]) -> Vec<f64> {
        perm.iter()
            .flat_map(|&i| flat[i * dim..(i + 1) * dim].to_vec())
            .collect()
    }

    /// The property the module exists for: the *sequence* of rows the permutation yields is the same
    /// whichever order the rows were handed over in.
    #[test]
    fn the_canonical_sequence_is_the_same_for_every_input_permutation() {
        let (n, dim) = (400, 6);
        let flat = rows(n, dim, 11);
        let canonical = |data: &[f64]| -> Vec<Vec<f64>> {
            canonical_permutation(data, n, dim)
                .into_iter()
                .map(|i| data[i as usize * dim..(i as usize + 1) * dim].to_vec())
                .collect()
        };
        let reference = canonical(&flat);
        let mut rng = SplitMix64::new(99);
        for _ in 0..4 {
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                perm.swap(i, (rng.next_u64() % (i as u64 + 1)) as usize);
            }
            assert_eq!(canonical(&permute(&flat, dim, &perm)), reference);
        }
    }

    /// A key alone is not a total order. Rows that collide in the code must still be ordered by
    /// their contents, or the caller's order leaks back in — which is how sorting by norm fails.
    #[test]
    fn colliding_codes_are_broken_by_the_row_so_duplicates_cannot_leak_the_input_order() {
        // Two distinct rows one quantisation step apart, plus an exact duplicate of each.
        let dim = 2;
        let a = [0.0, 0.0];
        let b = [1e-12, 0.0];
        let forward: Vec<f64> = [a, b, a, b].concat();
        let backward: Vec<f64> = [b, a, b, a].concat();
        let seq = |data: &[f64]| -> Vec<Vec<f64>> {
            canonical_permutation(data, 4, dim)
                .into_iter()
                .map(|i| data[i as usize * dim..(i as usize + 1) * dim].to_vec())
                .collect()
        };
        assert_eq!(seq(&forward), seq(&backward));
    }

    /// What the quantiser does to three collinear rows, derived rather than recorded.
    ///
    /// Each projection is ranged over its own observed span, so on `{-1, 0, +1}` scaled by the
    /// projection's own weight sum, one antipode minimises every projection and the other maximises
    /// it. Which of the two is which flips with the sign of that weight sum — so the codes are not
    /// `0` and `u64::MAX`, they are **bitwise complements**, and the sign pattern is what decides
    /// the halves. The centre row sits at exactly half of every span, level `round(0.5 · 255) = 128`
    /// in all eight projections; level 128 is bit 7 alone, and the interleaving puts projection `j`'s
    /// bit `b` at position `b · PROJECTIONS + j`, so its code is bits 56–63 and nothing else.
    ///
    /// It is here because the invariance tests cannot see the arithmetic at all: they hold for *any*
    /// deterministic key, so a quantiser that clamps to the wrong width, masks with the wrong
    /// operator or ranges against the wrong origin satisfies every one of them. The centre row is
    /// what catches an inverted mask specifically — at levels 0 and 255 an inversion maps the code
    /// set onto itself, and only an interior level can tell the two apart.
    #[test]
    fn three_collinear_rows_quantise_to_the_levels_the_ranging_implies() {
        let dim = 6;
        let flat: Vec<f64> = [vec![-1.0; dim], vec![0.0; dim], vec![1.0; dim]].concat();
        let codes = morton_codes(&flat, 3, dim);
        assert_eq!(codes[0] ^ codes[2], u64::MAX, "antipodes are complements");
        assert_eq!(
            codes[1], 0xFF00_0000_0000_0000,
            "the centre is level 128 in all 8"
        );
    }

    /// A change to the key is a change to every label the library has ever published, so it must not
    /// be possible to make one by accident. Nothing else in this module can catch it: the key's only
    /// requirement is determinism, and every property test here is satisfied by any deterministic
    /// key at all. This pins one fixture's answer so that a changed projection draw, a changed seed
    /// or a changed bit layout shows up as a failing test rather than as a silent relabelling.
    ///
    /// It is change-detection, not correctness — if it fails, the question is whether the change was
    /// intended, and the answer belongs in `CHANGELOG.md` under a version bump.
    #[test]
    fn the_key_is_pinned_so_a_change_to_it_cannot_pass_unnoticed() {
        let (n, dim) = (12, 5);
        let flat = rows(n, dim, 7);
        assert_eq!(
            canonical_permutation(&flat, n, dim),
            vec![4, 9, 11, 8, 5, 6, 1, 0, 3, 2, 7, 10],
        );
    }

    /// The tie-break decides the order of every row inside a collision, and on the sparse path it is
    /// a merge walk over two column lists — twenty of its mutants survived the first mutation run
    /// because no test ever reached it. Collisions are not rare on real data but they are hard to
    /// arrange on a small fixture, so this one forces them: a single far-away row stretches every
    /// projection's span so wide that the whole cluster quantises to level 0, and the comparator
    /// alone decides its order.
    ///
    /// Comparing against the dense path is what makes this a test rather than a snapshot — the two
    /// comparators are written independently (dense walks one slice, sparse merges two column
    /// lists), so agreement is evidence and not a recording.
    #[test]
    fn a_code_collision_is_broken_the_same_way_on_csr_as_on_dense() {
        let (n, dim) = (17, 4);
        let mut flat = rows(n - 1, dim, 21);
        flat.extend(std::iter::repeat_n(1e6, dim));

        let codes = morton_codes(&flat, n, dim);
        let distinct: std::collections::HashSet<u64> = codes.iter().copied().collect();
        assert!(
            distinct.len() < n,
            "the fixture must collide or it exercises nothing: {} distinct codes for {n} rows",
            distinct.len()
        );

        let (data, indices, indptr) = to_csr(&flat, n, dim);
        assert_eq!(
            canonical_permutation_csr(&data, &indices, &indptr, dim).unwrap(),
            canonical_permutation(&flat, n, dim),
        );
    }

    /// The shard count has to be a function of `n` and nothing else — a rule that consulted the
    /// thread count, the core count or `max_leaves` would put a machine or a knob back into the
    /// summary. The cases below are the three that carry that: small inputs stay on the sequential
    /// build, the count grows with `n`, and it stops growing before the merge does.
    #[test]
    fn the_shard_count_is_a_function_of_n_alone() {
        assert_eq!(canonical_shards(0), 1);
        assert_eq!(canonical_shards(ROWS_PER_SHARD - 1), 1);
        assert_eq!(canonical_shards(8 * ROWS_PER_SHARD), 8);
        assert_eq!(canonical_shards(usize::MAX), MAX_SHARDS);
        assert!(canonical_shards(1_000_000) >= canonical_shards(500_000));
    }

    #[test]
    fn the_permutation_is_a_permutation() {
        let (n, dim) = (257, 5);
        let flat = rows(n, dim, 3);
        let mut seen = canonical_permutation(&flat, n, dim);
        assert_eq!(seen.len(), n);
        seen.sort_unstable();
        assert_eq!(seen, (0..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn degenerate_shapes_return_the_identity_rather_than_failing() {
        assert_eq!(canonical_permutation::<f64>(&[], 0, 4), Vec::<u32>::new());
        assert_eq!(canonical_permutation(&[1.0, 2.0], 1, 2), vec![0]);
        assert_eq!(canonical_permutation::<f64>(&[], 3, 0), vec![0, 1, 2]);
    }

    /// A constant column contributes a zero-span projection; ranging it must not divide by zero.
    #[test]
    fn a_constant_column_does_not_produce_a_nan_code() {
        let (n, dim) = (64, 3);
        let mut rng = SplitMix64::new(5);
        let flat: Vec<f64> = (0..n * dim)
            .map(|i| if i % dim == 1 { 2.5 } else { rng.gauss() })
            .collect();
        let perm = canonical_permutation(&flat, n, dim);
        let mut seen = perm.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..n as u32).collect::<Vec<_>>());
    }

    fn to_csr(flat: &[f64], n: usize, dim: usize) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let (mut data, mut indices, mut indptr) = (vec![], vec![], vec![0i64]);
        for i in 0..n {
            for d in 0..dim {
                let v = flat[i * dim + d];
                if v != 0.0 {
                    data.push(v);
                    indices.push(d as i64);
                }
            }
            indptr.push(data.len() as i64);
        }
        (data, indices, indptr)
    }

    /// The sparse key reads only the non-zeros, so it has to land on the same order the dense key
    /// does — otherwise a CSR matrix and its own `.toarray()` would cluster differently, which is a
    /// bug a user would find before any test did.
    #[test]
    fn the_csr_order_matches_the_dense_order_on_the_same_matrix() {
        let (n, dim) = (300, 12);
        let mut rng = SplitMix64::new(21);
        // Genuinely sparse: about a fifth of the entries are non-zero, so implicit zeros dominate.
        let flat: Vec<f64> = (0..n * dim)
            .map(|_| {
                if rng.next_f64() < 0.2 {
                    rng.gauss()
                } else {
                    0.0
                }
            })
            .collect();
        let (data, indices, indptr) = to_csr(&flat, n, dim);
        assert_eq!(
            canonical_permutation_csr(&data, &indices, &indptr, dim).unwrap(),
            canonical_permutation(&flat, n, dim)
        );
    }

    /// An all-zero row is the zero vector, not a special case, and must sort where the zero vector
    /// belongs rather than wherever the caller left it.
    #[test]
    fn an_empty_csr_row_sorts_as_the_zero_vector() {
        let dim = 4;
        let flat = vec![
            0.0, 0.0, 0.0, 0.0, // an empty row
            1.0, 0.0, 0.0, 2.0, //
            0.0, 3.0, 0.0, 0.0, //
        ];
        let (data, indices, indptr) = to_csr(&flat, 3, dim);
        assert_eq!(
            canonical_permutation_csr(&data, &indices, &indptr, dim).unwrap(),
            canonical_permutation(&flat, 3, dim)
        );
    }

    /// Unsorted column indices would make the merge-walk compare the wrong columns and return a
    /// *nearly* canonical order — right on most inputs, wrong on some. Refuse instead.
    #[test]
    fn unsorted_csr_indices_are_refused_rather_than_silently_mis_ordered() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indptr = vec![0i64, 2, 4];
        assert!(canonical_permutation_csr(&data, &[1i64, 0, 0, 1], &indptr, 4).is_err());
        assert!(canonical_permutation_csr(&data, &[0i64, 0, 0, 1], &indptr, 4).is_err());
        assert!(canonical_permutation_csr(&data, &[0i64, 1, 0, 1], &indptr, 4).is_ok());
    }

    /// Locality is the reason for the projection: consecutive rows in canonical order should be
    /// closer together than consecutive rows in arrival order.
    #[test]
    fn the_canonical_order_places_neighbours_next_to_each_other() {
        let (n, dim) = (2000, 4);
        let flat = rows(n, dim, 7);
        let step = |seq: &[u32]| -> f64 {
            seq.windows(2)
                .map(|w| {
                    let (a, b) = (w[0] as usize * dim, w[1] as usize * dim);
                    (0..dim)
                        .map(|d| (flat[a + d] - flat[b + d]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .sum::<f64>()
                / (n - 1) as f64
        };
        let arrival: Vec<u32> = (0..n as u32).collect();
        assert!(step(&canonical_permutation(&flat, n, dim)) < 0.5 * step(&arrival));
    }
}
