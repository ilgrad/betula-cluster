//! `O(nnz)` sparse-native spherical clustering via flat leader summarisation.
//!
//! For very high-dimensional sparse data (text, one-hot, embeddings) the dense CF path costs `O(d)`
//! per row regardless of how few entries are non-zero. This module summarises CSR rows into spherical
//! micro-clusters touching only the non-zeros: a micro-cluster keeps `(n, ΣX, ‖ΣX‖², S)`, so the mean
//! `μ = ΣX / n`, the cached `‖μ‖² = ‖ΣX‖² / n²`, and the point-to-centroid distance
//! `‖x − μ‖² = ‖x‖² − 2⟨x, μ⟩ + ‖μ‖²` all update / evaluate in `O(nnz)`.
//!
//! **Numerical note.** `O(nnz)` updates are only possible via this *expanded* squared-distance form;
//! it is **not** the dense path's cancellation-free Welford computation. For sparse high-dimensional
//! data (rows far from the dense centroid) the expansion is accurate, but near-duplicate dense points
//! can lose precision in `‖x − μ‖²`. Use the dense path when the cancellation-free guarantee matters.
//! The mean itself (`ΣX / n`) is the classic sum form — stable for the centroid; only the scatter `S`
//! inherits the expansion's caveat. The resulting micro-clusters are materialised to dense
//! [`Spherical`] features once and handed to the ordinary Phase-3 heads.

use crate::feature::Spherical;

/// A sparse-native spherical accumulator: weight `n`, coordinate sum `ΣX`, cached `‖ΣX‖²`, scatter `S`.
struct SparseSpherical {
    n: f64,
    sumx: Vec<f64>,
    sumx_sq: f64,
    ssd: f64,
}

impl SparseSpherical {
    fn new(dim: usize) -> Self {
        Self {
            n: 0.0,
            sumx: vec![0.0; dim],
            sumx_sq: 0.0,
            ssd: 0.0,
        }
    }

    /// `⟨x, ΣX⟩` over the row's non-zeros.
    fn dot(&self, idx: &[usize], val: &[f64]) -> f64 {
        idx.iter().zip(val).map(|(&j, &v)| v * self.sumx[j]).sum()
    }

    /// Squared distance from the row `x` to this micro-cluster's centroid (`O(nnz)`).
    fn dist2(&self, idx: &[usize], val: &[f64], x_sq: f64) -> f64 {
        if self.n == 0.0 {
            return x_sq;
        }
        let dot = self.dot(idx, val);
        (x_sq - 2.0 * dot / self.n + self.sumx_sq / (self.n * self.n)).max(0.0)
    }

    /// Fold the sparse row `x` (`x_sq = ‖x‖²` precomputed) into the accumulator in `O(nnz)`.
    fn push(&mut self, idx: &[usize], val: &[f64], x_sq: f64) {
        if self.n == 0.0 {
            for (&j, &v) in idx.iter().zip(val) {
                self.sumx[j] = v;
            }
            self.sumx_sq = x_sq;
            self.ssd = 0.0;
            self.n = 1.0;
            return;
        }
        let dot = self.dot(idx, val);
        // ‖x − μ‖² with μ = ΣX/n (expanded form — see the module note on its numerical trade-off).
        let delta_sq = (x_sq - 2.0 * dot / self.n + self.sumx_sq / (self.n * self.n)).max(0.0);
        let w_new = self.n + 1.0;
        self.ssd += (self.n / w_new) * delta_sq; // Welford coefficient w·(1 − w/W') = n/(n+1)
        self.sumx_sq += 2.0 * dot + x_sq; // ‖ΣX + x‖² = ‖ΣX‖² + 2⟨ΣX, x⟩ + ‖x‖²
        for (&j, &v) in idx.iter().zip(val) {
            self.sumx[j] += v;
        }
        self.n = w_new;
    }

    /// Materialise into a dense spherical feature `(n, μ = ΣX/n, S)`.
    fn into_spherical(self) -> Spherical<f64> {
        let mean: Vec<f64> = self.sumx.iter().map(|&s| s / self.n).collect();
        Spherical::from_moments(self.n, mean, self.ssd)
    }
}

/// `‖x‖²` of a sparse row.
fn norm_sq(val: &[f64]) -> f64 {
    val.iter().map(|&v| v * v).sum()
}

/// Summarise CSR rows into spherical micro-clusters with a single `O(nnz)`-per-row leader pass: each
/// row joins the nearest leader whose centroid is within `threshold` (squared distance), otherwise it
/// seeds a new leader; once `max_leaders` is reached every further row joins its nearest leader
/// (bounded memory). Returns dense [`Spherical`] micro-clusters. Caller has validated the CSR arrays.
pub fn summarize_sparse(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    n_features: usize,
    threshold: f64,
    max_leaders: usize,
) -> Vec<Spherical<f64>> {
    let mut leaders: Vec<SparseSpherical> = Vec::new();
    let mut idx_buf: Vec<usize> = Vec::new();
    for w in indptr.windows(2) {
        let (lo, hi) = (w[0] as usize, w[1] as usize);
        let val = &data[lo..hi];
        idx_buf.clear();
        idx_buf.extend(indices[lo..hi].iter().map(|&c| c as usize));
        let x_sq = norm_sq(val);
        let mut best = usize::MAX;
        let mut bd = f64::INFINITY;
        for (li, l) in leaders.iter().enumerate() {
            let d = l.dist2(&idx_buf, val, x_sq);
            if d < bd {
                bd = d;
                best = li;
            }
        }
        if best != usize::MAX && (bd <= threshold || leaders.len() >= max_leaders) {
            leaders[best].push(&idx_buf, val, x_sq);
        } else {
            let mut l = SparseSpherical::new(n_features);
            l.push(&idx_buf, val, x_sq);
            leaders.push(l);
        }
    }
    leaders.into_iter().map(|l| l.into_spherical()).collect()
}

/// Index of the micro-cluster nearest to a sparse row, given precomputed dense means and `‖μ‖²`.
pub fn nearest_sparse(
    means: &[Vec<f64>],
    musq: &[f64],
    idx: &[usize],
    val: &[f64],
    x_sq: f64,
) -> usize {
    let mut best = 0;
    let mut bd = f64::INFINITY;
    for (i, mean) in means.iter().enumerate() {
        let dot: f64 = idx.iter().zip(val).map(|(&j, &v)| v * mean[j]).sum();
        let d = (x_sq - 2.0 * dot + musq[i]).max(0.0);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

/// Upper bound on `n_features` for the sparse-native path. This path materialises a **dense** centroid
/// per micro-cluster (`O(n_features)` memory each), so a feature count beyond this cannot fit in RAM
/// regardless of how sparse the rows are — reduce dimensionality first (e.g. TruncatedSVD, the
/// reduce-then-cluster path). The cap is also a hard trust-boundary guard: without it a caller could
/// pass a single-nonzero row with a huge `n_features` and force an unbounded allocation (`vec![0.0;
/// n_features]`). `2^30` features is already ~8 GB per centroid — far past where this path is viable.
pub const MAX_SPARSE_FEATURES: usize = 1 << 30;

/// Validate CSR arrays at the untrusted boundary so the `O(nnz)` row expansion can never index out of
/// bounds or force an unbounded allocation: `n_features ∈ 1..=MAX_SPARSE_FEATURES`, matched
/// `data`/`indices` lengths, an `indptr` that starts at 0, is non-decreasing, and ends at `nnz`,
/// in-range column indices, and finite values. Returns a human-readable message on failure (the caller
/// maps it to its own error type). Pure — no PyO3 — so it is reachable on stable Rust for fuzzing/tests.
pub fn validate_csr(
    data: &[f64],
    indices: &[i64],
    indptr: &[i64],
    n_features: usize,
) -> Result<(), String> {
    if n_features == 0 {
        return Err("n_features must be > 0".into());
    }
    if n_features > MAX_SPARSE_FEATURES {
        return Err(format!(
            "n_features {n_features} exceeds the sparse-native cap {MAX_SPARSE_FEATURES}: this path \
             materialises a dense centroid per micro-cluster, so reduce dimensionality first \
             (e.g. TruncatedSVD)"
        ));
    }
    if data.len() != indices.len() {
        return Err("CSR data and indices must have equal length".into());
    }
    if indptr.first() != Some(&0) || *indptr.last().unwrap_or(&-1) as usize != data.len() {
        return Err("CSR indptr must start at 0 and end at nnz".into());
    }
    if indptr.windows(2).any(|w| w[1] < w[0]) {
        return Err("CSR indptr must be non-decreasing".into());
    }
    if indices.iter().any(|&c| c < 0 || c as usize >= n_features) {
        return Err("CSR column index out of range".into());
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("data contains NaN or infinite values".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::ClusterFeature;
    use proptest::prelude::*;

    /// Dense reference: the spherical CF built from the same rows must match the sparse accumulator's
    /// weight, mean, and (to the expansion's accuracy) scatter.
    #[test]
    fn sparse_accumulator_matches_dense_moments() {
        let dim = 6;
        let rows: Vec<(Vec<usize>, Vec<f64>)> = vec![
            (vec![0, 3], vec![1.0, 2.0]),
            (vec![1, 3], vec![4.0, 1.0]),
            (vec![0, 5], vec![2.0, 3.0]),
        ];
        let mut sp = SparseSpherical::new(dim);
        let mut dense = Spherical::<f64>::new(dim);
        for (idx, val) in &rows {
            let x_sq = norm_sq(val);
            sp.push(idx, val, x_sq);
            let mut d = vec![0.0; dim];
            for (&j, &v) in idx.iter().zip(val) {
                d[j] = v;
            }
            dense.push(&d, 1.0);
        }
        let got = sp.into_spherical();
        assert!((got.weight() - dense.weight()).abs() < 1e-9);
        for (a, b) in got.mean().iter().zip(dense.mean()) {
            assert!((a - b).abs() < 1e-9, "mean {a} vs {b}");
        }
        assert!(
            (got.ssd() - dense.ssd()).abs() < 1e-6,
            "ssd {} vs {}",
            got.ssd(),
            dense.ssd()
        );
    }

    fn csr(rows: &[Vec<(usize, f64)>], n_features: usize) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0i64];
        for r in rows {
            for &(c, v) in r {
                assert!(c < n_features);
                indices.push(c as i64);
                data.push(v);
            }
            indptr.push(data.len() as i64);
        }
        (data, indices, indptr)
    }

    #[test]
    fn summarize_groups_repeated_rows() {
        // Two distinct sparse patterns, repeated; threshold 0 ⇒ each pattern is one leader.
        let mut rows = Vec::new();
        for _ in 0..5 {
            rows.push(vec![(0usize, 1.0), (1, 1.0)]);
        }
        for _ in 0..5 {
            rows.push(vec![(8usize, 1.0), (9, 1.0)]);
        }
        let (data, indices, indptr) = csr(&rows, 10);
        let micros = summarize_sparse(&data, &indices, &indptr, 10, 0.0, 64);
        assert_eq!(micros.len(), 2);
        let total: f64 = micros.iter().map(|m| m.weight()).sum();
        assert_eq!(total as i64, 10);
    }

    #[test]
    fn summarize_caps_leaders() {
        let rows: Vec<Vec<(usize, f64)>> =
            (0..200).map(|i| vec![(i % 50, 1.0 + i as f64)]).collect();
        let (data, indices, indptr) = csr(&rows, 50);
        let micros = summarize_sparse(&data, &indices, &indptr, 50, 0.0, 16);
        assert!(micros.len() <= 16);
        let total: f64 = micros.iter().map(|m| m.weight()).sum();
        assert_eq!(total as i64, 200); // mass conserved despite the cap
    }

    #[test]
    fn nearest_routes_to_closest_micro() {
        let means = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 5.0]];
        let musq: Vec<f64> = means
            .iter()
            .map(|m| m.iter().map(|v| v * v).sum())
            .collect();
        // a row close to micro 1 (large value on axis 2)
        assert_eq!(nearest_sparse(&means, &musq, &[2], &[4.5], 4.5 * 4.5), 1);
        assert_eq!(nearest_sparse(&means, &musq, &[0], &[1.2], 1.2 * 1.2), 0);
    }

    #[test]
    fn sparse_scatter_loses_precision_on_near_duplicate_dense_points() {
        // Documented regime (module note above): near-duplicate *dense* rows at large magnitude. The
        // mean (classic ΣX/n) stays exact; the scatter S — from the expanded ‖x−μ‖² — cancels
        // catastrophically (‖x‖² ≈ 2·(1e8)² has an ULP ~4, so the true scatter is quantised away).
        let dim = 2;
        let b = 1.0e8;
        let idx = [0usize, 1usize];
        let rows = [[b, b], [b + 1.0, b - 1.0]];

        let mut sp = SparseSpherical::new(dim);
        let mut dense = Spherical::<f64>::new(dim);
        for r in &rows {
            sp.push(&idx, r, norm_sq(r));
            dense.push(r, 1.0); // cancellation-free Welford reference
        }
        let sparse = sp.into_spherical();
        let true_ssd = 1.0; // mean [b+0.5, b−0.5]; each point contributes 0.5 ⇒ Σ = 1.0

        for (a, d) in sparse.mean().iter().zip(dense.mean()) {
            assert!((a - d).abs() < 1e-6, "means diverge: {a} vs {d}"); // mean stable in both paths
        }
        assert!(
            (dense.ssd() - true_ssd).abs() < 1e-6,
            "dense ssd = {}",
            dense.ssd()
        );
        // the O(nnz) expanded sparse path cannot recover it (documented trade-off), yet never negative
        assert!(
            (sparse.ssd() - true_ssd).abs() > 0.5,
            "sparse ssd = {}",
            sparse.ssd()
        );
        assert!(sparse.ssd() >= 0.0);
    }

    #[test]
    fn validate_csr_rejects_hostile_n_features() {
        // The trust-boundary DoS: a single non-zero with a huge n_features would force an ~8 EB
        // `vec![0.0; n_features]`. Must be rejected up front, not allocated.
        assert!(validate_csr(&[1.0], &[0], &[0, 1], usize::MAX / 2).is_err());
        assert!(validate_csr(&[1.0], &[0], &[0, 1], MAX_SPARSE_FEATURES + 1).is_err());
        assert!(validate_csr(&[1.0], &[0], &[0, 1], MAX_SPARSE_FEATURES).is_ok());
    }

    #[test]
    fn validate_csr_catches_malformed_arrays() {
        assert!(validate_csr(&[1.0], &[0], &[0, 1], 0).is_err()); // n_features == 0
        assert!(validate_csr(&[1.0, 2.0], &[0], &[0, 1], 4).is_err()); // data/indices length mismatch
        assert!(validate_csr(&[1.0], &[0], &[1, 1], 4).is_err()); // indptr[0] != 0
        assert!(validate_csr(&[1.0], &[0], &[0, 2], 4).is_err()); // indptr end != nnz
        assert!(validate_csr(&[1.0, 2.0], &[0, 1], &[0, 2, 1], 4).is_err()); // non-decreasing
        assert!(validate_csr(&[1.0], &[9], &[0, 1], 4).is_err()); // column index out of range
        assert!(validate_csr(&[1.0], &[-1], &[0, 1], 4).is_err()); // negative column index
        assert!(validate_csr(&[f64::NAN], &[0], &[0, 1], 4).is_err()); // non-finite value
        assert!(validate_csr(&[], &[], &[0], 4).is_ok()); // zero rows is valid
    }

    proptest! {
        // Any CSR that passes validation must summarise without panicking and conserve finite mass —
        // no out-of-bounds slice, no NaN centroid. Bounds keep the property test itself from OOMing;
        // most random inputs fail validation (that path must not panic either).
        #[test]
        fn validated_csr_never_panics(
            data in prop::collection::vec(-1e6f64..1e6, 0..48),
            indices in prop::collection::vec(-2i64..64, 0..48),
            indptr in prop::collection::vec(-2i64..96, 0..12),
            n_features in 0usize..64,
        ) {
            if validate_csr(&data, &indices, &indptr, n_features).is_ok() {
                let micros = summarize_sparse(&data, &indices, &indptr, n_features, 0.5, 32);
                let mass: f64 = micros.iter().map(|m| m.weight()).sum();
                prop_assert!(mass.is_finite());
                prop_assert!(micros.iter().all(|m| m.mean().iter().all(|v| v.is_finite())));
            }
        }
    }

    #[test]
    fn validate_csr_names_the_exact_malformation() {
        // Well-formed, including an *empty* row: `indptr` may repeat, so the non-decreasing test
        // must accept equality. Rejecting it would refuse every matrix with an all-zero row.
        assert!(validate_csr(&[1.0, 2.0], &[0, 2], &[0, 0, 2], 3).is_ok());
        assert!(
            validate_csr(&[], &[], &[0], 3).is_ok(),
            "an empty matrix is valid"
        );

        /// `(data, indices, indptr, n_features, expected error fragment)`
        type Case = (
            &'static [f64],
            &'static [i64],
            &'static [i64],
            usize,
            &'static str,
        );
        let cases: [Case; 6] = [
            (&[1.0], &[0], &[0, 1], 0, "n_features"),
            (&[1.0, 2.0], &[0], &[0, 2], 3, "equal length"),
            (&[1.0, 2.0], &[0, 1], &[1, 2], 3, "start at 0"),
            (&[1.0, 2.0], &[0, 1], &[0, 1], 3, "end at nnz"),
            (&[1.0, 2.0], &[0, 1], &[0, 2, 1, 2], 3, "non-decreasing"),
            (&[1.0, f64::NAN], &[0, 1], &[0, 2], 3, "NaN"),
        ];
        for (data, indices, indptr, n_features, needle) in cases {
            let err = validate_csr(data, indices, indptr, n_features)
                .expect_err(&format!("accepted a matrix that should fail on `{needle}`"));
            assert!(err.contains(needle), "wrong error for `{needle}`: {err}");
        }

        // Column bounds are checked on both sides: negative and past the end.
        assert!(
            validate_csr(&[1.0], &[-1], &[0, 1], 3).is_err(),
            "negative column accepted"
        );
        assert!(
            validate_csr(&[1.0], &[3], &[0, 1], 3).is_err(),
            "out-of-range column accepted"
        );
        assert!(
            validate_csr(&[1.0], &[2], &[0, 1], 3).is_ok(),
            "the last column was rejected"
        );

        // An empty `indptr` must not be read as a valid trailer.
        assert!(
            validate_csr(&[1.0], &[0], &[], 3).is_err(),
            "empty indptr accepted"
        );
    }

    #[test]
    fn nearest_sparse_expands_the_squared_distance_and_keeps_the_first_tie() {
        // ‖x − μ‖² = ‖x‖² − 2⟨x, μ⟩ + ‖μ‖². Row x = (1, 0, 2) against μ0 = (1, 0, 0) and
        // μ1 = (0, 0, 2): d0 = 5 − 2·1 + 1 = 4, d1 = 5 − 2·4 + 4 = 1, so μ1 wins.
        let means = vec![vec![1.0, 0.0, 0.0], vec![0.0, 0.0, 2.0]];
        let musq = vec![1.0, 4.0];
        let (idx, val) = (vec![0usize, 2], vec![1.0, 2.0]);
        assert_eq!(nearest_sparse(&means, &musq, &idx, &val, 5.0), 1);

        // Exact tie: μ0 = (1,0,2) and μ1 = (1,0,2) are both at distance 0; the first must win.
        let means = vec![vec![1.0, 0.0, 2.0], vec![1.0, 0.0, 2.0]];
        let musq = vec![5.0, 5.0];
        assert_eq!(nearest_sparse(&means, &musq, &idx, &val, 5.0), 0);

        // A single candidate is returned whatever the distance.
        let means = vec![vec![9.0, 9.0, 9.0]];
        assert_eq!(nearest_sparse(&means, &[243.0], &idx, &val, 5.0), 0);
    }

    #[test]
    fn the_sparse_accumulator_distance_uses_the_running_centroid() {
        // Two rows folded in: ΣX = (3, 0, 4), n = 2, so μ = (1.5, 0, 2) and ‖ΣX‖² = 25.
        // A third row x = (1, 0, 0) is at ‖x‖² − 2⟨ΣX, x⟩/n + ‖ΣX‖²/n² = 1 − 3 + 6.25 = 4.25,
        // which is exactly ‖x − μ‖² = 0.25 + 4.
        let mut acc = SparseSpherical::new(3);
        acc.push(&[0, 2], &[1.0, 2.0], 5.0);
        acc.push(&[0, 2], &[2.0, 2.0], 8.0);
        let d = acc.dist2(&[0], &[1.0], 1.0);
        assert!((d - 4.25).abs() < 1e-12, "dist2 = {d}");

        // An empty accumulator has no centroid, so the distance is the row's own norm.
        let empty = SparseSpherical::new(3);
        assert!((empty.dist2(&[0], &[1.0], 1.0) - 1.0).abs() < 1e-12);

        // The scatter matches the dense Welford value: two points 0.5 either side of the mean in
        // dim 0 give ssd = 0.5.
        let sph = acc.into_spherical();
        assert!((sph.weight() - 2.0).abs() < 1e-12);
        assert!((sph.mean()[0] - 1.5).abs() < 1e-12);
        assert!((sph.ssd() - 0.5).abs() < 1e-9, "ssd = {}", sph.ssd());
    }

    #[test]
    fn summarize_keeps_the_first_of_two_equally_near_leaders() {
        // Threshold 0 forces every distinct row into its own leader; the two seeds are equidistant
        // from the probe row, so only the scan's tie rule decides which absorbs it.
        let data = [1.0, 1.0, 1.0];
        let indices = [0i64, 2, 1];
        let indptr = [0i64, 1, 2, 3];
        let leaders = summarize_sparse(&data, &indices, &indptr, 3, 0.0, 8);
        assert_eq!(leaders.len(), 3, "distinct rows were merged at threshold 0");

        // With the leader cap reached, the nearest leader absorbs regardless of the threshold.
        let capped = summarize_sparse(&data, &indices, &indptr, 3, 0.0, 1);
        assert_eq!(capped.len(), 1, "max_leaders was not enforced");
        assert!((capped[0].weight() - 3.0).abs() < 1e-12);
    }
}
