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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::ClusterFeature;

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
}
