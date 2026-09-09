//! Small dense linear algebra for full-covariance features and Gaussian/Mahalanobis criteria.
//!
//! Dimensionality `d` is small/moderate for clustering, so hand-rolled `O(d^3)` Cholesky and
//! `O(d^2)` triangular solves are adequate and dependency-free (no LAPACK/BLAS). For larger or
//! batched linear algebra we will reach for `faer` (pure-Rust SIMD) later.

use crate::types::Real;

/// Lower-triangular Cholesky factor `L` (dense, row-major) with `A = L Lᵀ`.
/// `None` if `A` is not positive-definite (e.g. fewer points than dimensions, or a flat cluster),
/// and equally if a partial sum is not finite: `NaN <= 0` is false, so a non-finite entry passed
/// the definiteness test and left every downstream `logdet` and Mahalanobis distance `NaN`.
pub fn cholesky_lower<R: Real>(a: &[Vec<R>]) -> Option<Vec<Vec<R>>> {
    let d = a.len();
    let mut l = vec![vec![R::zero(); d]; d];
    for i in 0..d {
        for j in 0..=i {
            let dot: R = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            let sum = a[i][j] - dot;
            if !sum.is_finite() {
                return None;
            }
            if i == j {
                if sum <= R::zero() {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L y = b` for lower-triangular `L` (forward substitution).
pub fn solve_lower<R: Real>(l: &[Vec<R>], b: &[R]) -> Vec<R> {
    let d = l.len();
    let mut y = vec![R::zero(); d];
    for i in 0..d {
        let dot: R = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = (b[i] - dot) / l[i][i];
    }
    y
}

/// `log|A|` from its Cholesky factor `L`: `2 · Σ_i ln L[i][i]`.
pub fn logdet_from_chol<R: Real>(l: &[Vec<R>]) -> R {
    let two = R::one() + R::one();
    let mut s = R::zero();
    for (i, row) in l.iter().enumerate() {
        s = s + row[i].ln();
    }
    two * s
}

/// Squared Mahalanobis distance `δᵀ A⁻¹ δ` from the Cholesky factor `L` of `A` (`= ‖L⁻¹ δ‖²`).
pub fn mahalanobis_sq_from_chol<R: Real>(l: &[Vec<R>], delta: &[R]) -> R {
    solve_lower(l, delta).iter().copied().map(|v| v * v).sum()
}

/// Solve `Lᵀ x = b` for a lower-triangular `L` (back substitution).
pub fn solve_upper_t<R: Real>(l: &[Vec<R>], b: &[R]) -> Vec<R> {
    let d = l.len();
    let mut x = vec![R::zero(); d];
    for i in (0..d).rev() {
        let mut s = b[i];
        for k in (i + 1)..d {
            s = s - l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Inverse of `A = L Lᵀ` from its Cholesky factor `L` (solving `A x = e_j` per column).
pub fn inv_from_chol<R: Real>(l: &[Vec<R>]) -> Vec<Vec<R>> {
    let d = l.len();
    let mut inv = vec![vec![R::zero(); d]; d];
    let mut e = vec![R::zero(); d];
    for j in 0..d {
        e[j] = R::one();
        let col = solve_upper_t(l, &solve_lower(l, &e));
        for (i, &xi) in col.iter().enumerate() {
            inv[i][j] = xi;
        }
        e[j] = R::zero();
    }
    inv
}

/// Eigenvalues and eigenvectors of a small dense **symmetric** matrix via the cyclic Jacobi
/// algorithm. Returns `(eigenvalues, V)` where column `j` of `V` is the (unit) eigenvector for
/// `eigenvalues[j]`; values are not sorted. Robust and accurate for the small (a few dozen rows)
/// symmetric matrices used by the Frequent-Directions sketch.
///
/// Sweeps stop when `off(A) <= eps_R · ‖A‖_F`, so the answer is invariant under scaling the input
/// by a positive constant, in `f32` as in `f64`. A cap of 100 sweeps bounds the work; cyclic Jacobi
/// converges quadratically and reaches the tolerance in well under ten sweeps for every matrix this
/// crate builds, so the cap is a guard against a malformed (non-finite) input rather than a
/// documented iteration budget.
#[allow(clippy::needless_range_loop)] // matrix rotation is inherently (p, q)-index based
pub fn jacobi_eigen<R: Real>(matrix: &[Vec<R>]) -> (Vec<R>, Vec<Vec<R>>) {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut v = vec![vec![R::zero(); n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = R::one();
    }
    if n <= 1 {
        let eig = (0..n).map(|i| a[i][i]).collect();
        return (eig, v);
    }
    // The convergence test is relative to the matrix, not absolute: `off(A) <= eps_R * ‖A‖_F`
    // (Golub & Van Loan, Alg. 8.4.3). A fixed 1e-15 answered two different questions wrongly at
    // once — it is unreachable for a unit-scale matrix in either precision, since `off` sums
    // n(n-1)/2 entries and each has to fall to 1e-15/n first, so the loop always ran to the sweep
    // cap; and it is satisfied *before the first rotation* for a matrix whose entries are smaller
    // than 1e-15, where it returned the untouched diagonal as the spectrum.
    // `‖A‖_F` is invariant under the rotations, so it is computed once.
    let frob = a
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| x * x)
        .sum::<R>()
        .sqrt();
    let tol = R::epsilon() * frob;
    // An entry below `tol/n` cannot lift `off` above `tol`, so a sweep that rotates nothing is
    // immediately followed by the exit test above.
    let entry_tol = tol / R::from_usize(n).unwrap();
    let half = R::from_f64(0.5).unwrap();
    let one = R::one();
    for _sweep in 0..100 {
        let mut off = R::zero();
        for p in 0..n {
            for q in (p + 1)..n {
                off = off + a[p][q] * a[p][q];
            }
        }
        if off.sqrt() <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() <= entry_tol {
                    continue;
                }
                // Rotation angle that zeros a[p][q] (Numerical Recipes form).
                let theta = half * (a[q][q] - a[p][p]) / apq;
                let t = {
                    let mag = one / (theta.abs() + (theta * theta + one).sqrt());
                    if theta < R::zero() { -mag } else { mag }
                };
                let c = one / (t * t + one).sqrt();
                let s = t * c;
                let tau = s / (one + c);
                a[p][p] = a[p][p] - t * apq;
                a[q][q] = a[q][q] + t * apq;
                a[p][q] = R::zero();
                a[q][p] = R::zero();
                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[i][p];
                        let aiq = a[i][q];
                        let nip = aip - s * (aiq + tau * aip);
                        let niq = aiq + s * (aip - tau * aiq);
                        a[i][p] = nip;
                        a[p][i] = nip;
                        a[i][q] = niq;
                        a[q][i] = niq;
                    }
                }
                for row in v.iter_mut() {
                    let vip = row[p];
                    let viq = row[q];
                    row[p] = vip - s * (viq + tau * vip);
                    row[q] = viq + s * (vip - tau * viq);
                }
            }
        }
    }
    let eig = (0..n).map(|i| a[i][i]).collect();
    (eig, v)
}

/// Matrix logarithm of a symmetric positive-(semi)definite matrix `A`, via its symmetric
/// eigendecomposition: `log A = U · diag(ln max(λ_k, floor)) · Uᵀ`. Eigenvalues are floored at
/// `floor` before the log so a near-singular (ridge-regularized) covariance stays finite — the
/// log-Euclidean metric needs strictly positive eigenvalues. The result is symmetric.
#[allow(clippy::needless_range_loop)] // symmetric recomposition reads clearest with (i, j, k) indices
pub fn matrix_log<R: Real>(a: &[Vec<R>], floor: R) -> Vec<Vec<R>> {
    let n = a.len();
    let (eig, u) = jacobi_eigen(a);
    let logl: Vec<R> = eig.iter().map(|&l| l.max(floor).ln()).collect();
    let mut out = vec![vec![R::zero(); n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = R::zero();
            for k in 0..n {
                s = s + u[i][k] * logl[k] * u[j][k];
            }
            out[i][j] = s;
        }
    }
    out
}

/// Gram-Schmidt with one re-orthogonalisation pass, in place, over the **rows**.
///
/// A row that collapses is left at zero rather than filled with a random replacement: it then
/// carries no direction, which is the honest answer when the block has lower rank than it has rows.
/// The second pass is not decoration — one pass loses orthogonality to `O(κ)` on an ill-conditioned
/// block, and every caller here feeds the result to a Rayleigh quotient, where that loss shows up as
/// a wrong eigenvalue rather than as a warning.
pub fn orthonormalize_rows<R: Real>(rows: &mut [Vec<R>]) {
    let tiny = R::from_f64(1e-150).unwrap();
    for i in 0..rows.len() {
        for _ in 0..2 {
            for j in 0..i {
                let p = rows[i]
                    .iter()
                    .zip(&rows[j])
                    .map(|(&x, &y)| x * y)
                    .fold(R::zero(), |a, b| a + b);
                if p != R::zero() {
                    for d in 0..rows[i].len() {
                        rows[i][d] = rows[i][d] - p * rows[j][d];
                    }
                }
            }
        }
        let norm = rows[i]
            .iter()
            .map(|&x| x * x)
            .fold(R::zero(), |a, b| a + b)
            .sqrt();
        if norm > tiny {
            for v in rows[i].iter_mut() {
                *v = *v / norm;
            }
        } else {
            rows[i].iter_mut().for_each(|v| *v = R::zero());
        }
    }
}

/// Squared Frobenius distance `‖A − B‖²_F = Σ_{ij} (A_ij − B_ij)²` between two same-shape matrices.
pub fn frobenius_sq_diff<R: Real>(a: &[Vec<R>], b: &[Vec<R>]) -> R {
    a.iter()
        .zip(b)
        .flat_map(|(ar, br)| ar.iter().zip(br).map(|(&x, &y)| (x - y) * (x - y)))
        .fold(R::zero(), |acc, v| acc + v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn cholesky_reconstructs_and_solves() {
        // A = [[4,2],[2,3]] (SPD). L Lᵀ == A; logdet == ln(det); Mahalanobis matches A⁻¹.
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky_lower(&a).unwrap();
        let mut recon = vec![vec![0.0; 2]; 2];
        for (i, rrow) in recon.iter_mut().enumerate() {
            for (j, cell) in rrow.iter_mut().enumerate() {
                *cell = (0..2).map(|k| l[i][k] * l[j][k]).sum();
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                assert!(close(recon[i][j], a[i][j]));
            }
        }
        assert!(close(logdet_from_chol(&l), 8.0_f64.ln())); // det = 4*3 - 2*2 = 8
        // A⁻¹ = 1/8 [[3,-2],[-2,4]]; δ=[1,1] -> δᵀA⁻¹δ = (3-2-2+4)/8 = 3/8
        assert!(close(mahalanobis_sq_from_chol(&l, &[1.0, 1.0]), 3.0 / 8.0));
    }

    #[test]
    fn matrix_log_inverts_matrix_exp() {
        // exp(log A) == A for an SPD A: eigendecompose log A, exponentiate its eigenvalues, recompose.
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let la = matrix_log(&a, 1e-12);
        assert!(close(la[0][1], la[1][0]), "log A must stay symmetric");
        let (eig, u) = jacobi_eigen(&la);
        let e: Vec<f64> = eig.iter().map(|&l| l.exp()).collect();
        for i in 0..2 {
            for j in 0..2 {
                let recon: f64 = (0..2).map(|k| u[i][k] * e[k] * u[j][k]).sum();
                assert!(close(recon, a[i][j]), "exp(log A) != A at ({i},{j})");
            }
        }
    }

    #[test]
    fn matrix_log_of_identity_is_zero() {
        let l = matrix_log(&[vec![1.0, 0.0], vec![0.0, 1.0]], 1e-12);
        assert!(l.iter().flatten().all(|&v| close(v, 0.0)));
    }

    #[test]
    fn frobenius_sq_diff_matches_manual() {
        // diffs 1, 0, -2, 0 → 1 + 4 = 5.
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![0.0, 2.0], vec![5.0, 4.0]];
        assert!(close(frobenius_sq_diff(&a, &b), 5.0));
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // indefinite (det = -3)
        assert!(cholesky_lower(&a).is_none());
    }

    #[test]
    fn inverse_from_cholesky() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky_lower(&a).unwrap();
        let inv = inv_from_chol(&l); // A⁻¹ = 1/8 [[3,-2],[-2,4]]
        assert!(close(inv[0][0], 3.0 / 8.0));
        assert!(close(inv[0][1], -2.0 / 8.0));
        assert!(close(inv[1][0], -2.0 / 8.0));
        assert!(close(inv[1][1], 4.0 / 8.0));
    }

    /// `octave-cli`'s `eig` on the same matrix, at 17 significant digits, unscaled and at 1e-18.
    /// A second engine rather than a tolerance: the failure this pins is a *silent* one — the
    /// scaled matrix used to come back with its own diagonal as the spectrum, which no
    /// reconstruction test on the unscaled matrix can see.
    const OCTAVE_EIG: [f64; 4] = [
        -0.47549823443677247,
        3.389_528_888_999_065,
        4.071833859146313,
        7.2641354862913925,
    ];

    fn e6_matrix(scale: f64) -> Vec<Vec<f64>> {
        [
            [4.0, 1.0, -2.0, 0.5],
            [1.0, 3.0, 0.75, -1.25],
            [-2.0, 0.75, 5.5, 2.0],
            [0.5, -1.25, 2.0, 1.75],
        ]
        .iter()
        .map(|row| row.iter().map(|&v| v * scale).collect())
        .collect()
    }

    #[test]
    fn jacobi_agrees_with_octave_at_every_scale() {
        for scale in [1e18, 1.0, 1e-9, 1e-18, 1e-30] {
            let (mut eig, _) = jacobi_eigen(&e6_matrix(scale));
            eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (got, want) in eig.iter().zip(OCTAVE_EIG.iter()) {
                let rel = (got / scale - want).abs() / want.abs();
                assert!(
                    rel < 1e-13,
                    "scale {scale:e}: {got:e} vs {want} (rel {rel:e})"
                );
            }
        }
    }

    #[test]
    fn jacobi_converges_in_f32_where_an_absolute_tolerance_could_not() {
        // `eps_f32` is 1.2e-7, so a 1e-15 threshold was unreachable for a unit-scale f32 matrix and
        // the loop always spent its whole sweep budget. It reached the right answer anyway, so this
        // pins an invariant the module had no `f32` coverage for rather than the defect itself.
        let a: Vec<Vec<f32>> = e6_matrix(1.0)
            .iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let (mut eig, _) = jacobi_eigen(&a);
        eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (got, want) in eig.iter().zip(OCTAVE_EIG.iter()) {
            assert!(
                ((*got as f64) - want).abs() / want.abs() < 1e-6,
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn cholesky_rejects_a_non_finite_entry() {
        // `NaN <= 0.0` is false, so the definiteness test passed it straight through and every
        // `logdet` and Mahalanobis distance downstream came back NaN.
        for bad in [f64::NAN, f64::INFINITY] {
            let a = vec![vec![4.0, bad], vec![bad, 3.0]];
            assert!(cholesky_lower(&a).is_none(), "accepted {bad}");
        }
    }

    #[test]
    fn jacobi_symmetric_eigen() {
        // Symmetric A; verify V Λ Vᵀ == A, orthonormal V, and trace == Σ eigenvalues.
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let (eig, v) = jacobi_eigen(&a);
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let recon: f64 = (0..n).map(|k| v[i][k] * eig[k] * v[j][k]).sum();
                assert!(
                    close(recon, a[i][j]),
                    "A[{i}][{j}] = {recon} vs {}",
                    a[i][j]
                );
            }
        }
        for p in 0..n {
            for q in 0..n {
                let dot: f64 = (0..n).map(|i| v[i][p] * v[i][q]).sum();
                let want = if p == q { 1.0 } else { 0.0 };
                assert!(close(dot, want), "VᵀV[{p}][{q}] = {dot}");
            }
        }
        let trace: f64 = (0..n).map(|i| a[i][i]).sum();
        assert!(close(trace, eig.iter().sum()));
    }

    #[test]
    fn jacobi_eigen_handles_1x1() {
        let (eig, v) = jacobi_eigen(&[vec![3.5_f64]]);
        assert!((eig[0] - 3.5).abs() < 1e-12);
        assert!((v[0][0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_off_diagonal_norm_sums_squares_not_signed_entries() {
        // The strictly-upper entries cancel: any convergence test that adds them, rather than
        // their squares, reads zero on the first sweep and returns the untouched diagonal.
        let a = vec![
            vec![5.0, 1.0, -1.0],
            vec![1.0, 2.0, 0.0],
            vec![-1.0, 0.0, 3.0],
        ];
        let (mut eig, v) = jacobi_eigen(&a);
        for i in 0..3 {
            for j in 0..3 {
                let recon: f64 = (0..3).map(|k| v[i][k] * eig[k] * v[j][k]).sum();
                assert!(close(recon, a[i][j]), "A[{i}][{j}] = {recon}");
            }
        }
        eig.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let diag = [2.0, 3.0, 5.0];
        assert!(
            eig.iter().zip(&diag).any(|(e, d)| (e - d).abs() > 1e-6),
            "the sweep returned the diagonal unrotated: {eig:?}"
        );
    }

    #[test]
    fn a_zero_rotation_angle_takes_the_positive_root() {
        // Equal diagonal entries give theta == 0, where the sign rule decides which of the two
        // eigenvalues lands in slot p. Numerical Recipes takes t = +1 there.
        let (eig, v) = jacobi_eigen(&[vec![2.0, 1.0], vec![1.0, 2.0]]);
        assert!(close(eig[0], 1.0), "eig = {eig:?}");
        assert!(close(eig[1], 3.0), "eig = {eig:?}");
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!(close(v[0][0].abs(), s) && close(v[1][0].abs(), s), "{v:?}");
        assert!(close(v[0][0] * v[1][0], -0.5), "{v:?}");
    }
}
