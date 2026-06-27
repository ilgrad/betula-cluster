//! Small dense linear algebra for full-covariance features and Gaussian/Mahalanobis criteria.
//!
//! Dimensionality `d` is small/moderate for clustering, so hand-rolled `O(d^3)` Cholesky and
//! `O(d^2)` triangular solves are adequate and dependency-free (no LAPACK/BLAS). For larger or
//! batched linear algebra we will reach for `faer` (pure-Rust SIMD) later.

use crate::types::Real;

/// Lower-triangular Cholesky factor `L` (dense, row-major) with `A = L Lᵀ`.
/// `None` if `A` is not positive-definite (e.g. fewer points than dimensions, or a flat cluster).
pub fn cholesky_lower<R: Real>(a: &[Vec<R>]) -> Option<Vec<Vec<R>>> {
    let d = a.len();
    let mut l = vec![vec![R::zero(); d]; d];
    for i in 0..d {
        for j in 0..=i {
            let dot: R = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            let sum = a[i][j] - dot;
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
    let tol = R::from_f64(1e-15).unwrap();
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
                if apq.abs() <= tol {
                    continue;
                }
                // Rotation angle that zeros a[p][q] (Numerical Recipes form).
                let theta = half * (a[q][q] - a[p][p]) / apq;
                let t = {
                    let mag = one / (theta.abs() + (theta * theta + one).sqrt());
                    if theta < R::zero() {
                        -mag
                    } else {
                        mag
                    }
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
}
