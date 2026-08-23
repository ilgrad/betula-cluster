//! CF-weighted PCA of the leaf summary — the second Phase-3 projection, for signed data and for the
//! reduce-then-cluster text pipeline.
//!
//! Assigning every point in leaf `C_j` its leaf's code (the hard-leaf approximation every Phase-3 head
//! already makes), the total scatter of the data splits exactly by König-Huygens:
//! `T = Σ_j S_j + Σ_j n_j (μ_j − x̄)(μ_j − x̄)ᵀ = W + B`, with `x̄` the weighted grand mean. `B` — the
//! **between-leaf** scatter — is a function of `(n_j, μ_j)` alone, so it is exactly computable from the
//! summary; the leading eigenvectors of `B` are the top right singular vectors of the `M×d` matrix
//! `X̃_j = √n_j·(μ_j − x̄)`, which a randomized SVD reaches in `O(M·d·(r+p))` without ever forming a
//! `d×d` matrix. That matters: the text path runs at `d` in the thousands.
//!
//! **What the dropped `W` costs, exactly.** Under the spherical cluster feature — the one the sparse
//! and text paths use — the within-leaf model is isotropic, `S_j ≈ (S_j/d)·I`, so `W` is a multiple of
//! the identity and `eig(B + cI) = eig(B)`: the directions are unchanged and only the eigenvalues shift.
//! The weighted PCA of the centroids is then the data's PCA under the summary's own model, not an
//! approximation of it. For `diagonal` and `full` features `W` is not isotropic and the correction is
//! genuinely dropped — but its **trace** is still exact (`Σ_j ssd_j`), which is what bounds the
//! reconstruction error.
//!
//! Unlike NMF, this projection is a **linear map**: a raw row encodes as `(x − x̄)Vᵀ` in `O(d·r)`, so a
//! projected fit keeps the head's own point rule instead of falling back to the microcluster route.
//! Measured on 20-newsgroups that is worth 0.062 ARI — the whole of the gap this projection exists to
//! close (`plans/2026-08-23-supremacy-audit.md` §15).

// Reached only through the Python bindings, like the NMF projection beside it.
#![cfg_attr(not(feature = "python"), allow(dead_code))]

use crate::clustering::nmf::randomized_svd;
use crate::feature::ClusterFeature;
use crate::types::Real;

/// The linear encoder a weighted PCA leaves behind: subtract the weighted grand mean, then take
/// coordinates in an orthonormal basis of the leading between-leaf directions.
pub(crate) struct Pca<R: Real> {
    /// Weighted grand mean `x̄` (length `d`).
    pub centre: Vec<R>,
    /// Top-`rank` right singular vectors of `X̃`, one per row (`rank × d`), orthonormal.
    pub basis: Vec<Vec<R>>,
    /// Between-leaf scatter the basis captures, `Σ_{k<r} σ_k² / ‖X̃‖²_F`, in `[0, 1]`.
    pub captured: R,
}

impl<R: Real> Pca<R> {
    /// Encode one row: `(x − x̄)·Vᵀ`. Writes `rank` coordinates into `out`, reusing its allocation.
    pub fn encode(&self, x: &[R], out: &mut Vec<R>) {
        out.clear();
        out.extend(self.basis.iter().map(|v| {
            x.iter()
                .zip(&self.centre)
                .zip(v)
                .map(|((&xi, &ci), &vi)| (xi - ci) * vi)
                .fold(R::zero(), |a, b| a + b)
        }));
    }
}

/// Rank-`rank` CF-weighted PCA of the leaf summary. `rank` is clamped to what the summary can
/// support (`min(M, d)`); an empty summary yields an empty basis, which every caller treats as
/// "no projection ran".
pub(crate) fn weighted_pca<R, C>(feats: &[C], rank: usize, seed: u64) -> Pca<R>
where
    R: Real,
    C: ClusterFeature<R>,
{
    let d = feats.first().map_or(0, |f| f.dim());
    if d == 0 || rank == 0 {
        return Pca {
            centre: vec![R::zero(); d],
            basis: Vec::new(),
            captured: R::zero(),
        };
    }
    let weights: Vec<R> = feats.iter().map(|f| f.weight().max(R::zero())).collect();
    let total = weights.iter().copied().fold(R::zero(), |a, b| a + b);
    let mut centre = vec![R::zero(); d];
    if total > R::zero() {
        for (f, &w) in feats.iter().zip(&weights) {
            for (c, &m) in centre.iter_mut().zip(f.mean()) {
                *c = *c + w * m;
            }
        }
        centre.iter_mut().for_each(|c| *c = *c / total);
    }
    let x: Vec<Vec<R>> = feats
        .iter()
        .zip(&weights)
        .map(|(f, &w)| {
            let s = w.sqrt();
            f.mean()
                .iter()
                .zip(&centre)
                .map(|(&m, &c)| s * (m - c))
                .collect()
        })
        .collect();
    let energy = x
        .iter()
        .flatten()
        .map(|&v| v * v)
        .fold(R::zero(), |a, b| a + b);
    let rank = rank.min(feats.len()).min(d);
    let (sigma, _, basis) = randomized_svd(&x, rank, seed);
    let kept = sigma.iter().map(|&s| s * s).fold(R::zero(), |a, b| a + b);
    let captured = if energy > R::zero() {
        (kept / energy).min(R::one())
    } else {
        R::zero()
    };
    Pca {
        centre,
        basis,
        captured,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clustering::rng::SplitMix64;
    use crate::feature::Spherical;

    fn leaf<R: Real>(w: f64, mean: &[f64]) -> Spherical<R> {
        Spherical::from_moments(
            R::from_f64(w).unwrap(),
            mean.iter().map(|&v| R::from_f64(v).unwrap()).collect(),
            R::zero(),
        )
    }

    #[test]
    fn the_grand_mean_is_weighted_by_leaf_mass_not_by_leaf_count() {
        // Three leaves, one of them carrying 98 % of the points. An unweighted mean would sit near
        // (1, 1); the mass-weighted one has to sit essentially on the heavy leaf.
        let feats = vec![
            leaf::<f64>(1.0, &[0.0, 0.0]),
            leaf(1.0, &[3.0, 3.0]),
            leaf(98.0, &[1.0, 1.0]),
        ];
        let pca = weighted_pca(&feats, 2, 0);
        assert!((pca.centre[0] - 1.01).abs() < 1e-12, "{:?}", pca.centre);
        assert!((pca.centre[1] - 1.01).abs() < 1e-12, "{:?}", pca.centre);
    }

    #[test]
    fn the_basis_is_the_between_leaf_scatter_axis_and_encoding_is_its_coordinate() {
        // Leaf means strung along y = 2x with equal mass: the between-leaf scatter is rank 1 and its
        // axis is (1, 2)/√5. Rank 1 must therefore capture all of it, and encoding a point must give
        // its signed distance along that axis from the grand mean. The line is deliberately offset
        // from the origin — at `x̄ = 0` the centring is invisible, because `x − x̄` and `x + x̄` and
        // `μ − x̄` and `μ + x̄` all coincide.
        let feats: Vec<Spherical<f64>> = (-3..=3)
            .map(|t| leaf(1.0, &[t as f64 + 10.0, 2.0 * t as f64 + 20.0]))
            .collect();
        let pca = weighted_pca(&feats, 1, 7);
        assert!(
            (pca.centre[0] - 10.0).abs() < 1e-12 && (pca.centre[1] - 20.0).abs() < 1e-12,
            "{:?}",
            pca.centre
        );
        assert!(
            (pca.captured - 1.0).abs() < 1e-10,
            "captured {}",
            pca.captured
        );
        let axis = (1.0f64 / 5.0).sqrt();
        assert!(
            (pca.basis[0][0].abs() - axis).abs() < 1e-10,
            "{:?}",
            pca.basis
        );
        // The sign of a singular vector is arbitrary, so the closed form is asserted on |code| and the
        // sign is pinned against an independent dot product in whichever orientation the SVD returned.
        let mut code = Vec::new();
        pca.encode(&[11.0, 22.0], &mut code);
        assert_eq!(code.len(), 1);
        assert!((code[0].abs() - 5.0f64.sqrt()).abs() < 1e-10, "{code:?}");
        let want = pca.basis[0][0] + 2.0 * pca.basis[0][1];
        assert!((code[0] - want).abs() < 1e-10, "{code:?} vs {want}");
    }

    #[test]
    fn a_rank_below_the_true_one_captures_strictly_less_than_all_of_it() {
        // Two orthogonal planted directions with a 4:1 energy ratio. Rank 1 must take the larger and
        // leave the smaller, so `captured` has to land between the two — a routine that reported 1.0
        // regardless (or 0.0) would pass no part of this.
        let mut rng = SplitMix64::new(3);
        let feats: Vec<Spherical<f64>> = (0..64)
            .map(|_| {
                let (a, b) = (rng.gauss() * 2.0, rng.gauss());
                leaf(1.0, &[a, b, 0.0])
            })
            .collect();
        let pca = weighted_pca(&feats, 1, 1);
        assert!(
            (0.6..0.95).contains(&pca.captured),
            "captured {}",
            pca.captured
        );
        assert_eq!(weighted_pca(&feats, 3, 1).basis.len(), 3);
        assert!((weighted_pca(&feats, 3, 1).captured - 1.0).abs() < 1e-10);
    }

    #[test]
    fn an_empty_or_rankless_summary_yields_no_basis_rather_than_a_panic() {
        let none: Vec<Spherical<f64>> = Vec::new();
        assert!(weighted_pca(&none, 4, 0).basis.is_empty());
        let feats = vec![leaf::<f64>(1.0, &[1.0, 2.0]), leaf(2.0, &[3.0, 4.0])];
        assert!(weighted_pca(&feats, 0, 0).basis.is_empty());
        // Rank is clamped to what M×d can support, not to what the caller asked for.
        assert_eq!(weighted_pca(&feats, 9, 0).basis.len(), 2);
    }

    #[test]
    fn a_summary_with_no_mass_and_a_summary_with_no_spread_report_zero_rather_than_nan() {
        // The two guards in `weighted_pca` are both `x > 0` on a quantity that can be exactly 0, and
        // both divide by it. Massless leaves make the grand-mean divisor 0; co-located leaves make the
        // energy divisor 0. Either guard relaxed to `>=` yields NaN — and NaN is not loud here, since
        // `NaN.min(1)` is 1, so `captured` would silently claim the basis explains everything.
        let massless = vec![leaf::<f64>(0.0, &[3.0, 4.0]), leaf(0.0, &[-1.0, 5.0])];
        let pca = weighted_pca(&massless, 2, 0);
        assert_eq!(pca.centre, vec![0.0, 0.0], "{:?}", pca.centre);
        assert!(pca.captured.is_finite(), "captured {}", pca.captured);

        let flat = vec![leaf::<f64>(2.0, &[7.0, -1.0]); 5];
        let pca = weighted_pca(&flat, 2, 0);
        assert_eq!(pca.captured, 0.0, "captured {}", pca.captured);
    }
}
