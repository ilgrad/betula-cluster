//! Small special-function helpers for statistical absorption / merge gates.
//!
//! The χ² quantile is the threshold for a Mahalanobis absorption gate: a point is absorbed into a
//! cluster only if its squared Mahalanobis distance is below `chi2_quantile(d, p)` — a mass- and
//! scale-invariant criterion that fixes the BIRCH size-imbalance bug (scikit-learn #22854,
//! `../../math_improove/05`). Computed from the regularized lower incomplete gamma via
//! `χ²_d CDF(x) = P(d/2, x/2)` (DLMF 8.2.4); the inverse uses the Numerical-Recipes `invgammp`
//! scheme. All math is done in `f64` (this is config-time work, run once per tree).

use std::f64::consts::PI;

const EPS: f64 = 1e-14;
const FPMIN: f64 = 1e-300;

/// Lanczos approximation of `ln Γ(x)` (g = 7), with reflection for `x < 0.5`.
fn ln_gamma(x: f64) -> f64 {
    // Published Lanczos g=7 coefficients; the trailing digits beyond f64 precision are harmless.
    #[allow(clippy::excessive_precision)]
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Γ(x)Γ(1-x) = π / sin(πx).
        (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + 7.5;
        let mut a = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Series expansion of the regularized lower incomplete gamma `P(a, x)` for `x < a + 1`.
fn gser(a: f64, x: f64, gln: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..1000 {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            break;
        }
    }
    sum * (-x + a * x.ln() - gln).exp()
}

/// Continued-fraction expansion of the regularized upper incomplete gamma `Q(a, x)` for `x ≥ a+1`.
fn gcf(a: f64, x: f64, gln: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..1000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    (-x + a * x.ln() - gln).exp() * h
}

/// Regularized lower incomplete gamma `P(a, x) = γ(a, x) / Γ(a)` for `a > 0`, `x ≥ 0`.
fn reg_lower_gamma(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && x >= 0.0);
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        gser(a, x, gln)
    } else {
        1.0 - gcf(a, x, gln)
    }
}

/// Inverse of `P(a, x) = p` in `x` (Numerical Recipes `invgammp`): an analytic initial guess
/// refined by Halley steps. `p ∈ (0, 1)`, `a > 0`.
fn inv_reg_lower_gamma(a: f64, p: f64) -> f64 {
    let gln = ln_gamma(a);
    let a1 = a - 1.0;
    let lna1 = if a > 1.0 { a1.ln() } else { 0.0 };
    let afac = if a > 1.0 {
        (a1 * (lna1 - 1.0) - gln).exp()
    } else {
        0.0
    };

    // Initial guess.
    let mut x;
    if a > 1.0 {
        let pp = if p < 0.5 { p } else { 1.0 - p };
        let t = (-2.0 * pp.ln()).sqrt();
        let mut xx = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
        if p < 0.5 {
            xx = -xx;
        }
        x = (a * (1.0 - 1.0 / (9.0 * a) - xx / (3.0 * a.sqrt())).powi(3)).max(1e-3);
    } else {
        let t = 1.0 - a * (0.253 + a * 0.12);
        if p < t {
            x = (p / t).powf(1.0 / a);
        } else {
            x = 1.0 - (1.0 - (p - t) / (1.0 - t)).ln();
        }
    }

    // Halley refinement on err = P(a, x) - p.
    for _ in 0..12 {
        if x <= 0.0 {
            return 0.0;
        }
        let err = reg_lower_gamma(a, x) - p;
        let t = if a > 1.0 {
            afac * (-(x - a1) + a1 * (x.ln() - lna1)).exp()
        } else {
            (-x + a1 * x.ln() - gln).exp()
        };
        let u = err / t;
        let step = u / (1.0 - 0.5 * (u * (a1 / x - 1.0)).min(1.0));
        x -= step;
        if x <= 0.0 {
            x = 0.5 * (x + step);
        }
        if step.abs() < EPS * x {
            break;
        }
    }
    x
}

/// Quantile (inverse CDF) of the χ² distribution with `d` degrees of freedom at probability `p`.
///
/// Uses `χ²_d CDF(x) = P(d/2, x/2)`, so the quantile is `2 · P⁻¹(d/2, p)`. Returned value is the
/// Mahalanobis-distance² threshold for a `p`-level absorption / merge gate.
///
/// # Panics
/// Panics if `d == 0` or `p ∉ (0, 1)`.
pub fn chi2_quantile(d: usize, p: f64) -> f64 {
    assert!(d >= 1, "degrees of freedom must be >= 1");
    assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");
    2.0 * inv_reg_lower_gamma(d as f64 / 2.0, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn chi2_matches_known_quantiles() {
        // Authoritative χ² table values (upper-tail critical points).
        let cases = [
            (1, 0.95, 3.841_459),
            (2, 0.95, 5.991_465),
            (3, 0.95, 7.814_728),
            (5, 0.95, 11.070_498),
            (10, 0.95, 18.307_038),
            (1, 0.99, 6.634_897),
            (10, 0.99, 23.209_251),
            (2, 0.50, 1.386_294),
            (10, 0.90, 15.987_179),
            (4, 0.975, 11.143_287),
        ];
        for (d, p, want) in cases {
            let got = chi2_quantile(d, p);
            assert!(
                close(got, want, 1e-3),
                "chi2_quantile({d}, {p}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn cdf_inverts_quantile() {
        // P(d/2, q/2) should recover p for the q we computed.
        for &(d, p) in &[(1usize, 0.3f64), (3, 0.8), (7, 0.95), (20, 0.5)] {
            let q = chi2_quantile(d, p);
            let back = reg_lower_gamma(d as f64 / 2.0, q / 2.0);
            assert!(close(back, p, 1e-6), "d={d} p={p} q={q} back={back}");
        }
    }

    #[test]
    fn reg_lower_gamma_endpoints() {
        // P(a, 0) = 0 and P(a, ∞) → 1.
        assert!(close(reg_lower_gamma(1.5, 0.0), 0.0, 1e-12));
        assert!(close(reg_lower_gamma(1.5, 200.0), 1.0, 1e-9));
    }

    #[test]
    fn chi2_quantile_low_probability_branch() {
        // d > 2 (a > 1) with p < 0.5 exercises the inverse-incomplete-gamma sign flip.
        let q = chi2_quantile(6, 0.3);
        assert!(q > 0.0 && q.is_finite(), "q = {q}");
    }

    #[test]
    fn ln_gamma_uses_reflection_for_small_x() {
        // x < 0.5 hits the reflection branch; Γ(0.25) ≈ 3.6256099082.
        assert!(close(ln_gamma(0.25).exp(), 3.625_609_908_2, 1e-6));
    }
}
