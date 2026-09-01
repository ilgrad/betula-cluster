//! Small special-function helpers for statistical absorption / merge gates.
//!
//! The χ² quantile is the threshold for a Mahalanobis absorption gate: a point is absorbed into a
//! cluster only if its squared Mahalanobis distance is below `chi2_quantile(d, p)` — a mass- and
//! scale-invariant criterion that fixes the BIRCH size-imbalance bug (scikit-learn #22854,
//! `math_improove/05`, local-only notes). Computed from the regularized lower incomplete gamma via
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

    /// `inv_reg_lower_gamma` is a Halley iteration, and Halley is self-correcting: the existing χ²
    /// table test passes to 1e-3 even when the initial guess or the step denominator is corrupted,
    /// because twelve refinements still land close. Inverting the CDF and reading it back catches
    /// exactly that — a damaged iteration converges more slowly and leaves a bigger residual long
    /// before it moves the third decimal. Measured worst case on the unmutated code: 1.6e-13.
    #[test]
    fn the_gamma_quantile_inverts_its_own_cdf_to_full_precision() {
        for &a in &[0.2f64, 0.5, 1.0, 1.5, 3.0, 12.5, 60.0, 200.0] {
            for &p in &[
                1e-6f64,
                1e-3,
                0.05,
                0.25,
                0.5,
                0.75,
                0.95,
                0.999,
                1.0 - 1e-6,
            ] {
                let x = inv_reg_lower_gamma(a, p);
                assert!(x > 0.0 && x.is_finite(), "a={a} p={p}: quantile = {x}");
                let back = reg_lower_gamma(a, x);
                assert!(
                    (back - p).abs() <= 1e-11 * p,
                    "a={a} p={p}: P(a, {x}) = {back}, relative error {:e}",
                    (back - p).abs() / p
                );
            }
        }
    }

    #[test]
    fn the_gamma_quantile_is_strictly_increasing_in_p() {
        for &a in &[0.4f64, 1.0, 7.0, 45.0] {
            let mut prev = 0.0;
            for i in 1..200 {
                let p = i as f64 / 200.0;
                let x = inv_reg_lower_gamma(a, p);
                assert!(x > prev, "a={a}: quantile fell at p={p} ({prev} -> {x})");
                prev = x;
            }
        }
    }

    #[test]
    fn chi2_quantile_is_twice_the_gamma_quantile_at_half_the_degrees() {
        for d in [1usize, 2, 3, 7, 30] {
            for p in [0.05, 0.5, 0.95] {
                let got = chi2_quantile(d, p);
                let want = 2.0 * inv_reg_lower_gamma(d as f64 / 2.0, p);
                assert!(close(got, want, 1e-12), "d={d} p={p}: {got} vs {want}");
                // The χ²_d CDF read back at the quantile must return the probability asked for.
                assert!(
                    close(reg_lower_gamma(d as f64 / 2.0, got / 2.0), p, 1e-11),
                    "d={d} p={p}: CDF round-trip"
                );
            }
        }
    }

    /// `(a, x, P(a, x))` to full double precision, from mpmath at 80 digits. The grid
    /// straddles `x = a + 1` -- where the implementation switches from the series to the
    /// continued fraction -- at 0.999, 1.0 and 1.001 of it, so both branches and the
    /// boundary itself are covered at every `a`.
    ///
    /// **Evaluate the reference at `mpf(float(x))`, not at `mpf("x")`.** `P` has logarithmic
    /// sensitivity `≈ a` in the left tail, so the ~1e-17 gap between the decimal `x` and the
    /// `f64` this test actually passes to `reg_lower_gamma` is amplified by `a`: at `a = 200,
    /// x = 10.05` that is 61 ulp, and the reference then answers a question about a number the
    /// implementation is never given. The six left-tail rows were regenerated for this.
    const CDF_TABLE: &[(f64, f64, f64)] = &[
        (0.1, 0.055, 0.7826178064887638),
        (0.1, 0.44, 0.9336108910947823),
        (0.1, 1.0989, 0.9793638620479284),
        (0.1, 1.1, 0.9793992217906597),
        (0.1, 1.1011, 0.9794345108845673),
        (0.1, 2.2, 0.9956514277405353),
        (0.1, 6.6, 0.9999766684578184),
        (0.5, 0.075, 0.30146464169666126),
        (0.5, 0.6, 0.7266783217077019),
        (0.5, 1.4985, 0.9165811487106036),
        (0.5, 1.5, 0.9167354833364496),
        (0.5, 1.5015, 0.9168895096014817),
        (0.5, 3.0, 0.9856941215645704),
        (0.5, 9.0, 0.9999779095030014),
        (1.0, 0.1, 0.09516258196404043),
        (1.0, 0.8, 0.5506710358827784),
        (1.0, 1.998, 0.8643937753458103),
        (1.0, 2.0, 0.8646647167633873),
        (1.0, 2.002, 0.864935116839651),
        (1.0, 4.0, 0.9816843611112658),
        (1.0, 12.0, 0.9999938557876467),
        (1.5, 0.125, 0.03085959578372673),
        (1.5, 1.0, 0.4275932955291202),
        (1.5, 2.4975, 0.827836364801021),
        (1.5, 2.5, 0.8282028557032669),
        (1.5, 2.5025, 0.8285686143562626),
        (1.5, 5.0, 0.9814338645369568),
        (1.5, 15.0, 0.9999986199429687),
        (2.5, 0.175, 0.0034043674466371116),
        (2.5, 1.4, 0.26921351341124145),
        (2.5, 3.4965, 0.778838572373545),
        (2.5, 3.5, 0.7793596920632893),
        (2.5, 3.5035, 0.7798797705554588),
        (2.5, 7.0, 0.9843905838997331),
        (2.5, 21.0, 0.9999999410959279),
        (5.0, 0.3, 1.5785040541659963e-05),
        (5.0, 2.4, 0.09586859033639923),
        (5.0, 5.994, 0.7141395808632276),
        (5.0, 6.0, 0.7149434996833688),
        (5.0, 6.006, 0.7157458122737057),
        (5.0, 12.0, 0.992399609318933),
        (5.0, 36.0, 0.9999999999818046),
        (12.0, 0.65, 6.524144682539895e-12),
        (12.0, 5.2, 0.007310495624981595),
        (12.0, 12.987, 0.645514471643534),
        (12.0, 13.0, 0.6468350671487296),
        (12.0, 13.013, 0.6481530241071783),
        (12.0, 26.0, 0.9992176081886389),
        (50.0, 2.55, 5.73768331433484e-46),
        (50.0, 20.4, 2.276141792827706e-08),
        (50.0, 50.949, 0.5716034974248363),
        (50.0, 51.0, 0.5743948595168596),
        (50.0, 51.051, 0.5771806445092041),
        (50.0, 102.0, 0.9999999958642293),
        (200.0, 10.05, 1.562912316399334e-179),
        (200.0, 80.4, 2.8704844228061367e-29),
        (200.0, 200.799, 0.5318765976152702),
        (200.0, 201.0, 0.5375075509172329),
        (200.0, 201.201, 0.5431272539397605),
    ];

    /// `(a, p, P^-1(a, p))` to full double precision, from mpmath at 80 digits by bisection.
    /// `a` spans both initial-guess branches (`a > 1` and `a <= 1`), and `p` spans both
    /// halves of the `p < 0.5` reflection and both sides of the `p < t` split the small-`a`
    /// branch makes.
    /// `(a, p, P^-1(a, p))` to full double precision, from mpmath at 80 digits by bisection.
    /// The `p` grid reaches down to `1e-9` because the analytic initial guess is at its worst in
    /// the tails and the twelve Halley steps that follow hide a corrupted guess anywhere nearer
    /// the median. It stops at `1 - 1e-6` at the top: `P = 1 - Q` cancels there, so beyond that
    /// the table would measure the subtraction rather than the quantile.
    /// `(a, p, P^-1(a, p))` to full double precision, from mpmath at 60 digits by bisection **in
    /// log space** -- bisecting `x` directly cannot resolve the `1e-151` end of this grid at all, it
    /// returns its own floor instead. The `p` grid reaches `1e-14` and the `a` grid `1e4` because
    /// the analytic initial guess is at its worst in the tails, while the twelve Halley steps that
    /// follow are self-correcting enough to hide a corrupted guess anywhere nearer the median. It
    /// stops at `1 - 1e-6`: `P = 1 - Q` cancels beyond that, so the table would be measuring the
    /// subtraction. Rows whose answer underflows `f64` (`a = 0.02` below `p = 1e-3`) are absent.
    #[allow(clippy::approx_constant)] // `P^-1(1, 0.5)` really is `ln 2`.
    const QUANTILE_TABLE: &[(f64, f64, f64)] = &[
        (0.02, 0.001, 5.706812442247299e-151),
        (0.02, 0.05, 5.068667656480206e-66),
        (0.02, 0.2, 6.425299597094536e-36),
        (0.02, 0.4999, 5.018228551727737e-16),
        (0.02, 0.5, 5.068667656480183e-16),
        (0.02, 0.5001, 5.119603494398379e-16),
        (0.02, 0.75, 3.2318924992433466e-07),
        (0.02, 0.95, 0.04590982359974054),
        (0.02, 0.999, 2.0063208017766128),
        (0.02, 0.999999, 7.794654560684711),
        (0.1, 1e-14, 6.073048362407991e-141),
        (0.1, 1e-12, 6.073048362407975e-121),
        (0.1, 1e-09, 6.073048362407956e-91),
        (0.1, 1e-06, 6.0730483624079264e-61),
        (0.1, 0.001, 6.073048362407907e-31),
        (0.1, 0.05, 5.930711291414281e-14),
        (0.1, 0.2, 6.21880187468291e-08),
        (0.1, 0.4999, 0.0005922047512548213),
        (0.1, 0.5, 0.0005933911044602261),
        (0.1, 0.5001, 0.0005945795975899961),
        (0.1, 0.75, 0.0353063580735583),
        (0.1, 0.95, 0.5804351053231345),
        (0.1, 0.999, 3.3636770117187536),
        (0.1, 0.999999, 9.45702706045744),
        (0.25, 1e-14, 6.74969789311173e-57),
        (0.25, 1e-12, 6.74969789311173e-49),
        (0.25, 1e-09, 6.749697893111732e-37),
        (0.25, 1e-06, 6.749697893111729e-25),
        (0.25, 0.001, 6.749697893115375e-13),
        (0.25, 0.05, 4.218575420262992e-06),
        (0.25, 0.2, 0.0010808857306629767),
        (0.25, 0.4999, 0.04363763045181252),
        (0.25, 0.5, 0.04367380235287341),
        (0.25, 0.5001, 0.04370999804693902),
        (0.25, 0.75, 0.2606260019782326),
        (0.25, 0.95, 1.210116137444762),
        (0.25, 0.999, 4.3764442578866865),
        (0.25, 0.999999, 10.68781757641098),
        (0.5, 1e-14, 7.853981633974483e-29),
        (0.5, 1e-12, 7.853981633974483e-25),
        (0.5, 1e-09, 7.853981633974484e-19),
        (0.5, 1e-06, 7.853981633978594e-13),
        (0.5, 0.001, 7.85398574631245e-07),
        (0.5, 0.05, 0.0019660700000097616),
        (0.5, 0.2, 0.03209237733365079),
        (0.5, 0.4999, 0.22736210315538977),
        (0.5, 0.5, 0.2274682115597864),
        (0.5, 0.5001, 0.2275743559838986),
        (0.5, 0.75, 0.661651848465733),
        (0.5, 0.95, 1.9207294103470622),
        (0.5, 0.999, 5.4137830853313655),
        (0.5, 0.999999, 11.964063488439734),
        (0.75, 1e-14, 1.9251300738806556e-19),
        (0.75, 1e-12, 8.935662254176595e-17),
        (0.75, 1e-09, 8.935662254181158e-13),
        (0.75, 1e-06, 8.935662299802914e-09),
        (0.75, 0.001, 8.936118548088433e-05),
        (0.75, 0.05, 0.016616388156245176),
        (0.75, 0.2, 0.11129296135667874),
        (0.75, 0.4999, 0.454008570524411),
        (0.75, 0.5, 0.4541669783238561),
        (0.75, 0.5001, 0.45432542503855816),
        (0.75, 0.75, 1.0340914067757996),
        (0.75, 0.95, 2.490097642154325),
        (0.75, 0.999, 6.213083455477074),
        (0.75, 0.999999, 12.954021253340024),
        (0.9, 1e-14, 2.664603564600519e-16),
        (0.9, 1e-12, 4.44482663753542e-14),
        (0.9, 1e-09, 9.576088699566497e-11),
        (0.9, 1e-06, 2.0631059928342247e-07),
        (0.9, 0.001, 0.0004445866775525494),
        (0.9, 0.05, 0.034959469312141125),
        (0.9, 0.2, 0.17539136011397422),
        (0.9, 0.4999, 0.5965587507708906),
        (0.9, 0.5, 0.5967430489553945),
        (0.9, 0.5001, 0.5969273868061218),
        (0.9, 0.75, 1.2473282885862036),
        (0.9, 0.95, 2.798961058467819),
        (0.9, 0.999, 6.6388768279406944),
        (0.9, 0.999999, 13.482084784620758),
        (1.0, 1e-14, 1.000000000000005e-14),
        (1.0, 1e-12, 1.0000000000005e-12),
        (1.0, 1e-09, 1.0000000005000001e-09),
        (1.0, 1e-06, 1.0000005000003334e-06),
        (1.0, 0.001, 0.0010005003335835335),
        (1.0, 0.05, 0.051293294387550536),
        (1.0, 0.2, 0.22314355131420976),
        (1.0, 0.4999, 0.6929472005572791),
        (1.0, 0.5, 0.6931471805599453),
        (1.0, 0.5001, 0.6933472005626123),
        (1.0, 0.75, 1.3862943611198906),
        (1.0, 0.95, 2.99573227355399),
        (1.0, 0.999, 6.907755278982136),
        (1.0, 0.999999, 13.815510557935518),
        (1.001, 1e-14, 1.0331647572458654e-14),
        (1.001, 1e-12, 1.0284225276982838e-12),
        (1.001, 1e-09, 1.0213499654986867e-09),
        (1.001, 1e-06, 1.0143265551373011e-06),
        (1.001, 0.001, 0.0010078578857110824),
        (1.001, 0.05, 0.05147270543894933),
        (1.001, 0.2, 0.2236385049030028),
        (1.001, 0.4999, 0.6939151297415423),
        (1.001, 0.5, 0.6941152611609122),
        (1.001, 0.5001, 0.6943154325831584),
        (1.001, 0.75, 1.3876727532509323),
        (1.001, 0.95, 2.9976687755190228),
        (1.001, 0.999, 6.910392816904597),
        (1.001, 0.999999, 13.818780739521433),
        (1.05, 1e-14, 4.7395831884635004e-14),
        (1.05, 1e-12, 3.80629158755277e-12),
        (1.05, 1e-09, 2.739333526481915e-09),
        (1.05, 1e-06, 1.9714609858760984e-06),
        (1.05, 0.001, 0.001419813761682098),
        (1.05, 0.05, 0.06064322752465151),
        (1.05, 0.2, 0.24827971266935023),
        (1.05, 0.4999, 0.7414269940401486),
        (1.05, 0.5, 0.7416344252481228),
        (1.05, 0.5001, 0.7418418965906969),
        (1.05, 0.75, 1.4549604756872),
        (1.05, 0.95, 3.091865715923263),
        (1.05, 0.999, 7.03849522587578),
        (1.05, 0.999999, 13.977593070295415),
        (1.5, 1e-14, 5.611652891486994e-10),
        (1.5, 1e-12, 1.2089939713590179e-08),
        (1.5, 1e-09, 1.2089945501792994e-06),
        (1.5, 1e-06, 0.00012090524360062141),
        (1.5, 0.001, 0.012148792907846366),
        (1.5, 0.05, 0.1759231588746357),
        (1.5, 0.2, 0.5025870065261747),
        (1.5, 0.4999, 1.1827210005571056),
        (1.5, 0.5, 1.182986942187669),
        (1.5, 0.5001, 1.1832529246569214),
        (1.5, 0.75, 2.0541724678161586),
        (1.5, 0.95, 3.907363951625589),
        (1.5, 0.999, 8.133118098119064),
        (1.5, 0.999999, 15.332424853077134),
        (2.5, 1e-14, 4.060981277665631e-06),
        (2.5, 1e-12, 2.562321748415268e-05),
        (2.5, 1e-09, 0.00040614478283282704),
        (2.5, 1e-06, 0.006448080103248548),
        (2.5, 0.001, 0.10510630131460959),
        (2.5, 0.05, 0.5727381130308846),
        (2.5, 0.2, 1.1712671529205605),
        (2.5, 0.4999, 2.1753652478224934),
        (2.5, 0.5, 2.1757300955477636),
        (2.5, 0.5001, 2.176094984619723),
        (2.5, 0.75, 3.312839881914625),
        (2.5, 0.95, 5.535248846758176),
        (2.5, 0.999, 10.257502826216438),
        (2.5, 0.999999, 17.94409343980521),
        (5.0, 1e-14, 0.004131762040943242),
        (5.0, 1e-12, 0.0103893448525018),
        (5.0, 1e-09, 0.04157613724276548),
        (5.0, 1e-06, 0.16906300162147725),
        (5.0, 0.001, 0.7393717319178326),
        (5.0, 0.05, 1.97014956805953),
        (5.0, 0.2, 3.089539628019696),
        (5.0, 0.4999, 4.670370440927089),
        (5.0, 0.5, 4.670908882795984),
        (5.0, 0.5001, 4.671447366310887),
        (5.0, 0.75, 6.2744306984446885),
        (5.0, 0.95, 9.153519026637571),
        (5.0, 0.999, 14.794149222537209),
        (5.0, 0.999999, 23.43152342335784),
        (12.0, 1e-14, 0.37073817143471394),
        (12.0, 1e-12, 0.5517805131624026),
        (12.0, 1e-09, 1.0167891727378786),
        (12.0, 1e-06, 1.9399860087734906),
        (12.0, 0.001, 4.042440790424585),
        (12.0, 0.05, 6.924212513585107),
        (12.0, 0.2, 9.030902161693746),
        (12.0, 0.4999, 11.667508988541222),
        (12.0, 0.5, 11.668363153044766),
        (12.0, 0.5001, 11.669217359341626),
        (12.0, 0.75, 14.12057501276438),
        (12.0, 0.95, 18.207514250903653),
        (12.0, 0.999, 25.589298888688695),
        (12.0, 0.999999, 36.11442691003145),
        (50.0, 1e-14, 13.24755513025787),
        (50.0, 1e-12, 15.04208379308092),
        (50.0, 1e-09, 18.45464896859099),
        (50.0, 1e-06, 23.25066535794659),
        (50.0, 0.001, 30.95896960346831),
        (50.0, 0.05, 38.96473258250863),
        (50.0, 0.2, 43.97266796137551),
        (50.0, 0.4999, 49.66529908580119),
        (50.0, 0.5, 49.66706461799423),
        (50.0, 0.5001, 49.66883019205322),
        (50.0, 0.75, 54.5706205350403),
        (50.0, 0.95, 62.17105670200204),
        (50.0, 0.999, 74.72462638951936),
        (50.0, 0.999999, 91.06338855971308),
        (200.0, 1e-14, 110.13799652307603),
        (200.0, 1e-12, 116.04226006046942),
        (200.0, 1e-09, 126.4682680237736),
        (200.0, 1e-06, 139.8184115377329),
        (200.0, 0.001, 159.12980117448907),
        (200.0, 0.05, 177.32048703304125),
        (200.0, 0.2, 188.01091541157214),
        (200.0, 0.4999, 199.66322417309212),
        (200.0, 0.5, 199.66676561246567),
        (200.0, 0.5001, 199.6703070937217),
        (200.0, 0.75, 209.3484400855892),
        (200.0, 0.95, 223.8162339154042),
        (200.0, 0.999, 246.56587936997389),
        (200.0, 0.999999, 274.55761900238383),
        (1000.0, 1e-14, 776.8835024098369),
        (1000.0, 1e-12, 793.4392266170769),
        (1000.0, 1e-09, 821.8327535831709),
        (1000.0, 1e-06, 856.8146512793919),
        (1000.0, 0.001, 905.1207909349766),
        (1000.0, 0.05, 948.5598493836511),
        (1000.0, 0.2, 973.293038491621),
        (1000.0, 0.4999, 999.6587613346126),
        (1000.0, 0.5, 999.6666864269652),
        (1000.0, 0.5001, 999.6746115612046),
        (1000.0, 0.75, 1021.1436879071974),
        (1000.0, 0.95, 1052.5771180823206),
        (1000.0, 0.999, 1100.5780982933145),
        (1000.0, 0.999999, 1157.577911008724),
        (10000.0, 1e-14, 9254.003719701559),
        (10000.0, 1e-12, 9312.628969854226),
        (10000.0, 1e-09, 9411.828411445864),
        (10000.0, 1e-06, 9531.835117189807),
        (10000.0, 0.001, 9693.824385823726),
        (10000.0, 0.05, 9836.085110855192),
        (10000.0, 0.2, 9915.742124149649),
        (10000.0, 0.4999, 9999.641602867385),
        (10000.0, 0.5, 9999.666668642047),
        (10000.0, 0.5001, 9999.691734458598),
        (10000.0, 0.75, 10067.266062388988),
        (10000.0, 0.95, 10165.051911966126),
        (10000.0, 0.999, 10311.875224539537),
        (10000.0, 0.999999, 10482.561164638257),
    ];

    #[test]
    fn the_regularized_lower_gamma_matches_a_high_precision_reference() {
        // The series and the continued fraction each converge to full double precision, so the
        // only honest bound is one at that precision -- a table at 1e-9 tolerates a corrupted
        // recurrence that still converges to the right neighbourhood.
        let mut worst: f64 = 0.0;
        for &(a, x, want) in CDF_TABLE {
            let got = reg_lower_gamma(a, x);
            let rel = (got - want).abs() / want.abs();
            worst = worst.max(rel);
            assert!(
                rel < 5e-13,
                "P({a}, {x}) = {got:e} vs {want:e}, rel {rel:e}"
            );
        }
        // Measured worst on the unmutated code is 1.1e-13, so the bound above is four times it.
        assert!(
            worst > 1e-17,
            "the table is exact everywhere, which cannot be right"
        );
    }

    #[test]
    fn the_gamma_quantile_matches_a_high_precision_reference() {
        // Halley refinement is self-correcting, so a corrupted initial guess still lands near the
        // answer -- but not to the last few bits within the twelve steps it is allowed. Comparing
        // against mpmath at full double precision is what separates "converged" from "close".
        let mut worst: f64 = 0.0;
        for &(a, p, want) in QUANTILE_TABLE {
            let got = inv_reg_lower_gamma(a, p);
            let rel = (got - want).abs() / want.abs();
            worst = worst.max(rel);
            // Measured worst on the unmutated code is 4.6e-12, so the bound is four times it.
            assert!(
                rel < 2e-11,
                "P^-1({a}, {p}) = {got:e} vs {want:e}, rel {rel:e}"
            );
        }
        assert!(
            worst > 1e-17,
            "the table is exact everywhere, which cannot be right"
        );
    }

    #[test]
    fn the_quantile_inverts_the_cdf_to_the_last_few_bits() {
        // The table above pins the answer; this pins the property, and it holds where the table
        // cannot: near `p = 1` the reference comparison measures the cancellation in `1 - Q`,
        // while the round trip stays at the noise floor because it is relative to `p`, not `1-p`.
        let mut worst: f64 = 0.0;
        for &a in &[
            0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.001, 1.05, 1.5, 2.5, 5.0, 12.0, 50.0, 200.0,
            1000.0, 10000.0,
        ] {
            for &p in &[
                1e-14,
                1e-12,
                1e-9,
                1e-6,
                1e-3,
                0.01,
                0.05,
                0.2,
                0.4999,
                0.5,
                0.5001,
                0.75,
                0.95,
                0.999,
                0.999999,
                0.999999999,
            ] {
                let x = inv_reg_lower_gamma(a, p);
                // `x ~ (p/t)^(1/a)` underflows for a small enough shape in a far enough tail --
                // at `a = 0.02, p = 1e-14` the answer is `1e-700`. Nothing is wrong there; the
                // answer simply does not exist in `f64`, so there is no round trip to check.
                if x == 0.0 {
                    assert!(
                        a < 0.05 && p < 1e-3,
                        "P^-1({a}, {p}) underflowed unexpectedly"
                    );
                    continue;
                }
                // Express the residual back in `x`: dividing the relative error in `p` by the
                // local condition number `kappa = d ln P / d ln x` says how many ULP of `x` the
                // answer is out by, which is what the quantile actually promises. Comparing `p`
                // directly instead would only measure how steep the density is at `x`.
                let dens = (-x + (a - 1.0) * x.ln() - ln_gamma(a)).exp();
                let kappa = x * dens / p;
                let rel = (reg_lower_gamma(a, x) - p).abs() / p / kappa;
                assert!(
                    rel < 1e-3,
                    "P^-1({a}, {p}) = {x:e} is off by {rel:e} of itself"
                );
                worst = worst.max(rel);
            }
        }
        assert!(
            worst > 1e-17,
            "the round trip is exact everywhere, which cannot be right"
        );
    }
}
