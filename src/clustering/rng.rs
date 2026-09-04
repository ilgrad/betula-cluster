//! Tiny dependency-free PRNG (SplitMix64) for deterministic, seedable initialisation.

/// SplitMix64 — fast, well-distributed, fully deterministic.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Standard normal via Box–Muller.
    pub fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stream_matches_the_reference_splitmix64_vector() {
        // Vigna's splitmix64.c, seed 0: the state is advanced *before* the mixing rounds, so the
        // first output already carries the golden-ratio increment. Pinning the published vector is
        // what separates this from any other pair of shift-multiply rounds.
        let mut rng = SplitMix64::new(0);
        for want in [
            0xe220_a839_7b1d_cdafu64,
            0x6e78_9e6a_a1b9_65f4,
            0x06c4_5d18_8009_454f,
            0xf88b_b8a8_724c_81ec,
        ] {
            assert_eq!(rng.next_u64(), want);
        }
    }

    #[test]
    fn a_seed_reproduces_its_own_stream_and_two_seeds_do_not_share_one() {
        let take = |seed: u64| {
            let mut r = SplitMix64::new(seed);
            (0..8).map(|_| r.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(take(7), take(7));
        assert_ne!(take(7), take(8));
    }

    #[test]
    fn next_f64_stays_in_the_unit_interval() {
        let mut rng = SplitMix64::new(12345);
        let mut lo = 1.0f64;
        let mut hi = 0.0f64;
        for _ in 0..10_000 {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u), "u = {u}");
            lo = lo.min(u);
            hi = hi.max(u);
        }
        assert!(lo < 0.01 && hi > 0.99, "range [{lo}, {hi}] is not filled");
    }
}
