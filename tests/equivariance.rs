//! Equivariance audit: the symmetry group each Phase-3 head claims, enforced.
//!
//! A clustering head answers a question about the *data*, not about the coordinates it arrived in.
//! Which coordinate changes it is allowed to ignore is a property of its model, and it is a property
//! that is easy to lose by accident — a bandwidth hard-coded in data units, a covariance floor added
//! before whitening, an initialisation that reads coordinate 0. Each head below is run twice, on the
//! same points in two frames, and the partitions are compared up to relabelling.
//!
//! The groups, and why each is what it is:
//!
//! | head | translate | rotate | scale | swap axes |
//! |---|---|---|---|---|
//! | `kmeans`, `ward_hac`, `agglomerative`, `dyn_msc`, `hdbscan` | yes | yes | yes | yes |
//! | `gmm_full`, `mppca` | yes | yes | yes | yes |
//! | `gmm_diagonal` | yes | **no** | yes | yes |
//! | `spectral`, `scale_space` | yes | yes | yes | yes |
//! | `movmf`, `spherical_kmeans` | **no** | yes | yes | yes |
//!
//! `gmm_diagonal` is the interesting row: axis-aligned covariance is a statement *about the axes*, so
//! rotation is not a symmetry of the model and the head is not expected to be invariant under it.
//! Rotating a diagonal fit and expecting the same answer is the error, not the head's behaviour —
//! which is why that case is asserted to *differ* on a fixture built to make it differ, rather than
//! quietly omitted.
//!
//! The directional heads are the other asymmetry: they read a direction from the origin, so
//! translation moves the data relative to the thing they measure and cannot be a symmetry.
//!
//! The head is only half of the pair, though. A symmetry the *leaf summary* has already discarded
//! cannot be recovered by any head reading it, so the group of a (feature, head) pair is the
//! intersection of the two:
//!
//! | feature | keeps | rotate |
//! |---|---|---|
//! | `Spherical` | the trace of the scatter | yes — nothing anisotropic is left to rotate |
//! | `Diagonal` | its axis-aligned part | **no** — the off-diagonal scatter is gone |
//! | `Full`, `FdSketch` | the whole scatter (`FdSketch` up to sketch error) | yes |
//!
//! So `gmm_full`, a rotation-invariant model, has three different symmetry groups depending on what
//! it is handed — asserted directly in `the_leaf_summary_and_not_only_the_head_decides_the_symmetry_group`.
//!
//! Uniform scaling is the one every Euclidean head must survive and the one most easily lost, since
//! any absolute distance threshold breaks it. It is the reason `scale_space` derives its bandwidth
//! range from the data rather than taking a constant.

use betula_cluster::clustering::{
    Linkage, Selection, agglomerative, dyn_msc, gmm_diagonal, gmm_full, gmm_toeplitz,
    hdbscan_selected, kmeans, movmf, mppca, scale_space, spectral, spherical_kmeans, ward_hac,
};
use betula_cluster::feature::{ClusterFeature, Diagonal, Full, Spherical};

const K: usize = 4;
const SEED: u64 = 11;

/// Deterministic 2-D blobs, well separated so that no head is deciding a near-tie and a rounding
/// difference between two frames cannot flip a label.
fn points() -> Vec<Vec<f64>> {
    blobs(&[[0.0, 0.0], [14.0, 1.0], [1.0, 13.0], [15.0, 14.0]])
}

/// The same four groups with two of them superimposed: three real groups, still asked for four.
/// The control fixture — see `the_fixture_can_tell_two_frames_apart`.
fn merged_points() -> Vec<Vec<f64>> {
    blobs(&[[0.0, 0.0], [0.0, 0.0], [1.0, 13.0], [15.0, 14.0]])
}

fn blobs(centres: &[[f64; 2]]) -> Vec<Vec<f64>> {
    let mut pts = Vec::new();
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for ctr in centres {
        for _ in 0..40 {
            // xorshift64*, inline so the fixture owes nothing to a test helper.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let a =
                ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / (1u64 << 53) as f64;
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let b =
                ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / (1u64 << 53) as f64;
            pts.push(vec![ctr[0] + 1.4 * (a - 0.5), ctr[1] + 1.4 * (b - 0.5)]);
        }
    }
    pts
}

/// Points shifted off the origin, for the directional heads: a mixture of directions only exists if
/// the data is not centred on the point it is measured from.
fn directional_points() -> Vec<Vec<f64>> {
    let mut pts = points();
    for p in pts.iter_mut() {
        p[0] += 40.0;
        p[1] += 40.0;
    }
    pts
}

fn translate(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pts.iter()
        .map(|p| vec![p[0] - 137.5, p[1] + 62.25])
        .collect()
}

fn rotate(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (s, c) = (0.7f64).sin_cos();
    pts.iter()
        .map(|p| vec![c * p[0] - s * p[1], s * p[0] + c * p[1]])
        .collect()
}

fn scale(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pts.iter().map(|p| vec![p[0] * 37.5, p[1] * 37.5]).collect()
}

fn swap_axes(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pts.iter().map(|p| vec![p[1], p[0]]).collect()
}

/// One leaf per group of four consecutive points, so every head sees a genuine summary — second
/// moments included — rather than a point set wearing a feature's type.
fn leaves(pts: &[Vec<f64>]) -> Vec<Full<f64>> {
    leaves_as(pts)
}

fn leaves_as<C: ClusterFeature<f64>>(pts: &[Vec<f64>]) -> Vec<C> {
    pts.chunks(4)
        .map(|chunk| {
            let mut f = C::new(2);
            for p in chunk {
                f.push(p, 1.0);
            }
            f
        })
        .collect()
}

/// Two elongated, deliberately non-axis-aligned groups: the fixture that separates a model which
/// reads the off-diagonal scatter from one that does not.
fn elongated() -> Vec<Vec<f64>> {
    let mut pts = Vec::new();
    let mut t = 0.0f64;
    for centre in [[0.0, 0.0], [9.0, 9.0]] {
        for _ in 0..80 {
            t += 0.37;
            let along = 6.0 * (t.sin());
            let across = 0.25 * (t * 2.7).cos();
            pts.push(vec![centre[0] + along + across, centre[1] + along - across]);
        }
    }
    pts
}

/// Do two labellings induce the same partition? Labels are names, so only the equivalence relation
/// is comparable; `-1` is noise and matches only noise.
fn same_partition(a: &[i64], b: &[i64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut forward = std::collections::HashMap::new();
    let mut backward = std::collections::HashMap::new();
    for (&x, &y) in a.iter().zip(b) {
        if (x < 0) != (y < 0) {
            return false;
        }
        if *forward.entry(x).or_insert(y) != y || *backward.entry(y).or_insert(x) != x {
            return false;
        }
    }
    true
}

fn as_i64(labels: &[usize]) -> Vec<i64> {
    labels.iter().map(|&l| l as i64).collect()
}

/// Every head, run on one leaf set. Returns `(name, labels)` so a whole frame is one call.
fn all_heads(pts: &[Vec<f64>]) -> Vec<(&'static str, Vec<i64>)> {
    let lv = leaves(pts);
    vec![
        ("kmeans", as_i64(&kmeans(&lv, K, 100, 4, SEED).labels)),
        ("ward_hac", as_i64(&ward_hac(&lv, K).labels)),
        (
            "agglomerative-average",
            as_i64(&agglomerative(&lv, Linkage::Average, K).labels),
        ),
        (
            "agglomerative-weighted",
            as_i64(&agglomerative(&lv, Linkage::Weighted, K).labels),
        ),
        ("gmm_full", as_i64(&gmm_full(&lv, K, 100, SEED).labels)),
        ("mppca", as_i64(&mppca(&lv, K, 1, 100, SEED).labels)),
        ("dyn_msc", as_i64(&dyn_msc(&lv, 8, 100, SEED).labels)),
        ("spectral", as_i64(&spectral(&lv, K, 100, SEED).labels)),
        ("scale_space", as_i64(&scale_space(&lv, 15, 100).labels)),
        (
            "hdbscan",
            hdbscan_selected(
                &lv,
                8,
                Selection::ExcessOfMass {
                    min_cluster_size: 8,
                },
                0,
                SEED,
            )
            .labels,
        ),
        (
            "gmm_diagonal",
            as_i64(&gmm_diagonal(&lv, K, 100, SEED).labels),
        ),
    ]
}

fn check(frame: &str, transform: fn(&[Vec<f64>]) -> Vec<Vec<f64>>, skip: &[&str]) {
    let base = all_heads(&points());
    let moved = all_heads(&transform(&points()));
    for ((name, a), (other, b)) in base.iter().zip(&moved) {
        assert_eq!(name, other, "head order must line up");
        if skip.contains(name) {
            continue;
        }
        assert!(
            same_partition(a, b),
            "{name} is not invariant under {frame}:\n  {a:?}\n  {b:?}"
        );
    }
}

#[test]
fn every_euclidean_head_is_invariant_under_translation() {
    check("translation", translate, &[]);
}

#[test]
fn every_euclidean_head_but_the_diagonal_gmm_is_invariant_under_rotation() {
    check("rotation", rotate, &["gmm_diagonal"]);
}

#[test]
fn every_euclidean_head_is_invariant_under_uniform_scaling() {
    check("uniform scaling", scale, &[]);
}

#[test]
fn every_euclidean_head_is_invariant_under_swapping_the_axes() {
    check("an axis swap", swap_axes, &[]);
}

#[test]
fn the_directional_heads_are_invariant_under_rotation_and_not_under_translation() {
    let pts = directional_points();
    let base = leaves(&pts);
    let turned = leaves(&rotate(&pts));
    let shifted = leaves(&translate(&pts));

    for (name, run) in [
        (
            "spherical_kmeans",
            (|f: &[Full<f64>]| spherical_kmeans(f, K, 100, 4, SEED).labels)
                as fn(&[Full<f64>]) -> _,
        ),
        ("movmf", |f: &[Full<f64>]| movmf(f, K, 100, SEED).labels),
    ] {
        let a = as_i64(&run(&base));
        assert!(
            same_partition(&a, &as_i64(&run(&turned))),
            "{name} must be rotation invariant"
        );
        assert!(
            !same_partition(&a, &as_i64(&run(&shifted))),
            "{name} came out translation invariant, which a direction read from the origin \
             cannot be — the fixture has stopped testing anything"
        );
    }
}

#[test]
fn the_diagonal_gmm_is_not_rotation_invariant_and_the_fixture_can_see_it() {
    // Elongated, non-axis-aligned clusters: a diagonal covariance cannot describe them in both
    // frames, so the two fits genuinely disagree. Asserting the *absence* of a symmetry keeps the
    // exemption above honest — without this, `gmm_diagonal` could become rotation invariant by
    // accident and the skip list would hide it.
    let pts = elongated();
    let a = as_i64(&gmm_diagonal(&leaves(&pts), 2, 200, SEED).labels);
    let b = as_i64(&gmm_diagonal(&leaves(&rotate(&pts)), 2, 200, SEED).labels);
    assert!(
        !same_partition(&a, &b),
        "a diagonal covariance is a claim about the axes; if rotating the data changes nothing, \
         the fixture is axis-aligned and proves nothing"
    );
}

#[test]
fn the_fixture_can_tell_two_frames_apart() {
    // The control the four invariance tests above rest on. Blobs separated far enough that every
    // head finds them in any frame would pass those tests without exercising a single symmetry —
    // so the same fixture, under a transformation that is *not* a symmetry, has to come out
    // different. No anisotropic scaling can do it — the blobs are separated along each axis
    // independently, and scaling one axis divides that axis' separation and its spread alike — so
    // the control moves the data instead of the frame: superimpose two of the four groups and the
    // right answer has genuinely changed.
    let base = all_heads(&points());
    let merged = all_heads(&merged_points());
    let differing = base
        .iter()
        .zip(&merged)
        .filter(|((_, a), (_, b))| !same_partition(a, b))
        .count();
    assert!(
        differing >= base.len() / 2,
        "only {differing} of {} heads noticed two of the four groups being superimposed — the \
         harness is not reading the labels it compares, so the invariance tests mean nothing",
        base.len()
    );
}

#[test]
fn the_leaf_summary_and_not_only_the_head_decides_the_symmetry_group() {
    // The head is only half of the pair. A model cannot be invariant under a transformation its
    // leaves have already thrown away: `Full` keeps the whole scatter, `Diagonal` keeps only the
    // axis-aligned part of it, `Spherical` keeps only its trace. So one rotation-invariant head,
    // `gmm_full`, has three different symmetry groups depending on what it is handed — it loses
    // rotation invariance on `Diagonal` leaves, and gets it back on `Spherical` ones, which have
    // nothing left to rotate.
    let pts = elongated();
    let turned = rotate(&pts);

    let full = |p: &[Vec<f64>]| as_i64(&gmm_full(&leaves_as::<Full<f64>>(p), 2, 200, SEED).labels);
    assert!(
        same_partition(&full(&pts), &full(&turned)),
        "gmm_full on Full leaves keeps the whole scatter and must be rotation invariant"
    );

    let diag =
        |p: &[Vec<f64>]| as_i64(&gmm_full(&leaves_as::<Diagonal<f64>>(p), 2, 200, SEED).labels);
    assert!(
        !same_partition(&diag(&pts), &diag(&turned)),
        "gmm_full on Diagonal leaves reads diag(variance) — the off-diagonal scatter these \
         clusters live along is gone, so rotation invariance cannot survive the summary"
    );

    let sph =
        |p: &[Vec<f64>]| as_i64(&gmm_full(&leaves_as::<Spherical<f64>>(p), 2, 200, SEED).labels);
    assert!(
        same_partition(&sph(&pts), &sph(&turned)),
        "a Spherical leaf is isotropic, so there is nothing in it a rotation can change"
    );
}

/// Two AR(1) signals in four dimensions, one smooth and one alternating, offset from each other so
/// the head has a mean to split on. The offset is not incidental: with the two components sharing a
/// mean and differing only in lag structure, `gmm_toeplitz` collapses to a single cluster at every
/// leaf size tried — mean-seeded EM has no gradient to separate two co-located components. That is a
/// limitation of the head, not of this fixture, and it is why the assertion below is about the fit
/// rather than about the partition.
fn ar_signals() -> Vec<Vec<f64>> {
    let mut pts = Vec::new();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64) / (1u64 << 53) as f64 - 0.5
    };
    for (rho, shift) in [(0.92f64, 0.0f64), (-0.92, 20.0)] {
        for _ in 0..80 {
            let mut x = vec![noise()];
            for t in 1..4 {
                let prev = x[t - 1];
                x.push(rho * prev + 0.25 * noise());
            }
            pts.push(x.iter().map(|v| v + shift).collect());
        }
    }
    pts
}

/// `(0,1,2,3) -> (0,2,1,3)`: a permutation of the coordinates, which for `d >= 3` is not an
/// automorphism of a Toeplitz matrix. In two dimensions it would be — `[[s, r], [r, s]]` is
/// symmetric under the only non-trivial swap — which is why this fixture is four-dimensional.
fn permute_lags(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pts.iter().map(|p| vec![p[0], p[2], p[1], p[3]]).collect()
}

#[test]
fn the_toeplitz_fit_reads_the_coordinate_order_even_where_the_partition_does_not() {
    // The stationary head is the one row of the table that is neither Euclidean nor directional. An
    // AR(w) covariance says the coordinates are an evenly spaced *sequence*, so translation leaves
    // the model alone while any permutation of the axes is a different model — and for `d >= 3` a
    // permutation really is non-trivial, where in two dimensions `[[s, r], [r, s]]` is symmetric
    // under the only swap there is.
    //
    // The partition, though, is not where that shows. Assignment here is driven by the means, and
    // means are permutation equivariant, so the labels come out identical; it is the fitted
    // likelihood that moves. Asserting the labels differ would be asserting something false.
    let pts = ar_signals();
    let fit = |p: &[Vec<f64>]| {
        let g = gmm_toeplitz(&leaves_as::<Full<f64>>(p), 2, 200, SEED);
        (as_i64(&g.labels), g.loglik)
    };
    let (base, ll) = fit(&pts);

    let (moved, ll_moved) = fit(&translate4(&pts));
    assert!(
        same_partition(&base, &moved),
        "an AR covariance does not see the origin, so the head must be translation invariant"
    );
    assert!(
        (ll - ll_moved).abs() < 1e-6 * ll.abs().max(1.0),
        "translation is a measure-preserving change of variable; the log-likelihood must not move \
         ({ll} against {ll_moved})"
    );

    assert!(
        same_partition(&base, &fit(&scale4(&pts)).0),
        "uniform scaling multiplies every Toeplitz band alike and must not move the partition"
    );

    let (reordered, ll_reordered) = fit(&permute_lags(&pts));
    assert!(
        same_partition(&base, &reordered),
        "the partition is mean-driven here, and means are permutation equivariant"
    );
    assert!(
        (ll - ll_reordered).abs() > 1e-3 * ll.abs(),
        "reordering the coordinates reorders the lags; a fit that does not move ({ll} against \
         {ll_reordered}) is not reading the band structure it claims to model"
    );
}

fn translate4(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pts.iter()
        .map(|p| p.iter().map(|v| v + 31.75).collect())
        .collect()
}

fn scale4(pts: &[Vec<f64>]) -> Vec<Vec<f64>> {
    pts.iter()
        .map(|p| p.iter().map(|v| v * 37.5).collect())
        .collect()
}
