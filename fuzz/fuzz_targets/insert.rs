//! `CFTree::try_insert` on rows nothing has checked — NaN, ±inf, subnormals, values whose square
//! overflows — under every feature model, three absorption gates, both element types and each
//! optional knob, then `refit_leaves_beam` over the same rows. The Python boundary refuses a
//! non-finite row; the Rust API does not, so the tree has to survive one: stay a tree `validate`
//! accepts, keep every row's mass, and still answer a route.
#![no_main]

use std::ops::ControlFlow;

use betula_cluster::distance::{CFDistance, CentroidEuclidean, Radius, VarianceIncrease};
use betula_cluster::feature::{ClusterFeature, Diagonal, FdSketch, Full, Spherical};
use betula_cluster::tree::CFTree;
use betula_cluster::types::Real;
use libfuzzer_sys::arbitrary::{Arbitrary, Result, Unstructured};
use libfuzzer_sys::fuzz_target;

#[derive(Debug)]
struct Input {
    f32: bool,
    feature: u8,
    gate: u8,
    dim: usize,
    branching: usize,
    leaf_cap: usize,
    threshold: f64,
    max_leaves: usize,
    huber_k: Option<f64>,
    balance: Option<f64>,
    auto_balance: bool,
    beam: usize,
    /// Row-major, `dim` to a row. An `f32` tree's values are drawn as `f32`, so they cover its own
    /// subnormals and overflow rather than whatever survives narrowing an `f64`, and widen exactly.
    rows: Vec<f64>,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let f32 = u.arbitrary().unwrap_or(false);
        let feature = u.int_in_range(0..=3)?;
        let gate = u.int_in_range(0..=2)?;
        let dim = u.int_in_range(1..=4)?;
        let mut m = Input {
            f32,
            feature,
            gate,
            dim,
            branching: u.int_in_range(2..=5)?,
            leaf_cap: u.int_in_range(1..=5)?,
            threshold: u.arbitrary()?,
            max_leaves: u.int_in_range(1..=12)?,
            huber_k: if u.arbitrary()? {
                Some(u.arbitrary()?)
            } else {
                None
            },
            balance: if u.arbitrary()? {
                Some(u.arbitrary()?)
            } else {
                None
            },
            auto_balance: u.arbitrary()?,
            beam: u.int_in_range(0..=4)?,
            rows: Vec::new(),
        };
        // Running out of bytes mid-row ends the stream; the rows before it are the input.
        let _ = u.arbitrary_loop(None, Some(256), |u| {
            let row = (0..dim)
                .map(|_| match f32 {
                    true => u.arbitrary::<f32>().map(f64::from),
                    false => u.arbitrary::<f64>(),
                })
                .collect::<Result<Vec<f64>>>()?;
            m.rows.extend(row);
            Ok(ControlFlow::Continue(()))
        });
        Ok(m)
    }
}

/// A finite, non-negative knob: the domain `validate` accepts for the threshold, the Huber `k` and
/// the balance multiple, and the one inside which an insert is promised to terminate.
fn knob<R: Real>(v: f64) -> R {
    let v = R::from_f64(v).unwrap().abs();
    if v.is_finite() { v } else { R::zero() }
}

fn check<R: Real, C: ClusterFeature<R>, A: CFDistance<R, C>>(
    tree: &CFTree<R, C, CentroidEuclidean, A>,
    flat: &[R],
) {
    assert_eq!(tree.validate(), Ok(()));
    let mass: f64 = tree
        .leaf_features()
        .iter()
        .map(|cf| cf.weight().to_f64().unwrap())
        .sum();
    let n = flat.len() / tree.dim();
    assert_eq!(mass, n as f64, "every row is in exactly one leaf entry");
    if n == 0 {
        return;
    }
    for row in flat.chunks_exact(tree.dim()) {
        assert!(tree.nearest_entry(row) < tree.num_leaves());
        assert!(tree.nearest_entry_beam(row, 3) < tree.num_leaves());
    }
}

fn run<R: Real, C: ClusterFeature<R>, A: CFDistance<R, C>>(m: &Input, abs: A) {
    let mut tree: CFTree<R, C, CentroidEuclidean, A> = CFTree::new(
        m.dim,
        m.branching,
        m.leaf_cap,
        knob(m.threshold),
        m.max_leaves,
        CentroidEuclidean,
        abs,
    );
    tree.set_huber_k(m.huber_k.map(knob));
    tree.set_balance(m.balance.map(knob));
    tree.set_auto_balance(m.auto_balance);
    let flat: Vec<R> = m.rows.iter().map(|&v| R::from_f64(v).unwrap()).collect();
    for row in flat.chunks_exact(m.dim) {
        tree.try_insert(row).unwrap();
    }
    check(&tree, &flat);
    if m.beam > 0 {
        tree.refit_leaves_beam(&flat, flat.len() / m.dim, m.beam);
        check(&tree, &flat);
    }
}

fn gate<R: Real, C: ClusterFeature<R>>(m: &Input) {
    match m.gate {
        0 => run::<R, C, _>(m, CentroidEuclidean),
        1 => run::<R, C, _>(m, Radius),
        _ => run::<R, C, _>(m, VarianceIncrease),
    }
}

fn feature<R: Real>(m: &Input) {
    match m.feature {
        0 => gate::<R, Spherical<R>>(m),
        1 => gate::<R, Diagonal<R>>(m),
        2 => gate::<R, Full<R>>(m),
        _ => gate::<R, FdSketch<R>>(m),
    }
}

fuzz_target!(|m: Input| {
    if m.f32 {
        feature::<f32>(&m);
    } else {
        feature::<f64>(&m);
    }
});
