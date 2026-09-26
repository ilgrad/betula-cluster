//! A model file is input, and `CFTree::validate` is all that stands between its CBOR and every read
//! and write that indexes the arena. So the contract is checked by using an accepted tree the way a
//! loaded estimator does: route through it, read its leaves, and insert into it.
//!
//! Documents come from two places. Raw bytes let the fuzzer change a document's shape; a real
//! document with a few bytes overwritten reaches what a damaged file carries — an index, a length, a
//! float — without first rediscovering CBOR and every field name. The gzip frame the estimator's
//! `decode` peels off first is left out: `flate2` is fuzzed upstream, and a bare document is the
//! branch `decode` takes for every file written before 0.9.0.
#![no_main]

use std::ops::ControlFlow;

use betula_cluster::distance::CentroidEuclidean;
use betula_cluster::feature::Spherical;
use betula_cluster::tree::CFTree;
use libfuzzer_sys::arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;

type Tree = CFTree<f64, Spherical<f64>, CentroidEuclidean, CentroidEuclidean>;

fn damaged(u: &mut Unstructured) -> Result<Vec<u8>> {
    let dim = u.int_in_range(1..=3)?;
    let mut tree = Tree::new(
        dim,
        u.int_in_range(2..=4)?,
        u.int_in_range(1..=4)?,
        0.0,
        u.int_in_range(1..=8)?,
        CentroidEuclidean,
        CentroidEuclidean,
    );
    u.arbitrary_loop(None, Some(40), |u| {
        let row = (0..dim)
            .map(|_| Ok(f64::from(u.arbitrary::<i8>()?)))
            .collect::<Result<Vec<f64>>>()?;
        tree.try_insert(&row).unwrap();
        Ok(ControlFlow::Continue(()))
    })?;
    let mut doc = Vec::new();
    ciborium::into_writer(&tree, &mut doc).unwrap();
    u.arbitrary_loop(Some(1), Some(8), |u| {
        let at = u.choose_index(doc.len())?;
        doc[at] = u.arbitrary()?;
        Ok(ControlFlow::Continue(()))
    })?;
    Ok(doc)
}

fuzz_target!(|bytes: &[u8]| {
    let mut u = Unstructured::new(bytes);
    let doc = if u.arbitrary().unwrap_or(false) {
        match damaged(&mut u) {
            Ok(doc) => doc,
            Err(_) => return,
        }
    } else {
        u.take_rest().to_vec()
    };
    let Ok(mut tree) = ciborium::from_reader::<Tree, _>(doc.as_slice()) else {
        return;
    };
    if tree.validate().is_err() {
        return;
    }
    let dim = tree.dim();
    let probes: Vec<Vec<f64>> = (0..12)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * 7 + j * 3) % 11) as f64 - 5.0)
                .collect()
        })
        .collect();
    if tree.num_leaves() > 0 {
        for x in &probes {
            assert!(tree.nearest_entry(x) < tree.num_leaves());
            assert!(tree.nearest_entry_beam(x, 4) < tree.num_leaves());
        }
    }
    let _ = tree.top1_mass();
    for x in &probes {
        tree.try_insert(x).unwrap();
    }
    assert_eq!(
        tree.validate(),
        Ok(()),
        "an insert broke a tree validate accepted"
    );
    for x in &probes {
        assert!(tree.nearest_entry(x) < tree.num_leaves());
    }
});
