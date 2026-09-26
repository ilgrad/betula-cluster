//! `validate_csr` and `summarize_sparse` as the pair they are: whatever the validator accepts, the
//! summariser has to take without panicking, and the routes it returns have to agree with the
//! summary it built from them.
#![no_main]

use std::ops::ControlFlow;

use betula_cluster::feature::ClusterFeature;
use betula_cluster::sparse::{summarize_sparse, validate_csr};
use libfuzzer_sys::arbitrary::{Arbitrary, Result, Unstructured};
use libfuzzer_sys::fuzz_target;

#[derive(Debug)]
struct Input {
    /// A `u8`: the validator's own cap is `2^30`, and a centroid that wide is an 8 GB allocation
    /// by design, which is a documented cost and not something to rediscover here.
    n_features: usize,
    threshold: f64,
    max_leaders: usize,
    data: Vec<f64>,
    indices: Vec<i64>,
    indptr: Vec<i64>,
}

/// Either three arbitrary arrays, which is how a malformed `indptr` reaches the validator, or rows
/// appended one at a time, whose `indptr` is well formed by construction so that the run is spent
/// past the validator rather than on rediscovering a prefix sum.
impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        let mut m = Input {
            n_features: usize::from(u.arbitrary::<u8>()?),
            threshold: u.arbitrary()?,
            max_leaders: usize::from(u.arbitrary::<u8>()?),
            data: Vec::new(),
            indices: Vec::new(),
            indptr: vec![0],
        };
        if u.arbitrary()? {
            (m.data, m.indices, m.indptr) = u.arbitrary()?;
            return Ok(m);
        }
        u.arbitrary_loop(None, Some(64), |u| {
            for _ in 0..u.int_in_range(0..=8)? {
                m.indices.push(i64::from(u.arbitrary::<u8>()?));
                m.data.push(u.arbitrary()?);
            }
            m.indptr.push(m.data.len() as i64);
            Ok(ControlFlow::Continue(()))
        })?;
        Ok(m)
    }
}

fuzz_target!(|m: Input| {
    if validate_csr(&m.data, &m.indices, &m.indptr, m.n_features).is_err() {
        return;
    }
    let (micros, of_row) = summarize_sparse(
        &m.data,
        &m.indices,
        &m.indptr,
        m.n_features,
        m.threshold,
        m.max_leaders,
    );
    let n_rows = m.indptr.len() - 1;
    assert_eq!(of_row.len(), n_rows);
    assert_eq!(micros.is_empty(), n_rows == 0);
    assert!(micros.len() <= m.max_leaders.max(1));
    let mut members = vec![0usize; micros.len()];
    for &i in &of_row {
        members[i] += 1;
    }
    for (cf, &k) in micros.iter().zip(&members) {
        assert_eq!(cf.dim(), m.n_features);
        assert_eq!(
            cf.weight(),
            k as f64,
            "a micro-cluster weighs the rows routed to it"
        );
    }
});
