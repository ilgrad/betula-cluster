//! One clustering run per process, for `bench/rust_rivals.py`: betula through its documented Rust
//! API against linfa-clustering, clump, the `hdbscan` crate, anofox-ml-cluster, scirs2-cluster and
//! fuel, each at its documented defaults.
//!
//! Usage: `rust-rivals <arm> <rows.csv> <k> <seed> <min_points> [<eps>]`; prints one label per row,
//! `-1` for noise (the `linfa-optics` arm prints its ordering instead: `index reachability core`
//! per line, `inf` where undefined), then on stderr the seconds the fit took (CSV parsing
//! excluded), the heap allocations the fit made, and the peak RSS. `rust-rivals check` runs the
//! documented examples `bench/rust_rivals.py --check` reads.

use std::alloc::{GlobalAlloc, Layout, System};
use std::error::Error;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use anofox_ml_core::FitUnsupervised;
use betula_cluster::distance::CentroidEuclidean;
use betula_cluster::feature::{ClusterFeature, Diagonal, Spherical};
use betula_cluster::model::{Method, Model};
use betula_cluster::tree::CFTree;
use linfa::traits::{Fit, Predict, Transformer};
use linfa::DatasetBase;
use ndarray::Array2;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

/// The system allocator, counting every call that hands out memory and the bytes asked for, so a
/// row can say how many allocations a fit made beside its peak RSS. Memory a C library takes with
/// its own `malloc` (OpenBLAS under fuel) is not seen.
struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

fn count(bytes: usize) {
    ALLOCS.fetch_add(1, Relaxed);
    ALLOC_BYTES.fetch_add(bytes as u64, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

type Labels = Vec<i64>;

/// `C` is the leaf feature the Python wrapper builds for the head: `Spherical` for k-means, and
/// `Diagonal`, its default, for `gmm` and `gmm-full`, which it refuses on a `Spherical` tree —
/// that feature's one scalar of scatter is added to every dimension of every component.
fn betula<C: ClusterFeature<f64>>(
    rows: &[Vec<f64>],
    k: usize,
    seed: u64,
    method: Method,
) -> Result<Labels, Box<dyn Error>> {
    // The construction of `docs/USAGE.md`'s Rust section: branching and leaf capacity 32, a
    // threshold of 0 and a 2 000-leaf budget; `0` takes the default restart count and `auto_k_max`.
    let mut tree: CFTree<f64, C, _, _> = CFTree::new(
        rows[0].len(),
        32,
        32,
        0.0,
        2000,
        CentroidEuclidean,
        CentroidEuclidean,
    );
    for p in rows {
        tree.try_insert(p)?;
    }
    let model = Model::fit(tree, k, method, 100, 0, seed, 0);
    rows.iter()
        .map(|p| Ok(model.try_predict(p)? as i64))
        .collect()
}

fn records(rows: &[Vec<f64>]) -> Result<Array2<f64>, Box<dyn Error>> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Ok(Array2::from_shape_vec((rows.len(), rows[0].len()), flat)?)
}

fn linfa_kmeans(rows: &[Vec<f64>], k: usize, seed: u64) -> Result<Labels, Box<dyn Error>> {
    let data = DatasetBase::from(records(rows)?);
    let rng = Xoshiro256Plus::seed_from_u64(seed);
    let model = linfa_clustering::KMeans::params_with_rng(k, rng).fit(&data)?;
    Ok(model
        .predict(data.records())
        .iter()
        .map(|&l| l as i64)
        .collect())
}

fn linfa_gmm(rows: &[Vec<f64>], k: usize, seed: u64) -> Result<Labels, Box<dyn Error>> {
    let data = DatasetBase::from(records(rows)?);
    let rng = Xoshiro256Plus::seed_from_u64(seed);
    let model = linfa_clustering::GaussianMixtureModel::params(k)
        .with_rng(rng)
        .fit(&data)?;
    Ok(model
        .predict(data.records())
        .iter()
        .map(|&l| l as i64)
        .collect())
}

fn linfa_dbscan(rows: &[Vec<f64>], min_points: usize, eps: f64) -> Result<Labels, Box<dyn Error>> {
    let labels = linfa_clustering::Dbscan::params(min_points)
        .tolerance(eps)
        .transform(&records(rows)?)?;
    Ok(labels.iter().map(|l| l.map_or(-1, |c| c as i64)).collect())
}

/// linfa's OPTICS returns the augmented ordering, not labels: one `index reachability core` line
/// per row in visiting order, which the harness hands to scikit-learn's cluster extraction.
fn linfa_optics(rows: &[Vec<f64>], min_points: usize) -> Result<Vec<String>, Box<dyn Error>> {
    let data = records(rows)?;
    let analysis = linfa_clustering::Optics::params(min_points).transform(data.view())?;
    let show = |v: &Option<f64>| v.map_or("inf".to_string(), |d| d.to_string());
    Ok(analysis
        .iter()
        .map(|s| {
            format!(
                "{} {} {}",
                s.index(),
                show(s.reachability_distance()),
                show(s.core_distance())
            )
        })
        .collect())
}

fn as_f32(rows: &[Vec<f64>]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|r| r.iter().map(|&v| v as f32).collect())
        .collect()
}

fn clump_labels(labels: Vec<usize>) -> Labels {
    labels
        .into_iter()
        .map(|l| if l == clump::NOISE { -1 } else { l as i64 })
        .collect()
}

/// anofox's `Birch` at scikit-learn's defaults, which its documentation says it mirrors:
/// threshold 0.5, 50 branches, the count given.
fn anofox_birch(rows: &[Vec<f64>], k: usize) -> Result<Labels, Box<dyn Error>> {
    let birch = anofox_ml_cluster::Birch::new(0.5).with_n_clusters(Some(k));
    let fitted = FitUnsupervised::<f64>::fit_view(&birch, records(rows)?.view())?;
    Ok(fitted.labels.iter().map(|&l| l as i64).collect())
}

fn scirs2_data(rows: &[Vec<f64>]) -> Result<scirs2_core::ndarray::Array2<f64>, Box<dyn Error>> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Ok(scirs2_core::ndarray::Array2::from_shape_vec(
        (rows.len(), rows[0].len()),
        flat,
    )?)
}

/// scirs2's documented call, `birch(data, options)` with the count given and every other option
/// at its default (threshold 0.5, 50 branches).
fn scirs2_birch(rows: &[Vec<f64>], k: usize) -> Result<Labels, Box<dyn Error>> {
    let options = scirs2_cluster::birch::BirchOptions {
        n_clusters: Some(k),
        ..Default::default()
    };
    let (_, labels) = scirs2_cluster::birch::birch(scirs2_data(rows)?.view(), options)?;
    Ok(labels.iter().map(|&l| i64::from(l)).collect())
}

/// The same fit, with each row's leaf entry (what `birch` returns as the label) mapped to the
/// cluster `extract_clusters` gives that entry.
fn scirs2_birch_mapped(rows: &[Vec<f64>], k: usize) -> Result<Labels, Box<dyn Error>> {
    let data = scirs2_data(rows)?;
    let mut model = scirs2_cluster::birch::Birch::new(scirs2_cluster::birch::BirchOptions {
        n_clusters: Some(k),
        ..Default::default()
    });
    model.fit(data.view())?;
    let (_, entry_cluster) = model.extract_clusters()?;
    let entry = model.predict(data.view())?;
    Ok(entry
        .iter()
        .map(|&e| i64::from(entry_cluster[e as usize]))
        .collect())
}

type FuelTable<'a> = fuel::TableWithDistance<'a, f64, Vec<f64>, fuel::distance::Euclidean, f64>;

/// fuel's `kmeans` at the defaults of its Python wrapper: the simplified Elkan variant, a random
/// sample of rows as the initial centres, 300 iterations, run to convergence.
fn fuel_kmeans(rows: &[Vec<f64>], k: usize, seed: u64) -> Result<Labels, Box<dyn Error>> {
    use rand_pcg::rand_core::SeedableRng as _;

    let data: FuelTable = fuel::TableWithDistance::with_distance(rows, fuel::distance::Euclidean);
    let mut init =
        fuel::cluster::kmeans::init::RandomSample::new(rand_pcg::Pcg32::seed_from_u64(seed));
    let result = fuel::cluster::kmeans::simp_elkan(&data, k, &mut init, 300, 0.0)?;
    Ok(result.assignments.iter().map(|&l| l as i64).collect())
}

/// fuel's `hdbscan` at its wrapper's default variant (Prim's tree on the mutual reachability
/// distances), the flat clusters of ELKI's HDBSCAN extraction at the same minimum cluster size.
/// The first node is the root of the merge tree; the rows it keeps for itself are noise.
fn fuel_hdbscan(rows: &[Vec<f64>], min_points: usize) -> Result<Labels, Box<dyn Error>> {
    use fuel::cluster::hdbscan::extraction::extract_hdbscan_hierarchy_hdbscan;

    let data: FuelTable = fuel::TableWithDistance::with_distance(rows, fuel::distance::Euclidean);
    let hierarchy = fuel::cluster::hdbscan::hdbscan_prim(&data, min_points)?;
    let flat = extract_hdbscan_hierarchy_hdbscan(&hierarchy, min_points, false);
    let mut labels = vec![-1; rows.len()];
    for (id, node) in flat.hierarchy.nodes.iter().enumerate().skip(1) {
        for &p in &node.members {
            labels[p] = id as i64 - 1;
        }
    }
    Ok(labels)
}

/// The documented examples, one `name<TAB>published<TAB>reproduced` line each; the harness
/// compares the two.
fn check() -> Result<(), Box<dyn Error>> {
    // scikit-learn's Birch docstring, which anofox's Birch documents itself as mirroring.
    let x = vec![
        vec![0.0, 1.0],
        vec![0.3, 1.0],
        vec![-0.3, 1.0],
        vec![0.0, -1.0],
        vec![0.3, -1.0],
        vec![-0.3, -1.0],
    ];
    let birch = anofox_ml_cluster::Birch::new(0.5).with_n_clusters(None);
    let fitted = FitUnsupervised::<f64>::fit_view(&birch, records(&x)?.view())?;
    let got: Vec<i64> = fitted.labels.iter().map(|&l| l as i64).collect();
    println!("anofox-birch\t[0, 0, 0, 1, 1, 1]\t{got:?}");

    // scirs2's `birch` doc example; its unit test on the same rows asserts two centroids.
    let x = vec![
        vec![1.0, 2.0],
        vec![1.2, 1.8],
        vec![0.8, 1.9],
        vec![4.0, 5.0],
        vec![4.2, 4.8],
        vec![3.9, 5.1],
    ];
    let options = scirs2_cluster::birch::BirchOptions {
        n_clusters: Some(2),
        ..Default::default()
    };
    let (centroids, _) = scirs2_cluster::birch::birch(scirs2_data(&x)?.view(), options)?;
    println!("scirs2-birch\t2\t{}", centroids.nrows());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("check") {
        return check();
    }
    if !(6..=7).contains(&args.len()) {
        return Err(
            "usage: rust-rivals <arm> <rows.csv> <k> <seed> <min_points> [<eps>] | check".into(),
        );
    }
    let (arm, k, seed, mcs) = (
        args[1].as_str(),
        args[3].parse::<usize>()?,
        args[4].parse::<u64>()?,
        args[5].parse::<usize>()?,
    );
    let eps = || -> Result<f64, Box<dyn Error>> {
        Ok(args.get(6).ok_or("this arm needs <eps>")?.parse::<f64>()?)
    };
    let rows: Vec<Vec<f64>> = fs::read_to_string(&args[2])?
        .lines()
        .map(|line| line.split(',').map(str::parse).collect())
        .collect::<Result<_, _>>()?;

    let (allocs0, bytes0) = (ALLOCS.load(Relaxed), ALLOC_BYTES.load(Relaxed));
    let start = Instant::now();
    let out: Vec<String> = if arm == "linfa-optics" {
        linfa_optics(&rows, mcs)?
    } else {
        let labels = match arm {
            "betula-kmeans" => betula::<Spherical<f64>>(&rows, k, seed, Method::KMeans)?,
            "betula-gmm" => betula::<Diagonal<f64>>(&rows, k, seed, Method::Gmm)?,
            "betula-gmm-full" => betula::<Diagonal<f64>>(&rows, k, seed, Method::GmmFull)?,
            "linfa-kmeans" => linfa_kmeans(&rows, k, seed)?,
            "linfa-gmm" => linfa_gmm(&rows, k, seed)?,
            "linfa-dbscan" => linfa_dbscan(&rows, mcs, eps()?)?,
            "clump-kmeans" => clump_labels(
                clump::Kmeans::new(k)
                    .with_seed(seed)
                    .fit_predict(&as_f32(&rows))?,
            ),
            "clump-hdbscan" => clump_labels(
                clump::Hdbscan::new()
                    .with_min_cluster_size(mcs)
                    .fit_predict(&as_f32(&rows))?,
            ),
            "clump-dbscan" => {
                clump_labels(clump::Dbscan::new(eps()? as f32, mcs).fit_predict(&as_f32(&rows))?)
            }
            "hdbscan-crate" => {
                let params = hdbscan::HdbscanHyperParams::builder()
                    .min_cluster_size(mcs)
                    .build();
                hdbscan::Hdbscan::new(&rows, params)
                    .cluster()?
                    .into_iter()
                    .map(i64::from)
                    .collect()
            }
            "anofox-birch" => anofox_birch(&rows, k)?,
            "scirs2-birch" => scirs2_birch(&rows, k)?,
            "scirs2-birch-mapped" => scirs2_birch_mapped(&rows, k)?,
            "fuel-kmeans" => fuel_kmeans(&rows, k, seed)?,
            "fuel-hdbscan" => fuel_hdbscan(&rows, mcs)?,
            other => return Err(format!("unknown arm {other}").into()),
        };
        labels.iter().map(i64::to_string).collect()
    };
    let fit_s = start.elapsed().as_secs_f64();
    let (allocs, bytes) = (
        ALLOCS.load(Relaxed) - allocs0,
        ALLOC_BYTES.load(Relaxed) - bytes0,
    );
    println!("{}", out.join("\n"));
    // This process's own peak: `exec` gave it a fresh address space, whereas the `ru_maxrss` a
    // parent reads with `wait4` keeps the parent's high-water mark from before the fork.
    let status = fs::read_to_string("/proc/self/status")?;
    let hwm = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .ok_or("no VmHWM in /proc/self/status")?;
    eprintln!("fit_s {fit_s}");
    eprintln!("allocs {allocs}");
    eprintln!("alloc_bytes {bytes}");
    eprintln!("VmHWM {}", hwm.trim());
    Ok(())
}
