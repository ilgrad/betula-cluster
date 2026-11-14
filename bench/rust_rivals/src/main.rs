//! One clustering run per process, for `bench/rust_rivals.py`: betula through its documented Rust
//! API against linfa-clustering, clump and the `hdbscan` crate, each at its documented defaults.
//!
//! Usage: `rust-rivals <arm> <rows.csv> <k> <seed> <min_cluster_size>`; prints one label per row,
//! `-1` for noise.

use std::error::Error;
use std::fs;

use betula_cluster::distance::CentroidEuclidean;
use betula_cluster::feature::Spherical;
use betula_cluster::model::{Method, Model};
use betula_cluster::tree::CFTree;
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use ndarray::Array2;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

type Labels = Vec<i64>;

fn betula(
    rows: &[Vec<f64>],
    k: usize,
    seed: u64,
    method: Method,
) -> Result<Labels, Box<dyn Error>> {
    // The construction of `docs/USAGE.md`'s Rust section: branching and leaf capacity 32, a
    // threshold of 0 and a 2 000-leaf budget; `0` takes the default restart count and `auto_k_max`.
    let mut tree: CFTree<f64, Spherical<f64>, _, _> = CFTree::new(
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: rust-rivals <arm> <rows.csv> <k> <seed> <min_cluster_size>".into());
    }
    let (arm, k, seed, mcs) = (
        args[1].as_str(),
        args[3].parse::<usize>()?,
        args[4].parse::<u64>()?,
        args[5].parse::<usize>()?,
    );
    let rows: Vec<Vec<f64>> = fs::read_to_string(&args[2])?
        .lines()
        .map(|line| line.split(',').map(str::parse).collect())
        .collect::<Result<_, _>>()?;

    let labels = match arm {
        "betula-kmeans" => betula(&rows, k, seed, Method::KMeans)?,
        "betula-gmm" => betula(&rows, k, seed, Method::Gmm)?,
        "betula-gmm-full" => betula(&rows, k, seed, Method::GmmFull)?,
        "linfa-kmeans" => linfa_kmeans(&rows, k, seed)?,
        "linfa-gmm" => linfa_gmm(&rows, k, seed)?,
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
        other => return Err(format!("unknown arm {other}").into()),
    };
    let out: Vec<String> = labels.iter().map(i64::to_string).collect();
    println!("{}", out.join("\n"));
    // This process's own peak: `exec` gave it a fresh address space, whereas the `ru_maxrss` a
    // parent reads with `wait4` keeps the parent's high-water mark from before the fork.
    let status = fs::read_to_string("/proc/self/status")?;
    let hwm = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .ok_or("no VmHWM in /proc/self/status")?;
    eprintln!("VmHWM {}", hwm.trim());
    Ok(())
}
