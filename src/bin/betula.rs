//! `betula` — a minimal command-line interface for clustering a numeric matrix (behind the `cli`
//! feature). Reads delimited numeric rows from a file or stdin, clusters them with a parametric
//! Phase-3 head, and writes one integer label per row to stdout. Dependency-free (std only).

use betula_cluster::distance::CentroidEuclidean;
use betula_cluster::feature::{ClusterFeature, Diagonal, Full, Spherical};
use betula_cluster::model::{Method, Model};
use betula_cluster::tree::CFTree;
use std::io::{self, Read, Write};
use std::process::ExitCode;

const USAGE: &str = "\
betula — cluster a numeric matrix (CSV / delimited text)

USAGE:
    betula [OPTIONS] [INPUT]

INPUT:
    Path to a delimited numeric file; omit (or use '-') to read from stdin.

OPTIONS:
    -k, --clusters N      number of clusters (0 = auto via BIC / dendrogram cut) [default: 8]
        --method NAME     kmeans | gmm | gmm-full | ward [default: gmm]
        --feature NAME    spherical | diagonal | full [default: diagonal]
        --threshold F     CF-tree absorption threshold (squared distance) [default: 0.0]
        --branching N     max children per internal node [default: 32]
        --leaf-cap N      max entries per leaf [default: 32]
        --max-leaves N    leaf bound that triggers a rebuild [default: 2000]
        --max-iter N      max Lloyd / EM iterations [default: 100]
        --seed N          RNG seed [default: 0]
        --delimiter C     field delimiter character [default: ,]
        --header          skip the first (header) line
    -h, --help            print this help
";

struct Cfg {
    input: Option<String>,
    clusters: usize,
    method: Method,
    feature: String,
    threshold: f64,
    branching: usize,
    leaf_cap: usize,
    max_leaves: usize,
    max_iter: usize,
    seed: u64,
    delimiter: u8,
    header: bool,
}

impl Default for Cfg {
    fn default() -> Self {
        Self {
            input: None,
            clusters: 8,
            method: Method::Gmm,
            feature: "diagonal".to_string(),
            threshold: 0.0,
            branching: 32,
            leaf_cap: 32,
            max_leaves: 2000,
            max_iter: 100,
            seed: 0,
            delimiter: b',',
            header: false,
        }
    }
}

fn parse_method(s: &str) -> Result<Method, String> {
    match s {
        "kmeans" => Ok(Method::KMeans),
        "gmm" => Ok(Method::Gmm),
        "gmm-full" => Ok(Method::GmmFull),
        "ward" => Ok(Method::Ward),
        _ => Err(format!(
            "unknown method '{s}' (kmeans | gmm | gmm-full | ward)"
        )),
    }
}

fn validate_feature(s: &str) -> Result<(), String> {
    match s {
        "spherical" | "diagonal" | "full" => Ok(()),
        _ => Err(format!(
            "unknown feature '{s}' (spherical | diagonal | full)"
        )),
    }
}

/// Outcome of argument parsing: a config to run, or a request to print help.
enum Parsed {
    Run(Cfg),
    Help,
}

fn parse_args<I: Iterator<Item = String>>(args: I) -> Result<Parsed, String> {
    let mut cfg = Cfg::default();
    let mut args = args.peekable();
    let need = |a: &mut std::iter::Peekable<I>, flag: &str| -> Result<String, String> {
        a.next().ok_or_else(|| format!("{flag} requires a value"))
    };
    let int = |v: &str, flag: &str| {
        v.parse()
            .map_err(|_| format!("{flag} expects an integer, got '{v}'"))
    };
    let float = |v: &str, flag: &str| {
        v.parse()
            .map_err(|_| format!("{flag} expects a number, got '{v}'"))
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok(Parsed::Help),
            "-k" | "--clusters" => cfg.clusters = int(&need(&mut args, &arg)?, "--clusters")?,
            "--method" => cfg.method = parse_method(&need(&mut args, &arg)?)?,
            "--feature" => {
                let f = need(&mut args, &arg)?;
                validate_feature(&f)?;
                cfg.feature = f;
            }
            "--threshold" => cfg.threshold = float(&need(&mut args, &arg)?, "--threshold")?,
            "--branching" => cfg.branching = int(&need(&mut args, &arg)?, "--branching")?,
            "--leaf-cap" => cfg.leaf_cap = int(&need(&mut args, &arg)?, "--leaf-cap")?,
            "--max-leaves" => cfg.max_leaves = int(&need(&mut args, &arg)?, "--max-leaves")?,
            "--max-iter" => cfg.max_iter = int(&need(&mut args, &arg)?, "--max-iter")?,
            "--seed" => {
                let v = need(&mut args, &arg)?;
                cfg.seed = v
                    .parse()
                    .map_err(|_| format!("--seed expects an integer, got '{v}'"))?;
            }
            "--delimiter" => {
                let v = need(&mut args, &arg)?;
                let bytes = v.as_bytes();
                if bytes.len() != 1 {
                    return Err(format!("--delimiter expects a single character, got '{v}'"));
                }
                cfg.delimiter = bytes[0];
            }
            "--header" => cfg.header = true,
            "-" => cfg.input = None,
            s if s.starts_with('-') && s != "-" => return Err(format!("unknown option '{s}'")),
            _ => {
                if cfg.input.is_some() {
                    return Err(format!("unexpected extra argument '{arg}'"));
                }
                cfg.input = Some(arg);
            }
        }
    }
    Ok(Parsed::Run(cfg))
}

/// Parse delimited numeric rows; every row must have the same width and only finite values.
fn parse_rows(text: &str, delimiter: u8, header: bool) -> Result<Vec<Vec<f64>>, String> {
    let delim = delimiter as char;
    let mut rows = Vec::new();
    let mut width: Option<usize> = None;
    for (i, line) in text.lines().enumerate() {
        if header && i == 0 {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let row: Vec<f64> = line
            .split(delim)
            .map(|f| {
                let f = f.trim();
                f.parse::<f64>()
                    .map_err(|_| format!("line {}: cannot parse '{f}' as a number", i + 1))
                    .and_then(|v| {
                        if v.is_finite() {
                            Ok(v)
                        } else {
                            Err(format!("line {}: non-finite value '{f}'", i + 1))
                        }
                    })
            })
            .collect::<Result<_, _>>()?;
        match width {
            Some(w) if w != row.len() => {
                return Err(format!(
                    "line {}: expected {w} columns, found {}",
                    i + 1,
                    row.len()
                ));
            }
            None => width = Some(row.len()),
            _ => {}
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("no data rows found".to_string());
    }
    Ok(rows)
}

/// Build a tree over `rows`, cluster its leaves, and return one label per row.
fn run<C: ClusterFeature<f64>>(rows: &[Vec<f64>], cfg: &Cfg) -> Vec<usize> {
    let dim = rows[0].len();
    let mut tree: CFTree<f64, C, _, _> = CFTree::new(
        dim,
        cfg.branching,
        cfg.leaf_cap,
        cfg.threshold,
        cfg.max_leaves,
        CentroidEuclidean,
        CentroidEuclidean,
    );
    for r in rows {
        tree.insert(r);
    }
    let model = Model::fit(tree, cfg.clusters, cfg.method, cfg.max_iter, cfg.seed);
    rows.iter().map(|r| model.predict(r)).collect()
}

fn cluster(rows: &[Vec<f64>], cfg: &Cfg) -> Vec<usize> {
    match cfg.feature.as_str() {
        "spherical" => run::<Spherical<f64>>(rows, cfg),
        "full" => run::<Full<f64>>(rows, cfg),
        _ => run::<Diagonal<f64>>(rows, cfg),
    }
}

fn read_input(input: &Option<String>) -> Result<String, String> {
    match input {
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("cannot read '{p}': {e}")),
        None => {
            let mut s = String::new();
            io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            Ok(s)
        }
    }
}

fn write_labels(labels: &[usize]) -> Result<(), String> {
    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());
    for l in labels {
        writeln!(w, "{l}").map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn try_main() -> Result<(), String> {
    let cfg = match parse_args(std::env::args().skip(1))? {
        Parsed::Help => {
            print!("{USAGE}");
            return Ok(());
        }
        Parsed::Run(cfg) => cfg,
    };
    let text = read_input(&cfg.input)?;
    let rows = parse_rows(&text, cfg.delimiter, cfg.header)?;
    write_labels(&cluster(&rows, &cfg))
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("betula: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rows_reads_matrix() {
        let r = parse_rows("1,2,3\n4,5,6\n", b',', false).unwrap();
        assert_eq!(r, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn parse_rows_skips_header_and_blank_lines() {
        let r = parse_rows("a,b\n1,2\n\n3,4\n", b',', true).unwrap();
        assert_eq!(r, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn parse_rows_honours_delimiter() {
        let r = parse_rows("1\t2\n3\t4\n", b'\t', false).unwrap();
        assert_eq!(r, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    }

    #[test]
    fn parse_rows_rejects_ragged_rows() {
        assert!(parse_rows("1,2,3\n4,5\n", b',', false).is_err());
    }

    #[test]
    fn parse_rows_rejects_non_numeric_and_non_finite() {
        assert!(parse_rows("1,x\n", b',', false).is_err());
        assert!(parse_rows("1,inf\n", b',', false).is_err());
        assert!(parse_rows("", b',', false).is_err()); // no rows
    }

    #[test]
    fn parse_args_defaults_and_overrides() {
        let a = [
            "--clusters",
            "3",
            "--method",
            "kmeans",
            "--feature",
            "full",
            "data.csv",
        ];
        let Parsed::Run(cfg) = parse_args(a.iter().map(|s| s.to_string())).unwrap() else {
            panic!("expected Run");
        };
        assert_eq!(cfg.clusters, 3);
        assert!(matches!(cfg.method, Method::KMeans));
        assert_eq!(cfg.feature, "full");
        assert_eq!(cfg.input.as_deref(), Some("data.csv"));
    }

    #[test]
    fn parse_args_help_and_errors() {
        assert!(matches!(
            parse_args(["--help"].iter().map(|s| s.to_string())).unwrap(),
            Parsed::Help
        ));
        assert!(parse_args(["--method", "bogus"].iter().map(|s| s.to_string())).is_err());
        assert!(parse_args(["--feature", "bogus"].iter().map(|s| s.to_string())).is_err());
        assert!(parse_args(["--clusters"].iter().map(|s| s.to_string())).is_err()); // missing value
        assert!(parse_args(["--nope"].iter().map(|s| s.to_string())).is_err());
        assert!(parse_args(["--delimiter", ",,"].iter().map(|s| s.to_string())).is_err());
    }

    #[test]
    fn cluster_separates_two_blobs() {
        let mut rows = Vec::new();
        for i in 0..40 {
            let j = (i % 7) as f64 * 0.05;
            rows.push(vec![j, j]); // tight blob near origin
        }
        for i in 0..40 {
            let j = (i % 7) as f64 * 0.05;
            rows.push(vec![10.0 + j, 10.0 + j]); // tight blob far away
        }
        let cfg = Cfg {
            clusters: 2,
            feature: "spherical".to_string(),
            method: Method::KMeans,
            threshold: 0.05,
            seed: 1,
            ..Cfg::default()
        };
        let labels = cluster(&rows, &cfg);
        assert_eq!(labels.len(), 80);
        assert_eq!(labels[0], labels[39]); // same blob → same label
        assert_eq!(labels[40], labels[79]);
        assert_ne!(labels[0], labels[40]); // different blobs → different labels
    }
}
