---
title: 'betula-cluster: Memory-bounded clustering of large data with numerically stable BETULA CF-trees'
tags:
  - Python
  - Rust
  - clustering
  - unsupervised learning
  - streaming
  - machine learning
authors:
  - name: Ilia Gradina
    orcid: 0009-0008-0911-8765
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 5 July 2026
bibliography: paper.bib
---

# Summary

`betula-cluster` is a Python library for clustering large embedding and tabular
data sets under a fixed memory budget. It compresses the data into numerically
stable *clustering features* (CFs) — the BETULA triple $(n, \mu, S)$
[@lang2020betula; @lang2022betula] — held in a height-balanced CF-tree, and then runs a clustering
*head* on the resulting $M \ll N$ microclusters rather than on the raw points, so
cost scales with the microcluster count and not with the data set size. One
stable engine backs weighted k-means, Gaussian mixtures (diagonal and full
covariance, with BIC-based model selection), agglomerative Ward linkage, spectral
clustering [@ng2002spectral], Leiden community detection [@traag2019leiden], and
density-based HDBSCAN-style clustering [@mcinnes2017hdbscan], all behind a
scikit-learn-compatible API [@pedregosa2011scikit] with streaming `partial_fit`.
The performance-critical core is written from scratch in Rust and exposed through
PyO3; at runtime the package depends only on NumPy [@harris2020numpy].

# Statement of need

Clustering at scale forces a choice between three failure modes. First,
**numerical instability**: the classic BIRCH CF stores a sum of squared norms and
recovers the within-cluster variance as $\mathrm{SS}/n - \lVert\mu\rVert^2$
[@zhang1996birch], a difference of two large, nearly equal quantities that
catastrophically cancels once the data lie far from the origin — exactly the
regime of raw (unnormalized) embeddings, timestamps, or geographic coordinates.
BETULA [@lang2020betula; @lang2022betula] removes this by maintaining a centered
second moment and combining CFs with the numerically stable parallel-axis update
[@chan1983].
Second, **memory blow-up**: full Gaussian-mixture and HDBSCAN implementations must
hold the whole data set, and BIRCH-family trees can grow without bound in high
dimensions. Third, **fragmentation**: practitioners stitch together several
single-purpose libraries to move between k-means, mixtures, hierarchical,
spectral, graph, and density clustering.

`betula-cluster` addresses all three. CFs are updated with Welford/Chan
arithmetic and are positive semidefinite by construction; the CF-tree caps its
leaves (`max_leaves`, or an explicit `memory_budget_mb`) and rebuilds, so
streaming memory stays flat in $N$; and one compressed representation feeds every
head. The closest widely used tool, scikit-learn's `Birch`
[@pedregosa2011scikit], uses the classic unstable formulation, exposes a single
downstream head, and offers no bounded-memory streaming interface. Measured
against scikit-learn on standardized data, `betula-cluster` reaches
parity-or-better cluster quality (for example, k-means adjusted Rand index
$0.861$ vs $0.861$ on Gaussian blobs) while running 15–40$\times$ faster at
$N = 10^6$, and it clusters a $10^7$-point stream with peak memory held near
57 MB, where an in-core k-means requires about 5 GB. These properties make it useful for researchers
and engineers clustering large embedding corpora, high-throughput tabular
streams, and data sets that do not fit in memory.

# Functionality

Beyond the core heads, the package provides soft assignments and coresets,
outlier and near-duplicate detection, cluster-geometry inspection, a consensus
wrapper that quantifies per-point label stability across insertion orders, a
Mapper topological skeleton, evolving-stream density clustering (DenStream and
DbStream), mergeable quantile sketches, `scipy.sparse` input with an $O(\mathrm{nnz})$
sparse-native path, mixed numeric/categorical k-prototypes, must-link/cannot-link
constrained clustering, and memory-aware hyperparameter tuning. Prebuilt `abi3`
wheels ship for Linux, macOS, and Windows, and the Rust core is separately
reusable. Correctness is covered by a 100%-line-coverage Python test suite and an
extensive Rust test suite, and every benchmark figure — including the honest
losses — is reproducible from the repository.

# Acknowledgements

The clustering-feature formulation and its numerically stable variant are due to
@zhang1996birch and @lang2020betula, respectively.

# References
