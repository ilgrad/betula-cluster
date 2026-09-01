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
date: 1 September 2026
bibliography: paper.bib
---

# Summary

`betula-cluster` is a Python library for clustering large embedding and tabular
data sets under a fixed memory budget. It compresses the data into numerically
stable *clustering features* (CFs) — the BETULA triple $(n, \mu, S)$
[@lang2020betula; @lang2022betula] — held in a height-balanced CF-tree, and then runs a clustering
*head* on the resulting $M \ll N$ microclusters rather than on the raw points, so
cost scales with the microcluster count and not with the data set size. One
stable engine backs 27 heads: weighted k-means and k-medoids
[@schubert2021fasterpam], fuzzy c-means, Gaussian mixtures (diagonal, full,
low-rank subspace [@tipping1999mppca], and Toeplitz-structured for stationary
signals, with BIC-based model selection), five agglomerative linkages including
exact Ward, spectral clustering [@ng2002spectral], Leiden community detection
[@traag2019leiden], density-based HDBSCAN-style clustering
[@mcinnes2017hdbscan], directional mixtures on the unit sphere
[@banerjee2005vmf], and k-means on the Lorentz model of hyperbolic space — all behind a
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
head. Measured against scikit-learn on standardized data, `betula-cluster` reaches
parity-or-better cluster quality (for example, k-means adjusted Rand index
$0.793$ vs $0.794$ on Gaussian blobs, median of three seeds; full-covariance
mixtures $0.961$ vs $0.902$ on anisotropic data) while labelling $10^6$ points
$9\times$ faster than scikit-learn's `KMeans` and $30\times$ faster than its
`Birch`, and it clusters a $10^7$-point stream with peak memory held near 60 MB,
where an in-core k-means requires about 5 GB.

Two existing tools are close, in different directions. scikit-learn's `Birch`
[@pedregosa2011scikit] is the de-facto Python CF-tree, but it implements the
*classic* unstable formulation, exposes a single downstream head, and offers no
bounded-memory streaming interface. `betulars`, by a BETULA co-author, is a
faithful and fast Phase-1 CF-tree builder in Rust, but it produces leaf
statistics rather than labels and leaves the global clustering step to the user.
`betula-cluster` occupies the gap: the stable CF *and* the end-to-end pipeline
that turns it into per-point labels.

# Functionality

Absorption is a first-class choice: eight criteria are exposed, including the six
BIRCH distances and a mass-invariant Mahalanobis-$\chi^2$ gate, and the
benchmark quantifies which of them are immune to the BIRCH size-imbalance
pathology and which are not. Beyond the heads, the package provides soft
assignments, sensitivity-sampled $(k,\varepsilon)$-coresets with an explicit
summarization bound [@feldman2011coresets], outlier and near-duplicate detection,
cluster-geometry inspection, a consensus wrapper that quantifies per-point label
stability across insertion orders, a Mapper topological skeleton,
evolving-stream density clustering with ADWIN drift detection [@bifet2007adwin],
mergeable quantile sketches, `scipy.sparse` input with an $O(\mathrm{nnz})$
sparse-native path, mixed numeric/categorical/directional k-prototypes,
must-link/cannot-link constrained clustering, and memory-aware hyperparameter
tuning. Prebuilt `abi3` wheels ship for Linux, macOS, and Windows, and the Rust
core is separately reusable.

Correctness is covered by a 457-case Python test suite held at 100% statement
coverage of the wrapper, 728 Rust tests, and a mutation-testing baseline in
which every surviving mutant carries either a killing test or a written
equivalence argument. Every benchmark figure is the median of three seeds and is
reproducible from the repository — including the measured **losses**, and
including research probes that were implemented, measured, refuted, and reverted
rather than shipped.

# Acknowledgements

The clustering-feature formulation and its numerically stable variant are due to
@zhang1996birch and @lang2020betula, respectively.

# References
