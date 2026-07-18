# Examples

Executed Jupyter notebooks (with plots, tables, and graphs) demonstrating **every feature** of
`betula-cluster`. Each notebook is paired with a [jupytext](https://jupytext.readthedocs.io/) `.py`
source (the diff-friendly form); the `.ipynb` is the rendered, executed output you can read on GitHub.

| notebook | what it covers |
|----------|----------------|
| [`01_quickstart`](01_quickstart.ipynb) | one-shot `fit_predict`, every head (k-means / GMM / full-cov GMM / Ward / HDBSCAN), automatic `k`, the scikit-learn-style estimator |
| [`02_embeddings_and_inspection`](02_embeddings_and_inspection.ipynb) | `normalize=True` (cosine/direction clustering of embeddings); inspection API: `summary`, `outlier_scores`, `find_outliers`, `find_near_duplicates`, `sample_representatives`, microcluster geometry |
| [`03_streaming_and_persistence`](03_streaming_and_persistence.ipynb) | out-of-core `partial_fit`, EWMA `decay`, `save`/`load` + pickle, scikit-learn `Pipeline` / `GridSearchCV` |
| [`04_method_comparison`](04_method_comparison.ipynb) | every head across six dataset shapes; time-vs-`N` and the memory-vs-`N` headline |
| [`05_topology_mapper`](05_topology_mapper.ipynb) | the **Mapper** topological skeleton (`mapper()` → `MapperGraph`), bridges/branch points, `to_networkx()` |
| [`06_streaming_density`](06_streaming_density.ipynb) | **`DenStream`** & **`DbStream`** fading-microcluster streaming; shared-density connectivity vs proximity; the micro-cluster graph |
| [`07_mixed_data_kprototypes`](07_mixed_data_kprototypes.ipynb) | **`KPrototypes`** — mixed numeric + categorical (k-prototypes); cluster centroids + modes; numeric-only vs mixed |
| [`08_quantile_sketches`](08_quantile_sketches.ipynb) | **`KllSketch`** & **`DdSketch`** streaming quantiles; rank- vs relative-error; mergeable shards; footprint |
| [`09_semisupervised_constraints`](09_semisupervised_constraints.ipynb) | **must-link / cannot-link** (COP-KMeans) via `fit(X, must_link=, cannot_link=)`; the constraint graph; infeasible → `ValueError` |
| [`10_sparse_highdim`](10_sparse_highdim.ipynb) | `scipy.sparse` input (dense-tree path) and the `O(nnz)` **`fit_predict_sparse`**; sparsity pattern, speed + memory |
| [`11_soft_assignment_coreset_diagnostics`](11_soft_assignment_coreset_diagnostics.ipynb) | `predict_proba` / `assignment_confidence`; `export_coreset` (refit anything); `diagnostics`, `cluster_profile`, `representatives` |
| [`12_drift_robust_memory`](12_drift_robust_memory.ipynb) | `snapshot` + `compare_snapshots` drift; `active_learning_batch`; robust `huber_k`; `memory_budget_mb` |
| [`13_graph_clustering`](13_graph_clustering.ipynb) | graph & manifold heads — `method="spectral"` on moons/spirals, `method="leiden"` community detection (auto-count), and `consensus` per-point stability maps |
| [`14_directional_embeddings`](14_directional_embeddings.ipynb) | **directional** heads for L2-normalized embeddings — `method="vmf"` (von Mises–Fisher mixture: soft posterior, concentration `κ`, BIC auto-`k`) and `method="spherical-kmeans"`; why direction beats Euclidean distance when magnitude varies |
| [`15_geometry_aware_clustering`](15_geometry_aware_clustering.ipynb) | head-to-head vs scikit-learn: **`method="scale-space"`** is parameter-free *and* more accurate + faster than `MeanShift` (and beats wrong-`k` k-means), and **covariance-aware `gmm-full`** ties `GaussianMixture` while crushing spherical k-means on anisotropic clusters — plus the GeoBETULA `covariance_weight` / `tangent_weight` knobs |
| [`16_toeplitz_timeseries`](16_toeplitz_timeseries.ipynb) | **`method="gmm-toeplitz"`** — clustering ordered **time-series windows** by their autocovariance *shape* (AR / Toeplitz covariance). Where a diagonal model is blind and full covariance is singular (`N_k ≪ d`), the AR(w) head recovers the components and **beats scikit-learn's `GaussianMixture`** — and it sharpens as the window grows |
| [`17_nmf_topics`](17_nmf_topics.ipynb) | **`projection="weighted-nmf"`** — CF-weighted nonnegative matrix factorization for **nonnegative** data (TF-IDF / counts / spectrograms): factorizes the `M ≪ N` leaf centroids into interpretable nonnegative codes (König-Huygens), then clusters them — so NMF runs at BETULA scale + bounded memory where point-level NMF cannot stream; signed input is rejected |

## Use cases (concrete, end-to-end scenarios)

Applied walk-throughs that compose the features above into a real task, each scored against ground truth:

| use case | what it shows |
|----------|----------------|
| [`usecase_01_embedding_dedup`](usecases/usecase_01_embedding_dedup.ipynb) | deduplicating a repost-heavy embedding corpus — `normalize=True` + `find_near_duplicates`, scored for precision/recall, collapsed to one representative per group |
| [`usecase_02_log_anomaly_detection`](usecases/usecase_02_log_anomaly_detection.ipynb) | anomaly detection on log events — batch `outlier_scores` (ROC-AUC, precision@k) **and** streaming `DbStream` real-time noise flags |
| [`usecase_03_customer_segmentation`](usecases/usecase_03_customer_segmentation.ipynb) | mixed RFM + categorical segmentation with `KPrototypes` — a persona + action table, and why mixed beats numeric-only |
| [`usecase_04_rag_corpus_curation`](usecases/usecase_04_rag_corpus_curation.ipynb) | prepping an embedding store for RAG — junk removal (`outlier_scores`), topic coherence (`mapper_stability`, β₀ = #topics), and topic-leakage detection (Mapper) |
| [`usecase_05_real_data_clustering`](usecases/usecase_05_real_data_clustering.ipynb) | clustering a **real** 64-D dataset (handwritten `digits`) — betula's regularized `gmm` / `gmm-full` **lead scikit-learn on both mean ARI and worst-case stability** (the per-dimension covariance floor stops the high-dimensional component collapse), plus average-digit centroids, medoid exemplars, a refit-anything coreset |
| [`usecase_06_graph_communities`](usecases/usecase_06_graph_communities.ipynb) | community detection on an embedding corpus — **`method="leiden"`** discovers the group count with no `k`, `resolution` γ / CPM dial granularity, and `consensus` flags ambiguous items |

## Run / re-render

```bash
pip install betula-cluster jupytext nbconvert ipykernel \
            matplotlib seaborn pandas networkx scikit-learn scipy

# open interactively
jupytext --to ipynb 01_quickstart.py && jupyter lab 01_quickstart.ipynb

# or re-execute headless (regenerates outputs + plots in place)
jupytext --to ipynb --execute 01_quickstart.py
```

All plots use [seaborn](https://seaborn.pydata.org/) for a consistent look; graphs use
[networkx](https://networkx.org/) and tables use [pandas](https://pandas.pydata.org/). These are
**example-only** dependencies — the `betula-cluster` package itself requires none of them.
