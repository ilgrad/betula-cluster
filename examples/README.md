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
