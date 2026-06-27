# Examples

Executed Jupyter notebooks (with plots) demonstrating `betula-cluster`. Each notebook is paired with
a [jupytext](https://jupytext.readthedocs.io/) `.py` source (the diff-friendly form); the `.ipynb` is
the rendered, executed output you can read on GitHub.

| notebook | what it covers |
|----------|----------------|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | one-shot `fit_predict`, every head (k-means / GMM / full-cov GMM / Ward / HDBSCAN), automatic `k`, the scikit-learn-style estimator |
| [`02_embeddings_and_inspection.ipynb`](02_embeddings_and_inspection.ipynb) | `normalize=True` for direction/cosine clustering of embeddings, then the inspection API: `summary`, `outlier_scores`, `find_outliers`, `find_near_duplicates`, `sample_representatives`, microcluster geometry |
| [`03_streaming_and_persistence.ipynb`](03_streaming_and_persistence.ipynb) | out-of-core `partial_fit` streaming, `save`/`load` + pickle, scikit-learn `Pipeline` |

## Run / re-render

```bash
pip install betula-cluster jupytext nbconvert ipykernel matplotlib scikit-learn

# open interactively
jupytext --to ipynb 01_quickstart.py && jupyter lab 01_quickstart.ipynb

# or re-execute headless (regenerates outputs + plots in place)
jupytext --to ipynb --execute 01_quickstart.py
```
