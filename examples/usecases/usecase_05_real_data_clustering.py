# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (betula examples)
#     language: python
#     name: betula-examples
# ---

# %% [markdown]
# # Use case — clustering a real dataset (handwritten digits)
#
# A real, recognizable dataset end-to-end: scikit-learn's **`digits`** — 1797 handwritten digits as
# 8×8 grayscale images (64 features), 10 true classes (0–9). We cluster it with `betula-cluster`,
# check it matches scikit-learn's k-means on a real (imperfect) dataset, and use the inspection API to
# *see* what each cluster learned — its average-digit centroid and exemplar images.
#
# ```bash
# pip install betula-cluster matplotlib seaborn pandas scikit-learn
# ```

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import betula_cluster

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})

# %% [markdown]
# ## The data — real handwritten digits
#
# `X_raw` holds the 0–16 pixel intensities; `X` is the standardized matrix we cluster on; `y` is the
# true digit for scoring only (clustering is unsupervised).

# %%
X_raw, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X_raw).astype(np.float64)
print(f"digits: {X.shape[0]} images × {X.shape[1]} pixels (8×8), {len(set(y))} classes")

fig, axes = plt.subplots(2, 8, figsize=(11, 3))
for ax, img, lab in zip(axes.ravel(), X_raw, y, strict=False):
    ax.imshow(img.reshape(8, 8), cmap="gray_r")
    ax.set_title(int(lab), fontsize=9)
    ax.set_axis_off()
fig.suptitle("Sample digits", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Cluster it — betula's regularized GMM leads scikit-learn on real 64-D data
#
# Real data is imperfect: digits like 1/8/9 overlap, so no method reaches ARI 1. On this **64-D**
# dataset the telling comparison is the **Gaussian-mixture heads**. In high dimensions a mixture
# component's covariance is estimated from few effective points and can go near-singular, which makes
# the expected-log fit over-confident and swing wildly with the random seed. betula floors the
# per-dimension (co)variance, so its `gmm` / `gmm-full` heads are both **higher on average and far more
# stable across seeds** than scikit-learn's — a component is never starved into collapse. We score five
# seeds per model and report the mean and the worst case. k-means (no covariance to collapse) ties.

# %%
seeds = range(5)


def bet(seed, feat, meth):
    return np.asarray(
        betula_cluster.fit_predict(X, 10, feature=feat, method=meth, threshold=0.0, max_leaves=2000, seed=seed)
    )


specs = [
    ("k-means", lambda s: bet(s, "spherical", "kmeans"), lambda s: KMeans(10, n_init=10, random_state=s).fit_predict(X)),
    ("diagonal GMM", lambda s: bet(s, "diagonal", "gmm"), lambda s: GaussianMixture(10, covariance_type="diag", random_state=s).fit_predict(X)),
    ("full GMM", lambda s: bet(s, "full", "gmm-full"), lambda s: GaussianMixture(10, covariance_type="full", random_state=s).fit_predict(X)),
]
rows = []
for name, bfn, sfn in specs:
    ba = [adjusted_rand_score(y, bfn(s)) for s in seeds]
    sa = [adjusted_rand_score(y, sfn(s)) for s in seeds]
    rows.append(
        {"model": name, "betula mean": np.mean(ba), "betula worst": np.min(ba), "sklearn mean": np.mean(sa), "sklearn worst": np.min(sa)}
    )
cmp = pd.DataFrame(rows).set_index("model")

fig, ax = plt.subplots(figsize=(7.2, 3.9))
xs = np.arange(len(cmp))
c0, c1 = sns.color_palette("deep")[0], sns.color_palette("deep")[1]
ax.bar(xs - 0.2, cmp["betula mean"], 0.4, yerr=[cmp["betula mean"] - cmp["betula worst"], np.zeros(len(cmp))], capsize=4, label="betula", color=c0)
ax.bar(xs + 0.2, cmp["sklearn mean"], 0.4, yerr=[cmp["sklearn mean"] - cmp["sklearn worst"], np.zeros(len(cmp))], capsize=4, label="scikit-learn", color=c1)
ax.set(xticks=xs, ylabel="ARI (mean; whisker down to worst seed)", ylim=(0, 0.7), title="digits — ARI over 5 seeds (taller + shorter whisker = better)")
ax.set_xticklabels(cmp.index)
ax.legend()
fig.tight_layout()
plt.show()

# the single betula fit we inspect below (centroids / exemplars / coreset)
labels = np.asarray(bet(0, "spherical", "kmeans"))
cmp.round(3)

# %% [markdown]
# ## What did each cluster learn? — average-digit centroids
#
# Averaging the raw images in each cluster gives its "prototype" digit. Most clusters resolve to a
# clean, recognizable numeral — the unsupervised structure lines up with the real classes.

# %%
fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
for c, ax in enumerate(axes.ravel()):
    members = X_raw[labels == c]
    ax.imshow(members.mean(0).reshape(8, 8), cmap="gray_r")
    # the majority true digit in this cluster, for a sanity label
    maj = np.bincount(y[labels == c]).argmax()
    ax.set_title(f"cluster {c} → {maj}", fontsize=9)
    ax.set_axis_off()
fig.suptitle("Cluster centroids (average digit)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Exemplars — the most typical image per cluster
#
# `representatives(..., method="medoid")` returns the rows closest to a cluster's centroid: a compact,
# human-readable summary of each cluster without scanning thousands of images.

# %%
est = betula_cluster.Betula(n_clusters=10, feature="spherical", method="kmeans", threshold=0.0, max_leaves=2000, seed=0).fit(X)
fig, axes = plt.subplots(2, 5, figsize=(10, 4.2))
for c, ax in enumerate(axes.ravel()):
    reps = np.asarray(est.representatives(X, cluster_id=c, method="medoid", k=1))
    ax.imshow(X_raw[reps[0]].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"cluster {c}", fontsize=9)
    ax.set_axis_off()
fig.suptitle("Medoid exemplar per cluster", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A coreset reproduces the result at a fraction of the size
#
# `export_coreset()` returns the CF-tree leaves as weighted points. A deliberately **coarse** tree
# (256 leaves) summarizes the 1797 images; refitting scikit-learn k-means on those 256 weighted leaves
# gives a clustering that closely agrees with fitting on all 1797 images — at a fraction of the rows.

# %%
coarse = betula_cluster.Betula(
    n_clusters=10, feature="spherical", method="kmeans", threshold=0.0, max_leaves=256, seed=0
).fit(X)
core = coarse.export_coreset()
km_full = KMeans(10, n_init=10, random_state=0).fit(X)
km_core = KMeans(10, n_init=10, random_state=0).fit(core.centers, sample_weight=core.weights)
pd.DataFrame(
    {
        "metric": ["coreset size", "ARI: coreset labels vs full-data labels"],
        "value": [
            f"{len(core.centers)} weighted leaves vs {len(X)} images",
            round(adjusted_rand_score(km_full.predict(X), km_core.predict(X)), 3),
        ],
    }
)

# %% [markdown]
# ## Takeaway
#
# On a real 64-D dataset, `betula-cluster`'s regularized Gaussian-mixture heads **lead scikit-learn's
# both on average and — more importantly — in worst-case stability**: the per-dimension covariance
# floor stops the high-dimensional component collapse that makes an unregularized full GMM swing with
# the seed (scikit-learn's full-cov ARI ranges ~0.40–0.64 here; betula's never drops below ~0.51).
# k-means, which has no covariance to collapse, ties. And betula exposes the structure (centroids,
# exemplars, a refit-anything coreset) over the microclusters it already built, at bounded memory — so
# the identical code scales from 1797 images to tens of millions. For the at-scale numbers on real data
# (covtype, MNIST) see [`bench/RESULTS.md`](../../bench/RESULTS.md).
