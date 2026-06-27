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
# # Soft assignment, coresets & diagnostics
#
# Beyond hard labels, a fitted `Betula` exposes the structure it learned: soft posteriors and
# confidence, a weighted-point **coreset** you can refit anything on, and structural **diagnostics**
# and **profiles** for each cluster.
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
from sklearn.datasets import make_blobs

from betula_cluster import Betula

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cont = pd.crosstab(a, b).to_numpy().astype(float)
    comb = lambda m: (m * (m - 1) / 2).sum()
    s, sa, sb, t = comb(cont), comb(cont.sum(1)), comb(cont.sum(0)), comb(np.array([len(a)]))
    exp = sa * sb / t
    return float((s - exp) / (0.5 * (sa + sb) - exp))


X, y = make_blobs(n_samples=12_000, centers=4, cluster_std=1.3, random_state=0)
X = X.astype(np.float64)
est = Betula(n_clusters=4, feature="diagonal", method="gmm", threshold=0.1, seed=1).fit(X)

# %% [markdown]
# ## `predict_proba` & `assignment_confidence`
#
# The GMM heads return true posterior responsibilities. The **confidence** is the max per-row
# probability — low values flag boundary / ambiguous points (here, where blobs overlap).

# %%
conf = np.asarray(est.assignment_confidence(X))
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(conf, bins=40, ax=axes[0])
axes[0].set(title="Assignment confidence", xlabel="max posterior probability")
sc = axes[1].scatter(X[:, 0], X[:, 1], c=conf, cmap="viridis", s=10, linewidth=0)
axes[1].set_title("Low confidence = cluster boundaries")
plt.colorbar(sc, ax=axes[1], label="confidence")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## A streaming coreset — refit anything on a tiny weighted summary
#
# `export_coreset()` returns the CF-tree leaves as weighted points. Fitting scikit-learn `KMeans` on
# the coreset (a few hundred weighted rows) matches fitting it on all 12,000 points — at a fraction of
# the size.

# %%
core = est.export_coreset()
km_full = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
km_core = KMeans(n_clusters=4, n_init=10, random_state=0).fit(core.centers, sample_weight=core.weights)
pd.DataFrame(
    {
        "fit on": ["full data (12,000 pts)", f"coreset ({len(core.centers)} weighted pts)"],
        "rows used": [len(X), len(core.centers)],
        "ARI of labels vs each other": ["—", round(ari(km_full.predict(X), km_core.predict(X)), 3)],
    }
)

# %% [markdown]
# ## Structural diagnostics

# %%
diag = est.diagnostics()
pd.Series(diag).to_frame("value")

# %% [markdown]
# ## Per-cluster profile (JSON-able — e.g. to feed an LLM that names clusters)

# %%
prof = est.cluster_profile(0)
print("cluster 0:")
print("  size   :", prof["size"])
print("  radius :", round(prof["radius"], 3))
print("  center :", np.round(prof["center"], 2).tolist())
print("  nearest:", [(p["cluster_id"], round(p["distance"], 2)) for p in prof["nearest_clusters"]])

# %% [markdown]
# ## Representatives — exemplars to show a human
#
# Four selection strategies: the medoid (most typical), boundary cases, outliers, and a diverse spread.

# %%
reps = {m: est.representatives(X, cluster_id=0, method=m, k=4).tolist() for m in ["medoid", "boundary", "outlier", "diverse"]}
fig, ax = plt.subplots(figsize=(7, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=np.asarray(est.predict(X)), palette="tab10", s=8, linewidth=0, legend=False, ax=ax)
markers = {"medoid": ("*", "black"), "boundary": ("P", "crimson"), "outlier": ("X", "darkorange"), "diverse": ("D", "navy")}
for m, idx in reps.items():
    mk, col = markers[m]
    ax.scatter(*X[idx].T, marker=mk, c=col, s=160, edgecolor="white", linewidth=1.2, label=m, zorder=5)
ax.legend(title="representatives of cluster 0")
ax.set_title("Exemplar selection strategies")
plt.show()
pd.DataFrame({m: idx for m, idx in reps.items()})
