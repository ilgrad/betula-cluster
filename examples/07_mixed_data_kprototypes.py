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
# # Mixed numeric + categorical data — `KPrototypes`
#
# Real tabular data is rarely all-numeric. **k-prototypes** (Huang, 1997) clusters rows that are part
# numeric, part categorical with a single distance:
#
# $$ d(x, c) = \underbrace{\lVert x_\text{num} - \mu \rVert^2}_{\text{k-means}} \; + \; \gamma \cdot
# \underbrace{\sum_j [\,x_{\text{cat},j} \neq \text{mode}_j\,]}_{\text{k-modes}} $$
#
# Each cluster keeps a numeric **mean** and a per-attribute **mode** (the categorical centroid).
# `gamma` trades the two off; it defaults to Huang's heuristic (½·mean numeric σ).
#
# ```bash
# pip install betula-cluster matplotlib seaborn pandas scikit-learn
# ```

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from betula_cluster import KPrototypes

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold"})
rng = np.random.default_rng(7)

# %% [markdown]
# ## A small customer table
#
# Three latent segments that differ in **both** the numeric columns (age, monthly spend) **and** the
# categorical ones (city, plan): students, families, and retirees.

# %%
CITIES = ["Berlin", "Munich", "Hamburg"]
PLANS = ["free", "basic", "pro"]
segments = [
    {"name": "students", "age": (22, 3), "spend": (15, 5), "city": [0.2, 0.3, 0.5], "plan": [0.7, 0.25, 0.05]},
    {"name": "families", "age": (40, 5), "spend": (80, 15), "city": [0.5, 0.3, 0.2], "plan": [0.1, 0.6, 0.3]},
    {"name": "retirees", "age": (68, 6), "spend": (45, 10), "city": [0.3, 0.5, 0.2], "plan": [0.2, 0.5, 0.3]},
]
rows, truth = [], []
for s, seg in enumerate(segments):
    for _ in range(400):
        rows.append(
            {
                "age": rng.normal(*seg["age"]),
                "monthly_spend": rng.normal(*seg["spend"]),
                "city": rng.choice(3, p=seg["city"]),
                "plan": rng.choice(3, p=seg["plan"]),
            }
        )
        truth.append(s)
df = pd.DataFrame(rows)
truth = np.array(truth)
display = df.copy()
display["city"] = display["city"].map(dict(enumerate(CITIES)))
display["plan"] = display["plan"].map(dict(enumerate(PLANS)))
display.head()

# %% [markdown]
# ## Fit `KPrototypes`
#
# Columns 2 and 3 (`city`, `plan`) are categorical — their values are integer codes. Pass their
# indices via `categorical=`; the rest are treated as numeric.

# %%
X = df.to_numpy(dtype=np.float64)  # categorical cols hold integer codes
kp = KPrototypes(n_clusters=3, categorical=[2, 3], seed=1)
labels = np.asarray(kp.fit_predict(X))
print("clusters found:", kp.n_clusters_)

# %% [markdown]
# ## What each cluster looks like
#
# `cluster_centroids_` gives the numeric mean; `cluster_modes_` the categorical mode (decoded back to
# names). This profile table is what makes the result interpretable.

# %%
cent = kp.cluster_centroids_  # (k, 2): age, monthly_spend
modes = kp.cluster_modes_  # (k, 2): city, plan codes
profile = pd.DataFrame(
    {
        "size": [int((labels == c).sum()) for c in range(kp.n_clusters_)],
        "age": cent[:, 0].round(1),
        "monthly_spend": cent[:, 1].round(1),
        "city (mode)": [CITIES[m] for m in modes[:, 0]],
        "plan (mode)": [PLANS[m] for m in modes[:, 1]],
    }
)
profile

# %% [markdown]
# ## Visualise — numeric split + categorical composition

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(
    x=df["age"], y=df["monthly_spend"], hue=labels, palette="tab10", s=18, linewidth=0, ax=axes[0]
)
axes[0].set_title("Numeric view (age vs spend), coloured by k-prototypes cluster")
axes[0].legend(title="cluster", fontsize=8)

comp = (
    pd.crosstab(labels, df["plan"].map(dict(enumerate(PLANS))), normalize="index")
    .reindex(columns=PLANS)
    .fillna(0.0)
)
sns.heatmap(comp, annot=True, fmt=".0%", cmap="Blues", cbar=False, ax=axes[1])
axes[1].set_title("Plan composition per cluster")
axes[1].set(xlabel="plan", ylabel="cluster")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Why not just k-means on the numbers?
#
# Dropping the categorical signal blurs segments that overlap numerically. k-prototypes recovers the
# true segments better because it also uses `city` / `plan`.

# %%
from betula_cluster import fit_predict


def ari(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cont = pd.crosstab(a, b).to_numpy()
    comb = lambda m: (m * (m - 1) / 2).sum()
    s, sa, sb = comb(cont), comb(cont.sum(1)), comb(cont.sum(0))
    t = comb(np.array([len(a)]))
    exp = sa * sb / t
    return (s - exp) / (0.5 * (sa + sb) - exp)


num_only = fit_predict(df[["age", "monthly_spend"]].to_numpy(np.float64), 3, method="kmeans", seed=1)
pd.DataFrame(
    {
        "method": ["k-means (numeric only)", "k-prototypes (numeric + categorical)"],
        "ARI vs true segments": [round(ari(num_only, truth), 3), round(ari(labels, truth), 3)],
    }
)
