# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mixed numeric + categorical + directional data — `KPrototypes`
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
# A third block, for columns that are an **angle** rather than a length, is added at the end.
#
# ```bash
# pip install betula-cluster matplotlib seaborn pandas scikit-learn
# ```

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from betula_cluster import KPrototypes, __version__

sns.set_theme(style="whitegrid", context="notebook", palette="deep")

print("betula-cluster", __version__)
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
    {
        "name": "students",
        "age": (22, 3),
        "spend": (15, 5),
        "city": [0.2, 0.3, 0.5],
        "plan": [0.7, 0.25, 0.05],
    },
    {
        "name": "families",
        "age": (40, 5),
        "spend": (80, 15),
        "city": [0.5, 0.3, 0.2],
        "plan": [0.1, 0.6, 0.3],
    },
    {
        "name": "retirees",
        "age": (68, 6),
        "spend": (45, 10),
        "city": [0.3, 0.5, 0.2],
        "plan": [0.2, 0.5, 0.3],
    },
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


num_only = fit_predict(
    df[["age", "monthly_spend"]].to_numpy(np.float64), 3, method="kmeans", seed=1
)
pd.DataFrame(
    {
        "method": ["k-means (numeric only)", "k-prototypes (numeric + categorical)"],
        "ARI vs true segments": [round(ari(num_only, truth), 3), round(ari(labels, truth), 3)],
    }
)

# %% [markdown]
# ## A third block — directions
#
# Some columns are an **angle**, not a length. Give each customer a login-time vector: its direction
# is the hour of day they usually log in, its length is that month's session count. The habit is in
# the angle; the length is heavy-tailed noise that says nothing about the segment.
#
# Handed to k-prototypes as two ordinary numeric columns, that vector asks the distance to match
# session counts as well as habits. `directional=` routes a group of columns through a third block
# instead: the group is L2-normalised **per row**, so only the angle survives, and the cost gains
#
# $$ \gamma_\text{dir}\,\bigl(2 - 2\, u^\top c\bigr), \qquad u = \frac{x_\text{dir}}{\lVert
# x_\text{dir} \rVert} $$
#
# where `c` is the cluster's mean **resultant** direction — itself a unit vector, available as
# `cluster_directions_`. `gamma_dir` weights the block; it defaults to the mean numeric variance. A
# row whose directional part is all zeros has no direction, so its term is the same for every
# prototype and it abstains rather than voting arbitrarily.

# %%
LOGIN_HOUR = [23.0, 19.0, 9.0]  # students log in at night, families after dinner, retirees early


def with_login(seed, vol_sigma):
    """The customer table of the first cell, plus a login-time vector in columns 4 and 5."""
    r = np.random.default_rng(seed)
    out, lab = [], []
    for s, seg in enumerate(segments):
        for _ in range(400):
            theta = 2 * np.pi * r.normal(LOGIN_HOUR[s], 1.5) / 24.0
            sessions = 30.0 * r.lognormal(0.0, vol_sigma)
            out.append(
                [
                    r.normal(*seg["age"]),
                    r.normal(*seg["spend"]),
                    r.choice(3, p=seg["city"]),
                    r.choice(3, p=seg["plan"]),
                    sessions * np.cos(theta),
                    sessions * np.sin(theta),
                ]
            )
            lab.append(s)
    return np.array(out), np.array(lab)


def unit_rows(Z):
    """What a careful user does by hand: rescale each login vector to unit length."""
    Z = Z.copy()
    n = np.linalg.norm(Z[:, 4:6], axis=1, keepdims=True)
    Z[:, 4:6] /= np.where(n == 0.0, 1.0, n)
    return Z


def fit_ari(Z, t, **kw):
    return ari(KPrototypes(n_clusters=3, categorical=[2, 3], seed=1, **kw).fit_predict(Z), t)


# %% [markdown]
# ### Three ways to spend the same two columns
#
# ARI against the true segments, median of three seeds, as the session-count spread `σ` grows.
# `radius std` is the spread the login vector actually contributes, for comparison with the σ ≈ 20
# of `age` and σ ≈ 29 of `monthly_spend` printed further down.

# %%
sweep = []
for vol_sigma in (0.6, 0.9, 1.2):
    got = {"login dropped": [], "raw numeric": [], "unit rows": [], "directional=[4, 5]": []}
    rad = []
    for seed in (7, 8, 9):
        Z, t = with_login(seed, vol_sigma)
        rad.append(np.linalg.norm(Z[:, 4:6], axis=1).std())
        got["login dropped"].append(fit_ari(Z[:, :4], t))
        got["raw numeric"].append(fit_ari(Z, t))
        got["unit rows"].append(fit_ari(unit_rows(Z), t))
        got["directional=[4, 5]"].append(fit_ari(Z, t, directional=[4, 5]))
    sweep.append(
        {
            "σ": vol_sigma,
            "radius std": round(float(np.median(rad))),
            **{k: round(float(np.median(v)), 3) for k, v in got.items()},
        }
    )
pd.DataFrame(sweep).set_index("σ")

# %% [markdown]
# The directional row does not move at all — normalisation removes `σ` before the distance ever
# sees it. Normalising by hand removes it too, but pins the block's weight at 1 against numeric
# columns whose spread is tens of units, so the angle barely counts: that row reproduces the
# `login dropped` column to three decimals. The missing scale is exactly what `gamma_dir` is, and
# unlike the hand-rolled version it has a default.
#
# ### What breaks the raw-numeric fit
#
# Not the spread as such: at `σ = 0.9` the login radius already varies more than `monthly_spend`
# does and costs nothing, because a wedge of points at one hour is still a compact Euclidean
# cluster. It is the **tail**. Squared error is dominated by the longest vectors, so at `σ = 1.2`
# the fit spends clusters on a few dozen outliers and leaves the segments in the remainder:

# %%
Z12, _ = with_login(7, 1.2)
naive12 = np.asarray(KPrototypes(n_clusters=3, categorical=[2, 3], seed=1).fit_predict(Z12))
radius = np.linalg.norm(Z12[:, 4:6], axis=1)
print(f"σ of age {Z12[:, 0].std():.0f}, of monthly_spend {Z12[:, 1].std():.0f}")
pd.DataFrame(
    {
        "size": [int((naive12 == c).sum()) for c in range(naive12.max() + 1)],
        "median login radius": [
            round(float(np.median(radius[naive12 == c])), 1) for c in range(naive12.max() + 1)
        ],
    }
)

# %% [markdown]
# ### The directional prototype, read back as a clock
#
# `cluster_directions_` returns one unit vector per cluster. `arctan2` turns it back into the hour
# of day it stands for — the interpretable counterpart of `cluster_modes_` for the angular block.
# Same rows, same seed as the table above; only the block assignment differs.

# %%
kp_dir = KPrototypes(n_clusters=3, categorical=[2, 3], directional=[4, 5], seed=1)
lab_dir = np.asarray(kp_dir.fit_predict(Z12))
dirs = kp_dir.cluster_directions_  # (k, 2), unit norm

pd.DataFrame(
    {
        "size": [int((lab_dir == c).sum()) for c in range(kp_dir.n_clusters_)],
        "age": kp_dir.cluster_centroids_[:, 0].round(1),
        "monthly_spend": kp_dir.cluster_centroids_[:, 1].round(1),
        "plan (mode)": [PLANS[m] for m in kp_dir.cluster_modes_[:, 1]],
        "login hour": (np.arctan2(dirs[:, 1], dirs[:, 0]) * 24 / (2 * np.pi) % 24).round(1),
    }
)
