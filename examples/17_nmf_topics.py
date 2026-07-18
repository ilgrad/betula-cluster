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
# # Nonnegative data — `projection="weighted-nmf"`
#
# Some data is **nonnegative and additive**: TF-IDF / bag-of-words counts, event tallies, spectrogram
# magnitudes, pixel or colour histograms. There each object is a *sum of parts*, and a **nonnegative
# matrix factorization** `X ≈ W H` (`W, H ≥ 0`) is the natural representation — the parts `H` are
# interpretable, and the per-object codes `W` are what you cluster.
#
# Running NMF over all `N` rows defeats BETULA's compression (it is `O(N·d·r)` and holds an `N×r` code
# matrix). **`projection="weighted-nmf"`** factorizes the `M ≪ N` leaf **centroids** instead: by
# König-Huygens the weighted-centroid NMF equals the full-data one up to the within-leaf scatter
# constant, so NMF runs at BETULA scale and bounded memory, and any head then clusters the codes.
#
# All numbers are computed live. See [`docs/MATH.md`](../docs/MATH.md#cf-weighted-nmf-for-nonnegative-data).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score as ari

import betula_cluster

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})


def topic_corpus(n_per, d, k, seed):
    """`k` latent nonnegative "topics"; each document is a nonnegative mixture dominated by one."""
    rng = np.random.default_rng(seed)
    parts = np.abs(rng.normal(size=(k, d))) * (rng.random((k, d)) < 0.35)  # sparse nonneg parts
    xs, ys = [], []
    for c in range(k):
        w = rng.random((n_per, k)) * 0.15  # small cross-topic leakage
        w[:, c] += 1.0 + rng.random(n_per)  # dominant topic
        xs.append(w @ parts + 0.02 * rng.random((n_per, d)))
        ys += [c] * n_per
    return np.ascontiguousarray(np.vstack(xs)), np.array(ys), parts


d, k = 50, 4
X, y, parts = topic_corpus(600, d, k, seed=0)
print(f"corpus: {X.shape[0]} documents × {d} nonnegative features, {k} latent topics")

# %% [markdown]
# ## The data — nonnegative, additive, overlapping
#
# Each row is a nonnegative feature vector; documents of the same topic share a dominant set of active
# features, but every document mixes in a little of the others. Raw Euclidean k-means on the sparse
# high-`d` counts sees mostly magnitude, not topic.

# %%
fig, ax = plt.subplots(figsize=(7, 2.8))
order = np.argsort(y)
ax.imshow(X[order][::12].T, aspect="auto", cmap="magma")
ax.set(title="documents (sorted by topic) × features — blocks are the latent parts", xticks=[], yticks=[])
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Cluster it — reduce over the leaves, then k-means
#
# `projection="weighted-nmf"` factorizes the leaf centroids into `projection_dim` nonnegative codes and
# clusters those. We score it against raw k-means on the full counts and a diagonal GMM.

# %%
kw = dict(feature="spherical", threshold=0.0, seed=0, max_leaves=4000)
runs = {
    "raw k-means": betula_cluster.fit_predict(X, k, method="kmeans", **kw),
    "raw gmm (diag)": betula_cluster.fit_predict(
        X, k, method="gmm", feature="diagonal", threshold=0.0, seed=0
    ),
    "weighted-nmf → k-means": betula_cluster.fit_predict(
        X, k, method="kmeans", projection="weighted-nmf", projection_dim=8, **kw
    ),
}
scores = {name: ari(y, np.asarray(lab)) for name, lab in runs.items()}

fig, ax = plt.subplots(figsize=(6.4, 2.8))
names = list(scores)
colors = ["#bbb", "#bbb", "#2a9d8f"]
ax.barh(range(len(names)), [scores[n] for n in names], color=colors)
ax.set(yticks=range(len(names)), xlabel="ARI vs ground-truth topics", xlim=(0, 1.05))
ax.set_yticklabels(names)
ax.invert_yaxis()
for i, n in enumerate(names):
    ax.text(scores[n] + 0.02, i, f"{scores[n]:.2f}", va="center", fontsize=8)
fig.tight_layout()
plt.show()
scores

# %% [markdown]
# ## Why it is fast — NMF runs over the leaves, not the points
#
# The factorization cost is bounded by the leaf count `max_leaves`, not `N`. So the same call scales:
# doubling `N` leaves the NMF work unchanged (only the `O(N)` tree build grows). That is the whole point
# — point-level NMF is `O(N·d·r)` per iteration and cannot stream; this stays memory-bounded.
# `bench/nmf_cf_weighted.py` times it against `sklearn.decomposition.NMF → k-means` as `N` grows.

# %%
sizes = [400, 800, 1600, 3200]
curve = []
for per in sizes:
    Xn, yn, _ = topic_corpus(per, d, k, seed=1)
    lab = betula_cluster.fit_predict(
        Xn, k, method="kmeans", projection="weighted-nmf", projection_dim=8, **kw
    )
    curve.append(ari(yn, np.asarray(lab)))

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot([s * k for s in sizes], curve, "o-", lw=2, color="#2a9d8f")
ax.set(xlabel="N (documents)", ylabel="ARI", title="weighted-nmf: quality holds as N grows (NMF work is flat in N)", ylim=(0, 1.05))
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Count data — `projection="weighted-nmf-kl"` matches the Poisson noise model
#
# Frobenius NMF minimizes squared error — the right objective when the noise is Gaussian (TF-IDF,
# magnitudes). But **raw counts** (word tallies, event counts) are **Poisson**, whose variance grows
# with the mean, so squared error over-weights the large entries. The **KL-divergence** objective
# (`projection="weighted-nmf-kl"`, Lee-Seung multiplicative updates, leaf mass folded into the `H`
# update) is the maximum-likelihood fit under Poisson noise.
#
# The advantage is **largest where the counts are sparsest** — low rates are where Poisson noise
# departs most from Gaussian. As the mean count grows, the central-limit theorem pulls Poisson toward
# Gaussian and Frobenius catches up. The sweep below draws the same topic mixture at rising intensities.

# %%
def poisson_corpus(n_per, d, k, seed, scale):
    """`k` nonnegative topics; each document is a Poisson draw from its topic-mixture rate."""
    rng = np.random.default_rng(seed)
    parts = np.abs(rng.normal(size=(k, d))) * (rng.random((k, d)) < 0.35)
    xs, ys = [], []
    for c in range(k):
        w = rng.random((n_per, k)) * 0.15
        w[:, c] += 1.0 + rng.random(n_per)
        rate = scale * (w @ parts)  # nonnegative intensity
        xs.append(rng.poisson(rate).astype(np.float64))
        ys += [c] * n_per
    return np.ascontiguousarray(np.vstack(xs)), np.array(ys)


scales = [0.5, 1.0, 2.0, 4.0]
mean_counts, frob, kl_ = [], [], []
for sc in scales:
    Xc, yc = poisson_corpus(600, d, k, seed=0, scale=sc)
    mean_counts.append(Xc.mean())
    frob.append(ari(yc, np.asarray(betula_cluster.fit_predict(
        Xc, k, method="kmeans", projection="weighted-nmf", projection_dim=8, **kw))))
    kl_.append(ari(yc, np.asarray(betula_cluster.fit_predict(
        Xc, k, method="kmeans", projection="weighted-nmf-kl", projection_dim=8, **kw))))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(mean_counts, frob, "o-", lw=2, color="#bbb", label="weighted-nmf (Frobenius)")
ax.plot(mean_counts, kl_, "o-", lw=2, color="#e76f51", label="weighted-nmf-kl (Poisson)")
ax.set(xlabel="mean count per entry (sparser ←)", ylabel="ARI vs ground-truth topics",
       title="KL wins most on sparse counts; the gap closes as counts grow", ylim=(0, 1.05))
ax.set_xscale("log")
ax.legend()
fig.tight_layout()
plt.show()
pd.DataFrame({"mean_count": np.round(mean_counts, 2), "Frobenius": np.round(frob, 3),
              "KL": np.round(kl_, 3), "delta": np.round(np.array(kl_) - np.array(frob), 3)})

# %% [markdown]
# ## Nonnegative only — signed input is rejected, not shifted
#
# NMF is undefined for signed values, and shifting `X - X.min()` would change angles and the cosine
# geometry. The engine refuses signed data rather than silently corrupt it — for signed embeddings use
# the **directional** heads (`vmf` / `spherical-kmeans`) or reduce with PCA / TruncatedSVD first.

# %%
try:
    betula_cluster.fit_predict(X - 1.0, k, method="kmeans", projection="weighted-nmf", projection_dim=8)
    print("no error (unexpected)")
except ValueError as e:
    print("rejected signed input:", str(e)[:80])

# %% [markdown]
# **Takeaway.** For **nonnegative, additive** data, `projection="weighted-nmf"` learns interpretable
# nonnegative parts and clusters their codes — but factorizes the `M ≪ N` leaf centroids, so NMF runs at
# BETULA scale and bounded memory (point-level NMF cannot stream). The default Frobenius objective fits
# Gaussian-ish magnitudes (TF-IDF); **`projection="weighted-nmf-kl"`** switches to the KL divergence, the
# Poisson maximum-likelihood fit for raw **counts** — a large win on sparse counts, converging to
# Frobenius as counts grow. It is an **opt-in reducer** for nonnegative data only; for signed embeddings
# reach for `vmf` / `spherical-kmeans` or PCA / TruncatedSVD. See the [Usage guide](../docs/USAGE.md) and
# [`docs/MATH.md`](../docs/MATH.md#cf-weighted-nmf-for-nonnegative-data).
