"""Memory-safe quality (ARI) + speed + peak-RSS benchmark: betula-cluster vs scikit-learn.

Each (dataset, method) runs in its own spawned subprocess with an address-space cap (RLIMIT_AS)
and a wall-clock timeout, so a method that explodes in memory (e.g. sklearn Birch tiling a
high-dimensional Gaussian into a curse-of-dimensionality blow-up) fails *gracefully* as `OOM`
instead of taking down the whole run or the host. betula's CF-tree is memory-bounded by design
(`max_leaves`), which is exactly what this benchmark surfaces.

Usage:  python benchmark.py [--scale FLOAT]   (scale multiplies all sample counts)
"""

# Thread caps must be set before numpy/BLAS import so RLIMIT_AS isn't tripped by arena reservations.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import resource
import time

import numpy as np

MEM_CAP = 10 * 1024**3  # 10 GiB address-space cap per worker
TIMEOUT = 120.0  # seconds per method


def gen(spec):
    from sklearn.datasets import make_blobs, make_moons

    if spec["kind"] == "blobs":
        x, y = make_blobs(
            n_samples=spec["n"],
            centers=spec["k"],
            n_features=spec["d"],
            cluster_std=spec["std"],
            random_state=0,
        )
    elif spec["kind"] == "embed":
        # Embedding-like data: clusters share a *direction*, magnitude is noise (varying-norm
        # vectors). Cosine/direction structure — exactly where `normalize=True` matters. Shuffled
        # so insertion order is realistic.
        rng = np.random.default_rng(0)
        d, k, per = spec["d"], spec["k"], spec["n"] // spec["k"]
        centers = rng.standard_normal((k, d))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        xs, ys = [], []
        for c in range(k):
            dirs = centers[c] + spec["std"] * rng.standard_normal((per, d))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            xs.append(rng.lognormal(0.0, 1.0, (per, 1)) * dirs)  # wide magnitude spread
            ys += [c] * per
        x, y = np.vstack(xs), np.array(ys)
        perm = rng.permutation(len(x))
        return np.ascontiguousarray(x[perm], dtype=np.float64), y[perm]
    else:
        x, y = make_moons(n_samples=spec["n"], noise=spec["noise"], random_state=0)
    return np.ascontiguousarray(x, dtype=np.float64), y


def run_method(name, x, y, spec):
    import betula_cluster
    from sklearn.cluster import Birch, KMeans
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture

    k = spec["k"]
    if name == "betula-kmeans":
        labels = betula_cluster.fit_predict(
            x, k, feature="spherical", method="kmeans", threshold=0.0, max_leaves=2000
        )
    elif name == "betula-kmeans-par8":
        # parallel Phase-1 shard+merge build (8 workers); same quality, faster on large N
        labels = betula_cluster.fit_predict(
            x, k, feature="spherical", method="kmeans", threshold=0.0, max_leaves=2000, n_jobs=8
        )
    elif name == "betula-kmeans-f32":
        # float32 input clustered in f32 (half the working memory)
        labels = betula_cluster.fit_predict(
            x.astype(np.float32),
            k,
            feature="spherical",
            method="kmeans",
            threshold=0.0,
            max_leaves=2000,
        )
    elif name == "betula-gmm":
        labels = betula_cluster.fit_predict(
            x, k, feature="diagonal", method="gmm", threshold=0.0, max_leaves=2000
        )
    elif name == "betula-gmm-full":
        # diagonal leaves + full-cov GMM: cheap O(d) build, but the *component* covariance is full
        # (built from the between-leaf spread), so it captures rotated clusters at a fraction of the
        # cost of full-covariance leaves.
        labels = betula_cluster.fit_predict(
            x, k, feature="diagonal", method="gmm-full", threshold=0.0, max_leaves=2000
        )
    elif name == "betula-gmm-full-par8":
        labels = betula_cluster.fit_predict(
            x, k, feature="diagonal", method="gmm-full", threshold=0.0, max_leaves=2000, n_jobs=8
        )
    elif name == "betula-ward":
        labels = betula_cluster.fit_predict(
            x, k, feature="diagonal", method="ward", threshold=0.0, max_leaves=2000
        )
    elif name == "betula-fd":
        # Frequent-Directions sketch: O(ℓ·d) per leaf instead of d×d (for high d)
        labels = betula_cluster.fit_predict(
            x, k, feature="fd", method="gmm-full", threshold=0.0, max_leaves=1000
        )
    elif name == "betula-hdbscan":
        labels = betula_cluster.fit_predict(
            x, method="hdbscan", threshold=0.0, max_leaves=2000, min_samples=10, min_cluster_size=25
        )
    elif name == "betula-kmeans-raw":
        # raw Euclidean on varying-norm vectors: magnitude dominates → fails (the point of normalize)
        labels = betula_cluster.fit_predict(
            x, k, feature="diagonal", method="kmeans", max_leaves=4000
        )
    elif name == "betula-kmeans-norm":
        # normalize=True ⇒ cluster by direction (cosine) on the unit sphere
        labels = betula_cluster.fit_predict(
            x, k, feature="diagonal", method="kmeans", max_leaves=4000, normalize=True
        )
    elif name == "sklearn-kmeans-norm":
        from sklearn.preprocessing import normalize as sk_normalize

        labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(sk_normalize(x))
    elif name == "sklearn-kmeans":
        labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(x)
    elif name == "sklearn-birch":
        # scale the radius threshold to the data so it does not explode subclusters in high-d
        thr = spec["std"] * (spec["d"] ** 0.5) * 0.5
        labels = Birch(n_clusters=k, threshold=thr).fit_predict(x)
    elif name == "sklearn-birch-naive":
        # the original benchmark setting: a fixed small radius explodes subclusters in high-d
        labels = Birch(n_clusters=k, threshold=0.5).fit_predict(x)
    elif name == "sklearn-gmm":
        labels = GaussianMixture(n_components=k, random_state=0).fit_predict(x)
    elif name == "sklearn-hdbscan":
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_samples=10, min_cluster_size=25).fit_predict(x)
    else:
        raise ValueError(name)
    return float(adjusted_rand_score(y, labels))


def worker(q, name, spec):
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (MEM_CAP, hard))
        x, y = gen(spec)
        t0 = time.perf_counter()
        score = run_method(name, x, y, spec)
        dt = time.perf_counter() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KiB -> MiB on Linux
        q.put(("ok", score, dt, rss))
    except MemoryError:
        q.put(("OOM", 0.0, 0.0, 0.0))
    except Exception as exc:
        q.put((f"err:{type(exc).__name__}", 0.0, 0.0, 0.0))


def measure(name, spec):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=worker, args=(q, name, spec))
    p.start()
    p.join(TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return ("timeout", 0.0, 0.0, 0.0)
    try:
        return q.get_nowait()
    except Exception:
        return ("killed", 0.0, 0.0, 0.0)


def bench(title, spec, methods):
    print(f"\n== {title} ==")
    print(f"  {'method':<18}{'ARI':>8}{'time(s)':>10}{'peakRSS(MB)':>13}{'  status'}")
    for name in methods:
        status, ari, dt, rss = measure(name, spec)
        if status == "ok":
            print(f"  {name:<18}{ari:>8.3f}{dt:>10.3f}{rss:>13.0f}{'  ok'}")
        else:
            print(f"  {name:<18}{'-':>8}{'-':>10}{'-':>13}{'  ' + status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all sample counts")
    args = ap.parse_args()
    s = args.scale

    bench(
        f"blobs  n={int(100_000 * s)}  k=10  d=10  (high-dim; memory matters)",
        dict(kind="blobs", n=int(100_000 * s), k=10, d=10, std=1.0),
        [
            "betula-kmeans",
            "betula-kmeans-par8",
            "betula-kmeans-f32",
            "betula-gmm",
            "betula-gmm-full",
            "betula-ward",
            "sklearn-kmeans",
            "sklearn-birch",
            "sklearn-birch-naive",
            "sklearn-gmm",
        ],
    )
    bench(
        f"blobs  n={int(20_000 * s)}  k=6  d=2",
        dict(kind="blobs", n=int(20_000 * s), k=6, d=2, std=1.0),
        [
            "betula-kmeans",
            "betula-gmm",
            "betula-ward",
            "sklearn-kmeans",
            "sklearn-birch",
            "sklearn-gmm",
        ],
    )
    bench(
        f"two-moons  n={int(20_000 * s)}  (non-convex)",
        dict(kind="moons", n=int(20_000 * s), k=2, d=2, std=1.0, noise=0.08),
        ["betula-hdbscan", "sklearn-kmeans", "sklearn-hdbscan"],
    )
    bench(
        f"parallel build  blobs  n={int(400_000 * s)}  k=16  d=16  (Phase-1 n_jobs speedup)",
        dict(kind="blobs", n=int(400_000 * s), k=16, d=16, std=1.0),
        ["betula-kmeans", "betula-kmeans-par8"],
    )
    bench(
        f"high-d full-cov  blobs  n={int(30_000 * s)}  k=8  d=64  (diagonal leaves + full-cov GMM)",
        dict(kind="blobs", n=int(30_000 * s), k=8, d=64, std=1.0),
        ["betula-gmm-full", "betula-gmm-full-par8", "sklearn-gmm"],
    )
    bench(
        f"embeddings  n={int(24_000 * s)}  k=8  d=64  (direction clusters, varying norm — cosine)",
        dict(kind="embed", n=int(24_000 * s), k=8, d=64, std=0.1),
        [
            "betula-kmeans-raw",
            "betula-kmeans-norm",
            "sklearn-kmeans",
            "sklearn-kmeans-norm",
        ],
    )
    print("\ndone")


if __name__ == "__main__":
    main()
