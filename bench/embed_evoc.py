"""The two `evoc_bench_*` sets of the EVoC benchmark: embedding boards of `bench/rivals.py`.

- `news-mpnet`: `lmcinnes/evoc_bench_20newsgroups`, 20 newsgroups under a second sentence model,
  18 119 × 768, 20 classes. The card names no model; its first rows equal those of
  `lmcinnes/20newsgroups_embedded` to about 1e-7, whose card names `all-mpnet-base-v2`.
- `cifar100-clip`: `lmcinnes/evoc_bench_cifar100`, CIFAR-100 as CLIP image embeddings,
  60 000 × 1 280, 100 classes. The EVoC benchmark page says "CLIP" and names no variant. Its last
  row (class 0) is all zeros, which no direction can represent, so the cache holds 59 999 rows.

Both are published as unit-norm float64. The cache holds them as float32, re-normalised after the
cast, so Euclidean neighbours are cosine neighbours, as in `bench/embed_news.py`. Neither card
declares a licence: the cache is for local runs, and nothing derived from the rows is committed
beyond the scores. The download is pinned to the dataset revisions read on 2026-11-15.

Neither huggingface_hub nor pyarrow is a project dependency. Build the caches once:

    uv run --no-sync --with huggingface_hub --with pyarrow --with scikit-learn \\
        python bench/embed_evoc.py [news-mpnet|cifar100-clip]

It writes `evoc_<name>.npz` under scikit-learn's data home, beside `news_minilm.npz`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SETS = {
    "news-mpnet": (
        "lmcinnes/evoc_bench_20newsgroups",
        "fb0d6029218aeb288d02405094ac53a4a1f3dd73",
        ["data/train-00000-of-00001.parquet"],
    ),
    "cifar100-clip": (
        "lmcinnes/evoc_bench_cifar100",
        "f54bfc106190cf921415ea57c171883dbcf3d104",
        ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"],
    ),
}


def path(name: str) -> Path:
    from sklearn.datasets import get_data_home

    return Path(get_data_home()) / "betula_bench" / f"evoc_{name}.npz"


def cached(name: str) -> tuple[np.ndarray, np.ndarray]:
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; build it with bench/embed_evoc.py {name}")
    with np.load(p) as d:
        return d["X"], d["y"]


def build(name: str) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, revision, files = SETS[name]
    xs, ys = [], []
    for f in files:
        local = hf_hub_download(repo, f, repo_type="dataset", revision=revision)
        table = pq.read_table(local, columns=["embedding", "target"])
        emb = table.column("embedding").combine_chunks()
        xs.append(emb.flatten().to_numpy().reshape(len(emb), -1))
        ys.append(table.column("target").to_numpy())
    x = np.concatenate(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.int64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    zero = norm[:, 0] == 0
    x, y = x[~zero] / norm[~zero], y[~zero]
    p = path(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=x, y=y)
    print(
        f"wrote {p}: {x.shape[0]} × {x.shape[1]}, {len(np.unique(y))} classes,"
        f" {zero.sum()} all-zero rows dropped"
    )


if __name__ == "__main__":
    for name in sys.argv[1:] or list(SETS):
        build(name)
