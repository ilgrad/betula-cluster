"""20 newsgroups as sentence embeddings: the text-embedding fixture of `bench/rivals.py`.

Headers, footers and quoted replies are removed, as scikit-learn's documentation advises for a
topic task; posts shorter than 50 characters are dropped, and each post is cut at 2 000 characters.
The model is `sentence-transformers/all-MiniLM-L6-v2`, BERTopic's default, run through fastembed
(ONNX, no torch). Rows are L2-normalised, so Euclidean neighbours are cosine neighbours.

Neither fastembed nor the model is a project dependency. Build the cache once:

    uv run --no-sync --with fastembed --with scikit-learn python bench/embed_news.py

It writes `news_minilm.npz` (17 901 × 384, 20 classes on 2026-11-13) under scikit-learn's data
home, beside the 20 newsgroups download it is made from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def path() -> Path:
    from sklearn.datasets import get_data_home

    return Path(get_data_home()) / "betula_bench" / "news_minilm.npz"


def cached() -> tuple[np.ndarray, np.ndarray]:
    p = path()
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; build it with bench/embed_news.py")
    with np.load(p) as d:
        return d["X"], d["y"]


def build() -> None:
    from fastembed import TextEmbedding
    from sklearn.datasets import fetch_20newsgroups

    d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    keep = [i for i, t in enumerate(d.data) if len(t.strip()) >= 50]
    texts = [d.data[i][:2000] for i in keep]
    y = np.asarray(d.target)[keep]
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    x = np.asarray(list(model.embed(texts, batch_size=64)), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=x, y=y)
    print(f"wrote {p}: {x.shape[0]} × {x.shape[1]}, {len(np.unique(y))} classes")


if __name__ == "__main__":
    build()
