"""Five small labelled text corpora as sentence embeddings: the guard set for neighbour smoothing.

The corpora are those of the archive that ships with Fettal, Labiod and Nadif, "More
Discriminative Sentence Embeddings via Semantic Graph Smoothing" (EACL 2024), read from `data.zip`
of `github.com/chakib401/smoothing_sentence_embeddings` at its last commit, `data/text/<name>.csv`,
column `text` (the other four corpora of that archive hold index lines, not documents):

| name | rows | classes |
|---|---|---|
| `ag_news` | 8 000 | 4 |
| `bbc` | 2 225 | 5 |
| `classic3` | 3 891 | 3 |
| `classic4` | 7 095 | 4 |
| `dbpedia` | 12 000 | 14 |

The archive has no licence file. The documents are read in memory and never written; the cache
holds only the vectors and the labels, for local runs, and nothing derived from it is committed
beyond the scores. Every row is kept. The embedding is the one `bench/embed_news.py` makes: each
text cut at 2 000 characters, `sentence-transformers/all-MiniLM-L6-v2` through fastembed (ONNX,
no torch), rows L2-normalised.

Neither fastembed nor the model is a project dependency. Build the caches once:

    uv run --no-sync --with fastembed --with scikit-learn python bench/embed_corpora.py

It writes `corpus_<name>_minilm.npz` under scikit-learn's data home, beside `news_minilm.npz`.
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

NAMES = ("ag_news", "bbc", "classic3", "classic4", "dbpedia")
ARCHIVE = (
    "https://raw.githubusercontent.com/chakib401/smoothing_sentence_embeddings/"
    "8d897f4e56982be83a2567fc34c9f21a7c640d19/data.zip"
)


def path(name: str) -> Path:
    from sklearn.datasets import get_data_home

    return Path(get_data_home()) / "betula_bench" / f"corpus_{name}_minilm.npz"


def cached(name: str) -> tuple[np.ndarray, np.ndarray]:
    p = path(name)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; build it with bench/embed_corpora.py")
    with np.load(p) as d:
        return d["X"], d["y"]


def build() -> None:
    from fastembed import TextEmbedding

    with urllib.request.urlopen(ARCHIVE, timeout=120) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))
    csv.field_size_limit(sys.maxsize)
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    for name in NAMES:
        with archive.open(f"data/text/{name}.csv") as f:
            rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
        texts = [row["text"][:2000] for row in rows]
        _, y = np.unique([row["label"] for row in rows], return_inverse=True)
        x = np.asarray(list(model.embed(texts, batch_size=64)), dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        p = path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, X=x, y=y)
        print(f"wrote {p}: {x.shape[0]} × {x.shape[1]}, {len(np.unique(y))} classes")


if __name__ == "__main__":
    build()
