"""`arxiv-1m`: a labelled million-row embedding set, the timing campaign's large real input.

The first 1 000 000 rows of `lmcinnes/arxiv_archive` (2 477 303 × 768 at the pinned revision),
labelled by `primary_field`. The rows are arXiv abstracts as embeddings; the card names no model
and declares no licence. The cache is for local runs: nothing derived from the rows is committed
beyond the scores.

The set is 73 parquet shards of 33 936 rows; the build downloads only the 30 that hold the first
million rows (5.7 GB of 14.1 GB, into the Hugging Face cache) and reads two of their columns. The
embeddings are published as float64. The cache holds them as float32; rows published at unit norm
are re-normalised after the cast, so Euclidean neighbours are cosine neighbours, as in
`bench/embed_evoc.py`, and an all-zero row, which no direction can represent, is dropped.

Neither huggingface_hub nor pyarrow is a project dependency. Build the cache once:

    uv run --no-sync --with huggingface_hub --with pyarrow --with scikit-learn \\
        python bench/embed_arxiv.py

It writes `arxiv_1m.npz` (uncompressed, 3.1 GB: float32 embeddings compress by a few per cent and
decompress slower than they read) under scikit-learn's data home, beside `news_minilm.npz`, and
prints the label count and the class balance. `bench/rivals.py` loads it as `arxiv-1m`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = "lmcinnes/arxiv_archive"
REVISION = "17f683983592a43b8a40e1db3f4952f914895d79"  # read on 2026-11-18
SHARDS = 73
N_ROWS = 1_000_000
DIM = 768
# Published rows whose norm is this close to 1 count as unit-norm.
UNIT_TOL = 1e-6


def path() -> Path:
    from sklearn.datasets import get_data_home

    return Path(get_data_home()) / "betula_bench" / "arxiv_1m.npz"


def cached() -> tuple[np.ndarray, np.ndarray]:
    """`(X float32, y int64)`; `y` indexes the sorted `primary_field` names stored as `names`."""
    p = path()
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; build it with bench/embed_arxiv.py")
    with np.load(p) as d:
        return d["X"], d["y"]


def build() -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    x = np.empty((N_ROWS, DIM), dtype=np.float32)
    fields: list[str] = []
    norms = np.empty(N_ROWS)
    filled = 0
    for shard in range(SHARDS):
        if filled == N_ROWS:
            break
        name = f"data/train-{shard:05d}-of-{SHARDS:05d}.parquet"
        local = hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION)
        table = pq.read_table(local, columns=["embedding", "primary_field"])
        take = min(len(table), N_ROWS - filled)
        table = table.slice(0, take)
        field = table.column("primary_field")
        if field.null_count:
            raise ValueError(f"{name}: {field.null_count} rows have no primary_field")
        emb = table.column("embedding").combine_chunks()
        rows = emb.flatten().to_numpy().reshape(take, -1)
        if rows.shape[1] != DIM:
            raise ValueError(f"{name}: rows have {rows.shape[1]} coordinates, not {DIM}")
        norms[filled : filled + take] = np.linalg.norm(rows, axis=1)
        x[filled : filled + take] = rows
        fields.extend(field.to_pylist())
        filled += take
        print(f"{name}: {filled} rows", flush=True)

    zero = norms == 0
    unit = bool(np.all(np.abs(norms[~zero] - 1.0) <= UNIT_TOL))
    names, y = np.unique(np.asarray(fields), return_inverse=True)
    y = y.astype(np.int64)
    if zero.any():  # a boolean index copies all 3 GB, so only when there is a row to drop
        x, y = x[~zero], y[~zero]
    if unit:
        # Chunked, so the re-normalisation needs no second 3 GB array.
        for start in range(0, len(x), 65_536):
            block = x[start : start + 65_536]
            block /= np.linalg.norm(block, axis=1, keepdims=True)
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=x, y=y, names=names)
    spread = np.abs(norms[~zero] - 1.0).max()
    print(
        f"wrote {p}: {x.shape[0]} × {x.shape[1]}, {len(names)} classes, {zero.sum()} all-zero rows"
        f" dropped; published norms within {spread:.1e} of 1"
        + (", re-normalised after the float32 cast" if unit else ", not unit-norm: kept as cast")
    )
    counts = np.bincount(y, minlength=len(names))
    for i in np.argsort(-counts):
        print(f"  {counts[i]:>8}  {counts[i] / len(y):6.2%}  {names[i]}")


if __name__ == "__main__":
    build()
