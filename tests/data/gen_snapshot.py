"""Write a persistence snapshot plus the attributes it should reproduce, for
`test_a_snapshot_written_by_0_6_0_still_loads`.

Run it against a *released* wheel, from a directory outside this repository:

    cp tests/data/gen_snapshot.py /tmp && cd /tmp
    uv run --no-project --with 'betula-cluster==X.Y.Z' --with numpy \
        python gen_snapshot.py vN_X.Y.Z.betula vN_X.Y.Z.json

Outside, because `--no-project` only skips *syncing* the project: run it inside the repository and
uv still resolves the project's own `.venv`, so the "released" snapshot is the local build. The
test guards against exactly that, by checking the blob for a field 0.6.0 could not have written.
"""

import json
import sys

import betula_cluster
import numpy as np

centres = [(0.0, 0.0), (6.0, 0.0), (0.0, 6.0)]
x = np.array(
    [[cx + i * 0.1, cy + j * 0.1] for cx, cy in centres for i in range(10) for j in range(10)],
    dtype=np.float64,
)

est = betula_cluster.Betula(
    n_clusters=3, feature="diagonal", method="gmm", threshold=0.05, max_leaves=50, seed=1
)
est.fit(x)
est.save(sys.argv[1])

with open(sys.argv[2], "w") as fh:
    json.dump(
        {
            "version": betula_cluster.__version__,
            "n_clusters_": int(est.n_clusters_),
            "n_leaves_": int(est.n_leaves_),
            "labels": np.asarray(est.predict(x)).tolist(),
            "cluster_sizes_": np.asarray(est.cluster_sizes_).tolist(),
            "cluster_centers_": np.asarray(est.cluster_centers_).tolist(),
        },
        fh,
        indent=1,
    )
print("wrote", sys.argv[1], sys.argv[2], "with", betula_cluster.__version__)
