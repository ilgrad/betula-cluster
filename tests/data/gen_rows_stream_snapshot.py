"""Write a `leaf_graph="rows"` stream saved mid-stream by a build that kept no counts, for
`test_a_stream_file_from_a_build_before_the_counts_loads_and_refuses_the_head`.

No release has the rows graph, so the file comes from a build of the commit before the counts
(`a2c42d8`), from a directory outside this repository:

    cp tests/data/gen_rows_stream_snapshot.py /tmp && cd /tmp
    PYTHONPATH=<a2c42d8 build>/python uv run --no-project --with numpy \
        python gen_rows_stream_snapshot.py v2_rows_stream_a2c42d8.betula

The test checks the file for the field that build could not have written.
"""

import sys

import betula_cluster
import numpy as np

x = np.random.default_rng(0).normal(size=(200, 2))
est = betula_cluster.Betula(method="leiden", leaf_graph="rows", max_leaves=50, seed=0)
est.partial_fit(x)
est.save(sys.argv[1])
print("wrote", sys.argv[1], "with", betula_cluster.__version__)
