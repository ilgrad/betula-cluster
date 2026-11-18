# Comparison with other clustering libraries

This page answers two questions: which method to use for a task, betula's or another library's, and
what each library can do. Every number comes from
[`bench/RESULTS.md`](https://github.com/ilgrad/betula-cluster/blob/main/bench/RESULTS.md), whose
*census* section states the rules (seeds, the tie rule, one betula head per contest) and holds the
tables. Where a rival wins, this page says to use the rival.

## Which method for which task

ARI is the median of seeds 0, 1 and 2 against the labels; RSS is the peak resident memory of the
whole process. A difference smaller than the wider seed range of the two is a tie.

| task | use | the measurement behind it |
|---|---|---|
| Sentence embeddings, count given | **UMAP + k-means** | 20 newsgroups under MiniLM (17 901 posts) and under mpnet (18 119): 0.451 and 0.472 against betula `leiden`, `leaf_graph="rows"`, `resolution="auto"`, 0.374 and 0.433 (`vmf` reads 0.382 and 0.400). Under mpnet, PaCMAP and openTSNE + k-means tie betula, and it beats scikit-learn's k-means (0.404). betula's process is 243 and 276 MB against 640 and 694 MB, and has no layout step. |
| Sentence embeddings, count unknown | **the CluProp recipe** (union kNN-8 + Leiden), **PhenoGraph** or **kNN + Leiden** | MiniLM: 0.419, 0.395 and 0.386; mpnet: 0.426, 0.429 and 0.397; against `leiden` with `leaf_graph="rows"` at 0.330 and 0.367, which finds 13–14 communities for 20 classes where the recipe finds 25–30. CoreSPECT (`cplearn`) reads 0.360 and 0.354. BERTopic's default UMAP + HDBSCAN reads 0.086 and 0.081 and marks a third of the posts as noise. |
| Image embeddings (CIFAR-100 under CLIP, 59 999 rows, 100 classes), count given | **UMAP + k-means** or **openTSNE + k-means** | 0.721 and 0.696 against betula `leiden`, `leaf_graph="rows"`, `resolution="auto"`, 0.666, which ties UMAP 0.6 + k-means (0.667) and scikit-learn's k-means (0.637) and beats PaCMAP + k-means (0.634). betula uses 690 MB against 1 098–1 649 MB. |
| Image embeddings (CIFAR-100 under CLIP), count unknown | **CoreSPECT** (`cplearn`) or **kNN + Leiden** | 0.626 and 0.561 against `leiden` with `leaf_graph="rows"` 0.370, which finds 51 communities for 100 classes; PhenoGraph reads 0.529. CoreSPECT peaks at 1 556 MB against betula's 689 MB. EVoC reads 0.198 with 21 % of the rows marked noise (0.835 on the rows it clusters); DelTriC runs out of memory under a 13 GB cap. |
| Images (raw MNIST), count unknown, 70 000 rows | **UMAP + HDBSCAN** (the UMAP docs' recipe) or **EVoC** | 0.920 and 0.899 against `leiden` with `leaf_graph="rows"` 0.777 at 14 000 leaves (0.788 at 20 000). The CluProp recipe, the same graph on rows instead of leaves, reads 0.865 in 940 MB. betula uses 563 MB against 1 096 and 1 408 MB. |
| Images (raw MNIST), count unknown, 20 000 rows | **the CluProp recipe** (union kNN-8 + Leiden) or **EVoC** | 0.854 and 0.815, level by EVoC's seed range, against `leiden` with `leaf_graph="rows"` 0.762, which is level with EVoC and PhenoGraph (0.766) and ahead of UMAP + HDBSCAN (0.536). betula uses 292 MB against 608 and 1 028 MB. |
| Images (raw MNIST), count given | **betula** `leiden`, `resolution="auto"` | 0.835 at 70 000 rows and 14 000 leaves (seeds 0.806–0.884): level with UMAP 0.6 + k-means (0.804), PaCMAP + k-means (0.795) and UMAP 0.5 + k-means (0.769), ahead of openTSNE + k-means (0.745). 20 000 leaves read 0.881 for 58 MB more, 8 000 leaves 0.788. |
| Low-dimensional data, density, count unknown | **hdbscan**; betula `gmm-full` with `n_clusters=0` if the clusters are Gaussian | Mean ARI over seven fixtures 0.652 against betula's `hdbscan` head 0.616. DelTriC reads 0.657, all of its lead on `digits` (0.727 against 0.087), and trails `hdbscan` on five of the other six fixtures. On blobs, aniso and `varied` the `gmm-full` BIC sweep reads 0.863, 0.902 and 0.757 against `hdbscan`'s 0.472, 0.500 and 0.505; on rings it reads 0.11 and on `digits` 0.152, so it is no general choice. |
| Low-dimensional data, count given | **betula** `leiden`, `leaf_graph="rows"`, `resolution="auto"`, or **Genie** (genieclust) | Mean ARI over seven fixtures 0.838 against 0.839, a tie. betula wins `digits` (0.840 against 0.643) and loses aniso (0.545 against 0.750; lumbermark 0.843). Merging the `gmm-full` BIC fit down to the count ([a recipe](USAGE.md#merging-an-over-fitted-mixture-to-a-known-count)) reads a mean of 0.871 and 0.902 on aniso, but 0.575 on `digits`. |
| Benchmark datasets, count given | **betula** `leiden`, `resolution="auto"`, or **Lumbermark** / **GIc** | Gagolewski's suite, 81 datasets: mean accuracy 0.774 against 0.786 and 0.784, a tie. `leiden` cannot reach the count on 3 of the 81. |
| Mixed numeric and categorical columns | either: betula `KPrototypes` or kmodes | 0.028 against 0.022 on mixed covtype: a tie, and both near 0. |
| A stream of rows | **betula** `partial_fit` | After each chunk the labels equal a refit on every row seen (75 of 75 checkpoints, 20 newsgroups). UMAP + k-means refitted at each checkpoint is level at 20 and 40 % of the stream and ahead from 60 % on (0.429–0.448 against 0.383–0.392), at the cost of a refit each time. Every rival that learns in chunks reads below betula at every checkpoint: BICO 0.345–0.364, scikit-learn's `MiniBatchKMeans` 0.320–0.347 and `Birch` 0.299–0.349, BERTopic's online recipe 0.21–0.22, River's `DenStream` 0.18 at best. |
| Input larger than memory | **betula**, fed in chunks | 500 000 to 10 M rows (20-D) in 55 MB, against 0.41–5.0 GB for an in-core scikit-learn `KMeans`; betula holding the whole array needs 0.13–1.65 GB. scikit-learn's `MiniBatchKMeans` and BICO stay flat too, at 180 and 175 MB, most of it their imports. `Birch` at its defaults keeps one subcluster per row of that data and did not finish 500 000 rows in the harness's time cap. |
| k-means on a million rows, speed first | **FAISS** | The two sides do different jobs. FAISS fits its centres on 2 048 of the rows (256 per centre) and labels every row with one search: median ARI 0.835 over eleven seeds at its defaults, 1.000 seeded with k-means++. betula summarises every row into at most 2 000 leaves, fits the centres on the leaves and labels every row: 1.000. Work per fit at one thread: FAISS 0.73 G instructions, betula 4.92 G (4.01 G at `n_shards=8`). Given FAISS's shortcut, a fit on a seeded 2 048-row sample and `predict` on every row, betula also reads 1.000, at 0.96 G. A sample is not free: a cluster of 0.1 % of the rows is found by betula's full fit in 21 of 22 runs and from a 2 048-row sample in 7. The one time is from 2026-10-20: FAISS's defaults 0.07 s against betula's full fit 0.42 s, both multi-threaded; the other arms are not yet timed. |
| Outliers inside clusters of unequal density | **betula** `outlier_scores` | ROC-AUC / average precision 1.000 / 0.998 against IsolationForest 0.887 / 0.175, LOF 0.993 / 0.915 and PyOD's cluster-based CBLOF 0.557 / 0.055 (median of seeds 0/1/2, `bench/results_outliers.csv`). |
| Global outliers, or sheared clusters | **IsolationForest**, or PyOD's `KNN` | 1.000 / 1.000 for both on global outliers; on sheared clusters `KNN` 1.000 / 0.999 and IsolationForest 0.980 / 0.698, against betula 0.990 / 0.943 and 0.711 / 0.154 (`metric="mahalanobis"`). Same CSV as the row above. |
| A mixture of stationary signals | **betula** `gmm-toeplitz` | 0.764 on an AR mixture against tslearn's k-Shape 0.057 and a diagonal GMM 0.018. |
| A GPU is available | the same choices | cuML's k-means, UMAP + k-means and DBSCAN, FAISS's k-means and fastkmeans's PyTorch path find what their CPU versions find; cuML's HDBSCAN on raw MNIST calls 83–85 % of the rows noise (ARI 0.046–0.052), and fastkmeans's default Triton path did not finish on this card (*GPU rivals* in the census). |

## What each library can do

Each rival cell cites the release's documentation or source. "Not documented" means that release
was read and no such feature was found. Each betula cell names a test, a script or a bench row.

| | betula-cluster `main` (1.5.0 and `[Unreleased]`) | umap-learn 0.6.0 + hdbscan 0.8.44 | EVoC 0.3.2 | fast_hdbscan 0.3.4 | scikit-learn 1.9.1 | FAISS 1.16.0 |
|---|---|---|---|---|---|---|
| **Learn from a stream** | `partial_fit` on chunks, then `partial_fit()` (`test_streaming_partial_fit_matches_oneshot`) | UMAP: [`update(X)`][u-update] retrains with the new rows; hdbscan: not documented | not documented | not documented | `partial_fit` on `MiniBatchKMeans` and [`Birch`][s-birch]; not on `HDBSCAN` | not documented |
| **Input larger than memory** | chunked `partial_fit`; 55 MB at 0.5–10 M rows (*Memory — streaming stays bounded*) | UMAP: [`low_memory`][u-lowmem] covers the neighbour search only; hdbscan: not documented | its [example][e-chunks] refits each chunk | not documented | `MiniBatchKMeans`, `Birch` ([incremental estimators][s-scaling]) | k-means [trains on a subsample][x-subsample] |
| **Sample weights** | no | no ([UMAP][u-fit], [hdbscan][h-fit]) | no ([`fit`][e-fit] takes `X` only) | [`sample_weight`][f-weight], [Euclidean only][f-weight-metric] | `KMeans`, `MiniBatchKMeans`; not `Birch` or `HDBSCAN` | [`train(x, weights=)`][x-weights] |
| **Label new rows** | `predict` (`test_estimator_predict_on_new_points`) | UMAP: [`transform`][u-transform]; hdbscan: [`approximate_predict`][h-predict], with `prediction_data=True` | not documented | not documented | `predict` on `KMeans`, `MiniBatchKMeans`, `Birch`; `HDBSCAN` has none ([transductive][s-transductive]) | [`Kmeans.assign`][x-assign] |
| **Same labels at any thread count** | every head, compared by digest at 1 and 8 threads (`scripts/thread_determinism.py`) | UMAP: a seed [forces one thread][u-seed]; hdbscan: no seed | a seed selects a deterministic mode, but `approx_n_clusters` [hard-codes `reproducible=False`][e-repro] | [`reproducible=False`][f-repro] by default | `KMeans`, `MiniBatchKMeans` [tested at 1 and 2 threads][s-threads] | [bitwise at the same thread count][x-threads] |
| **Save a fitted model** | `save` / `load` (gzip-framed CBOR) and pickle (`test_save_load_roundtrip`, `test_pickle_roundtrip`); MNIST 70 000 × 784 at 2 000 spherical leaves: 5.3 MB ([Saving and loading a model](USAGE.md#saving-and-loading-a-model)) | UMAP: [pickle][u-pickle]; hdbscan: not documented | not documented | not documented | [pickle, joblib, skops, ONNX][s-persist] | an index serialises; the `Kmeans` object: not documented |
| **Hard dependencies** (PyPI) | numpy | numpy, scipy, scikit-learn, numba, pynndescent, tqdm; joblib | numpy, scikit-learn, numba, tqdm | numpy, numba, scikit-learn | numpy, scipy, joblib, narwhals, threadpoolctl | numpy, packaging (`faiss-cpu`) |
| **Free-threaded CPython wheels** | 3.14t, and `abi3t` for 3.15t onward | UMAP is pure Python on numba 0.68.1, which has 3.14t and no 3.15t; hdbscan has no 3.14t wheel | as UMAP (numba) | as UMAP (numba) | 3.14t and 3.15t | none (`abi3` and `cp314`) |
| **Licence** | MIT | BSD-3-Clause; BSD-3-Clause | BSD-2-Clause | BSD-2-Clause | BSD-3-Clause | MIT |
| **Noise and the count** | noise (−1) from the `hdbscan` head; `leiden` and `hdbscan` find the count, `n_clusters=0` sweeps it for `gmm` by BIC | UMAP embeds only; hdbscan marks noise and finds the count | noise, count found | noise, count found | `HDBSCAN` marks noise and finds the count; `KMeans` needs it | no noise; the count is required |

The wheel and dependency rows were read from PyPI on 2026-11-14; the rest at each release's tag.

[u-update]: https://github.com/lmcinnes/umap/blob/release-0.6.0/umap/umap_.py#L4751
[u-lowmem]: https://github.com/lmcinnes/umap/blob/release-0.6.0/doc/faq.rst#L89
[u-fit]: https://github.com/lmcinnes/umap/blob/release-0.6.0/umap/umap_.py#L3506
[u-transform]: https://github.com/lmcinnes/umap/blob/release-0.6.0/umap/umap_.py#L4178
[u-seed]: https://github.com/lmcinnes/umap/blob/release-0.6.0/umap/umap_.py#L3100
[u-pickle]: https://github.com/lmcinnes/umap/blob/release-0.6.0/doc/parametric_umap.rst#L76
[h-fit]: https://github.com/scikit-learn-contrib/hdbscan/blob/release-0.8.44/hdbscan/hdbscan_.py#L1202
[h-predict]: https://github.com/scikit-learn-contrib/hdbscan/blob/release-0.8.44/hdbscan/prediction.py#L416
[e-chunks]: https://github.com/TutteInstitute/evoc/blob/release-0.3.2/doc/source/examples.rst#L181
[e-fit]: https://github.com/TutteInstitute/evoc/blob/release-0.3.2/evoc/clustering.py#L651
[e-repro]: https://github.com/TutteInstitute/evoc/blob/release-0.3.2/evoc/clustering_utilities.py#L151
[f-weight]: https://github.com/TutteInstitute/fast_hdbscan/blob/release-0.3.4/fast_hdbscan/hdbscan.py#L689
[f-weight-metric]: https://github.com/TutteInstitute/fast_hdbscan/blob/release-0.3.4/fast_hdbscan/hdbscan.py#L452
[f-repro]: https://github.com/TutteInstitute/fast_hdbscan/blob/release-0.3.4/fast_hdbscan/hdbscan.py#L621
[s-birch]: https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/cluster/_birch.py#L611
[s-scaling]: https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/computing/scaling_strategies.rst#L59
[s-transductive]: https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/clustering.rst#L139
[s-threads]: https://github.com/scikit-learn/scikit-learn/blob/1.9.1/sklearn/cluster/tests/test_k_means.py#L979
[s-persist]: https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/model_persistence.rst#L7
[x-subsample]: https://github.com/facebookresearch/faiss/blob/v1.16.0/faiss/Clustering.h#L45
[x-weights]: https://github.com/facebookresearch/faiss/blob/v1.16.0/faiss/python/extra_wrappers.py#L586
[x-assign]: https://github.com/facebookresearch/faiss/blob/v1.16.0/faiss/python/extra_wrappers.py#L648
[x-threads]: https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls

## Every verdict, rival by rival

Generated from the bench CSVs by `bench/scoreboard.py --claims`; CI fails when this block and the
CSVs disagree. The betula side of each cell is the one head its contest chose. On GitHub the block
is not inlined: read it in `bench/RESULTS.md`, *Every verdict, rival by rival*.

--8<-- "bench/RESULTS.md:claims"
