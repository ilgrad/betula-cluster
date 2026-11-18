# Scoreboard fixture: the speed axis read from the timing campaign

Made-up numbers in the format of `bench/results_rivals.csv` and `bench/results_timings.csv`, so
that the reader of the speed axis can be checked before the campaign writes a row. What each row
tests:

- `sklearn-kmeans` round 3 is `timing_valid=False` with a 99 s second fit: it must not move the
  median (4.00 s warm, 6.10 s cold).
- The `threads` stage times `betula-leiden-rows-auto` at 30 s on one thread: it must stay out of
  the census, which reads the `rivals` stage only.
- `opentsne+kmeans` did not finish: it gets no speed cell, and keeps its quality and memory cells.
- Ratios against the champion's 10 s: `sklearn-kmeans` 2.5× faster (loss), `umap-0.6+kmeans` 1.3×
  slower (level, so a tie), `pacmap+kmeans` 1.8× slower (win), `evoc` 2× faster (loss).

    uv run --no-sync --with pandas python bench/scoreboard.py --data bench/fixtures/scoreboard \
        --claims-check bench/fixtures/scoreboard/claims.md

<!-- claims:begin -->
- **evoc** — 2 win · 0 tie · 1 loss
  - win, memory, `mnist/20000/no-k` (rivals): betula 500 MB (`betula-leiden-rows`) against 800 MB
  - win, quality, `mnist/20000/no-k` (rivals): betula 0.750 ARI (`betula-leiden-rows`) against 0.700 ARI
  - loss, speed, `mnist/20000/no-k` (rivals): betula 10.00 s warm, 12.00 s cold (`betula-leiden-rows`) against 5.00 s warm, 9.00 s cold
- **opentsne+kmeans** — 2 win · 0 tie · 0 loss
  - win, memory, `mnist/20000/told-k` (rivals): betula 500 MB (`betula-leiden-rows-auto`) against 1200 MB
  - win, quality, `mnist/20000/told-k` (rivals): betula 0.800 ARI (`betula-leiden-rows-auto`) against 0.600 ARI
- **pacmap+kmeans** — 3 win · 0 tie · 0 loss
  - win, memory, `mnist/20000/told-k` (rivals): betula 500 MB (`betula-leiden-rows-auto`) against 700 MB
  - win, quality, `mnist/20000/told-k` (rivals): betula 0.800 ARI (`betula-leiden-rows-auto`) against 0.650 ARI
  - win, speed, `mnist/20000/told-k` (rivals): betula 10.00 s warm, 12.00 s cold (`betula-leiden-rows-auto`) against 18.00 s warm, 20.00 s cold
- **sklearn-kmeans** — 1 win · 0 tie · 2 loss
  - win, quality, `mnist/20000/told-k` (rivals): betula 0.800 ARI (`betula-leiden-rows-auto`) against 0.360 ARI
  - loss, memory, `mnist/20000/told-k` (rivals): betula 500 MB (`betula-leiden-rows-auto`) against 360 MB
  - loss, speed, `mnist/20000/told-k` (rivals): betula 10.00 s warm, 12.00 s cold (`betula-leiden-rows-auto`) against 4.00 s warm, 6.10 s cold
- **umap-0.6+kmeans** — 2 win · 1 tie · 0 loss
  - win, memory, `mnist/20000/told-k` (rivals): betula 500 MB (`betula-leiden-rows-auto`) against 900 MB
  - win, quality, `mnist/20000/told-k` (rivals): betula 0.800 ARI (`betula-leiden-rows-auto`) against 0.700 ARI
  - tie, speed, `mnist/20000/told-k` (rivals): betula 10.00 s warm, 12.00 s cold (`betula-leiden-rows-auto`) against 13.00 s warm, 16.00 s cold
<!-- claims:end -->
