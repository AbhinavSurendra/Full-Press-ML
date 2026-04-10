# This is a running report of our findings so far

## Initial Training Data Creation and Benchmarks

The tracking-data pipeline is now running end to end on the tiny dataset. It builds normalized `events.csv`, `frames.csv`, and `possessions.csv`, attaches possession labels to frames, and supports both a tabular baseline path and an LSTM sequence path.

Current tiny-dataset counts:

- `995` total possessions
- `943` usable possessions
- `996,784` frame rows

Initial smoke-test model results on the held-out `test` split:

- Raw possession-metadata logistic baseline: `0.3788` accuracy
- Aggregated-frame logistic baseline: `0.5354` accuracy
- One-epoch LSTM on frame sequences: `0.3939` accuracy

The main conclusion from this first pass is that the pipeline is functional, and simple frame aggregation already outperforms the minimally trained sequence model. The LSTM path is now runnable, but it is still undertrained and not yet using a stronger feature/normalization setup, so its current result should be treated as a wiring check rather than a meaningful benchmark.

Notable class-level behavior:

- The aggregated-frame baseline performs best on `missed_shot` and `made_2`.
- `made_3` remains difficult with the current tiny split and coarse features.
- The one-epoch LSTM mostly collapses toward `missed_shot`, which suggests more training, better balancing, and stronger input preprocessing are still needed.
