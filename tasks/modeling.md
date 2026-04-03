# Modeling Tasks

## Objective

Build one strong tabular baseline and one PyTorch sequence model.

## Tasks

1. Train multinomial logistic regression.
2. Train `XGBoost`.
3. Compare baseline performance across horizons.
4. Train the PyTorch `LSTM` on possession-prefix sequences.
5. Check whether sequence order improves on the engineered-feature baseline.

## Main Files

- `src/full_press_ml/models/baselines.py`
- `src/full_press_ml/models/lstm_model.py`
- `src/full_press_ml/training/train_baseline.py`
- `src/full_press_ml/training/train_lstm.py`

