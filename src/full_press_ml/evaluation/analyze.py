"""Post-training analysis helpers."""

from __future__ import annotations

import pandas as pd


def summarize_class_balance(df: pd.DataFrame, label_column: str) -> pd.Series:
    return df[label_column].value_counts(normalize=True).sort_index()


def horizon_slice(df: pd.DataFrame, horizon_seconds: int) -> pd.DataFrame:
    if "seconds_elapsed" not in df.columns:
        raise ValueError("Expected a seconds_elapsed column for horizon slicing.")
    return df[df["seconds_elapsed"] <= horizon_seconds].copy()

