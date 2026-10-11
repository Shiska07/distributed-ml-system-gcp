"""
Tests for spark_jobs/build_training_dataset.py helpers.
The full job (join, filters, manifest) is checked on the real run via the manifest numbers.
"""
from datetime import date, datetime

import pandas as pd

from spark_jobs.build_training_dataset import (
    LABELED_COLUMNS, add_label, compute_split_dates, newest_per_key,
)


def test_newest_per_key_keeps_one_row_from_the_newest_run(spark):
    df = spark.createDataFrame(
        [("a", "2022-01-01_2022-01-02_20260101-000000", 1.0),
         ("a", "2022-01-01_2022-01-02_20260201-000000", 2.0),   # newer rerun → wins
         ("b", "2022-01-01_2022-01-02_20260101-000000", 3.0)],
        ["unique_key", "run_id", "value"])
    out = {r["unique_key"]: r["value"] for r in newest_per_key(df, "run_id").collect()}
    assert out == {"a": 2.0, "b": 3.0}


def test_split_dates_are_in_order_and_inside_the_data(spark):
    days = [(datetime(2022, 1, d, 12),) for d in range(1, 31)]          # one trip per day
    df = spark.createDataFrame(days, ["trip_start_ts"])
    val_start, test_start = compute_split_dates(df, val_frac=0.1, test_frac=0.1)
    assert date(2022, 1, 1) < val_start < test_start <= date(2022, 1, 30)


def test_add_label_returns_the_schema_columns_and_labels():
    pdf = pd.DataFrame({
        "unique_key": ["a", "b"], "feature_run_id": ["r1", "r1"],
        "trip_start_ts": pd.to_datetime(["2022-01-01", "2022-01-02"]),
        "fare": [20.0, 20.0], "tips": [5.0, 2.0],                       # 25% → 1, 10% → 0
        "split": ["train", "train"], "outcome_run_id": ["o1", "o1"],     # extra column must be dropped
    })
    out = next(add_label(iter([pdf])))
    assert list(out.columns) == [name for name, _ in LABELED_COLUMNS]
    assert list(out["label"]) == [1, 0]
    