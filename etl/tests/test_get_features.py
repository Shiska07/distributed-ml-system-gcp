"""
Tests for spark_jobs/get_features.py: the Spark output schema must match common/.
Run: pytest etl -q
"""
import os
import sys
import pytest
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.api import types as ptypes

from spark_jobs.get_features import FEATURES_COLUMNS, parse_run_path, to_features
from taxi_features.features import FEATURE_COLUMNS, ID_COLUMNS

# how a Spark type must look in pandas
TYPE_CHECKS = {
    "string":    ptypes.is_object_dtype,
    "timestamp": ptypes.is_datetime64_any_dtype,
    "bigint":    ptypes.is_integer_dtype,
    "double":    ptypes.is_float_dtype,
}


@pytest.fixture
def raw():
    return pd.DataFrame({
        "unique_key": ["a", "b"], "taxi_id": ["t1", "t2"],
        "trip_start_ts": [datetime(2022, 1, 7, 18, 30), datetime(2022, 1, 8, 2, 0)],
        "trip_seconds": [900, 0], "trip_miles": [3.0, 0.0], "fare": [12.0, 3.25],
        "pickup_community_area": [8, None], "dropoff_community_area": [32, None],
        "tolls": [0.0, None], "extras": [1.0, None], "company": ["Flash Cab", None],
    })


def test_schema_lists_the_same_columns_as_common():
    assert [name for name, _ in FEATURES_COLUMNS] == ID_COLUMNS + FEATURE_COLUMNS


def test_pandas_output_types_match_the_spark_schema(raw):
    out = next(to_features(iter([raw])))           # same function Spark calls
    for name, spark_type in FEATURES_COLUMNS:
        assert TYPE_CHECKS[spark_type](out[name]), f"{name}: {out[name].dtype} is not {spark_type}"


def test_parse_run_path():
    path = "gs://lake/silver/trips/source=live/run_id=2022-04-01_2022-04-08_20261011-120000/"
    assert parse_run_path(path) == ("live", "2022-04-01_2022-04-08_20261011-120000")


def test_parse_run_path_rejects_wrong_layout():
    with pytest.raises(ValueError):
        parse_run_path("gs://lake/silver/trips/run_id=2022-04-01_2022-04-08_x")


def test_spark_end_to_end_local(raw):
    """Runs the real mapInPandas step on a local Spark (slower, ~20 s)."""
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession
    from spark_jobs.get_features import FEATURES_SCHEMA

    # Spark's Python workers are separate processes: give them the same Python
    # and the same import path as the tests (etl/ contains spark_jobs)
    etl_dir = str(Path(__file__).resolve().parents[1])
    os.environ["PYTHONPATH"] = etl_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYSPARK_PYTHON"] = sys.executable

    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    try:
        out = (spark.createDataFrame(raw)
               .mapInPandas(to_features, schema=FEATURES_SCHEMA)
               .toPandas())
        assert len(out) == len(raw)
        assert list(out.columns) == ID_COLUMNS + FEATURE_COLUMNS
    finally:
        spark.stop()
              