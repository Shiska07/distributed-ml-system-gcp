"""
create_features.py
Build features for ONE silver trips run. Used for training and inference alike:
no labels, no splits, nothing source-specific.

  silver/trips/source=<S>/run_id=<R>/   →  build_features() (common/)  →  gold/features/source=<S>/run_id=<R>/
                                                                         + _manifest.json (written last)

The output keeps the same source and run_id as its input, so every feature run
traces back 1:1 to the silver run it came from.
"""

import argparse
import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F

from taxi_features.features import FEATURE_COLUMNS, ID_COLUMNS, build_features

# Output schema for mapInPandas: one (name, Spark type) per column.
# Must match ID_COLUMNS + FEATURE_COLUMNS from common/ (checked in main()).
FEATURES_COLUMNS = [
    # keys (not model inputs; used for joins and time filters)
    ("unique_key",     "string"),
    ("trip_start_ts",  "timestamp"),
    # time
    ("hour",           "bigint"),
    ("day_of_week",    "bigint"),
    ("is_weekend",     "bigint"),
    # trip
    ("trip_miles",     "double"),
    ("trip_minutes",   "double"),
    ("avg_speed_mph",  "double"),
    ("fare",           "double"),
    ("fare_per_mile",  "double"),
    ("extras",         "double"),
    ("tolls",          "double"),
    ("has_extras",     "bigint"),
    # place
    ("pickup_area",    "bigint"),
    ("dropoff_area",   "bigint"),
    ("is_airport",     "bigint"),
    ("is_downtown",    "bigint"),
    ("same_area",      "bigint"),
    # operator
    ("company",        "string"),
]
FEATURES_SCHEMA = ", ".join(f"{name} {dtype}" for name, dtype in FEATURES_COLUMNS)


# ----------------------------- HELPERS ---------------------------------------

def parse_run_path(path: str) -> tuple[str, str]:
    """'…/silver/trips/source=live/run_id=2022-04-01_2022-04-08_20261011-120000/' → ('live', '2022-04-01_…')"""
    m = re.search(r"source=([^/]+)/run_id=([^/]+)", path)
    if not m:
        raise ValueError(f"Path must contain source=<…>/run_id=<…>: {path}")
    return m.group(1), m.group(2)


def to_features(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """mapInPandas function: each batch of silver rows → ID columns + features."""
    for pdf in batches:
        yield build_features(pdf, keep=ID_COLUMNS)


def write_text(spark: SparkSession, path: str, text: str) -> None:
    """Write one small text file (e.g. _manifest.json) to GCS or local disk.
    Uses Spark's Hadoop file system, so 'gs://…' and local paths both work.
    Fails if the file already exists."""
    jvm_path = spark._jvm.org.apache.hadoop.fs.Path(path)
    fs = jvm_path.getFileSystem(spark._jsc.hadoopConfiguration())
    stream = fs.create(jvm_path, False)       # False = do not overwrite
    try:
        stream.write(bytearray(text.encode("utf-8")))
    finally:
        stream.close()


# ----------------------------- MAIN ------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build features for one silver trips run.")
    p.add_argument("--trips_path", required=True,
                   help="One silver trips run, e.g. gs://…/silver/trips/source=live/run_id=…")
    p.add_argument("--features_root", required=True,
                   help="Parent folder for features, e.g. gs://…-lake/gold/features")
    p.add_argument("--git_commit", default="unknown",
                   help="Code version that built these features (recorded in the manifest)")
    args = p.parse_args()

    # the schema above must list exactly the columns common/ produces, in the same order
    expected = ID_COLUMNS + FEATURE_COLUMNS
    if [name for name, _ in FEATURES_COLUMNS] != expected:
        raise ValueError(f"FEATURES_COLUMNS out of sync with common/: expected {expected}")

    source, run_id = parse_run_path(args.trips_path)
    out_path = f"{args.features_root.rstrip('/')}/source={source}/run_id={run_id}"
    print(f">> Input:  {args.trips_path}")
    print(f">> Output: {out_path}")

    # ------------------------- START SPARK -----------------------------
    spark = SparkSession.builder.appName(f"create-features-{source}-{run_id}").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    trips: DataFrame = spark.read.parquet(args.trips_path)

    # Beam wrote timestamps without a time zone (TIMESTAMP_NTZ). Convert to a regular
    # TIMESTAMP; with the session time zone set to UTC, the values stay unchanged.
    trips = trips.withColumn("trip_start_ts", F.col("trip_start_ts").cast("timestamp"))

    # -------------------------- FEATURES -------------------------------
    features = trips.mapInPandas(to_features, schema=FEATURES_SCHEMA).cache()
    #     └── SPARK in ──► [ pandas: build_features ] ──► SPARK out ──┘

    # ------------------------- WRITE -----------------------------------
    (features.write
        .mode("errorifexists")        # one feature run per silver run; never overwritten
        .parquet(out_path))

    # ------------------------- SUMMARY + MANIFEST ----------------------
    s = features.agg(
        F.count("*").alias("rows"),
        F.min("trip_start_ts").alias("first_trip"),
        F.max("trip_start_ts").alias("last_trip"),
    ).collect()[0]
    print(f">> Rows: {s['rows']} | trips {s['first_trip']} → {s['last_trip']}")

    manifest = {
        "dataset": "features",
        "source": source,
        "run_id": run_id,
        "input_trips_path": args.trips_path,
        "output_path": out_path,
        "git_commit": args.git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "id_columns": ID_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "rows": s["rows"],
        "first_trip": str(s["first_trip"]),
        "last_trip": str(s["last_trip"]),
    }
    # written last: a feature run without _manifest.json is incomplete
    write_text(spark, f"{out_path}/_manifest.json", json.dumps(manifest, indent=2))
    print(f">> Manifest: {out_path}/_manifest.json")

    spark.stop()


if __name__ == "__main__":
    main()
    