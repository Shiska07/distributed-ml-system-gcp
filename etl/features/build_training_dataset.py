"""
build_training_dataset.py
Build a training dataset VERSION: labels + split for feature rows. Features are NOT copied.

  gold/features/source=<S>/run_id=*/     (all feature runs of the source)
  silver/outcomes/source=<S>/run_id=*/   (all outcome runs of the source)
      │  keep trips in [start_date, end_date) → dedupe per unique_key (newest run wins)
      │  inner join on unique_key → drop unrealistic tips → label (common/) → time split
      ▼
  gold/labels/<version>/split=train|val|test/   unique_key, feature_run_id, label
  gold/labels/<version>/_manifest.json          the dataset definition (written last)

Training reads features + labels by joining on (unique_key, feature_run_id).
"""

import argparse
import json
from collections.abc import Iterator
from datetime import date, datetime, timezone

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window, functions as F

from taxi_features.features import FEATURE_COLUMNS, TIP_THRESHOLD, create_label

# Output schema of the label step (mapInPandas)
LABELED_COLUMNS = [
    ("unique_key",     "string"),
    ("feature_run_id", "string"),
    ("trip_start_ts",  "timestamp"),
    ("split",          "string"),
    ("label",          "bigint"),
]
LABELED_SCHEMA = ", ".join(f"{name} {dtype}" for name, dtype in LABELED_COLUMNS)

# Columns written to gold/labels (split becomes the folder name)
OUTPUT_COLUMNS = ["unique_key", "feature_run_id", "label", "split"]


# ----------------------------- HELPERS ---------------------------------------

def newest_per_key(df: DataFrame, run_col: str) -> DataFrame:
    """Keep one row per unique_key: the one from the newest run (run_ids sort by time)."""
    w = Window.partitionBy("unique_key").orderBy(F.col(run_col).desc())
    return df.withColumn("_rn", F.row_number().over(w)).filter("_rn = 1").drop("_rn")


def compute_split_dates(df: DataFrame, val_frac: float, test_frac: float) -> tuple[date, date]:
    """(val_start, test_start) so that ~val_frac / test_frac of rows (by trip time) fall
    into val / test. Cuts are rounded to midnight."""
    secs = F.col("trip_start_ts").cast("long")
    q_val, q_test = (df.select(secs.alias("s"))
                     .approxQuantile("s", [1 - val_frac - test_frac, 1 - test_frac], 0.001))
    to_date = lambda s: datetime.fromtimestamp(s, tz=timezone.utc).date()
    return to_date(q_val), to_date(q_test)


def add_label(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """mapInPandas function: label with the shared rule from common/."""
    for pdf in batches:
        pdf["label"] = create_label(pdf)
        yield pdf[[name for name, _ in LABELED_COLUMNS]]


def write_text(spark: SparkSession, path: str, text: str) -> None:
    """Write one small text file (e.g. _manifest.json); fails if it already exists."""
    jvm_path = spark._jvm.org.apache.hadoop.fs.Path(path)
    fs = jvm_path.getFileSystem(spark._jsc.hadoopConfiguration())
    stream = fs.create(jvm_path, False)
    try:
        stream.write(bytearray(text.encode("utf-8")))
    finally:
        stream.close()


# ----------------------------- MAIN ------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Build a training dataset version (labels + split).")
    p.add_argument("--lake", required=True, help="Lake bucket, e.g. gs://…-lake")
    p.add_argument("--source", required=True, choices=["historical", "live"])
    p.add_argument("--start_date", type=date.fromisoformat, required=True, help="trip window start, inclusive")
    p.add_argument("--end_date", type=date.fromisoformat, required=True, help="trip window end, exclusive")
    p.add_argument("--version", required=True, help="Dataset version, e.g. v1 (written once)")
    p.add_argument("--git_commit", default="unknown")
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--max_tip_ratio", type=float, default=1.0,
                   help="Drop trips whose tip is more than this share of the fare (data errors)")
    args = p.parse_args()

    if args.start_date >= args.end_date:
        raise ValueError("start_date must be before end_date")
    if not 0 < args.val_frac + args.test_frac < 1:
        raise ValueError("val_frac + test_frac must be between 0 and 1")

    features_root = f"{args.lake}/gold/features"
    outcomes_root = f"{args.lake}/silver/outcomes"
    out_path = f"{args.lake}/gold/labels/{args.version}"
    print(f">> Source {args.source} | trips {args.start_date} → {args.end_date} | output {out_path}")

    spark = SparkSession.builder.appName(f"training-dataset-{args.version}").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # ------------------------- FEATURES (only what's needed for labels) ---
    # basePath makes Spark add the folder value run_id=… as a column
    features = (spark.read.option("basePath", features_root)
                .parquet(f"{features_root}/source={args.source}/")
                .select("unique_key",
                        F.col("trip_start_ts").cast("timestamp").alias("trip_start_ts"),
                        "fare",
                        F.col("run_id").alias("feature_run_id")))

    start = F.lit(str(args.start_date)).cast("timestamp")
    end = F.lit(str(args.end_date)).cast("timestamp")
    features = features.filter((F.col("trip_start_ts") >= start) & (F.col("trip_start_ts") < end))
    features = newest_per_key(features, "feature_run_id").cache()

    # ------------------------- OUTCOMES -----------------------------------
    outcomes = (spark.read.option("basePath", outcomes_root)
                .parquet(f"{outcomes_root}/source={args.source}/")
                .select("unique_key", "tips", F.col("run_id").alias("outcome_run_id")))
    outcomes = newest_per_key(outcomes, "outcome_run_id")

    # ------------------------- JOIN + SANITY FILTER ------------------------
    joined = features.join(outcomes, on="unique_key", how="inner").cache()
    plausible = joined.filter(F.col("tips") <= F.col("fare") * args.max_tip_ratio)

    n_features = features.count()
    n_joined = joined.count()
    n_plausible = plausible.count()
    if n_plausible == 0:
        raise ValueError("No labeled rows: check source, dates and that outcomes exist")

    # ------------------------- SPLIT + LABEL ------------------------------
    val_start, test_start = compute_split_dates(plausible, args.val_frac, args.test_frac)
    print(f">> Split dates: val from {val_start}, test from {test_start}")

    ts = F.col("trip_start_ts")
    split_col = (F.when(ts < F.lit(str(val_start)).cast("timestamp"), "train")
                  .when(ts < F.lit(str(test_start)).cast("timestamp"), "val")
                  .otherwise("test"))

    labeled = (plausible.withColumn("split", split_col)
               .mapInPandas(add_label, schema=LABELED_SCHEMA)
               .cache())

    # ------------------------- WRITE --------------------------------------
    (labeled.select(*OUTPUT_COLUMNS)
        .write.mode("errorifexists")        # a version is written once
        .partitionBy("split")
        .parquet(out_path))

    # ------------------------- SUMMARY + MANIFEST -------------------------
    per_split = (labeled.groupBy("split")
                 .agg(F.count("*").alias("rows"),
                      F.round(F.avg("label"), 4).alias("label_rate"),
                      F.min("trip_start_ts").alias("first_trip"),
                      F.max("trip_start_ts").alias("last_trip"))
                 .orderBy("first_trip"))
    per_split.show(truncate=False)

    feature_runs = sorted(r["feature_run_id"] for r in labeled.select("feature_run_id").distinct().collect())
    outcome_runs = sorted(r["outcome_run_id"] for r in joined.select("outcome_run_id").distinct().collect())

    manifest = {
        "dataset": "training_labels",
        "version": args.version,
        "source": args.source,
        "trip_window": {"start": str(args.start_date), "end_exclusive": str(args.end_date)},
        "inputs": {
            "features_root": f"{features_root}/source={args.source}",
            "feature_runs": feature_runs,
            "outcomes_root": f"{outcomes_root}/source={args.source}",
            "outcome_runs": outcome_runs,
        },
        "feature_columns": FEATURE_COLUMNS,
        "label_rule": f"tips / fare > {TIP_THRESHOLD}",
        "max_tip_ratio": args.max_tip_ratio,
        "counts": {
            "features_in_window": n_features,
            "with_outcome": n_joined,
            "match_rate": round(n_joined / n_features, 4) if n_features else None,
            "dropped_implausible_tips": n_joined - n_plausible,
            "labeled": n_plausible,
        },
        "split": {
            "val_start": str(val_start), "test_start": str(test_start),
            "val_frac": args.val_frac, "test_frac": args.test_frac,
            "per_split": [{k: str(v) if isinstance(v, datetime) else v for k, v in r.asDict().items()}
                          for r in per_split.collect()],
        },
        "git_commit": args.git_commit,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_text(spark, f"{out_path}/_manifest.json", json.dumps(manifest, indent=2))
    print(f">> Manifest: {out_path}/_manifest.json")

    spark.stop()


if __name__ == "__main__":
    main()
    