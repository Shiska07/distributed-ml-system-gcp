"""
pipeline.py
Beam ETL: raw data → clean, remove duplicates → Parquet in the GCS lake (silver layer).

  read (sources.py) → {"trips": ..., "outcomes": ...}
  each dataset → Clean (valid / rejected) → dedupe on unique_key → silver Parquet + rejects JSON

  historical   → trips + outcomes
  live         → trips
  ground_truth → outcomes

Output (one immutable folder per dataset, source and run):
  silver/<dataset>/source=<source>/run_id=<start>_<end>_<ts>/part-*.parquet
  rejected/<dataset>/source=<source>/run_id=<start>_<end>_<ts>/part-*.jsonl

Run from the etl/ folder:
  python -m beam_etl.pipeline --source historical --start_date 2022-01-01 --end_date 2022-01-02 \
      --out_lake $LAKE --project $PROJECT_ID --temp_location $TEMP/beam-local
"""

import argparse
import json
import logging
from datetime import date, datetime, timezone

import apache_beam as beam
import pyarrow as pa
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions

from beam_etl.sources import SOURCES
from beam_etl.transforms import REJECTED, CleanOutcome, CleanTrip

# Schemas of the silver Parquet files; must match what the DoFns emit
TRIP_SCHEMA = pa.schema([
    ("unique_key", pa.string()),
    ("taxi_id", pa.string()),
    ("trip_start_ts", pa.timestamp("us")),
    ("trip_seconds", pa.int64()),
    ("trip_miles", pa.float64()),
    ("fare", pa.float64()),
    ("pickup_community_area", pa.int64()),
    ("dropoff_community_area", pa.int64()),   # nullable (trips ending outside the city)
    ("tolls", pa.float64()),
    ("extras", pa.float64()),
    ("company", pa.string()),
])

OUTCOME_SCHEMA = pa.schema([
    ("unique_key", pa.string()),
    ("tips", pa.float64()),
])

# dataset name → (cleaning DoFn class, silver schema)
DATASETS = {
    "trips":    (CleanTrip,    TRIP_SCHEMA),
    "outcomes": (CleanOutcome, OUTCOME_SCHEMA),
}


def valid_date(s: str) -> date:
    try:
        return date.fromisoformat(s)          # "2022-01-01" → date(2022, 1, 1)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{s}' is not a valid date (expected YYYY-MM-DD)")


def keep_first(key_and_rows):
    """After grouping by unique_key (key, Iterable[row]), keep a single row per record."""
    _key, rows = key_and_rows
    return next(iter(rows))


def clean_and_write(raw, name: str, silver_dir: str, rejects_dir: str):
    """Clean one dataset, remove duplicates, write silver Parquet and rejects JSON."""
    clean_fn, schema = DATASETS[name]
    label = name.capitalize()                 # unique step names per dataset, e.g. "CleanTrips"

    cleaned = raw | f"Clean{label}" >> beam.ParDo(clean_fn()).with_outputs(REJECTED, main="valid")

    # remove duplicates
    deduped = (
        cleaned.valid                                                          # rows as dicts
        | f"Key{label}"   >> beam.Map(lambda r: (r["unique_key"], r))          # (unique_key, row)
        | f"Group{label}" >> beam.GroupByKey()                                 # (unique_key, Iterable[row])
        | f"First{label}" >> beam.Map(keep_first)                              # one row per key
    )

    # write clean data
    deduped | f"Write{label}" >> beam.io.WriteToParquet(
        file_path_prefix=f"{silver_dir}/part",
        schema=schema,
        file_name_suffix=".parquet",
    )

    # write rejected data as JSON lines
    (cleaned[REJECTED]
     | f"{label}RejectsToJson" >> beam.Map(json.dumps, default=str)
     | f"Write{label}Rejects"  >> beam.io.WriteToText(
           file_path_prefix=f"{rejects_dir}/part",
           file_name_suffix=".jsonl",
       ))


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES.keys(), default="historical",
                        help="historical = BigQuery, live = landing zone")
    parser.add_argument("--dataset", choices=["trips", "outcomes", "all"], default="all",
                        help="which dataset(s) to produce; live trips and outcomes arrive at "
                             "different times, so they are usually run separately")
    parser.add_argument("--start_date", type=valid_date, required=True, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end_date", type=valid_date, required=True, help="YYYY-MM-DD, exclusive")
    parser.add_argument("--out_lake", required=True, help="Lake bucket, e.g. gs://<project>-lake")
    parser.add_argument("--run_ts", default=None,
                        help="UTC run timestamp YYYYMMDD-HHMMSS; generated if not given")

    # parse_known_args: our args go to `args`; everything else (--project, --runner, ...) goes to Beam
    args, beam_args = parser.parse_known_args(argv)

    if args.start_date >= args.end_date:
        parser.error("--start_date must be before --end_date")

    reader = SOURCES[args.source]

    # one run_id shared by every dataset this run writes
    run_ts = args.run_ts or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{args.start_date}_{args.end_date}_{run_ts}"

    options = PipelineOptions(beam_args)

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]

    # job name (Dataflow: lowercase letters, digits and hyphens only)
    gcp = options.view_as(GoogleCloudOptions)
    if not gcp.job_name:
        gcp.job_name = f"taxi-clean-{args.source}-{run_id}".replace("_", "-")

    print(f">> Source: {args.source}")
    print(f">> Run ID: {run_id}")

    with beam.Pipeline(options=options) as p:

        # read: the only source-specific step; returns only the datasets this source has
        datasets_in = reader(p, args, datasets)

        for name, raw in datasets_in.items():
            silver_dir = f"{args.out_lake}/silver/{name}/source={args.source}/run_id={run_id}"
            rejects_dir = f"{args.out_lake}/rejected/{name}/source={args.source}/run_id={run_id}"
            print(f">> {name}: {silver_dir}")
            clean_and_write(raw, name, silver_dir, rejects_dir)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
