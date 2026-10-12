"""
sources.py
How raw data is read for each source. Every reader returns a dict of PCollections:

  {"trips": ..., "outcomes": ...}   (only the keys that source has)

  trips:    request-time fields only (no tips): what the model sees
  outcomes: unique_key + tips: the ground truth, joined to trips later in Spark

All trips have the same field names and types, and so do all outcomes, so the rest
of the pipeline doesn't care where the data came from.

  historical:   BigQuery public table via SQL (sql_queries/02_load_table.sql) → trips + outcomes
  live:         landing/live_trips/   JSON (production requests)              → trips
  ground_truth: landing/ground_truth/ JSON (outcomes, arrive later)           → outcomes
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import apache_beam as beam

SQL_DIR = Path(__file__).resolve().parent.parent / "sql_queries"
HISTORICAL_QUERY = "02_load_table.sql"

# Field lists for each dataset. Trips must never contain the label (tips).
TRIP_FIELDS = [
    "unique_key", "taxi_id", "trip_start_timestamp", "trip_seconds", "trip_miles", "fare",
    "pickup_community_area", "dropoff_community_area", "tolls", "extras",
    "company", "payment_type",
]
OUTCOME_FIELDS = ["unique_key", "tips"]


def pick(row: dict, fields: list[str]) -> dict:
    """Keep only the given fields (missing ones become None)."""
    return {f: row.get(f) for f in fields}


# ----------------------------- HISTORICAL ------------------------------------

def read_historical(p, args, datasets):
    """Read [start_date, end_date) from BigQuery once, then split each row into trip + outcome."""
    query = (SQL_DIR / HISTORICAL_QUERY).read_text().format(
        start_date=args.start_date, end_date=args.end_date)
    rows = p | "ReadBigQuery" >> beam.io.ReadFromBigQuery(query=query, use_standard_sql=True)
    out = {}
    if "trips" in datasets:
        out["trips"] = rows | "ToTrip" >> beam.Map(pick, TRIP_FIELDS)
    if "outcomes" in datasets:
        out["outcomes"] = rows | "ToOutcome" >> beam.Map(pick, OUTCOME_FIELDS)
    return out


# ----------------------------- LANDING (BATCHED PRODUCTION DATA) -------------

def landing_paths(lake: str, folder: str, start: date, end: date) -> list[str]:
    """One glob per day in [start, end), e.g. …/landing/live_trips/event_date=2022-04-01/*.json"""
    paths, d = [], start
    while d < end:
        paths.append(f"{lake}/landing/{folder}/event_date={d}/*.json")
        d += timedelta(days=1)
    return paths


def _to_int(v):
    return None if v in (None, "") else int(float(v))     # BigQuery exports INT64 as JSON strings


def _to_float(v):
    return None if v in (None, "") else float(v)


def _to_ts(v):
    """'2022-04-01 00:15:00 UTC' → naive UTC datetime (same as the silver timestamp type)."""
    if not v:
        return None
    return datetime.fromisoformat(v.replace(" UTC", "").replace("T", " ").rstrip("Z"))


def landing_to_trip(row: dict) -> dict:
    """Convert a landing request to the same field names/types as historical trips."""
    return {
        "unique_key": row.get("unique_key"),
        "taxi_id": row.get("taxi_id"),
        "trip_start_timestamp": _to_ts(row.get("trip_start_timestamp")),
        "trip_seconds": _to_int(row.get("trip_seconds")),
        "trip_miles": _to_float(row.get("trip_miles")),
        "fare": _to_float(row.get("fare")),
        "pickup_community_area": _to_int(row.get("pickup_community_area")),
        "dropoff_community_area": _to_int(row.get("dropoff_community_area")),
        "tolls": _to_float(row.get("tolls")),
        "extras": _to_float(row.get("extras")),
        "company": row.get("company"),
        "payment_type": row.get("payment_type"),
    }


def landing_to_outcome(row: dict) -> dict:
    """Convert a landing outcome to the same field names/types as historical outcomes."""
    return {
        "unique_key": row.get("unique_key"),
        "tips": _to_float(row.get("tips")),
    }


def read_landing(p, args, folder: str, name: str):
    """Read and parse all landing JSON files for [start_date, end_date)."""
    return (p
            | f"{name}Paths" >> beam.Create(
                  landing_paths(args.out_lake, folder, args.start_date, args.end_date))
            | f"Read{name}"  >> beam.io.ReadAllFromText()      # days with no files are skipped
            | f"Parse{name}" >> beam.Map(json.loads))


# landing folder for each dataset of the live source
LANDING_FOLDERS = {"trips": "live_trips", "outcomes": "ground_truth"}


def read_live(p, args, datasets):
    """Production data from the landing zone. Trips and outcomes arrive at different times,
    so each is read only when requested."""
    out = {}
    if "trips" in datasets:
        raw = read_landing(p, args, LANDING_FOLDERS["trips"], "Trips")
        out["trips"] = raw | "ToTrip" >> beam.Map(landing_to_trip)
    if "outcomes" in datasets:
        raw = read_landing(p, args, LANDING_FOLDERS["outcomes"], "Outcomes")
        out["outcomes"] = raw | "ToOutcome" >> beam.Map(landing_to_outcome)
    return out



# ----------------------------- REGISTRY --------------------------------------

# source name → reader(p, args, datasets) returning {"trips": ..., "outcomes": ...}
SOURCES = {
    "historical": read_historical,
    "live":       read_live,
}