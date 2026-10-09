"""
transforms.py
Validation rules and cleaning DoFns. Kept separate from pipeline I/O so they can be
unit tested without GCP.

Two datasets, two rule sets:
  trips:    request-time fields (what the model sees). No tips here.
  outcomes: unique_key + tips (the ground truth), joined to trips later in Spark.

Each DoFn has two outputs:
  main     → clean, typed record
  REJECTED → record that failed a rule, with the list of problems
"""

from datetime import datetime

import apache_beam as beam
from apache_beam.metrics import Metrics

# name of the second output for bad records
REJECTED = "rejected"

# trip validation thresholds
MIN_SECONDS, MAX_SECONDS = 60, 4 * 3600
MIN_MILES, MAX_MILES = 0.1, 100
MIN_FARE, MAX_FARE = 3.25, 500
MIN_AREA, MAX_AREA = 1, 77

# only card trips are in scope (tips aren't recorded for cash)
CARD_PAYMENT = "Credit Card"


# ----------------------------- TYPE HELPERS ----------------------------------

def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    f = to_float(value)
    return None if f is None else int(f)


def to_naive_datetime(value):
    """
    BigQuery gives timestamps as timezone-aware datetimes (UTC).
    Landing files and tests may give strings. Return a datetime with no timezone, or None.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)    # remove timezone info
    try:
        return datetime.fromisoformat(str(value).replace(" UTC", ""))
    except ValueError:
        return None


def rejected_record(row, problems):
    """Shape of a rejected record: key, reasons, and the raw row as strings (JSON-safe)."""
    return {
        "unique_key": row.get("unique_key"),
        "problems": problems,
        "raw": {k: None if v is None else str(v) for k, v in row.items()},
    }


# ----------------------------- RULES -----------------------------------------

def validate_trip(row):
    """Check request-time fields. Returns a list of problems (empty = valid)."""
    problems = []

    if not row.get("unique_key"):
        problems.append("missing_key")

    if to_naive_datetime(row.get("trip_start_timestamp")) is None:
        problems.append("missing_start_ts")

    seconds = to_float(row.get("trip_seconds"))
    if seconds is None or not (MIN_SECONDS <= seconds <= MAX_SECONDS):
        problems.append("bad_duration")

    miles = to_float(row.get("trip_miles"))
    if miles is None or not (MIN_MILES <= miles <= MAX_MILES):
        problems.append("bad_distance")

    fare = to_float(row.get("fare"))
    if fare is None or not (MIN_FARE <= fare <= MAX_FARE):
        problems.append("bad_fare")

    pickup = to_int(row.get("pickup_community_area"))
    if pickup is None or not (MIN_AREA <= pickup <= MAX_AREA):
        problems.append("bad_pickup_area")

    return problems


def validate_outcome(row):
    """Check the ground truth on its own. Checks that need fare (tip %) happen in Spark after the join."""
    problems = []

    if not row.get("unique_key"):
        problems.append("missing_key")

    tips = to_float(row.get("tips"))
    if tips is None or tips < 0:
        problems.append("bad_tips")

    return problems


# ----------------------------- DOFNS -----------------------------------------

class CleanTrip(beam.DoFn):
    """
    For each trip:
    - not a card payment   → dropped (out of scope)
    - fails a trip rule    → REJECTED output
    - otherwise            → clean, typed trip on the main output
    """
    def __init__(self):
        super().__init__()
        self.valid = Metrics.counter("clean_trip", "valid_rows")
        self.rejected = Metrics.counter("clean_trip", "rejected_rows")
        self.filtered = Metrics.counter("clean_trip", "filtered_not_credit_card")

    def process(self, row):
        # Beam calls process() once per element
        if row.get("payment_type") != CARD_PAYMENT:
            self.filtered.inc()
            return                      # no output for this row

        problems = validate_trip(row)
        if problems:
            self.rejected.inc()
            # TaggedOutput sends the record to the named REJECTED output instead of the main one
            yield beam.pvalue.TaggedOutput(REJECTED, rejected_record(row, problems))
            return

        self.valid.inc()
        # a plain yield goes to the main (valid) output
        yield {
            "unique_key": row["unique_key"],
            "taxi_id": row.get("taxi_id") or "unknown",
            "trip_start_ts": to_naive_datetime(row["trip_start_timestamp"]),
            "trip_seconds": to_int(row["trip_seconds"]),
            "trip_miles": to_float(row["trip_miles"]),
            "fare": to_float(row["fare"]),
            "pickup_community_area": to_int(row["pickup_community_area"]),
            "dropoff_community_area": to_int(row.get("dropoff_community_area")),
            "tolls": to_float(row.get("tolls")) or 0.0,
            "extras": to_float(row.get("extras")) or 0.0,
            "company": (row.get("company") or "Unknown").strip(),
        }


class CleanOutcome(beam.DoFn):
    """
    For each outcome:
    - fails an outcome rule → REJECTED output
    - otherwise             → clean, typed outcome on the main output
    No payment filter here: outcomes have no payment_type. Outcomes of cash trips
    simply find no matching trip in the Spark join.
    """
    def __init__(self):
        super().__init__()
        self.valid = Metrics.counter("clean_outcome", "valid_rows")
        self.rejected = Metrics.counter("clean_outcome", "rejected_rows")

    def process(self, row):
        problems = validate_outcome(row)
        if problems:
            self.rejected.inc()
            yield beam.pvalue.TaggedOutput(REJECTED, rejected_record(row, problems))
            return

        self.valid.inc()
        yield {
            "unique_key": row["unique_key"],
            "tips": to_float(row["tips"]),
        }
        