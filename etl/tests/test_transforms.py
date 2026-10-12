"""
Tests for ingestion/transforms.py
  - validate_trip() / validate_outcome(): plain Python unit tests (fast, no Beam)
  - CleanTrip / CleanOutcome: run in a small local Beam pipeline (TestPipeline)
Run from the repo root:  pytest etl -v
"""
from datetime import datetime, timezone

import apache_beam as beam
import pytest
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from ingestion.transforms import (
    REJECTED, CleanOutcome, CleanTrip, validate_outcome, validate_trip,
)


# ----------------------------- SAMPLE DATA -----------------------------------
# A fixture is reusable test data: pytest passes it into any test that lists it as an argument.

@pytest.fixture
def good_trip():
    """A valid raw trip (request-time fields only, no tips)."""
    return {
        "unique_key": "trip_1", "taxi_id": "taxi_9", "payment_type": "Credit Card",
        "trip_start_timestamp": "2022-01-01T08:00:00", "trip_seconds": 900,
        "trip_miles": 3.2, "fare": 14.5,
        "pickup_community_area": 8, "dropoff_community_area": 32,
        "tolls": 0, "extras": 1, "company": "Flash Cab",
    }


@pytest.fixture
def good_outcome():
    """A valid raw outcome (ground truth)."""
    return {"unique_key": "trip_1", "tips": 3.0}


# ----------------------------- validate_trip() -------------------------------

def test_valid_trip_has_no_problems(good_trip):
    assert validate_trip(good_trip) == []


def test_missing_key_is_rejected(good_trip):
    assert validate_trip({**good_trip, "unique_key": None}) == ["missing_key"]


def test_zero_second_trip_is_bad_duration(good_trip):
    assert validate_trip({**good_trip, "trip_seconds": 0}) == ["bad_duration"]


def test_missing_fare_is_bad_fare(good_trip):
    assert validate_trip({**good_trip, "fare": None}) == ["bad_fare"]


def test_pickup_area_out_of_range(good_trip):
    assert validate_trip({**good_trip, "pickup_community_area": 99}) == ["bad_pickup_area"]


def test_collects_all_problems_not_just_first(good_trip):
    bad = {**good_trip, "trip_seconds": 0, "fare": None, "trip_start_timestamp": None}
    assert set(validate_trip(bad)) == {"missing_start_ts", "bad_duration", "bad_fare"}


def test_accepts_timezone_aware_datetime_from_bigquery(good_trip):
    # BigQuery returns real datetime objects with UTC attached, not strings
    ts = datetime(2022, 1, 1, 8, 0, tzinfo=timezone.utc)
    assert validate_trip({**good_trip, "trip_start_timestamp": ts}) == []


def test_cash_trip_passes_validate_trip(good_trip):
    # Payment type is a business filter handled in CleanTrip, not a data-quality rule
    assert validate_trip({**good_trip, "payment_type": "Cash"}) == []


def test_trip_without_tips_is_valid(good_trip):
    # Trips never carry the label; a missing tip must not be a trip problem
    assert "tips" not in good_trip
    assert validate_trip(good_trip) == []


# ----------------------------- validate_outcome() ----------------------------

def test_valid_outcome_has_no_problems(good_outcome):
    assert validate_outcome(good_outcome) == []


def test_zero_tip_is_valid(good_outcome):
    # A tip of 0 is a real outcome (label = 0), not a data problem
    assert validate_outcome({**good_outcome, "tips": 0}) == []


def test_negative_tip_is_bad_tips(good_outcome):
    assert validate_outcome({**good_outcome, "tips": -1}) == ["bad_tips"]


def test_missing_tip_is_bad_tips(good_outcome):
    assert validate_outcome({**good_outcome, "tips": None}) == ["bad_tips"]


def test_outcome_missing_key(good_outcome):
    assert validate_outcome({**good_outcome, "unique_key": ""}) == ["missing_key"]


# ----------------------------- CleanTrip (Beam) ------------------------------

def test_clean_trip_routes_rows_to_correct_outputs(good_trip):
    cash = {**good_trip, "unique_key": "trip_2", "payment_type": "Cash"}
    zero_seconds = {**good_trip, "unique_key": "trip_3", "trip_seconds": 0}
    no_fare = {**good_trip, "unique_key": "trip_4", "fare": None}

    with TestPipeline() as p:
        # with_outputs splits the DoFn output into two PCollections:
        #   results.valid     ← plain yields
        #   results[REJECTED] ← yields wrapped in TaggedOutput(REJECTED, ...)
        results = (
            p
            | beam.Create([good_trip, cash, zero_seconds, no_fare])
            | beam.ParDo(CleanTrip()).with_outputs(REJECTED, main="valid")
        )

        # assert_that checks a PCollection's contents when the pipeline runs (labels must be unique)
        valid_keys = results.valid | "ValidKeys" >> beam.Map(lambda r: r["unique_key"])
        assert_that(valid_keys, equal_to(["trip_1"]), label="CheckValid")

        rejected_keys = results[REJECTED] | "RejectedKeys" >> beam.Map(lambda r: r["unique_key"])
        assert_that(rejected_keys, equal_to(["trip_3", "trip_4"]), label="CheckRejected")
        # trip_2 (cash) appears in neither: it was filtered out


def test_clean_trip_record_has_expected_types_and_no_label(good_trip):
    def check(records):
        # assert_that also accepts a function that receives the full list of outputs
        assert len(records) == 1
        r = records[0]
        assert isinstance(r["trip_start_ts"], datetime) and r["trip_start_ts"].tzinfo is None
        assert isinstance(r["trip_seconds"], int)
        assert isinstance(r["fare"], float)
        assert "payment_type" not in r       # filter column is not kept
        assert "tips" not in r               # the label never goes into trips

    with TestPipeline() as p:
        results = (
            p
            | beam.Create([good_trip])
            | beam.ParDo(CleanTrip()).with_outputs(REJECTED, main="valid")
        )
        assert_that(results.valid, check)


# ----------------------------- CleanOutcome (Beam) ---------------------------

def test_clean_outcome_routes_rows_to_correct_outputs(good_outcome):
    negative = {"unique_key": "trip_2", "tips": -5}
    missing = {"unique_key": "trip_3", "tips": None}

    with TestPipeline() as p:
        results = (
            p
            | beam.Create([good_outcome, negative, missing])
            | beam.ParDo(CleanOutcome()).with_outputs(REJECTED, main="valid")
        )
        assert_that(results.valid, equal_to([{"unique_key": "trip_1", "tips": 3.0}]),
                    label="CheckValid")

        rejected_keys = results[REJECTED] | "RejectedKeys" >> beam.Map(lambda r: r["unique_key"])
        assert_that(rejected_keys, equal_to(["trip_2", "trip_3"]), label="CheckRejected")


def test_clean_outcome_converts_string_tips(good_outcome):
    # Landing JSON may hold numbers as strings
    with TestPipeline() as p:
        results = (
            p
            | beam.Create([{**good_outcome, "tips": "2.5"}])
            | beam.ParDo(CleanOutcome()).with_outputs(REJECTED, main="valid")
        )
        assert_that(results.valid, equal_to([{"unique_key": "trip_1", "tips": 2.5}]))
        