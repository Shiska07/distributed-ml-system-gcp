from datetime import date, datetime

from ingestion.sources import (
    OUTCOME_FIELDS, TRIP_FIELDS, landing_paths, landing_to_outcome, landing_to_trip, pick,
)
from ingestion.transforms import validate_trip


def test_landing_paths_one_per_day_end_exclusive():
    paths = landing_paths("gs://lake", "live_trips", date(2022, 4, 1), date(2022, 4, 3))
    assert paths == ["gs://lake/landing/live_trips/event_date=2022-04-01/*.json",
                     "gs://lake/landing/live_trips/event_date=2022-04-02/*.json"]


def test_trip_fields_never_contain_the_label():
    assert "tips" not in TRIP_FIELDS
    assert OUTCOME_FIELDS == ["unique_key", "tips"]


def test_pick_keeps_only_given_fields_and_fills_missing():
    row = {"unique_key": "a1", "tips": 3.0, "fare": 10.0}
    assert pick(row, OUTCOME_FIELDS) == {"unique_key": "a1", "tips": 3.0}
    assert pick({"unique_key": "a1"}, OUTCOME_FIELDS) == {"unique_key": "a1", "tips": None}


def test_landing_to_trip_converts_types_and_has_no_tips():
    row = landing_to_trip({"unique_key": "a1", "trip_start_timestamp": "2022-04-01 08:30:00 UTC",
                           "trip_seconds": "600", "fare": 12.5, "pickup_community_area": "8",
                           "tips": 2.0})                     # tips in the input must be ignored
    assert row["trip_start_timestamp"] == datetime(2022, 4, 1, 8, 30)
    assert row["trip_seconds"] == 600 and row["pickup_community_area"] == 8 and row["fare"] == 12.5
    assert "tips" not in row


def test_landing_to_trip_has_same_fields_as_historical():
    assert set(landing_to_trip({})) == set(TRIP_FIELDS)


def test_landing_to_outcome_converts_tips():
    assert landing_to_outcome({"unique_key": "a1", "tips": "3.5"}) == {"unique_key": "a1", "tips": 3.5}
    assert landing_to_outcome({"unique_key": "a1", "tips": ""}) == {"unique_key": "a1", "tips": None}


def test_landing_trip_passes_trip_validation():
    """A good landing record must come out in a shape that validate_trip accepts."""
    row = landing_to_trip({"unique_key": "a1", "trip_start_timestamp": "2022-04-01 08:30:00 UTC",
                           "trip_seconds": "600", "trip_miles": "2.5", "fare": "12.5",
                           "pickup_community_area": "8", "payment_type": "Credit Card"})
    assert validate_trip(row) == []
    