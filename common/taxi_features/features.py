"""
Row-level feature and label logic: the single definition used everywhere.

Used by:
  build_features → Spark create_features (training + batch inference), online serving
  create_label   → build_training_dataset (training labels), monitoring (live labels)

Rules:
- Only use values known at payment time (end of trip).
- Never use tips or trip_total as features (they contain the answer).
- Only fixed, stateless rules live here. Anything learned from data (company vocabulary,
  scaling statistics) is fitted on the train split during training and saved with the model:
  learned values would leak evaluation data if computed here, and they are model-specific.


Output features
---------------
Passed through (cleaned, nulls filled):
    trip_miles, fare, extras, tolls, pickup_area, dropoff_area, company
Engineered (derived from other columns):
    hour, day_of_week, is_weekend       <- trip_start_ts
    trip_minutes                        <- trip_seconds
    avg_speed_mph                       <- trip_miles, trip_seconds
    fare_per_mile                       <- fare, trip_miles
    has_extras                          <- extras
    is_airport, is_downtown, same_area  <- pickup/dropoff community area

"""

import pandas as pd

TIP_THRESHOLD = 0.20

AIRPORT_AREAS = {56, 76}          # Midway (Garfield Ridge), O'Hare
DOWNTOWN_AREAS = {8, 32, 33}      # Near North Side, Loop, Near South Side
UNKNOWN_AREA = -1              # outside Chicago / missing
UNKNOWN_COMPANY = "Unknown"

FEATURE_COLUMNS = [
    "hour", "day_of_week", "is_weekend",
    "trip_miles", "trip_minutes", "avg_speed_mph",
    "fare", "fare_per_mile", "extras", "tolls", "has_extras",
    "pickup_area", "dropoff_area", "is_airport", "is_downtown", "same_area",
    "company",
]

# Columns that identify a trip. Never used as model input, but needed to join
ID_COLUMNS = ["unique_key", "trip_start_ts"]

def build_features(df: pd.DataFrame, keep: list[str] | None = None):
    """Return a DataFrame with exactly FEATURE_COLUMNS, one row per input row."""
    out = pd.DataFrame(index=df.index)

    # ── Time ──
    ts = pd.to_datetime(df["trip_start_ts"])
    out["hour"] = ts.dt.hour                                 # nightlife vs commute vs daytime riders tip differently
    out["day_of_week"] = ts.dt.dayofweek                     # Monday = 0; weekday business vs weekend leisure
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)  # explicit flag makes the weekend effect easy to learn

    # ── Trip ──
    miles = df["trip_miles"].astype(float)
    seconds = df["trip_seconds"].astype(float)
    hours = seconds / 3600
    out["trip_miles"] = miles                                # tip % often drops on longer trips
    out["trip_minutes"] = seconds / 60                       # minutes are easier to read than seconds
    out["avg_speed_mph"] = (miles / hours).where(hours > 0, 0.0)  # stuck in traffic vs smooth ride → satisfaction

    fare = df["fare"].astype(float)
    out["fare"] = fare                                       # riders tip less generously on expensive fares
    out["fare_per_mile"] = (fare / miles).where(miles > 0, 0.0)   # how pricey the trip felt (traffic, short hops)
    out["extras"] = df["extras"].fillna(0).astype(float)     # extra charges can reduce willingness to tip
    out["tolls"] = df["tolls"].fillna(0).astype(float)       # same idea as extras; missing means none
    out["has_extras"] = (out["extras"] > 0).astype(int)      # whether any extra was charged may matter more than the amount

    # ── Place ──
    pickup = df["pickup_community_area"].fillna(UNKNOWN_AREA).astype(int)
    dropoff = df["dropoff_community_area"].fillna(UNKNOWN_AREA).astype(int)
    out["pickup_area"] = pickup                              # neighbourhood effects (downtown, nightlife, residential)
    out["dropoff_area"] = dropoff
    out["is_airport"] = (pickup.isin(AIRPORT_AREAS) | dropoff.isin(AIRPORT_AREAS)).astype(int)     # airport riders (travellers, expense accounts) behave differently
    out["is_downtown"] = (pickup.isin(DOWNTOWN_AREAS) | dropoff.isin(DOWNTOWN_AREAS)).astype(int)  # tourists and business travellers
    out["same_area"] = ((pickup == dropoff) & (pickup != UNKNOWN_AREA)).astype(int)                # short local hop vs cross-city trip

    # ── Operator ──
    # Companies use different payment terminals with different preset tip
    # buttons (e.g. 15/20/25%), likely a strong signal. Kept as the raw name:
    # grouping rare companies into "other" is learned from data (rule 3).
    out["company"] = df["company"].fillna(UNKNOWN_COMPANY).str.strip()
    
    features = out[FEATURE_COLUMNS]
    if keep:
        return pd.concat([df[keep], features], axis=1)
    return features


def create_label(df: pd.DataFrame) -> pd.Series:
    """1 if the tip is more than 20% of the fare, else 0. Shared by training and monitoring."""
    if df["tips"].isna().any() or (df["fare"] <= 0).any():
        raise ValueError("create_label needs non-null tips and fare > 0")
    return ((df["tips"] / df["fare"]) > TIP_THRESHOLD).astype(int)
