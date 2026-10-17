"""Encoding utilities for taxi features."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from taxi_features.features import FEATURE_COLUMNS

# Skewed, non-negative amounts -> log1p, then standardize
NUMERIC_COLUMNS = [
    "trip_miles", "trip_minutes", "avg_speed_mph",
    "fare", "fare_per_mile", "extras", "tolls",
]

# Discrete values with no numeric order -> one-hot
CATEGORICAL_COLUMNS = ["hour", "day_of_week", "pickup_area", "dropoff_area", "company"]

# Already 0/1 -> passed through unchanged
BINARY_COLUMNS = ["is_weekend", "has_extras", "is_airport", "is_downtown", "same_area"]

def log1p_clipped(x):
    return np.log1p(np.clip(x, 0, None))  # clip negative values to 0 before log1p

def build_encoder_lr(min_frequency: int = 100) -> ColumnTransformer:
    """Unfitted encoder for linear models (logistic regression).

    min_frequency: categories seen fewer times in training (e.g. rare companies) are
    grouped into one 'infrequent' column. Unseen categories at inference map to it too.
    """
    
    # declare transformations for each type of feature
    numeric = make_pipeline(
        FunctionTransformer(log1p_clipped, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
    )
    
    # select columns and apply transformations, dropping any unlisted columns
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("binary", "passthrough", BINARY_COLUMNS),              # already 0/1 so no need to transform
        ],
        remainder="drop",
    )

def build_encoder_trees(min_frequency: int = 100) -> ColumnTransformer:
    """Unfitted encoder for tree models (random forest, XGBoost).

    Trees split on thresholds, so numeric columns need no log or scaling.
    Categories still need one-hot encoding: area 32 is not 'more' than area 8.
    """
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
    )

    return ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_COLUMNS),            # raw values; trees are scale-invariant
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("binary", "passthrough", BINARY_COLUMNS),
        ],
        remainder="drop",
    )
      