import numpy as np
import pandas as pd
import pytest

from taxi_features.encoding import (
    BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, build_encoder_lr, log1p_clipped,
)
from taxi_features.features import FEATURE_COLUMNS

GROUPS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BINARY_COLUMNS


def make_features(n: int, company: str = "Flash Cab") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({col: rng.uniform(0, 30, n) for col in NUMERIC_COLUMNS})
    df["hour"] = rng.integers(0, 24, n)
    df["day_of_week"] = rng.integers(0, 7, n)
    df["pickup_area"] = rng.integers(1, 78, n)
    df["dropoff_area"] = rng.integers(1, 78, n)
    df["company"] = company
    for col in BINARY_COLUMNS:
        df[col] = rng.integers(0, 2, n)
    return df[FEATURE_COLUMNS]


def test_every_feature_has_exactly_one_group():
    assert sorted(GROUPS) == sorted(FEATURE_COLUMNS)
    assert len(GROUPS) == len(set(GROUPS))


def test_log1p_clipped_handles_negatives():
    out = log1p_clipped(np.array([-5.0, 0.0, np.e - 1]))
    assert np.allclose(out, [0.0, 0.0, 1.0])


def test_fit_transform_has_no_nan():
    encoder = build_encoder_lr(min_frequency=1)
    X = encoder.fit_transform(make_features(200))
    dense = X.toarray() if hasattr(X, "toarray") else X
    assert dense.shape[0] == 200
    assert not np.isnan(dense).any()


def test_unseen_company_does_not_raise():
    encoder = build_encoder_lr(min_frequency=1).fit(make_features(200))
    X = encoder.transform(make_features(5, company="Brand New Cab Co"))
    assert X.shape[0] == 5


def test_output_column_names_available():
    encoder = build_encoder_lr(min_frequency=1).fit(make_features(200))
    names = encoder.get_feature_names_out()
    assert any(name.startswith("numeric__fare") for name in names)
    assert any(name.startswith("categorical__company") for name in names)
    