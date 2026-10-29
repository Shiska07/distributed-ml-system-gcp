import json

import numpy as np
import pandas as pd
import pytest

from load_dataset import load_data, shrink, split_xy
from taxi_features.features import FEATURE_COLUMNS

RUN_1 = "2022-01-01_2022-01-06_20260101-000000"
RUN_2 = "2022-01-01_2022-01-06_20260102-000000"   # a later reprocessing of the same trips
SPLIT_KEYS = {"train": ["a", "b", "c"], "val": ["d"], "test": ["e"]}
ALL_KEYS = ["a", "b", "c", "d", "e"]


def write_feature_run(features_dir, run_id, keys, value):
    df = pd.DataFrame({
        "unique_key": keys,
        "trip_start_ts": pd.date_range("2022-01-01", periods=len(keys), freq="D"),
        **{col: [value] * len(keys) for col in FEATURE_COLUMNS},
    })
    run_dir = features_dir / f"run_id={run_id}"
    run_dir.mkdir(parents=True)
    df.to_parquet(run_dir / "part-0.parquet")


def build_lake(tmp_path, split_keys=SPLIT_KEYS, feature_columns=FEATURE_COLUMNS):
    features_dir = tmp_path / "gold/features/source=historical"
    write_feature_run(features_dir, RUN_1, ALL_KEYS, value=1.0)
    write_feature_run(features_dir, RUN_2, ["a"], value=2.0)   # same key, different run

    version_dir = tmp_path / "gold/labels/v1"
    for split, keys in split_keys.items():
        split_dir = version_dir / f"split={split}"
        split_dir.mkdir(parents=True)
        pd.DataFrame({"unique_key": keys, "feature_run_id": RUN_1,
                      "label": [1] * len(keys)}).to_parquet(split_dir / "part-0.parquet")

    manifest = {"source": "historical",
                "inputs": {"features_root": str(features_dir)},   # .../gold/features/source=historical
                "feature_columns": feature_columns}
    (version_dir / "_manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path)


@pytest.fixture
def lake(tmp_path):
    return build_lake(tmp_path)


# --- load_data ---

def test_loads_all_splits_and_manifest(lake):
    splits, manifest = load_data(lake, "v1")
    assert set(splits) == {"train", "val", "test"}
    assert manifest["source"] == "historical"
    for split, keys in SPLIT_KEYS.items():
        assert sorted(splits[split]["unique_key"]) == keys


def test_features_come_from_the_labelled_run_only(lake):
    """Key 'a' exists in two feature runs; labels point to RUN_1."""
    splits, _ = load_data(lake, "v1")
    train = splits["train"]
    assert len(train) == 3                                   # no duplicate rows
    assert (train[FEATURE_COLUMNS[0]] == 1.0).all()          # RUN_1 values, not RUN_2


def test_feature_columns_mismatch_raises(tmp_path):
    lake = build_lake(tmp_path, feature_columns=["wrong"])
    with pytest.raises(ValueError):
        load_data(lake, "v1")


def test_label_without_features_raises(tmp_path):
    lake = build_lake(tmp_path, split_keys={**SPLIT_KEYS, "train": ["a", "b", "z"]})
    with pytest.raises(ValueError):
        load_data(lake, "v1")


def test_same_seed_same_order(lake):
    first, _ = load_data(lake, "v1", seed=1)
    second, _ = load_data(lake, "v1", seed=1)
    assert list(first["train"]["unique_key"]) == list(second["train"]["unique_key"])


def test_features_are_float32(lake):
    splits, _ = load_data(lake, "v1")
    assert splits["train"][FEATURE_COLUMNS[0]].dtype == np.float32


# --- helpers ---

def test_shrink_only_changes_float64():
    df = shrink(pd.DataFrame({"f": [1.5], "i": [1], "s": ["x"]}))
    assert df["f"].dtype == np.float32
    assert df["i"].dtype == np.int64
    assert df["s"].dtype == object


def test_split_xy(lake):
    splits, _ = load_data(lake, "v1")
    X, y = split_xy(splits["test"])
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(y) == 1


def test_split_xy_without_label_raises():
    with pytest.raises(ValueError):
        split_xy(pd.DataFrame({col: [1.0] for col in FEATURE_COLUMNS}))
        