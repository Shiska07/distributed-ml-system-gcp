"""Load a gold training split (labels + features) into pandas.

Gold layout:
    gold/labels/<version>/_manifest.json
    gold/labels/<version>/split=train|val|test/*.parquet   -> unique_key, feature_run_id, label
    gold/features/source=<S>/run_id=<R>/*.parquet          -> unique_key, trip_start_ts, features

Labels point to their feature run. Features are read only for the keys in the split,
so a feature run is never loaded in full.
"""
import json
import fsspec
import pandas as pd

import pyarrow.dataset as ds

from taxi_features.features import FEATURE_COLUMNS, ID_COLUMNS

LABEL = "label"
SPLITS = ("train", "val", "test")
LABEL_COLUMNS = ["unique_key", "feature_run_id", LABEL]

def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 to halve memory for numeric features."""
    floats = df.select_dtypes("float64").columns
    df[floats] = df[floats].astype("float32")
    return df

def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into X (features) and y (labels)."""
    if LABEL not in df.columns:
        raise ValueError(f"DataFrame missing required column: {LABEL}")
    X = df[FEATURE_COLUMNS]
    y = df[LABEL]
    return X, y

def load_data(lake: str,
              version: str,
              seed: int= 42):
    """Load a gold training split (labels + features) into pandas."""
    
    version_dir = f"{lake}/gold/labels/{version}"
    
    # read manifest file
    with fsspec.open(f"{version_dir}/_manifest.json", "r") as f:
        manifest = json.load(f)
    if manifest["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Dataset feature_columns do not match taxi_features.FEATURE_COLUMNS")
    
    features_dir = manifest["inputs"]["features_root"]
    
    splits = {}
    for split in SPLITS:
        # read labels for this split
        labels = pd.read_parquet(f"{version_dir}/split={split}")
        if not all(c in labels.columns for c in LABEL_COLUMNS):
            raise ValueError(f"Dataset {split} split missing required columns: {LABEL_COLUMNS}")
        
        # read features for the keys in this split
        # read features only for these keys, only from the runs the labels point to
        keys = labels["unique_key"].to_list()
        run_ids = labels["feature_run_id"].unique().tolist()
        features = (
            ds.dataset(features_dir, format="parquet", partitioning="hive")
            .to_table(
                columns=ID_COLUMNS + FEATURE_COLUMNS + ["run_id"],
                filter=ds.field("unique_key").isin(keys) & ds.field("run_id").isin(run_ids),
            )
            .to_pandas()
            .rename(columns={"run_id": "feature_run_id"})
        )
        
        # join on key AND run, so a reprocessed trip is never duplicated
        df = pd.merge(labels, features, on=["unique_key", "feature_run_id"], how="inner")
        if len(df) != len(labels):
            raise ValueError(f"{len(labels) - len(df)} labels in split={split} have no feature row")
        
        # shuffle the split to avoid any ordering bias
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        splits[split] = shrink(df)
        
    return splits, manifest
    
    