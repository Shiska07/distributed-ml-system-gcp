import json
import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

import train
from models import get_model
from taxi_features.encoding import BINARY_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from taxi_features.features import FEATURE_COLUMNS

RUN_ID = "2022-01-01_2022-02-01_20260101-000000"



# Fake data
# ---------------------------------------------------------------------------

def make_features(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.uniform(0, 30, n) for col in NUMERIC_COLUMNS})
    df["hour"] = rng.integers(0, 24, n)
    df["day_of_week"] = rng.integers(0, 7, n)
    df["pickup_area"] = rng.integers(1, 78, n)
    df["dropoff_area"] = rng.integers(1, 78, n)
    df["company"] = rng.choice(["Flash Cab", "Sun Taxi", "Unknown"], n)
    for col in BINARY_COLUMNS:
        df[col] = rng.integers(0, 2, n)
    return df[FEATURE_COLUMNS]


def build_lake(root, n=300):
    """Tiny gold dataset: one feature run, labels split 60/20/20 by time."""
    keys = [f"k{i}" for i in range(n)]
    features = make_features(n)
    features.insert(0, "unique_key", keys)
    features.insert(1, "trip_start_ts", pd.date_range("2022-01-01", periods=n, freq="h"))

    run_dir = root / "gold/features/source=historical" / f"run_id={RUN_ID}"
    run_dir.mkdir(parents=True)
    features.to_parquet(run_dir / "part-0.parquet")

    labels = np.random.default_rng(1).integers(0, 2, n)
    cuts = {"train": slice(0, int(n * 0.6)), "val": slice(int(n * 0.6), int(n * 0.8)),
            "test": slice(int(n * 0.8), n)}
    version_dir = root / "gold/labels/v1"
    for split, rows in cuts.items():
        split_dir = version_dir / f"split={split}"
        split_dir.mkdir(parents=True)
        pd.DataFrame({"unique_key": keys[rows], "feature_run_id": RUN_ID,
                      "label": labels[rows]}).to_parquet(split_dir / "part-0.parquet")

    manifest = {"source": "historical", "features_root": str(root / "gold/features"),
                "feature_columns": FEATURE_COLUMNS}
    (version_dir / "_manifest.json").write_text(json.dumps(manifest))
    return str(root)



# Unit tests
# ---------------------------------------------------------------------------

def test_pick_threshold_separates_perfect_scores():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert 0.2 < train.pick_threshold(y, proba) <= 0.8


def test_evaluate_returns_all_metrics_in_range():
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.2, 0.7, 0.4, 0.9, 0.6])
    metrics = train.evaluate(y, proba, threshold=0.5)
    assert set(metrics) == {"pr_auc", "roc_auc", "log_loss", "f1", "precision", "recall"}
    assert all(np.isfinite(v) for v in metrics.values())



# Every model type: builds, accepts its own grid keys, fits, predicts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["LR", "RF", "XGB"])
def test_model_accepts_grid_and_fits(model_type):
    if model_type == "XGB":
        pytest.importorskip("xgboost")
    model, grid = get_model(model_type)
    first_combo = {key: values[0] for key, values in grid.items()}
    model.set_params(**first_combo)                     # fails if a grid key is wrong

    X = make_features(200)
    y = np.random.default_rng(0).integers(0, 2, 200)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (200,) and ((proba >= 0) & (proba <= 1)).all()



# full training run with a throwaway MLflow store
# ---------------------------------------------------------------------------

@pytest.fixture
def mlflow_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                         # MLflow artifacts land in tmp, not the repo
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    return uri


def run_main(monkeypatch, lake, model_type="LR"):
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--lake", lake, "--version", "v1", "--model", model_type,
        "--cv_frac", "1.0", "--experiment", "test", "--git_commit", "test",
    ])
    train.main()


def test_training_run_logs_and_registers(tmp_path, monkeypatch, mlflow_store):
    lake = build_lake(tmp_path / "lake")
    run_main(monkeypatch, lake)

    client = MlflowClient()
    run = mlflow.search_runs(experiment_names=["test"], output_format="list")[0]
    assert run.data.tags["model_type"] == "LR"
    assert run.data.tags["dataset_version"] == "v1"
    assert "test_pr_auc" in run.data.metrics
    assert "threshold" in run.data.params

    champion = client.get_model_version_by_alias(train.MODEL_NAME, "champion")
    assert champion.tags["model_type"] == "LR"

    # the registered model loads and predicts on raw features
    model = mlflow.pyfunc.load_model(f"models:/{train.MODEL_NAME}@champion")
    assert len(model.predict(make_features(5).astype({c: "float64" for c in NUMERIC_COLUMNS}))) == 5


def test_second_run_becomes_challenger(tmp_path, monkeypatch, mlflow_store):
    lake = build_lake(tmp_path / "lake")
    run_main(monkeypatch, lake)
    run_main(monkeypatch, lake)

    client = MlflowClient()
    assert client.get_model_version_by_alias(train.MODEL_NAME, "champion").version == 1
    assert client.get_model_version_by_alias(train.MODEL_NAME, "challenger").version == 2
    