"""Prediction server for Vertex AI (custom container).

Vertex sets these environment variables when it starts the container:
    AIP_STORAGE_URI    GCS folder with the model files (copied there by Vertex)
    AIP_HTTP_PORT      port to listen on (used in the Dockerfile)
    AIP_HEALTH_ROUTE   path Vertex calls to check the server is ready
    AIP_PREDICT_ROUTE  path Vertex sends prediction requests to

Model folder (written by deploy.py):
    model/          MLflow sklearn model: encoder + estimator pipeline
    serving.json    {"model_version": "3", "threshold": 0.41}

Request:   {"instances": [{"hour": 14, "fare": 12.5, ..., "company": "Flash Cab"}, ...]}
Response:  {"predictions": [{"probability": 0.73, "tip_over_20": true}, ...]}
"""
import json
import os
import tempfile

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from google.cloud import storage

from taxi_features.features import FEATURE_COLUMNS


def download_model(gcs_uri: str) -> str:
    """Copy the model folder from GCS to local disk and return the local path."""
    bucket_name, prefix = gcs_uri.removeprefix("gs://").split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    local_dir = tempfile.mkdtemp()
    for blob in storage.Client().list_blobs(bucket_name, prefix=prefix):
        if blob.name.endswith("/"):
            continue                                     # folder placeholder, not a file
        path = os.path.join(local_dir, blob.name[len(prefix):])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob.download_to_filename(path)
    return local_dir


# --- Load once at startup ---------------------------------------------------------
MODEL_DIR = download_model(os.environ["AIP_STORAGE_URI"])
MODEL = mlflow.sklearn.load_model(os.path.join(MODEL_DIR, "model"))
with open(os.path.join(MODEL_DIR, "serving.json")) as f:
    META = json.load(f)
THRESHOLD = float(META["threshold"])

app = FastAPI()


@app.get(os.environ.get("AIP_HEALTH_ROUTE", "/health"))
def health():
    return {"status": "ok", "model_version": META["model_version"]}


@app.post(os.environ.get("AIP_PREDICT_ROUTE", "/predict"))
async def predict(request: Request):
    body = await request.json()
    X = pd.DataFrame(body.get("instances", []))

    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if X.empty or missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing or 'no instances'}")

    proba = MODEL.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    return {
        "predictions": [
            {"probability": float(p), "tip_over_20": bool(p >= THRESHOLD)} for p in proba
        ],
        "model_version": META["model_version"],
    }
    