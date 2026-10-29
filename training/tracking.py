"""Experiment tracking: MLflow, Vertex AI Experiments when enabled.

train.py calls these functions instead of MLflow/Vertex directly, so every
param and metric is logged to both places in the same way.
"""
import os

import google.auth.transport.requests
import google.oauth2.id_token
import mlflow
from google.cloud import aiplatform

_vertex_enabled = False     # switched on by init_vertex() when an experiment name is given


# --- MLflow server authentication -----------------------------------------------

def refresh_mlflow_token():
    """Identity token for the private MLflow server on Cloud Run.

    On Vertex AI the token comes from the job's service account. Tokens expire after
    1 hour, so this is called again before the final logging steps of long runs.
    Locally (user login), fetching fails; set MLFLOW_TRACKING_TOKEN yourself instead.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("https://"):
        return                                   # local server or sqlite: no token needed
    try:
        token = google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), uri)
        os.environ["MLFLOW_TRACKING_TOKEN"] = token
    except Exception:
        if "MLFLOW_TRACKING_TOKEN" not in os.environ:
            raise


# --- Vertex AI Experiments (optional) --------------------------------------------

def init_vertex(experiment: str):
    """Enable Vertex AI Experiments. Empty name = MLflow only (tests, local runs)."""
    global _vertex_enabled
    if not experiment:
        return
    aiplatform.init(
        project=os.environ["PROJECT_ID"],
        location=os.environ.get("REGION", "us-central1"),
        experiment=experiment,
    )
    _vertex_enabled = True


def start_vertex_run(run_name: str, mlflow_run_id: str):
    """Start the matching Vertex run, linked to the MLflow run by its id."""
    if not _vertex_enabled:
        return
    # Vertex run names: unique, lowercase letters, digits and hyphens only
    name = f"{run_name}-{mlflow_run_id[:8]}".lower().replace("_", "-")
    aiplatform.start_run(name)
    aiplatform.log_params({"mlflow_run_id": mlflow_run_id})


def end_vertex_run():
    if _vertex_enabled:
        aiplatform.end_run()


# --- Logging: both systems ----------------------------------------------------------

def log_params(params: dict):
    """Log params to MLflow, and to Vertex AI Experiments if enabled."""
    mlflow.log_params(params)
    if _vertex_enabled:
        # Vertex accepts only str / int / float values (e.g. None becomes "None")
        aiplatform.log_params({k: v if isinstance(v, (str, int, float)) else str(v)
                               for k, v in params.items()})


def log_metrics(metrics: dict):
    """Log metrics to MLflow, and to Vertex AI Experiments if enabled."""
    metrics = {k: float(v) for k, v in metrics.items()}
    mlflow.log_metrics(metrics)
    if _vertex_enabled:
        aiplatform.log_metrics(metrics)
        