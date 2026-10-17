#!/usr/bin/env bash
# One-time setup: MLflow tracking server on Cloud Run + Cloud SQL + GCS.
# Follows https://mlflow.org/docs/latest/self-hosting/deploy-to-cloud/gcp/
# Safe to rerun: existing resources are skipped, IAM bindings don't duplicate.
# Requires: infra/03_ml_service_account.sh has been run (ml-platform SA exists).

set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
source "$ROOT/config/dev.env"

# --- APIs -------------------------------------------------------------------
gcloud services enable sqladmin.googleapis.com run.googleapis.com \
  secretmanager.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com

# --- Artifact bucket: model files, plots, manifests ---------------------------
if ! gcloud storage buckets describe "$MLFLOW_BUCKET" >/dev/null 2>&1; then
  gcloud storage buckets create "$MLFLOW_BUCKET" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

gcloud storage buckets add-iam-policy-binding "$MLFLOW_BUCKET" \
  --member="serviceAccount:$ML_SA" \
  --role="roles/storage.objectAdmin"

# --- DB password in Secret Manager (hex: no characters that break the DB URL) --
if ! gcloud secrets describe mlflow-db-password >/dev/null 2>&1; then
  openssl rand -hex 16 | tr -d '\n' | gcloud secrets versions add mlflow-db-password --data-file=-
fi
gcloud secrets add-iam-policy-binding mlflow-db-password \
  --member="serviceAccount:$ML_SA" \
  --role="roles/secretmanager.secretAccessor"

# --- Cloud SQL Postgres: runs, metrics, registry (first creation ~5-10 min) ---
if ! gcloud sql instances describe "$MLFLOW_SQL_INSTANCE" >/dev/null 2>&1; then
  gcloud sql instances create "$MLFLOW_SQL_INSTANCE" \
    --database-version=POSTGRES_16 --edition=enterprise --tier=db-f1-micro \
    --region="$REGION" --storage-size=10 --storage-type=HDD
  gcloud sql databases create mlflow --instance="$MLFLOW_SQL_INSTANCE"
  gcloud sql users create mlflow --instance="$MLFLOW_SQL_INSTANCE" \
    --password="$(gcloud secrets versions access latest --secret=mlflow-db-password)"
fi

SQL_CONN=$(gcloud sql instances describe "$MLFLOW_SQL_INSTANCE" --format='value(connectionName)')

# --- Server image: official MLflow -full image + google-cloud-storage ----------
gcloud builds submit "$ROOT/infra/mlflow" --tag "$MLFLOW_IMAGE"

# --- Cloud Run (private, scales to zero) -------------------------------------
gcloud run deploy "$MLFLOW_SERVICE" \
  --image="$MLFLOW_IMAGE" \
  --region="$REGION" \
  --service-account="$ML_SA" \
  --no-allow-unauthenticated \
  --port=5000 \
  --memory=2Gi --cpu=1 \
  --min-instances=0 --max-instances=1 \
  --add-cloudsql-instances="$SQL_CONN" \
  --set-env-vars="CLOUDSQL_CONNECTION=$SQL_CONN,ARTIFACT_BUCKET=$MLFLOW_BUCKET" \
  --set-secrets="DB_PASSWORD=mlflow-db-password:latest"

echo "MLflow URL:"
gcloud run services describe "$MLFLOW_SERVICE" --region="$REGION" --format='value(status.url)'
