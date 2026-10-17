#!/usr/bin/env bash
# One service account for training, MLflow server and serving (split later via dev.env).
set -euo pipefail
source "$(dirname "$0")/../config/dev.env"
: "${PROJECT_ID:?}" "${LAKE:?}" "${ML_SA:?}"

gcloud iam service-accounts describe "$ML_SA" >/dev/null 2>&1 || \
  gcloud iam service-accounts create ml-platform --display-name="ML: training, MLflow, serving"

# read gold data (bucket level only)
gcloud storage buckets add-iam-policy-binding "$LAKE" \
  --member="serviceAccount:$ML_SA" --role="roles/storage.objectViewer"

# project-level roles
for role in roles/aiplatform.user roles/artifactregistry.reader roles/logging.logWriter \
            roles/cloudsql.client roles/run.invoker; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$ML_SA" --role="$role" --condition=None >/dev/null
done

# let you submit jobs/services that run as this account
gcloud iam service-accounts add-iam-policy-binding "$ML_SA" \
  --member="user:$(gcloud config get-value account)" --role="roles/iam.serviceAccountUser"

echo ">> $ML_SA ready"
