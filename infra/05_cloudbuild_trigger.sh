#!/usr/bin/env bash
# One-time setup: GitHub → Cloud Build trigger that tests, builds and pushes the
# training image on every push to main that changes training/ or common/.
# Run twice the first time: the first run creates the GitHub connection and prints a
# link to authorize it in the browser; the second run finishes the setup.
set -euo pipefail
source "$(git rev-parse --show-toplevel)/config/dev.env"

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
CLOUDBUILD_AGENT="service-${PROJECT_NUMBER}@gcp-sa-cloudbuild.iam.gserviceaccount.com"

# --- APIs ---------------------------------------------------------------------
gcloud services enable cloudbuild.googleapis.com secretmanager.googleapis.com

# --- GitHub connection (stores the GitHub token in Secret Manager) -------------
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$CLOUDBUILD_AGENT" --role="roles/secretmanager.admin" \
  --condition=None >/dev/null

if ! gcloud builds connections describe "$GITHUB_CONNECTION" --region="$REGION" >/dev/null 2>&1; then
  gcloud builds connections create github "$GITHUB_CONNECTION" --region="$REGION"
fi

STAGE=$(gcloud builds connections describe "$GITHUB_CONNECTION" --region="$REGION" \
  --format='value(installationState.stage)')
if [[ "$STAGE" != "COMPLETE" ]]; then
  echo "Authorize the GitHub connection in your browser:"
  gcloud builds connections describe "$GITHUB_CONNECTION" --region="$REGION" \
    --format='value(installationState.actionUri)'
  echo "Then run this script again."
  exit 0
fi

# --- Link the repository --------------------------------------------------------
if ! gcloud builds repositories describe "$GITHUB_REPO" \
     --connection="$GITHUB_CONNECTION" --region="$REGION" >/dev/null 2>&1; then
  gcloud builds repositories create "$GITHUB_REPO" \
    --remote-uri="https://github.com/${GITHUB_OWNER}/${GITHUB_REPO}.git" \
    --connection="$GITHUB_CONNECTION" --region="$REGION"
fi

# --- Build service account: push images, write logs ----------------------------
if ! gcloud iam service-accounts describe "$BUILD_SA" >/dev/null 2>&1; then
  gcloud iam service-accounts create cloud-build-train --display-name="Cloud Build: training image"
  sleep 60
fi
for ROLE in roles/artifactregistry.writer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$BUILD_SA" --role="$ROLE" --condition=None >/dev/null
done

# --- Trigger --------------------------------------------------------------------
if ! gcloud builds triggers describe train-image --region="$REGION" >/dev/null 2>&1; then
  gcloud builds triggers create github \
    --name=train-image \
    --region="$REGION" \
    --repository="projects/${PROJECT_ID}/locations/${REGION}/connections/${GITHUB_CONNECTION}/repositories/${GITHUB_REPO}" \
    --branch-pattern='^master$' \
    --build-config=training/cloudbuild.yaml \
    --included-files='training/**,common/**' \
    --service-account="projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}"
fi

echo "Trigger ready:"
gcloud builds triggers describe train-image --region="$REGION" --format='value(name,github.push.branch)'
