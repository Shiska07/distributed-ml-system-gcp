#!/usr/bin/env bash
# 01_build_beam_template.sh
# Builds the Beam cleaning pipeline as a Dataflow Flex Template named "clean-trips":
#   1) launcher image → Artifact Registry: $AR_REPO/clean-trips:<commit>
#   2) template spec  → $CODE/templates/clean-trips/<commit>.json (+ latest.json)
# Usage: bash etl/scripts/01_build_beam_template.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ETL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ETL_DIR")"

source "$REPO_DIR/config/dev.env"
: "${PROJECT_ID:?}" "${CODE:?}" "${AR_REPO:?AR_REPO not set in config/dev.env}"

TEMPLATE_NAME="clean-trips"
BEAM_VERSION="2.76.0"           # must match the version you tested with
PY_VERSION="312"

GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
if [[ -n "$(git -C "$REPO_DIR" status --porcelain -- etl)" ]]; then
  echo "WARNING: uncommitted changes in etl/ — image tag $GIT_COMMIT won't match the code exactly" >&2
fi

IMAGE="$AR_REPO/$TEMPLATE_NAME:$GIT_COMMIT"
SPEC="$CODE/templates/$TEMPLATE_NAME/$GIT_COMMIT.json"
LATEST="$CODE/templates/$TEMPLATE_NAME/latest.json"

echo ">> Image: $IMAGE"
echo ">> Spec:  $SPEC"

# 1) Build and push the launcher image (build context = etl/)
gcloud builds submit "$ETL_DIR" \
  --project "$PROJECT_ID" \
  --config "$ETL_DIR/ingestion/cloudbuild.yaml" \
  --substitutions "_IMAGE=$IMAGE,_BEAM_VERSION=$BEAM_VERSION,_PY_VERSION=$PY_VERSION"

# 2) Write the template spec (points to the image + declares parameters)
gcloud dataflow flex-template build "$SPEC" \
  --project "$PROJECT_ID" \
  --image "$IMAGE" \
  --sdk-language PYTHON \
  --metadata-file "$ETL_DIR/ingestion/metadata.json"

# 3) "latest" pointer for manual runs; automation should use the commit-tagged spec
gcloud storage cp "$SPEC" "$LATEST"

echo ">> Done. Launch with: --template-file-gcs-location $SPEC"
