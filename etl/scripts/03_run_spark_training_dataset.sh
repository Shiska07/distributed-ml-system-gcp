#!/usr/bin/env bash
# 03_run_spark_training_dataset.sh
# Builds a training dataset version (labels + split) on Dataproc Serverless.
#
# Usage:   bash etl/scripts/03_run_spark_training_dataset.sh SOURCE START_DATE END_DATE VERSION
# Example: bash etl/scripts/03_run_spark_training_dataset.sh historical 2022-01-01 2022-04-01 v1
# Optional: VAL_FRAC=0.15 TEST_FRAC=0.15 MAX_TIP_RATIO=1.0 bash ...

set -euo pipefail

USAGE="Usage: 03_run_spark_training_dataset.sh historical|live START_DATE END_DATE VERSION"
SOURCE=${1:?$USAGE}
START_DATE=${2:?$USAGE}
END_DATE=${3:?$USAGE}
VERSION=${4:?$USAGE}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ETL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ETL_DIR")"

JOB_FILE="build_training_dataset.py"
DATAPROC_VERSION="2.2"
JOB="$ETL_DIR/spark_jobs/$JOB_FILE"
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"
BATCH_ID="training-${VERSION}-$(date -u +%Y%m%d-%H%M%S)"

source "$REPO_DIR/config/dev.env"
: "${LAKE:?}" "${PROJECT_ID:?}" "${REGION:?}" "${TEMP:?}" "${SA:?}"

echo ">> Batch:  $BATCH_ID"
echo ">> Output: $LAKE/gold/labels/$VERSION"

gcloud dataproc batches submit pyspark "$JOB" \
  --batch "$BATCH_ID" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --service-account "$SA" \
  --version "$DATAPROC_VERSION" \
  --deps-bucket "$TEMP" \
  --py-files "$REPO_DIR/dist/taxi_features.zip" \
  -- \
  --lake "$LAKE" \
  --source "$SOURCE" \
  --start_date "$START_DATE" \
  --end_date "$END_DATE" \
  --version "$VERSION" \
  --git_commit "$GIT_COMMIT" \
  --val_frac "${VAL_FRAC:-0.10}" \
  --test_frac "${TEST_FRAC:-0.10}" \
  --max_tip_ratio "${MAX_TIP_RATIO:-1.0}"
