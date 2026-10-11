#!/usr/bin/env bash
# 02_run_spark_features.sh
# Builds features for ONE silver trips run on Dataproc Serverless (historical or live).
#   silver/trips/source=<S>/run_id=<R>/  →  gold/features/source=<S>/run_id=<R>/
#
# Usage:   bash etl/scripts/02_run_spark_features.sh SOURCE_OR_PATH
#   SOURCE_OR_PATH:
#     historical | live          → uses the NEWEST silver trips run of that source
#     gs://…/source=…/run_id=…   → uses exactly that run
#
# Examples:
#   bash etl/scripts/02_run_spark_features.sh historical
#   bash etl/scripts/02_run_spark_features.sh "$LAKE/silver/trips/source=live/run_id=2022-04-01_2022-05-01_20261011-120000"

set -euo pipefail

JOB_FILE="get_features.py"

USAGE="Usage: 02_run_spark_features.sh historical|live|<silver trips run path>"
INPUT=${1:?$USAGE}

JOB_FILE="get_features.py"
DATAPROC_VERSION="2.2" 

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ETL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ETL_DIR")"

source "$REPO_DIR/config/dev.env"
# Fail early with a clear message if config is missing
: "${LAKE:?LAKE not set in config/dev.env}"
: "${PROJECT_ID:?PROJECT_ID not set in config/dev.env}"
: "${REGION:?REGION not set in config/dev.env}"
: "${TEMP:?TEMP not set in config/dev.env}"
: "${SA:?SA not set in config/dev.env}"

JOB="$ETL_DIR/spark_jobs/$JOB_FILE"
[[ -f "$JOB" ]] || { echo "Job file not found: $JOB" >&2; exit 1; }

# ---------------- Decide which silver run to process ----------------
# You can pass either:
#   a source name (historical | live) → use the NEWEST run of that source
#   a full run path (gs://…/source=…/run_id=…) → use exactly that run
# After this block, two variables are set: TRIPS_PATH and SOURCE.

case "$INPUT" in

  # --- Option 1: a source name was given ---
  historical|live)
    SOURCE="$INPUT"

    # List all runs of this source, sort them (run_ids sort by date and time),
    # and keep the last one = the newest run
    TRIPS_PATH=$(gcloud storage ls "$LAKE/silver/trips/source=$SOURCE/" | sort | tail -1)

    # Stop if no run was found (empty result)
    if [[ -z "$TRIPS_PATH" ]]; then
      echo "No silver trips runs found for source=$SOURCE" >&2
      exit 1
    fi
    ;;

  # --- Option 2: a full run path was given ---
  # The pattern only matches paths that contain source=…/run_id=…
  gs://*source=*/run_id=*)
    TRIPS_PATH="$INPUT"

    # Take the source name out of the path, e.g. …/source=live/run_id=… → live
    SOURCE=$(echo "$INPUT" | sed -E 's|.*source=([^/]+)/.*|\1|')
    ;;

  # --- Anything else: show how to use the script and stop ---
  *)
    echo "$USAGE" >&2
    exit 1
    ;;

esac

FEATURES_ROOT="$LAKE/gold/features"
GIT_COMMIT="$(git -C "$REPO_DIR" rev-parse --short HEAD)"   # automatically get latest id
BATCH_ID="features-${SOURCE}-$(date -u +%Y%m%d-%H%M%S)" 

# Each serverless batch starts clean, so ship the shared feature code with it
mkdir -p "$REPO_DIR/dist"
rm -f "$REPO_DIR/dist/taxi_features.zip"
(cd "$REPO_DIR/common" && zip -qr "$REPO_DIR/dist/taxi_features.zip" taxi_features)

echo ">> Batch:  $BATCH_ID"
echo ">> Input:  $TRIPS_PATH"
echo ">> Output: $FEATURES_ROOT/source=$SOURCE/<same run_id>"
echo ">> Commit: $GIT_COMMIT"

gcloud dataproc batches submit pyspark "$JOB" \
  --batch "$BATCH_ID" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --service-account "$SA" \
  --version "$DATAPROC_VERSION" \
  --deps-bucket "$TEMP" \
  --py-files "$REPO_DIR/dist/taxi_features.zip" \
  -- \
  --trips_path "$TRIPS_PATH" \
  --features_root "$FEATURES_ROOT" \
  --git_commit "$GIT_COMMIT"
