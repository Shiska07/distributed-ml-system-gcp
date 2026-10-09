#!/usr/bin/env bash
# 00_run_beam_pipeline.sh
# Runs the Beam cleaning pipeline on Dataflow from Cloud Shell (manual runs).
# Automated runs use the Flex Template instead (01_build_beam_template.sh).
#
# Usage:   bash etl/scripts/00_run_beam_pipeline.sh START_DATE END_DATE [SOURCE] [DATASET]
#   bash etl/scripts/00_run_beam_pipeline.sh $HOLDOUT_START $HOLDOUT_END live trips
#   bash etl/scripts/00_run_beam_pipeline.sh $HOLDOUT_START $HOLDOUT_END live outcomes
# Optional: MAX_WORKERS=4 bash etl/scripts/00_run_beam_pipeline.sh 2022-01-01 2022-01-08
#
# SOURCE (default: historical)
#   historical   → BigQuery            → silver/trips + silver/outcomes
#   live         → landing/live_trips   → silver/trips
#   ground_truth → landing/ground_truth → silver/outcomes

set -euo pipefail

USAGE="Usage: 00_run_beam_pipeline.sh START_DATE END_DATE [historical|live|ground_truth]"
START_DATE=${1:?$USAGE}
END_DATE=${2:?$USAGE}
SOURCE=${3:-historical}
DATASET=${4:-all}

case "$SOURCE"  in historical|live) ;;          *) echo "Bad SOURCE. $USAGE" >&2; exit 1 ;; esac
case "$DATASET" in trips|outcomes|all) ;;       *) echo "Bad DATASET. $USAGE" >&2; exit 1 ;; esac

# Resolve paths relative to this script, so it works from any folder
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ETL_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ETL_DIR")"

source "$REPO_DIR/config/dev.env"

# Activate the venv if it isn't already active (Beam builds the job graph locally)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  source "$REPO_DIR/.venv/bin/activate"
fi

# Fail early with a clear message if config is missing
: "${LAKE:?LAKE not set in config/dev.env}"
: "${PROJECT_ID:?PROJECT_ID not set in config/dev.env}"
: "${REGION:?REGION not set in config/dev.env}"
: "${TEMP:?TEMP not set in config/dev.env}"
: "${SA:?SA not set in config/dev.env}"

echo ">> Dataflow | source: $SOURCE | $START_DATE -> $END_DATE | lake: $LAKE"

# Run from etl/ so "python -m beam_etl.pipeline" and ./setup.py resolve
cd "$ETL_DIR"
python -m beam_etl.pipeline \
  --source "$SOURCE" \
  --start_date "$START_DATE" \
  --end_date "$END_DATE" \
  --out_lake "$LAKE" \
  --project "$PROJECT_ID" \
  --runner DataflowRunner \
  --region "$REGION" \
  --temp_location "$TEMP/dataflow/temp" \
  --staging_location "$TEMP/dataflow/staging" \
  --service_account_email "$SA" \
  --setup_file ./setup.py \
  --no_use_public_ips \
  --max_num_workers "${MAX_WORKERS:-2}"

# ------------------- COMMON RUNS ----------------------
# Training data (after checking a small run with sql_queries/03_validate_silver.sql):
#   MAX_WORKERS=4 bash etl/scripts/00_run_beam_pipeline.sh $TRAIN_START $TRAIN_END
#
# Production batches (held-out window seeded by simulation/00_seed_landing.sh):
#   bash etl/scripts/00_run_beam_pipeline.sh $HOLDOUT_START $HOLDOUT_END live
#   bash etl/scripts/00_run_beam_pipeline.sh $HOLDOUT_START $HOLDOUT_END ground_truth
