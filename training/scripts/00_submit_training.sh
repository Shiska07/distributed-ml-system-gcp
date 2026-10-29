#!/usr/bin/env bash
# Submit a training run to Vertex AI as a custom job.
#
#   bash training/scripts/submit_training.sh MODEL VERSION EXPERIMENT MACHINE [IMAGE_TAG] [CV_FRAC]
#
#   MODEL       LR | RF | XGB
#   VERSION     gold dataset version, e.g. v1
#   EXPERIMENT  experiment name, used in MLflow and Vertex AI Experiments
#               (lowercase letters, digits, hyphens), e.g. taxi-tips
#   MACHINE     Vertex machine type, e.g. n1-highmem-8 (8 vCPU, 52 GB RAM)
#   IMAGE_TAG   training image tag: commit SHA or "latest"   (default: latest)
#   CV_FRAC     fraction of train used for tuning            (default: 0.2)
#
# Examples:
#   # logistic regression, newest image, quick tuning on 5% of train
#   bash training/scripts/00_submit_training.sh LR v1 taxi-tips n1-highmem-16 latest 0.2
#
#   # XGBoost from an exact build (commit 47731f8), default tuning fraction
#   bash training/scripts/00_submit_training.sh XGB v1 taxi-tips n1-highmem-16 47731f8

set -euo pipefail
source "$(git rev-parse --show-toplevel)/config/dev.env"

MODEL=$1 VERSION=$2 EXPERIMENT=$3 MACHINE=$4
TAG=${5:-latest} CV_FRAC=${6:-0.2}
CONFIG=$(mktemp)

cat > "$CONFIG" <<EOF
workerPoolSpecs:
  - machineSpec: {machineType: $MACHINE}
    replicaCount: 1
    containerSpec:
      imageUri: $TRAIN_IMAGE:$TAG
      args: [--lake=$LAKE, --version=$VERSION, --model=$MODEL, --cv_frac=$CV_FRAC,
             --experiment=$EXPERIMENT, --vertex_experiment=$EXPERIMENT]
      env:
        - {name: MLFLOW_TRACKING_URI, value: $MLFLOW_TRACKING_URI}
        - {name: PROJECT_ID, value: $PROJECT_ID}
        - {name: REGION, value: $REGION}
serviceAccount: $ML_SA
EOF

gcloud ai custom-jobs create --region="$REGION" --config="$CONFIG" \
  --display-name="train-${MODEL,,}-$VERSION-$(date -u +%Y%m%d-%H%M%S)"
rm -f "$CONFIG"
