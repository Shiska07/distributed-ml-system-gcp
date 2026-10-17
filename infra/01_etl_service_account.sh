#!/usr/bin/env bash

set -euo pipefail

# set environment file source
source "$(dirname "$0")/../config/dev.env"
gcloud config set project $PROJECT_ID

# Create the service account if it doesn't exist yet
if gcloud iam service-accounts describe "$SA" >/dev/null 2>&1; then
  echo "Service account exists, skipping: $SA"
else
  gcloud iam service-accounts create taxi-etl --display-name=="Taxi ETL pieplines"
  echo ">>> Created: $SA"
  echo "Waiting 30s for the new account to propagate..."
  sleep 30
fi



# Project-level roles:
#   dataflow.worker          - run Beam workers on Dataflow
#   dataproc.worker          - run Spark on Dataproc Serverless
#   bigquery.jobUser         - run queries (Beam reads the public taxi table with a query)
#   bigquery.dataEditor      - Beam creates a temporary dataset for query results
#   bigquery.readSessionUser - fast reads through the BigQuery Storage API (Spark join)
for ROLE in dataflow.worker dataproc.worker bigquery.jobUser bigquery.dataEditor bigquery.readSessionUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA" \
    --role="roles/$ROLE" \
    --condition=None \
    --quiet >/dev/null
  echo "Granted roles/$ROLE"
done

# Bucket-level access: read/write objects in our three buckets only,
for BUCKET in "$LAKE" "$CODE" "$TEMP"; do
  gcloud storage buckets add-iam-policy-binding "$BUCKET" \
    --member="serviceAccount:$SA" \
    --role="roles/storage.objectAdmin" >/dev/null
  echo "Granted storage.objectAdmin on $BUCKET"
done

# Let you (the logged-in user) launch jobs that run as this service account.
# This is redundant here but useful when assigning service account to other team members
ME=$(gcloud config get-value account)
gcloud iam service-accounts add-iam-policy-binding "$SA" \
  --member="user:$ME" \
  --role="roles/iam.serviceAccountUser" >/dev/null
echo "Allowed $ME to run jobs as $SA"

# Dataproc Serverless workers have no public IPs, so the subnet needs
# Private Google Access to reach Cloud Storage and BigQuery.
gcloud compute networks subnets update default \
  --region="$REGION" \
  --enable-private-ip-google-access
echo "Enabled Private Google Access on the default subnet in $REGION"

