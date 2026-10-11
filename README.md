# End-to-End ML System on Google Cloud

A production-style machine learning system on **Google Cloud Platform**. It covers the full
lifecycle: data pipelines, model training, deployment, and monitoring against production traffic.

The use case is predicting whether a Chicago taxi rider tips more than 20%. The focus is the
**infrastructure and MLOps workflow for automation**.

## Tech Stack

- **Data:** BigQuery · SQL · Apache Beam · Dataflow · Apache Spark · Dataproc Serverless · Parquet
- **ML:** Vertex AI Training · Model Registry · Endpoints · MLflow · Pipelines
- **Infrastructure:** Cloud Storage · IAM · gcloud CLI · Bash · Docker · Cloud Build · Artifact Registry
- **CI/CD:** Python · pytest · Git · GitHub Actions

## Status

| Part | Status |
|---|---|
| GCP foundation (buckets, IAM, networking) 
| Cleaning pipeline (Beam → silver), packaged as a Dataflow Flex Template 
| Feature pipeline (Spark → gold/features) 
| Training dataset builder (Spark → gold/labels) 
| Pipeline automation (Workflows + Scheduler), CI/CD | Planned |
| Training, registry, serving, monitoring | Planned |

## Architecture

```
BigQuery (historical trips)        GCS landing/ (production batches)
           └────────────────┬────────────────┘
                            │  Beam on Dataflow   → validate, clean, dedupe; split trips / outcomes
                            ▼
silver/trips     (what is known at payment)           silver/outcomes   (tips: the ground truth)
        │  Spark on Dataproc → features                      │
        ▼                                                    │
gold/features    (stored once; training + inference)         │
        │                                                    │
        ├──► batch / online inference                        │
        │                                                    │
        └──► Spark: join + label + split ◄───────────────────┘
                            ▼
gold/labels/<version>  (labels + split + manifest; no feature copies)
                            │  Vertex AI Training → train and evaluate
                            ▼
Vertex AI Model Registry   versioned models with data lineage
                            │
                            ▼
Vertex AI Endpoint  ◄── Pub/Sub ◄── held-out data replayed as production traffic
                            │
                            ▼
BigQuery prediction log + ground truth → live evaluation, drift monitoring, retraining
```

## Key Design Choices

- **Features and labels separation.** Cleaning splits each record into a trip and its
  outcome. The label is created only when a training dataset is built, so it can't leak into
  features, and the label rule can change without reprocessing.
- **One feature pipeline for training and serving.** The same code (`common/`) builds features
  for historical and live data, avoiding training/serving skew.
- **Features stored once.** Training datasets hold only labels and the split, and point to the
  feature rows they use. No data is duplicated per version.
- **Immutable, traceable data.** Every run writes to its own folder and is never overwritten.
  Manifest files record inputs, code version and row counts.
- **Beam for cleaning, Spark for dataset-wide work.** Beam handles record-level checks and can
  move to streaming with the same code; Spark handles joins, splits and aggregations.
- **Realistic evaluation.** A later time window is held out from training and replayed as
  production traffic, so live predictions can be scored against known outcomes.
- **Serverless where possible.** Dataflow, Dataproc Serverless and Vertex AI: no clusters to manage.

## Repository Structure

```
.
├── config/dev.env      # project, region, buckets, service account, date windows
├── infra/              # one-time setup: buckets, IAM, networking, Artifact Registry
├── common/             # shared feature and label logic (training, serving, monitoring)
├── simulation/         # writes held-out data into the landing zone as production batches
└── etl/                # data pipelines: Beam (silver), Spark (features, training datasets)
```

## Documentation

- [ETL pipelines](etl/README.md): cleaning, features, training datasets, how to run

## Quick Start

```bash
git clone https://github.com/Shiska07/gcp-mlops-tabular.git
cd gcp-mlops-tabular
source config/dev.env

# One-time infrastructure setup
bash infra/00_create_buckets.sh
bash infra/01_service_account.sh
bash infra/02_artifact_registry.sh

# Python environment and tests
python3 -m venv .venv && source .venv/bin/activate
pip install -r etl/requirements.txt
pip install -e common/
pytest common etl -q
```

See [etl/README.md](etl/README.md) to run the pipelines.