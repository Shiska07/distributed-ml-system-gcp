# End-to-End ML System on Google Cloud

A production-style machine learning system on **Google Cloud Platform**. It covers the full
lifecycle: data pipelines, model training, deployment, and monitoring against production traffic.

The use case is predicting whether a Chicago taxi rider tips more than 20%. The focus is the
**infrastructure and MLOps workflow for automation**.

## Tech Stack

- **Data:** BigQuery · SQL · Apache Beam · Dataflow · Apache Spark · Dataproc Serverless · Parquet
- **ML:** scikit-learn · XGBoost · MLflow · Vertex AI Training · Model Registry · Endpoints
- **Infrastructure:** Cloud Storage · Cloud SQL · Cloud Run · Secret Manager · IAM · Docker · Cloud Build · Artifact Registry · gcloud CLI · Bash
- **CI/CD:** Python · pytest · Git · GitHub Actions

## Status

| Part | Status |
|---|---|
| GCP foundation (buckets, IAM, networking) 
| Cleaning pipeline (Beam → silver), packaged as a Dataflow Flex Template 
| Feature pipeline (Spark → gold/features) 
| Training dataset builder (Spark → gold/labels) 
| Training code (LR / RF / XGBoost, tuning, MLflow logging + registry), tests 
| MLflow tracking server (Cloud Run + Cloud SQL + GCS) | In progress |
| Training on Vertex AI, CI (GitHub Actions) | Planned |
| Serving (endpoint + streaming inference), monitoring | Planned |
| Pipeline automation (Workflows + Scheduler) | Planned |

## Architecture

```
BigQuery (historical trips)        GCS landing/ (production batches)
           └────────────────┬────────────────┘
                            │  Beam on Dataflow   → validate, clean, dedupe; split trips / outcomes
                            ▼
silver/trips     (what is known at payment)           silver/outcomes   (tips: the ground truth)
        │  Spark on Dataproc → features                      │
        ▼                                                    │
gold/features    (stored once)                               │
        │                                                    │
        └──► Spark: join + label + split ◄───────────────────┘
                            ▼
gold/labels/<version>  (labels + split + manifest; no feature copies)
                            │  Vertex AI Training: tune → fit → evaluate
                            ▼
MLflow (Cloud Run)  runs, metrics, lineage → Cloud SQL · model files → GCS
  └─ Model Registry: @champion / @challenger
                            │  champion deployed
                            ▼
Vertex AI Endpoint (encoder + model in one pipeline)
        ▲
        │  Dataflow streaming: same cleaning + feature code
Pub/Sub ◄── held-out raw trips replayed as production requests
                            │
                            ▼
BigQuery prediction log + ground truth → live evaluation, drift monitoring, retraining
```

## Key Design Choices

- **Features and labels separation.** Cleaning splits each record into a trip and its
  outcome. The label is created only when a training dataset is built, so it can't leak into
  features, and the label rule can change without reprocessing.
- **One code path for training and serving.** The same cleaning (Beam) and feature code
  (`common/`) runs in batch for training and in streaming for live requests, avoiding
  training/serving skew.
- **Encoding saved with the model.** Each model is an sklearn Pipeline (encoder + estimator).
  Category vocabularies and scaling are fitted on the train split only and travel with the model.
- **Features stored once.** Training datasets hold only labels and the split, and point to the
  feature rows they use. No data is duplicated per version.
- **Immutable, traceable data and models.** Every run writes to its own folder; manifests record
  inputs, code version and row counts. Each MLflow run logs the dataset manifest and git commit.
- **Time-based evaluation.** Splits and cross-validation respect time order; the decision
  threshold is chosen on validation and test is used only for reporting.
- **Champion / challenger.** New models are registered as `@challenger` and replace
  `@champion` only through an explicit promotion rule.
- **Beam for cleaning, Spark for dataset-wide work.** Beam handles record-level checks and runs
  in batch or streaming with the same code; Spark handles joins, splits and aggregations.
- **Serverless where possible.** Dataflow, Dataproc Serverless, Cloud Run and Vertex AI: no
  clusters to manage, and idle services scale to zero.

## Repository Structure

```
.
├── config/dev.env      # project, region, buckets, service accounts, MLflow settings
├── infra/              # one-time setup: buckets, IAM, Artifact Registry, MLflow server
│   └── mlflow/         # MLflow server image (Cloud Run)
├── common/             # shared features, labels and encoding (training, serving, monitoring)
├── etl/                # data pipelines: Beam (silver), Spark (features, training datasets)
├── training/           # data loading, models, tuning grids, train.py (MLflow logging + registry)
└── simulation/         # writes held-out data into the landing zone as production batches
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
bash infra/04_ml_service_account.sh
bash infra/05_mlflow_server.sh          # MLflow on Cloud Run + Cloud SQL + GCS

# Python environments: one per component (different dependencies)
uv venv .venv --python 3.12
uv pip install --python .venv -r etl/requirements.txt -e common/

uv venv training/.venv --python 3.12
uv pip install --python training/.venv -r training/requirements.txt -e common/

# Tests: run each component separately
.venv/bin/python -m pytest etl -q
training/.venv/bin/python -m pytest common -q
training/.venv/bin/python -m pytest training -q
```

See [etl/README.md](etl/README.md) to run the pipelines.