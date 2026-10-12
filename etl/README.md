# ETL Pipelines

Three steps turn raw taxi data into training and inference data:

| Step | Tool | Input | Output |
|---|---|---|---|
| Cleaning | Apache Beam on Dataflow (Flex Template) | BigQuery or the landing zone | `silver/trips`, `silver/outcomes` |
| Features | Apache Spark on Dataproc Serverless | One silver trips run | `gold/features` |
| Training dataset | Apache Spark on Dataproc Serverless | Features + outcomes | `gold/labels/<version>` |

```
landing/   raw production batches (JSON)
silver/    cleaned trips and outcomes, one folder per run
rejected/  records that failed validation, with reasons
gold/      features (for training and inference) and training labels
```

---

## Key Design Decisions

- **Features and labels are kept separate.** Beam splits every record into a *trip*
  (what is known at payment time) and an *outcome* (the tip). The label only meets the
  features when a training dataset is built, so it can never leak into the features.
- **One feature pipeline for training and serving.** The same feature job runs on historical
  and live data, using one shared feature module (`common/`). This avoids training/serving skew.
- **Features are stored once.** A training dataset stores only labels and the split, and points
  to the feature rows it uses. No feature data is copied per dataset version.
- **Immutable, traceable runs.** Every run writes to a new folder and never overwrites. Each
  output has a `_manifest.json` (inputs, git commit, row counts), written last, so a folder
  without a manifest is incomplete.
- **Labels are derived late.** Silver keeps raw tips. The label rule (`tips / fare > 0.20`)
  is applied when building a training dataset, so it can change without reprocessing.

---

## 1. Data Exploration

Profiling `bigquery-public-data.chicago_taxi_trips` found missing values, duplicate trip IDs,
impossible trips, and that **tips are only recorded for card payments**. Cash trips are
excluded, so the model learns real tipping behavior instead of a recording gap.

---

## 2. Cleaning (Beam → silver)

### Sources and datasets

`--source` says where data comes from; `--dataset` says what to produce.

| Source | Reads from | Produces |
|---|---|---|
| `historical` | BigQuery (`sql_queries/02_load_table.sql`) | trips + outcomes (one read) |
| `live` | `landing/live_trips/` | trips |
| `live` | `landing/ground_truth/` | outcomes |

Live trips and outcomes arrive at different times, so they are usually run separately
(`--dataset trips` / `--dataset outcomes`). A date range becomes one file pattern per day,
so no extra tables are needed.

### Steps

1. **Read** and convert to one common shape per dataset.
2. **Validate:** `CleanTrip` (trip rules, card-only filter) and `CleanOutcome` (tip rules).
   Failures go to `rejected/` with the reasons.
3. **Deduplicate** on `unique_key`.
4. **Write** Parquet to silver.

### Output

```
silver/trips/source=<historical|live>/run_id=<start>_<end>_<ts>/
silver/outcomes/source=<historical|live>/run_id=<start>_<end>_<ts>/
```

Trips and outcomes are joined later by `unique_key`, never by folder.

### Flex Template

The pipeline is packaged as a **Dataflow Flex Template** (image in Artifact Registry + spec
in GCS), tagged with the git commit. It runs from parameters only: `source`, `dataset`,
`start_date`, `end_date`, `out_lake`, optional `run_ts`.

| File | Purpose |
|---|---|
| `ingestion/sources.py` | Readers per source; splits rows into trips and outcomes |
| `ingestion/transforms.py` | Validation rules, `CleanTrip`, `CleanOutcome` |
| `ingestion/pipeline.py` | Arguments, paths, clean → dedupe → write per dataset |
| `ingestion/Dockerfile`, `cloudbuild.yaml`, `metadata.json` | Flex Template build and parameters |

---

## 3. Features (Spark → gold/features)

**Prediction moment:** end of the trip, at payment. Only values known then are used.

| Group | Features |
|---|---|
| Time | `hour`, `day_of_week`, `is_weekend` |
| Trip | `trip_miles`, `trip_minutes`, `avg_speed_mph` |
| Cost | `fare`, `fare_per_mile`, `extras`, `tolls`, `has_extras` |
| Place | `pickup_area`, `dropoff_area`, `is_airport`, `is_downtown`, `same_area` |
| Operator | `company` |

- All features are row-level and stateless, so serving needs no lookups.
- Spark runs the shared pandas code from `common/` with `mapInPandas`.
- The output keeps the silver run's source and `run_id`:
  `gold/features/source=<S>/run_id=<R>/`.
- Learned encodings (company vocabulary, scaling) are **not** stored here. They are fitted
  during training on the train split only and saved with the model.

---

## 4. Training Dataset (Spark → gold/labels)

1. Read all feature and outcome runs of a source; keep trips in the date window.
2. Keep the newest row per `unique_key` (handles reruns).
3. Inner join on `unique_key`; drop unrealistic tips.
4. Create the label with the shared rule from `common/`.
5. Time-based split (~80/10/10): val and test always come after train.

Output: `gold/labels/<version>/split=train|val|test/` with `unique_key`, `feature_run_id`
and `label`. The manifest lists the feature and outcome runs used, the match rate, split
dates, and row count and label rate per split. Training joins features and labels on
`unique_key` + `feature_run_id`.

| File | Purpose |
|---|---|
| `features/get_features.py` | Silver trips run → features + manifest |
| `features/build_training_dataset.py` | Features + outcomes → labels/split + manifest |
| `../common/taxi_features/` | Shared feature and label code |

---

## 5. Validation and Tests

- **SQL checks** per silver run (duplicates, nulls, date window) as `ASSERT`s in
  `sql_queries/03_validate_silver.sql`.
- **Unit tests** for validation rules, readers, features, the Spark schema and the training
  helpers. They run without GCP.

```bash
pytest common etl -q
```

---

## 6. How to Run

From the repo root, after `source config/dev.env`:

| Script | What it does |
|---|---|
| `simulation/00_seed_landing.sh [START END]` | Seeds the landing zone with held-out data |
| `etl/scripts/01_build_beam_template.sh` | Builds the Flex Template (image + spec) |
| `etl/scripts/02_run_spark_features.sh historical\|live\|<run path>` | Builds features for a silver run |
| `etl/scripts/03_run_spark_training_dataset.sh SOURCE START END VERSION` | Builds a training dataset |

**Clean data with the template:**

```bash
RUN_TS=$(date -u +%Y%m%d-%H%M%S)
gcloud dataflow flex-template run "clean-historical-$RUN_TS" \
  --template-file-gcs-location "$CODE/templates/clean-trips/latest.json" \
  --region "$REGION" --service-account-email "$SA" \
  --staging-location "$TEMP/dataflow/staging" --temp-location "$TEMP/dataflow/temp" \
  --disable-public-ips --max-workers 4 \
  --parameters "source=historical,dataset=all,start_date=$TRAIN_START,end_date=$TRAIN_END,out_lake=$LAKE,run_ts=$RUN_TS"
```

**Then features and a training dataset:**

```bash
bash etl/scripts/02_run_spark_features.sh historical
bash etl/scripts/03_run_spark_training_dataset.sh historical "$TRAIN_START" "$TRAIN_END" v1
```