ARG MLFLOW_VERSION=3.4.0
FROM ghcr.io/mlflow/mlflow:v${MLFLOW_VERSION}

# Freeze the base image's packages as constraints, so installing the extras
# cannot upgrade anything MLflow depends on (e.g. protobuf, opentelemetry).
RUN pip freeze > /tmp/constraints.txt && \
    pip install --no-cache-dir -c /tmp/constraints.txt psycopg2-binary google-cloud-storage

# Same server arguments as the MLflow guide. The DB password comes from Secret Manager
# (DB_PASSWORD), so the command runs through a shell to fill in the variables.
CMD exec mlflow server \
    --backend-store-uri "postgresql://mlflow:${DB_PASSWORD}@/mlflow?host=/cloudsql/${CLOUDSQL_CONNECTION}" \
    --artifacts-destination "${ARTIFACT_BUCKET}" \
    --host 0.0.0.0 \
    --port 5000 
    