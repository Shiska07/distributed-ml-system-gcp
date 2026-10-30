FROM python:3.12-slim

WORKDIR /app

# 1. Dependencies first (cached when only code changes)
COPY serving/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2. Shared package: features + encoding (the pickled model refers to it)
COPY common/ /opt/common/
RUN pip install --no-cache-dir /opt/common

# 3. Prediction server
COPY serving/app.py /app/app.py

# Vertex sets AIP_HTTP_PORT (8080 by default); the shell form fills it in at start.
CMD exec uvicorn app:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}
