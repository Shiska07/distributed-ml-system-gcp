FROM python:3.12-slim

WORKDIR /app

# 1. Dependencies first: this layer is reused when only code changes
COPY training/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2. Shared feature/encoding package
COPY common/ /opt/common/
RUN pip install --no-cache-dir /opt/common

# 3. Training code
COPY training/ /app/

# Commit the image was built from, logged to MLflow by train.py
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

ENTRYPOINT ["python", "train.py"]
