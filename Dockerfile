FROM python:3.11-slim

WORKDIR /app

ARG REQ_FILE=requirements_hops.txt
COPY ${REQ_FILE} /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libc-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

COPY src/features/ /app/src/features/

COPY Novel_Pred.csv src/features/just_lemme_see.csv /app/
COPY api/ ./api/
COPY test/ ./test/

# Exposing my FASTAPI PORT
EXPOSE 8000

ENTRYPOINT ["python3"]
CMD ["src/features/feature_pipeline.py"]
