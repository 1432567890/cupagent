FROM python:3.12-slim AS base

WORKDIR /app

# Install build deps for asyncpg (libpq-dev) and TgCrypto (libc6-dev for stdint.h)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
