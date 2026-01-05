# =========================
# Stage 1: Builder
# =========================
FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# =========================
# Stage 2: Runtime
# =========================
FROM python:3.10-slim

# Create non-root user
RUN useradd -m appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src
COPY models ./models
COPY .env .env

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl --fail http://localhost:8000/ || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
