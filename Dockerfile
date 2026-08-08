# --- Build stage: train the model as part of the image build ---------------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY data/ data/

# Train once at build time so the served image already has a model baked in.
# (For a real production setup, you'd instead pull a versioned model
# artifact from a model registry / object store at container startup.)
RUN python train.py --n_estimators 200 --max_depth 5

# --- Runtime stage: slim image with only what's needed to serve -------------
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY --from=builder /app/models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
