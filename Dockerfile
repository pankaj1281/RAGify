# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies required by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer-caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/faiss_index data/embedding_cache

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
