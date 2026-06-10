FROM python:3.12-slim

WORKDIR /app

# System deps for lxml (used by trafilatura)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY datasets/ ./datasets/
COPY sources.yaml .

# Build the Chroma store at image-build time so it's baked into the image.
# (No GROQ_API_KEY needed for ingestion — only embeddings run here.)
RUN python -m app.build_kb --reset

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
