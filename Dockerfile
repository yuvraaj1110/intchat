FROM python:3.12-slim

# System deps for lxml (used by trafilatura). Must run as root, before USER.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000. Create that user so the
# Chroma SQLite store and the embedding-model cache are writable at runtime.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    ANONYMIZED_TELEMETRY=False

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user app/ ./app/
COPY --chown=user datasets/ ./datasets/
COPY --chown=user sources.yaml .

# Build the Chroma store at image-build time so it's baked into the image.
# Downloads the embedding model to HF_HOME (owned by `user`, persists at runtime).
# No GROQ_API_KEY needed here — only embeddings run during ingestion.
RUN python -m app.build_kb --reset

# HF Spaces expects the app on port 7860 (override with $PORT elsewhere).
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
