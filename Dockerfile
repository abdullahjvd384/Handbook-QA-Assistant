# Production image (deployed on Azure Container Apps; runs on any container host).
#   docker build -t handbook-qa .
#   docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... handbook-qa
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HANDBOOK_MODEL_CACHE=/models \
    HANDBOOK_VECTOR_STORE=memory \
    HANDBOOK_VECTOR_CACHE=/models/vectors.npz

WORKDIR /app
COPY requirements.txt .
# Runtime dependencies only (no test runner, no ablation-only embedder).
RUN grep -vE '^(pytest|model2vec)==' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# Bake the embedding model into the image so a cold start never downloads it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models')"

COPY app ./app
COPY data ./data
# Embed every chunk now; startup then reuses these vectors (same content hash as the Qdrant sync).
RUN python -c "from app.config import Settings; from app.search.index import KnowledgeBase; KnowledgeBase(Settings()).load()"
RUN useradd --create-home --uid 1000 app && chown -R app /app/data /models
USER app

# The host may set $PORT; default 8080.
CMD ["sh", "-c", "exec uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
