# SafeX Solutions FAQ Chatbot — container image
FROM python:3.12-slim

WORKDIR /app

# Install system deps needed by torch/sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set defaults; override at `docker run` time with -e
ENV EMBEDDING_BACKEND=sentence-transformers
ENV GENERATION_MODE=extractive
# ENV ANTHROPIC_API_KEY=  # pass at runtime, never bake secrets into the image

EXPOSE 8000

# Basic container health check hitting our /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
