FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU wheel first so the core dependency resolver keeps it.
RUN python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-postgres.txt requirements-ollama.txt requirements-gemini.txt requirements-transformers.txt requirements-ocr.txt ./
RUN python -m pip install --no-cache-dir \
    -r requirements-postgres.txt \
    -r requirements-ollama.txt \
    -r requirements-gemini.txt \
    -r requirements-transformers.txt \
    -r requirements-ocr.txt

COPY src ./src
COPY web ./web

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data /cache/huggingface \
    && chown -R appuser:appuser /data /cache

USER appuser

EXPOSE 8000

CMD ["fastapi", "run", "src/app.py", "--host", "0.0.0.0", "--port", "8000"]
