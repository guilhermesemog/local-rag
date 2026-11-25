This project is a **RAG (Retrieval-Augmented Generation)** application built with **FastAPI**, designed to allow the ingestion of PDF documents and asking questions about their content using Large Language Models (LLMs).

The system processes documents, generates vector embeddings, and stores them in SQLite by default or PostgreSQL with the **pgvector** extension. Queries can use Gemini, Ollama, Hugging Face Transformers, or the dummy provider for initial setup.

## Features

*   **Document Upload**: Accepts PDF files, extracts text, and splits it into chunks.
*   **Semantic Search**: Uses embeddings (via `sentence-transformers`) to find relevant sections in the documents.
*   **LLM Integration**: Google Gemini, Ollama, Hugging Face Transformers, or a no-credentials dummy provider for initial setup.
*   **REST API**: Simple interface built with FastAPI for upload and queries.
*   **Web Interface:** There's a simple web interface built to communicate with the system effortlessly.
*   **Progressive answers**: The web interface streams answers as they are generated, shows sources by page, and offers a copy button.
*   **Local conversation history**: The interface keeps a per-session transcript in the browser, reuses recent turns as context, and lets you clear the conversation without touching the document index.
*   **Document selection**: Questions can be filtered to one or more indexed documents, and the interface shows document/chunk counts plus remove/replace actions.

## Demo

You can have a preview without download anything in the following [notebook](https://colab.research.google.com/drive/13Fen8ZioJPHrGCFYvW5CSztQWw2TRoTC?usp=sharing)!

## Prerequisites

*   **Docker** and **Docker Compose**.
*   **Python 3.10+**.
*   **Ollama** (optional, if you wish to use local models).
*   **Google API Key** (optional, if you wish to use Gemini).

## Installation and Execution

### 1. Setup the repository

```bash
git clone https://github.com/dionae1/local-rag.git
cd local-rag
```

### 2. Configure the Python Environment

It is recommended to use a virtual environment:

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install PyTorch first so the CPU-only build is used by default, then install the core application dependencies. `fastapi[standard]` includes the `fastapi` CLI used by the startup command below, along with Uvicorn:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build matching your CUDA version using the [official PyTorch selector](https://pytorch.org/get-started/locally/) before `requirements.txt`. Install only the optional dependency groups you need:

```bash
pip install -r requirements-dev.txt          # tests and Ruff
pip install -r requirements-gemini.txt        # Gemini provider
pip install -r requirements-ollama.txt        # Ollama provider
pip install -r requirements-transformers.txt # Transformers provider
pip install -r requirements-postgres.txt      # PostgreSQL and pgvector
pip install -r requirements-ocr.txt           # OCR Python packages (also needs Tesseract)
```

### 3. Configure and run with Docker Compose

Copy `.env.example` to `.env`. The defaults use SQLite and `LLM_MODEL=dummy`, so no provider credentials or database server are needed for initial startup. Build and start the application:

```bash
docker compose up --build -d
```

Open the web interface at `http://localhost:8000/`, the API docs at `http://localhost:8000/docs`, and the health endpoint at `http://localhost:8000/health`. The embedding model is downloaded the first time it is needed, so the machine needs internet access for the initial upload.

The SQLite database and downloaded embedding model are stored in named Docker volumes. Stop the services with `docker compose down`; this preserves both. `docker compose down -v` also deletes the volumes and indexed documents.

#### Optional PostgreSQL/pgvector

Set a local `DB_PASSWORD` in `.env`, then start the PostgreSQL overlay:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml up --build -d
```

This configures the API to use PostgreSQL on the Compose network and stores its data in a separate named volume. The database is also published on host port 5433 by default; set `POSTGRES_HOST_PORT` to change it. Use the same two Compose files with `down` to stop this stack. Data is removed only with `down -v`.

#### Optional Ollama service

Set `LLM_MODEL=ollama` in `.env`, then start the Ollama profile:

```bash
docker compose --profile ollama up --build -d
docker compose --profile ollama exec ollama ollama pull llama3
```

The default Ollama model is `llama3`; change `OLLAMA_MODEL` and the pull command together if you select another model. Model files persist in a named volume. To combine Ollama and PostgreSQL, add both Compose files and the profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.postgres.yml --profile ollama up --build -d
```

#### Run directly without Docker

Copy `.env.example` to `.env`, install the dependencies as described above, and run:

```bash
fastapi dev src/app.py
```

The API and web interface are served from `http://localhost:8000/`. Do not open `web/index.html` directly with `file://`.

## How to Use

### RAG evaluation

The reproducible benchmark uses `samples/sample.pdf`, an isolated temporary SQLite database, the configured embedding model, and the configured real LLM. Set `LLM_MODEL` to `gemini`, `ollama`, or `transformers` and configure its model settings before running; the dummy provider is rejected so it cannot produce misleading quality metrics.

```bash
python -m evaluation.run
python -m evaluation.run --chunk-sizes 600 1000 --chunk-overlap 150
python -m evaluation.run --embedding-models all-MiniLM-L6-v2 other-compatible-model
```

The command prints the report and writes `evaluation-report.json` by default (override with `--output`). It reports expected-page recall at `k=6`, evidence and answer term coverage, heuristic refusal detection, and indexing/query embedding, retrieval, and generation latency. Term coverage is a local diagnostic rather than a semantic or factuality guarantee; inspect the per-question answers, evidence and sources manually. The configured embedding model must produce 384-dimensional vectors for the current vector stores. Provider calls can consume local compute, API quota, or cost.

### Automated Tests

Run the unit and API tests with `pytest -q`. The suite uses temporary SQLite databases and mocks external model calls. PostgreSQL integration tests are skipped by default; set `LOCAL_RAG_TEST_POSTGRES=1` and configure the database environment variables to run them against a dedicated test database.

GitHub Actions runs focused Ruff checks and the automated tests on pull requests and pushes to `main`. The workflow installs `requirements-ci.txt`; locally, install `requirements-dev.txt` after the CPU PyTorch setup, then run `ruff check --select E4,E7,E9,F src tests`, `ruff format --check tests`, and `python -m pytest -q` to match CI. Formatting checks currently cover tests; formatting existing application code is deferred to avoid a broad unrelated diff.

### Database and embedding index compatibility

SQLite and PostgreSQL schemas are initialized through numbered SQL migrations without deleting existing documents. PostgreSQL uses a shared thread-safe pool configured by `POSTGRES_POOL_MIN` and `POSTGRES_POOL_MAX`. Configure `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` to match the embedding model. The index records both values and rejects queries or uploads if they change while vectors remain; clear the database and re-ingest documents to rebuild the index. Existing databases without recorded identity are preserved but require clearing and reindexing before vector search.

You can interact with the API through the automatic documentation (Swagger UI) at `http://localhost:8000/docs`.

1.  **PDF Upload**:
    *   Use the `POST /upload-file` endpoint with a multipart `file` field to upload a PDF file (maximum size: 20 MiB). It returns HTTP 202 with a task `id`; poll `GET /jobs/{id}` for processing progress and completion.
    *   The endpoint requires a `.pdf` filename, `application/pdf` MIME type, and a valid `%PDF-` signature. Invalid files return HTTP 400; files over the limit return HTTP 413.

2.  **Ask Questions**:
    *   Use `POST /query` with `{"question": "Your question here"}`. It returns HTTP 202 with a task `id`; poll `GET /jobs/{id}` until it completes.
    *   Add `"document_ids": ["<id>", "..."]` to restrict retrieval to selected documents; omit it or send an empty list to search the whole index. Add `"history": [{"question": "...", "answer": "..."}]` to provide up to four previous turns as conversation context. The web interface fills both fields automatically.
    *   A completed task contains `result.results` (answer text) and `result.sources` (document name, page, chunk, excerpt, and similarity for each retrieved passage). Failed tasks expose a safe `error` message.
    *   Task states are `queued`, `processing`, `completed`, `failed`, or `cancelled`. `progress` is a coarse percentage and `phase` describes the current stage. `DELETE /jobs/{id}` cancels a task only while it is queued; tasks already running return HTTP 409.
    *   `DELETE /database` returns HTTP 409 while any task is queued or processing, preventing a background task from repopulating a database that was just cleared. Both this endpoint and `DELETE /jobs/{id}` require the `X-Admin-Key` header.
    *   Uploads and queries have separate single-worker queues, allowing queries to continue while a PDF is being indexed. Each queue accepts up to 20 waiting tasks; when full, new submissions return HTTP 503. The queue and task states use the configured SQLite or PostgreSQL database. After restart, queued tasks resume and interrupted running tasks are marked failed.
    *   Staged PDFs are kept under `JOB_UPLOAD_DIR` until ingestion finishes (default: `data/uploads`; Docker Compose: `/data/uploads` in the persistent SQLite data volume).

### Documents and conversation history

*   `GET /documents` lists indexed documents with their chunk and page counts, plus the totals shown by the interface.
*   `DELETE /documents/{id}` removes all chunks of one document. It requires `X-Admin-Key`, returns `404` when the document is not indexed, and reports the number of removed chunks.
*   `POST /documents/{id}/replace` accepts a multipart `file` field, requires `X-Admin-Key`, and returns HTTP 202 with an upload task. The new PDF is processed first; the old chunks are replaced atomically when the job succeeds, so a failed replacement leaves the previous document intact. A document's identity is the `document_id` stored in its chunk metadata.
*   Conversation history is stored in the browser's `localStorage`, keyed by a generated session identifier. The last four completed exchanges are sent with each question; at most 50 exchanges are kept per session. **Clear conversation** removes only that local transcript and starts a new session; it does not delete documents or embeddings. Clearing site data also clears the transcript.
*   Changing documents or filters does not erase the conversation. Filters are stored separately from the document index and are pruned automatically when a selected document is removed. Clearing the document index does not erase the local transcript either.

### Streaming answers

*   `POST /query/stream` creates the same background query task and responds with `text/event-stream`. The events are `job`, `phase`, `sources`, `token`, `done`, and `error`; `sources` is sent as soon as retrieval finishes, before generation starts.
*   `GET /jobs/{id}/stream` follows or replays an existing task, so the interface can reconnect after a reload. If the in-memory channel is unavailable (for example after a restart), the endpoint falls back to the persisted task state.
*   The web interface prefers the streaming endpoint and falls back to `POST /query` polling when streaming is unavailable. Closing the tab does not cancel the task; cancellation still applies only to queued tasks.
*   Streaming buffers live in memory, so the API must run as one process (the Compose setup does). Put public deployments behind a proxy configured to disable response buffering; the API sends `X-Accel-Buffering: no` and periodic heartbeat comments.

### Security configuration

Set a unique `ADMIN_API_KEY` generated from at least 32 random bytes through the deployment environment or its secret manager to enable database clearing and task cancellation. There is no built-in key; when it is unset, these actions fail closed with HTTP 503. For local development, a value may be placed in the ignored `.env` file, but never commit it. The web interface requests the key when clearing the database and keeps it only in memory for that request. The Compose listener serves plain HTTP; put public deployments behind a trusted TLS-terminating reverse proxy and never send the admin key over plain HTTP.

Rate limits are per client IP and per process: 5 uploads, 20 queries, 10 administrative actions, and 120 status reads per minute. Excess requests receive HTTP 429 with `Retry-After`. The service uses the direct connection IP and does not trust `X-Forwarded-For`; deployments behind a proxy must configure a trusted proxy setup before relying on per-client limits. Set `CORS_ALLOWED_ORIGINS` to a comma-separated allowlist for cross-origin development or deployment; same-origin Docker usage does not require CORS. PDF files remain limited to 20 MiB, with multipart overhead capped at 64 KiB and upload body receipt limited to 300 seconds. Query bodies are capped at 64 KiB and 30 seconds; questions are limited to 2,000 characters, document filters to 50 identifiers, and conversation history to four turns totaling 6,000 characters. Gemini uses `GEMINI_REQUEST_TIMEOUT` (30 seconds by default), Ollama uses `OLLAMA_REQUEST_TIMEOUT` (120 seconds), and Transformers output is limited to 512 generated tokens.

### PDF parsing and OCR

Chunking is configured with `CHUNK_SIZE` and `CHUNK_OVERLAP` (defaults: 1000 and 200 characters). Text extraction keeps pages separate, preserves line boundaries, and drops empty or repeated chunks within a page. Scanned pages are reported as having no processable text unless OCR is enabled. For local Python runs, install `requirements-ocr.txt` and the Tesseract system executable with the language data selected by `OCR_LANG`. The Docker image includes Tesseract English data; set `OCR_ENABLED=true` in `.env` to enable it there. Other OCR languages need their Tesseract language data added to the image. OCR defaults to disabled. Each stored source records whether it was extracted from PDF text or OCR.

3. **Changing Models**
    *   Change `.env` `LLM_MODEL` to `dummy`, `gemini`, `ollama`, or `transformers`.
    *   For Gemini, configure `GEMINI_MODEL` and `GEMINI_API_KEY`.
    *   For Ollama, configure `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_TEMPERATURE`.
    *   For Transformers, set `LLM_MODEL=transformers` and configure `TRANSFORMERS_MODEL`.
    *   Gemini requests use `GEMINI_REQUEST_TIMEOUT` and `GEMINI_RETRIES`; the defaults are 30 seconds and 1 retry.

3.  **Clear Database**:
    *   Use `DELETE /database` to remove all documents and embeddings. `GET /database` reports whether the index is empty.

### Database and embedding index compatibility

SQLite and PostgreSQL schemas are initialized through numbered SQL migrations without deleting existing documents. PostgreSQL uses a shared thread-safe pool configured by `POSTGRES_POOL_MIN` and `POSTGRES_POOL_MAX`. Configure `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` to match the embedding model. The index records both values and rejects queries or uploads if they change while vectors remain; clear the database and re-ingest documents to rebuild the index. Existing databases without recorded identity are preserved but require clearing and reindexing before vector search.
