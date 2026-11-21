This project is a **RAG (Retrieval-Augmented Generation)** application built with **FastAPI**, designed to allow the ingestion of PDF documents and asking questions about their content using Large Language Models (LLMs).

The system processes documents, generates vector embeddings, and stores them in a **PostgreSQL** database with the **pgvector** extension. Queries can be answered using a lot of differents models such as Gemini, Ollama models and Hugging Face Transformers models.

## Features

*   **Document Upload**: Accepts PDF files, extracts text, and splits it into chunks.
*   **Semantic Search**: Uses embeddings (via `sentence-transformers`) to find relevant sections in the documents.
*   **LLM Integration**: Cloud models like Google Gemini and Local models with Ollama and Hugging Face Transformers Models
*   **REST API**: Simple interface built with FastAPI for upload and queries.
*   **Web Interface:** There's a simple web interface built to communicate with the system effortlessly.

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

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root following `.env.example` template:

### 4. Run the Application

Run the FastAPI server:

```bash
fastapi dev src/app.py
```
The API will be available at `http://localhost:8000/`.

### OPTIONAL. Configure Ollama 

If you plan to use ollama models, ensure you have Ollama installed and running locally. The project is configured to use the `llama3` model by default.

```bash
ollama pull llama3
ollama serve
```

### OPTIONAL. Configure the Database

If you want to persist the data you produce, the project uses Docker Compose to spin up a PostgreSQL instance with the `pgvector` extension enabled.

```bash
docker-compose up -d
```

## Web Interface
```bash
cd web/
python3 -m http.server 8002
# or use an extension like live server, it will work the same.
```

Open `http://localhost:8002` in the browser. Do not open `web/index.html`
directly with `file://`, since browsers treat that page as `origin: null`.

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

GitHub Actions runs focused Ruff checks and the automated tests on pull requests and pushes to `main`. The workflow installs the lightweight `requirements-ci.txt`; run `ruff check --select E4,E7,E9,F src tests`, `ruff format --check tests`, and `python -m pytest -q` locally to match CI. Formatting checks currently cover tests; formatting existing application code is deferred to avoid a broad unrelated diff.

### Database and embedding index compatibility

SQLite and PostgreSQL schemas are initialized through numbered SQL migrations without deleting existing documents. PostgreSQL uses a shared thread-safe pool configured by `POSTGRES_POOL_MIN` and `POSTGRES_POOL_MAX`. Configure `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` to match the embedding model. The index records both values and rejects queries or uploads if they change while vectors remain; clear the database and re-ingest documents to rebuild the index. Existing databases without recorded identity are preserved but require clearing and reindexing before vector search.

You can interact with the API through the automatic documentation (Swagger UI) at `http://localhost:8000/docs`.

1.  **PDF Upload**:
    *   Use the `POST /upload-file/` endpoint with a multipart `file` field to upload a PDF file (maximum size: 20 MiB).
    *   The endpoint requires a `.pdf` filename, `application/pdf` MIME type, and a valid `%PDF-` signature. Invalid files return HTTP 400; files over the limit return HTTP 413.

2.  **Ask Questions**:
    *   Use `POST /query/` with `{"question": "Your question here"}`.
    *   Successful responses include `results` (answer text) and `sources` (document name, page, chunk, excerpt, and similarity for each retrieved passage).

### PDF parsing and OCR

Chunking is configured with `CHUNK_SIZE` and `CHUNK_OVERLAP` (defaults: 1000 and 200 characters). Text extraction keeps pages separate, preserves line boundaries, and drops empty or repeated chunks within a page. Scanned pages are reported as having no processable text unless OCR is enabled. To enable OCR, install `requirements-ocr.txt`, install the Tesseract system executable and the language data selected by `OCR_LANG`, then set `OCR_ENABLED=true`. OCR defaults to disabled. Each stored source records whether it was extracted from PDF text or OCR.

3. **Changing Models**
    *   Change the .env `LLM_MODEL` variable to change the provider in use. Available now: [`gemini`, `ollama`, `hugging-face-model-id`].
    *   For Gemini, configure `GEMINI_MODEL`.
    *   For Ollama, configure `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, and `OLLAMA_TEMPERATURE`.
    *   For Transformers, set `LLM_MODEL=transformers` and configure `TRANSFORMERS_MODEL`.
    *   Gemini requests use `GEMINI_REQUEST_TIMEOUT` and `GEMINI_RETRIES`; the defaults are 30 seconds and 1 retry.

3.  **Clear Database**:
    *   Use `DELETE /clear-database/` to remove all documents and embeddings.

### Database and embedding index compatibility

SQLite and PostgreSQL schemas are initialized through numbered SQL migrations without deleting existing documents. PostgreSQL uses a shared thread-safe pool configured by `POSTGRES_POOL_MIN` and `POSTGRES_POOL_MAX`. Configure `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` to match the embedding model. The index records both values and rejects queries or uploads if they change while vectors remain; clear the database and re-ingest documents to rebuild the index. Existing databases without recorded identity are preserved but require clearing and reindexing before vector search.
