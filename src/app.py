from services import UploadService, LLMService, DBService, build_llm
from embedding_engine import get_embedding_engine

import tempfile
import logging
import os
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from errors import AppError

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

llm = build_llm()

app = FastAPI()


@app.exception_handler(AppError)
async def handle_app_error(_request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.public_message},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(_request, exc: Exception):
    logger.exception("Unhandled API error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )

origins = [
    "null",
    "http://localhost",
    "http://localhost:8002",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    "http://0.0.0.0",
    "http://0.0.0.0:8002",
    "http://127.0.0.1",
    "http://127.0.0.1:8002"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str


MAX_UPLOAD_SIZE = 20 * 1024 * 1024
PDF_SIGNATURE = b"%PDF-"
UPLOAD_CHUNK_SIZE = 1024 * 1024


@app.post("/upload-file/")
def upload_file(file: UploadFile | None = None):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="A PDF file is required.")
    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="The uploaded file must have MIME type application/pdf.")

    temp_file_path = None
    total_size = 0
    signature = b""
    try:
        with tempfile.NamedTemporaryFile(prefix="localrag-", suffix=".pdf", delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)
            while True:
                chunk = file.file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(status_code=413, detail="PDF exceeds the 20 MiB upload limit.")
                if len(signature) < len(PDF_SIGNATURE):
                    signature += chunk[:len(PDF_SIGNATURE) - len(signature)]
                temp_file.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        if not signature.startswith(PDF_SIGNATURE):
            raise HTTPException(status_code=400, detail="The uploaded file does not have a valid PDF signature.")

        service = UploadService(str(temp_file_path), engine=get_embedding_engine())
        service.insert_documents()
        return {"status": "Documents inserted into the vector database."}
    finally:
        if temp_file_path is not None:
            temp_file_path.unlink(missing_ok=True)
        file.file.close()


@app.post("/query/")
def query_llm(request: QueryRequest):
    service = LLMService(
        llm,
        embedding_engine=get_embedding_engine(),
    )
    results = service.answer(request.question)
    return {"results": results}


@app.delete("/database/")
def clear_database():
    service = DBService()
    service.clear_database()
    return {"status": "Vector database cleared."}


@app.get("/database/")
def is_database_empty():
    service = DBService()
    empty = service.check_empty()
    return {"is_empty": empty}
