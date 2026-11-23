import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db.factory_db import close_vector_db
from errors import AppError
from job_manager import JobManager, JobQueueFull, public_job
from services import DBService, build_llm
from settings import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

llm = build_llm(settings)


@asynccontextmanager
async def lifespan(_app):
    manager = JobManager(llm, settings.job_upload_dir)
    manager.start()
    _app.state.job_manager = manager
    try:
        yield
    finally:
        manager.stop()
        close_vector_db()


app = FastAPI(lifespan=lifespan)


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
    "http://127.0.0.1:8002",
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


@app.post("/upload-file", status_code=202)
async def upload_file(file: UploadFile | None = None):
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="A PDF file is required.")
    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="The uploaded file must have MIME type application/pdf.",
        )

    temp_file_path = Path(settings.job_upload_dir) / f"{uuid.uuid4().hex}.pdf"
    total_size = 0
    signature = b""
    queued = False
    try:
        temp_file_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_file_path.open("wb") as temp_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413, detail="PDF exceeds the 20 MiB upload limit."
                    )
                if len(signature) < len(PDF_SIGNATURE):
                    signature += chunk[: len(PDF_SIGNATURE) - len(signature)]
                temp_file.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
        if not signature.startswith(PDF_SIGNATURE):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file does not have a valid PDF signature.",
            )

        job_id = uuid.uuid4().hex
        try:
            job = app.state.job_manager.submit(
                job_id,
                "upload",
                {"file_path": str(temp_file_path)},
            )
        except JobQueueFull as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        queued = True
        return public_job(job)
    finally:
        if not queued:
            temp_file_path.unlink(missing_ok=True)
        await file.close()


@app.post("/query", status_code=202)
def query_llm(request: QueryRequest):
    if DBService().check_empty():
        raise HTTPException(
            status_code=400, detail="Upload a document before querying."
        )
    try:
        job = app.state.job_manager.submit(
            uuid.uuid4().hex,
            "query",
            {"question": request.question},
        )
    except JobQueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return public_job(job)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = app.state.job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Processing task not found.")
    return public_job(job)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    job = app.state.job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Processing task not found.")
    if not app.state.job_manager.cancel(job_id):
        raise HTTPException(
            status_code=409,
            detail="Only tasks that have not started can be cancelled.",
        )
    job = app.state.job_manager.get(job_id)
    return public_job(job)


@app.delete("/database")
def clear_database():
    if not app.state.job_manager.clear_database_if_idle():
        raise HTTPException(
            status_code=409,
            detail="Wait for or cancel queued tasks before clearing the database.",
        )
    return {"status": "Vector database cleared."}


@app.get("/database")
def is_database_empty():
    service = DBService()
    empty = service.check_empty()
    return {"is_empty": empty}


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.mount(
    "/",
    StaticFiles(directory=Path(__file__).resolve().parents[1] / "web", html=True),
    name="web",
)
