import logging
import uuid
from contextlib import asynccontextmanager
from hmac import compare_digest
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from db.factory_db import close_vector_db
from errors import AppError
from job_manager import JobManager, JobQueueFull, public_job
from security import RateLimiter, RateLimitMiddleware, RequestLimitsMiddleware
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
rate_limiter = RateLimiter()


@app.exception_handler(AppError)
async def handle_app_error(_request, exc: AppError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.public_message},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(_request, exc: Exception):
    logger.error("Unhandled API error (%s)", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )


origins = [
    origin.strip()
    for origin in settings.cors_allowed_origins.split(",")
    if origin.strip()
]

app.add_middleware(RequestLimitsMiddleware)
app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


def require_admin(x_admin_key: str | None = Header(default=None)):
    expected = settings.admin_api_key
    if expected is None or not expected.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="Administrative actions are unavailable until ADMIN_API_KEY is configured.",
        )
    if x_admin_key is None or not compare_digest(
        x_admin_key.encode("utf-8"), expected.get_secret_value().encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail="A valid X-Admin-Key header is required.",
        )


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


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_admin)])
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


@app.delete("/database", dependencies=[Depends(require_admin)])
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
