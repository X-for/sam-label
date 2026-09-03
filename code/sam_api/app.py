from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, AsyncIterator
from uuid import uuid4

import aiofiles
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from .config import Settings
from .model_runtime import Sam3Runtime
from .schemas import JobCreate, JobList, JobProgress, JobStatus, JobView, ModelStatus, UploadedImage
from .store import JobStore
from .worker import JobWorker

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
UI_PATH = Path(__file__).with_name("static") / "index.html"
UI_SCRIPT_PATH = Path(__file__).with_name("static") / "ui.js"


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = JobStore(settings.data_dir)
        self.runtime = Sam3Runtime(settings)
        self.worker = JobWorker(self.store, self.runtime)
        self.job_locks: dict[str, asyncio.Lock] = {}

    def job_lock(self, job_id: str) -> asyncio.Lock:
        return self.job_locks.setdefault(job_id, asyncio.Lock())


async def idle_unloader(runtime: Sam3Runtime) -> None:
    while True:
        await asyncio.sleep(15)
        await runtime.maybe_unload_idle()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.result_dir.mkdir(parents=True, exist_ok=True)
    services = Services(settings)
    await services.store.initialize()
    app.state.services = services
    worker_task = asyncio.create_task(services.worker.run(), name="sam3-job-worker")
    unload_task = asyncio.create_task(idle_unloader(services.runtime), name="sam3-idle-unloader")
    if settings.lifecycle == "resident":
        asyncio.create_task(services.runtime.prewarm(), name="sam3-startup-prewarm")
    try:
        yield
    finally:
        await services.worker.queue.put(None)
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(worker_task, timeout=5)
        unload_task.cancel()
        with suppress(asyncio.CancelledError):
            await unload_task
        await services.runtime.close()


app = FastAPI(
    title="SAM3 remote pre-annotation API",
    version="1.0.0",
    lifespan=lifespan,
)


def services(request: Request) -> Services:
    return request.app.state.services


def authorize(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    expected = services(request).settings.api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


def require_job(record):
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return record


@app.get("/ui", include_in_schema=False)
async def web_ui() -> FileResponse:
    return FileResponse(UI_PATH, media_type="text/html")


@app.get("/ui.js", include_in_schema=False)
async def web_ui_script() -> FileResponse:
    return FileResponse(UI_SCRIPT_PATH, media_type="text/javascript")


async def save_uploads(job_id: str, files: list[UploadFile], svc: Services) -> JobView:
    # Sharing this lock with commit prevents a commit from racing an upload
    # between file write and metadata persistence.
    async with svc.job_lock(job_id):
        record = require_job(await svc.store.get(job_id))
        if record.status != JobStatus.UPLOADING:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job no longer accepts uploads")
        if not files:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="at least one image is required")

        total = sum(image.size_bytes for image in record.images)
        job_dir = svc.settings.upload_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        for upload in files:
            if upload.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"unsupported image type: {upload.content_type}",
                )
            suffix = Path(upload.filename or "image").suffix.lower()[:10]
            image_id = str(uuid4())
            destination = job_dir / f"{image_id}{suffix}"
            written = 0
            try:
                async with aiofiles.open(destination, "wb") as output:
                    while chunk := await upload.read(svc.settings.upload_chunk_bytes):
                        written += len(chunk)
                        total += len(chunk)
                        if written > svc.settings.max_file_bytes:
                            raise HTTPException(status_code=413, detail=f"file too large: {upload.filename}")
                        if total > svc.settings.max_job_bytes:
                            raise HTTPException(status_code=413, detail="job upload limit exceeded")
                        await output.write(chunk)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            finally:
                await upload.close()
            record = await svc.store.add_image(
                job_id,
                UploadedImage(
                    image_id=image_id,
                    original_name=Path(upload.filename or f"{image_id}{suffix}").name,
                    stored_path=str(destination.resolve()),
                    size_bytes=written,
                    content_type=upload.content_type,
                ),
            )
        return JobView.from_record(record)


@app.get("/healthz", dependencies=[Depends(authorize)])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/model", response_model=ModelStatus, dependencies=[Depends(authorize)])
async def model_status(request: Request) -> ModelStatus:
    return services(request).runtime.status()


@app.post("/v1/jobs", response_model=JobView, status_code=201, dependencies=[Depends(authorize)])
async def create_job(payload: JobCreate, request: Request) -> JobView:
    svc = services(request)
    record = await svc.store.create(payload)
    return JobView.from_record(record)


@app.get("/v1/jobs", response_model=JobList, dependencies=[Depends(authorize)])
async def list_jobs(
    request: Request,
    job_status: JobStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> JobList:
    records, total = await services(request).store.list_jobs(
        status=job_status,
        limit=limit,
        offset=offset,
    )
    return JobList(
        items=[JobView.from_record(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.post("/v1/jobs/{job_id}/images", response_model=JobView, dependencies=[Depends(authorize)])
async def upload_images(
    job_id: str,
    request: Request,
    files: Annotated[list[UploadFile], File(description="One or more image files")],
) -> JobView:
    return await save_uploads(job_id, files, services(request))


@app.post("/v1/jobs/{job_id}/commit", response_model=JobView, dependencies=[Depends(authorize)])
async def commit_job(job_id: str, request: Request) -> JobView:
    svc = services(request)
    async with svc.job_lock(job_id):
        record = require_job(await svc.store.get(job_id))
        if record.status != JobStatus.UPLOADING:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job was already committed")
        if not record.images:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="upload at least one image first")
        record = await svc.store.set_status(job_id, JobStatus.QUEUED)
        await svc.worker.enqueue(job_id)
    return JobView.from_record(record)


@app.get("/v1/jobs/{job_id}", response_model=JobView, dependencies=[Depends(authorize)])
async def get_job(job_id: str, request: Request) -> JobView:
    return JobView.from_record(require_job(await services(request).store.get(job_id)))


@app.get(
    "/v1/jobs/{job_id}/progress",
    response_model=JobProgress,
    dependencies=[Depends(authorize)],
)
async def get_job_progress(job_id: str, request: Request) -> JobProgress:
    record = require_job(await services(request).store.get(job_id))
    return JobProgress.from_record(record)


@app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(authorize)])
async def get_result(job_id: str, request: Request) -> FileResponse:
    record = require_job(await services(request).store.get(job_id))
    if record.status != JobStatus.SUCCEEDED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="result is not ready")
    if not record.result_path or not Path(record.result_path).is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="result not found")
    if record.config.output_format.value == "yolo":
        return FileResponse(record.result_path, media_type="application/zip", filename=f"{job_id}-yolo.zip")
    return FileResponse(record.result_path, media_type="application/json", filename=f"{job_id}-coco.json")


@app.delete(
    "/v1/jobs/{job_id}/result",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(authorize)],
)
async def delete_result(job_id: str, request: Request) -> Response:
    svc = services(request)
    async with svc.job_lock(job_id):
        record = require_job(await svc.store.get(job_id))
        if not record.result_path:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="result not found")
        await svc.store.delete_result(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/v1/predict", response_model=JobView, status_code=202, dependencies=[Depends(authorize)])
async def predict_one_shot(
    request: Request,
    task: Annotated[str, Form(description="JSON-encoded JobCreate object")],
    files: Annotated[list[UploadFile], File()],
) -> JobView:
    try:
        payload = JobCreate.model_validate(json.loads(task))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    svc = services(request)
    record = await svc.store.create(payload)
    await save_uploads(record.id, files, svc)
    record = await svc.store.set_status(record.id, JobStatus.QUEUED)
    await svc.worker.enqueue(record.id)
    return JobView.from_record(record)
