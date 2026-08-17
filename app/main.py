from __future__ import annotations

import hmac
import io
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .jobs import JobManager, QueueFullError
from .models import (
    CreateJobResponse,
    Direction,
    HealthResponse,
    JobRecord,
    JobStatus,
    RegionOrder,
    RenderOptions,
    SegmentationMode,
)
from .storage import JobStorage


settings = Settings.from_env()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
storage = JobStorage(settings.data_dir, settings.job_ttl_hours)
manager = JobManager(settings, storage)


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    yield
    manager.shutdown()


app = FastAPI(
    title="Organic Reveal API",
    version="0.3.0",
    description="Convierte una imagen en un video de revelado progresivo por detalles coloreados y pincel orgánico.",
    lifespan=lifespan,
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key is None or not any(
        hmac.compare_digest(x_api_key, expected) for expected in settings.api_keys
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key inválida")


def _parse_direction(value: str) -> Direction:
    try:
        return Direction(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Direction)
        raise HTTPException(status_code=422, detail=f"direction debe ser: {allowed}") from exc


def _parse_segmentation_mode(value: str) -> SegmentationMode:
    try:
        return SegmentationMode(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SegmentationMode)
        raise HTTPException(status_code=422, detail=f"segmentation_mode debe ser: {allowed}") from exc


def _parse_region_order(value: str) -> RegionOrder:
    try:
        return RegionOrder(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RegionOrder)
        raise HTTPException(status_code=422, detail=f"region_order debe ser: {allowed}") from exc


async def _read_and_normalize(upload: UploadFile) -> tuple[bytes, int, int]:
    data = bytearray()
    while chunk := await upload.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="La imagen excede MAX_UPLOAD_MB")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > settings.max_pixels:
                raise HTTPException(status_code=413, detail="La imagen excede MAX_PIXELS")
            if probe.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(status_code=415, detail="Solo se admite PNG, JPEG o WebP")
            probe.verify()
        with Image.open(io.BytesIO(data)) as source:
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            output = io.BytesIO()
            normalized.save(output, format="PNG", optimize=True)
            width, height = normalized.size
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=415, detail="El archivo no es una imagen válida") from exc
    return output.getvalue(), width, height


@app.get("/healthz", response_model=HealthResponse, include_in_schema=False)
def health() -> HealthResponse:
    active, queued = manager.counts()
    return HealthResponse(
        status="ok",
        concurrent_limit=settings.max_concurrent_renders,
        active_jobs=active,
        queued_jobs=queued,
    )


@app.post(
    "/v1/renders",
    response_model=CreateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
async def create_render(
    image: UploadFile = File(...),
    duration: float = Form(10.0, ge=2.0, le=60.0),
    fps: int = Form(30, ge=12, le=60),
    detail_ratio: float = Form(0.45, ge=0.15, le=0.8),
    fill_overlap: float = Form(0.08, ge=0.0, le=0.25),
    final_hold: float = Form(0.6, ge=0.0, le=5.0),
    brush_radius: float = Form(0.12, ge=0.03, le=0.30),
    brush_feather: float = Form(0.16, ge=0.01, le=0.60),
    fill_brushes: int = Form(3, ge=1, le=5),
    direction: str = Form("reading-order"),
    segmentation_mode: str = Form("auto"),
    region_order: str = Form("saliency"),
    max_regions: int = Form(12, ge=2, le=32),
    min_region_area: float = Form(0.002, ge=0.0002, le=0.05),
    background: str = Form("#f1efe9", pattern=r"^#[0-9a-fA-F]{6}$"),
    output_width: int = Form(0, ge=0, le=3840),
    seed: int = Form(3842, ge=0, le=2_147_483_647),
) -> CreateJobResponse:
    normalized, _, _ = await _read_and_normalize(image)
    if output_width and output_width < 64:
        raise HTTPException(status_code=422, detail="output_width debe ser 0 o al menos 64")
    options = RenderOptions(
        duration=duration,
        fps=fps,
        detail_ratio=detail_ratio,
        fill_overlap=fill_overlap,
        final_hold=final_hold,
        brush_radius=brush_radius,
        brush_feather=brush_feather,
        fill_brushes=fill_brushes,
        direction=_parse_direction(direction),
        segmentation_mode=_parse_segmentation_mode(segmentation_mode),
        region_order=_parse_region_order(region_order),
        max_regions=max_regions,
        min_region_area=min_region_area,
        background=background,
        output_width=output_width,
        seed=seed,
    )
    job_id = secrets.token_hex(16)
    record = JobRecord(
        id=job_id,
        status=JobStatus.queued,
        progress=0.0,
        created_at=datetime.now(timezone.utc).isoformat(),
        original_filename=Path(image.filename or "image").name[:200],
        options=options,
    )
    directory = storage.create(record)
    (directory / "input.png").write_bytes(normalized)
    try:
        manager.submit(record)
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return CreateJobResponse(
        id=job_id,
        status=JobStatus.queued,
        status_url=f"/v1/renders/{job_id}",
    )


@app.get(
    "/v1/renders/{job_id}",
    dependencies=[Depends(require_api_key)],
)
def get_render(job_id: str) -> JSONResponse:
    try:
        record = manager.get(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Render no encontrado") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Render no encontrado")
    return JSONResponse(record.public())


@app.get(
    "/v1/renders/{job_id}/video",
    dependencies=[Depends(require_api_key)],
    response_class=FileResponse,
)
def download_video(job_id: str) -> FileResponse:
    try:
        record = manager.get(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Render no encontrado") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Render no encontrado")
    if record.status != JobStatus.completed:
        raise HTTPException(status_code=409, detail=f"El render está {record.status.value}")
    path = storage.job_dir(job_id) / "output.mp4"
    if not path.is_file():
        raise HTTPException(status_code=410, detail="El video ya no está disponible")
    return FileResponse(path, media_type="video/mp4", filename=f"organic-reveal-{job_id}.mp4")


@app.get(
    "/v1/renders/{job_id}/regions",
    dependencies=[Depends(require_api_key)],
    response_class=FileResponse,
)
def download_region_preview(job_id: str) -> FileResponse:
    try:
        record = manager.get(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Render no encontrado") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Render no encontrado")
    if record.status != JobStatus.completed:
        raise HTTPException(status_code=409, detail=f"El render está {record.status.value}")
    path = storage.job_dir(job_id) / "regions.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Este render no utilizó segmentación automática")
    return FileResponse(path, media_type="image/png", filename=f"regions-{job_id}.png")


@app.delete(
    "/v1/renders/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_api_key)],
    response_class=Response,
)
def delete_render(job_id: str) -> Response:
    try:
        deleted = manager.delete(job_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Render no encontrado")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
