from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class Direction(str, Enum):
    right_to_left = "right-to-left"
    left_to_right = "left-to-right"
    top_to_bottom = "top-to-bottom"
    bottom_to_top = "bottom-to-top"
    center_out = "center-out"
    organic = "organic"


class RenderOptions(BaseModel):
    duration: float = Field(default=10.0, ge=2.0, le=60.0)
    fps: int = Field(default=30, ge=12, le=60)
    detail_ratio: float = Field(default=0.45, ge=0.15, le=0.8)
    fill_overlap: float = Field(default=0.08, ge=0.0, le=0.25)
    final_hold: float = Field(default=0.6, ge=0.0, le=5.0)
    brush_radius: float = Field(default=0.12, ge=0.03, le=0.30)
    brush_feather: float = Field(default=0.16, ge=0.01, le=0.60)
    direction: Direction = Direction.right_to_left
    background: str = Field(default="#f1efe9", pattern=r"^#[0-9a-fA-F]{6}$")
    output_width: int = Field(default=0, ge=0, le=3840)
    seed: int = Field(default=3842, ge=0, le=2_147_483_647)


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    expires_at: str | None = None
    original_filename: str
    options: RenderOptions
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if self.status == JobStatus.completed:
            payload["video_url"] = f"/v1/renders/{self.id}/video"
        return payload


class CreateJobResponse(BaseModel):
    id: str
    status: JobStatus
    status_url: str


class HealthResponse(BaseModel):
    status: str
    concurrent_limit: int
    active_jobs: int
    queued_jobs: int

