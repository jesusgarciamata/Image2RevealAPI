from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} debe ser un entero") from exc
    if value <= 0:
        raise RuntimeError(f"{name} debe ser mayor que cero")
    return value


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_keys: frozenset[str]
    max_concurrent_renders: int
    max_queued_jobs: int
    max_upload_bytes: int
    max_pixels: int
    job_ttl_hours: int
    purge_interval_minutes: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        keys = frozenset(
            value.strip()
            for value in os.getenv("API_KEYS", "").split(",")
            if value.strip()
        )
        if not keys:
            raise RuntimeError("API_KEYS es obligatorio")

        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "/app/data")).resolve(),
            api_keys=keys,
            max_concurrent_renders=_positive_int("MAX_CONCURRENT_RENDERS", 2),
            max_queued_jobs=_positive_int("MAX_QUEUED_JOBS", 50),
            max_upload_bytes=_positive_int("MAX_UPLOAD_MB", 20) * 1024 * 1024,
            max_pixels=_positive_int("MAX_PIXELS", 20_000_000),
            job_ttl_hours=_positive_int("JOB_TTL_HOURS", 24),
            purge_interval_minutes=_positive_int("PURGE_INTERVAL_MINUTES", 15),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

