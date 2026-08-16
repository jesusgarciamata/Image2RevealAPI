from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import JobRecord, JobStatus


class JobStorage:
    def __init__(self, root: Path, ttl_hours: int) -> None:
        self.root = root
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        if not job_id or any(ch not in "0123456789abcdef" for ch in job_id):
            raise ValueError("Identificador de trabajo inválido")
        path = (self.root / job_id).resolve()
        if path.parent != self.root:
            raise ValueError("Ruta de trabajo inválida")
        return path

    def create(self, record: JobRecord) -> Path:
        with self.lock:
            directory = self.job_dir(record.id)
            directory.mkdir(mode=0o700, parents=False, exist_ok=False)
            self.save(record)
            return directory

    def save(self, record: JobRecord) -> None:
        with self.lock:
            directory = self.job_dir(record.id)
            directory.mkdir(mode=0o700, parents=False, exist_ok=True)
            target = directory / "job.json"
            temporary = directory / "job.json.tmp"
            temporary.write_text(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, target)

    def load(self, job_id: str) -> JobRecord | None:
        with self.lock:
            path = self.job_dir(job_id) / "job.json"
            if not path.is_file():
                return None
            return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def recover(self) -> list[JobRecord]:
        recovered: list[JobRecord] = []
        with self.lock:
            for metadata in self.root.glob("*/job.json"):
                try:
                    record = JobRecord.model_validate_json(metadata.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if record.status in {JobStatus.queued, JobStatus.running}:
                    record.status = JobStatus.failed
                    record.error = "El servicio se reinició antes de terminar el render"
                    record.finished_at = datetime.now(timezone.utc).isoformat()
                    record.expires_at = (datetime.now(timezone.utc) + self.ttl).isoformat()
                    self.save(record)
                recovered.append(record)
        return recovered

    def delete(self, job_id: str) -> bool:
        with self.lock:
            directory = self.job_dir(job_id)
            if not directory.exists():
                return False
            shutil.rmtree(directory)
            return True

    def purge_expired(self, active_ids: set[str]) -> int:
        now = datetime.now(timezone.utc)
        removed = 0
        with self.lock:
            for metadata in self.root.glob("*/job.json"):
                try:
                    record = JobRecord.model_validate_json(metadata.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if record.id in active_ids or record.status in {JobStatus.queued, JobStatus.running}:
                    continue
                reference_text = record.finished_at or record.created_at
                reference = datetime.fromisoformat(reference_text)
                if now - reference >= self.ttl:
                    self.delete(record.id)
                    removed += 1
        return removed

