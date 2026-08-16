from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timedelta, timezone

from .config import Settings
from .models import JobRecord, JobStatus
from .renderer import render_video
from .storage import JobStorage


logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    pass


class JobManager:
    def __init__(self, settings: Settings, storage: JobStorage) -> None:
        self.settings = settings
        self.storage = storage
        self.pending: queue.Queue[str | None] = queue.Queue(maxsize=settings.max_queued_jobs)
        self.records: dict[str, JobRecord] = {record.id: record for record in storage.recover()}
        self.active: set[str] = set()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"render-{index + 1}",
                daemon=True,
            )
            for index in range(settings.max_concurrent_renders)
        ]
        self.cleaner = threading.Thread(target=self._cleanup_loop, name="cleaner", daemon=True)

    def start(self) -> None:
        for worker in self.workers:
            worker.start()
        self.cleaner.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        for _ in self.workers:
            try:
                self.pending.put_nowait(None)
            except queue.Full:
                break

    def submit(self, record: JobRecord) -> None:
        with self.lock:
            self.records[record.id] = record
        try:
            self.pending.put_nowait(record.id)
        except queue.Full as exc:
            with self.lock:
                self.records.pop(record.id, None)
            self.storage.delete(record.id)
            raise QueueFullError("La cola de render está llena") from exc

    def get(self, job_id: str) -> JobRecord | None:
        with self.lock:
            record = self.records.get(job_id)
        if record is not None:
            return record.model_copy(deep=True)
        record = self.storage.load(job_id)
        if record:
            with self.lock:
                self.records[job_id] = record
        return record

    def delete(self, job_id: str) -> bool:
        with self.lock:
            record = self.records.get(job_id)
            if job_id in self.active or (
                record is not None and record.status in {JobStatus.queued, JobStatus.running}
            ):
                raise RuntimeError("No se puede borrar un render activo")
            self.records.pop(job_id, None)
        return self.storage.delete(job_id)

    def counts(self) -> tuple[int, int]:
        with self.lock:
            return len(self.active), self.pending.qsize()

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                job_id = self.pending.get(timeout=1.0)
            except queue.Empty:
                continue
            if job_id is None:
                self.pending.task_done()
                break
            try:
                self._run(job_id)
            finally:
                self.pending.task_done()

    def _run(self, job_id: str) -> None:
        with self.lock:
            record = self.records[job_id]
            self.active.add(job_id)
            record.status = JobStatus.running
            record.started_at = datetime.now(timezone.utc).isoformat()
            record.progress = 0.0
            self.storage.save(record)

        last_saved = -1.0

        def update_progress(value: float) -> None:
            nonlocal last_saved
            with self.lock:
                current = self.records[job_id]
                current.progress = min(max(value, 0.0), 1.0)
                if current.progress - last_saved >= 0.02 or current.progress >= 1.0:
                    self.storage.save(current)
                    last_saved = current.progress

        directory = self.storage.job_dir(job_id)
        try:
            width, height, frames = render_video(
                directory / "input.png",
                directory / "output.mp4",
                record.options,
                update_progress,
            )
            now = datetime.now(timezone.utc)
            with self.lock:
                record = self.records[job_id]
                record.status = JobStatus.completed
                record.progress = 1.0
                record.width = width
                record.height = height
                record.frames = frames
                record.finished_at = now.isoformat()
                record.expires_at = (now + timedelta(hours=self.settings.job_ttl_hours)).isoformat()
                self.storage.save(record)
        except Exception as exc:
            logger.exception("Render %s failed", job_id)
            now = datetime.now(timezone.utc)
            with self.lock:
                record = self.records[job_id]
                record.status = JobStatus.failed
                record.error = str(exc)[:1200]
                record.finished_at = now.isoformat()
                record.expires_at = (now + timedelta(hours=self.settings.job_ttl_hours)).isoformat()
                self.storage.save(record)
        finally:
            with self.lock:
                self.active.discard(job_id)

    def _cleanup_loop(self) -> None:
        interval = self.settings.purge_interval_minutes * 60
        while not self.stop_event.wait(interval):
            with self.lock:
                active_ids = set(self.active)
            removed = self.storage.purge_expired(active_ids)
            if removed:
                logger.info("Purged %d expired jobs", removed)
                with self.lock:
                    existing_ids = {
                        path.parent.name for path in self.storage.root.glob("*/job.json")
                    }
                    self.records = {
                        job_id: record
                        for job_id, record in self.records.items()
                        if job_id in existing_ids or job_id in active_ids
                    }
