from __future__ import annotations

import threading
import time
from pathlib import Path

import app.jobs as jobs_module
from app.config import Settings
from app.jobs import JobManager
from app.models import JobRecord, JobStatus, RenderOptions, utc_now_iso
from app.storage import JobStorage


def test_configured_concurrency_is_respected(tmp_path, monkeypatch) -> None:
    settings = Settings(
        data_dir=tmp_path,
        api_keys=frozenset({"test"}),
        max_concurrent_renders=2,
        max_queued_jobs=10,
        max_upload_bytes=1024,
        max_pixels=1000,
        job_ttl_hours=24,
        purge_interval_minutes=60,
        log_level="INFO",
    )
    storage = JobStorage(tmp_path, ttl_hours=24)
    manager = JobManager(settings, storage)
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum = 0

    def fake_render(input_path, output_path, options, progress):
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        release.wait(timeout=3)
        progress(1.0)
        with state_lock:
            active -= 1
        return 100, 60, 24

    monkeypatch.setattr(jobs_module, "render_video", fake_render)
    manager.start()
    try:
        ids = [f"{index:032x}" for index in range(1, 5)]
        for job_id in ids:
            item = JobRecord(
                id=job_id,
                status=JobStatus.queued,
                progress=0,
                created_at=utc_now_iso(),
                original_filename="input.png",
                options=RenderOptions(),
            )
            directory = storage.create(item)
            (directory / "input.png").write_bytes(b"test")
            manager.submit(item)

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            running, queued = manager.counts()
            if running == 2 and queued == 2:
                break
            time.sleep(0.01)
        assert manager.counts() == (2, 2)
        assert maximum == 2
        release.set()

        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if all(manager.get(job_id).status == JobStatus.completed for job_id in ids):
                break
            time.sleep(0.01)
        assert all(manager.get(job_id).status == JobStatus.completed for job_id in ids)
        assert maximum == 2
    finally:
        release.set()
        manager.shutdown()
