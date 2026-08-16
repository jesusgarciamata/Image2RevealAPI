from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.models import JobRecord, JobStatus, RenderOptions
from app.storage import JobStorage


def record(job_id: str, finished_hours_ago: int = 0) -> JobRecord:
    finished = datetime.now(timezone.utc) - timedelta(hours=finished_hours_ago)
    return JobRecord(
        id=job_id,
        status=JobStatus.completed,
        progress=1.0,
        created_at=finished.isoformat(),
        finished_at=finished.isoformat(),
        original_filename="test.png",
        options=RenderOptions(),
    )


def test_save_load_and_safe_delete(tmp_path) -> None:
    storage = JobStorage(tmp_path, ttl_hours=24)
    item = record("a" * 32)
    storage.create(item)
    loaded = storage.load(item.id)
    assert loaded is not None and loaded.id == item.id
    assert storage.delete(item.id)
    assert storage.load(item.id) is None


def test_purge_only_expired_non_active_jobs(tmp_path) -> None:
    storage = JobStorage(tmp_path, ttl_hours=24)
    expired = record("a" * 32, finished_hours_ago=25)
    active = record("b" * 32, finished_hours_ago=25)
    fresh = record("c" * 32, finished_hours_ago=2)
    for item in (expired, active, fresh):
        storage.create(item)
    assert storage.purge_expired({active.id}) == 1
    assert storage.load(expired.id) is None
    assert storage.load(active.id) is not None
    assert storage.load(fresh.id) is not None


def test_rejects_path_traversal(tmp_path) -> None:
    storage = JobStorage(tmp_path, ttl_hours=24)
    for invalid in ("../outside", "abc/def", "", "not-hex"):
        try:
            storage.job_dir(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted unsafe id: {invalid}")

