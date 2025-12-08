from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from modules.utils.logger import get_logger


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class BackgroundJob:
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class BackgroundJobManager:
    """
    Light-weight background job orchestrator for the UI process.
    """

    def __init__(self, max_workers: int = 2, logger=None):
        self.logger = logger or get_logger()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, BackgroundJob] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit(self, name: str, func: Callable, *args, **kwargs) -> str:
        job_id = uuid.uuid4().hex
        job = BackgroundJob(id=job_id, name=name)
        with self._lock:
            self._jobs[job_id] = job
        future = self._executor.submit(self._run_job, job_id, func, args, kwargs)
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def _run_job(self, job_id: str, func: Callable, args: tuple, kwargs: dict):
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.started_at = time.time()
        job.status = JobStatus.RUNNING
        try:
            job.result = func(*args, **kwargs)
            job.status = JobStatus.SUCCEEDED
            return job.result
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            self.logger.error(f"后台任务 {job.name}({job_id}) 执行失败：{exc}", exc_info=True)
            raise
        finally:
            job.finished_at = time.time()

    def wait_for_result(self, job_id: str, timeout: Optional[float] = None):
        future = self._futures.get(job_id)
        if future is None:
            raise KeyError(f"Job {job_id} 不存在")
        return future.result(timeout=timeout)

    def run_sync(self, name: str, func: Callable, *args, **kwargs):
        job_id = self.submit(name, func, *args, **kwargs)
        return self.wait_for_result(job_id), job_id

    def get_job(self, job_id: str) -> Optional[BackgroundJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_info(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get_job(job_id)
        return job.to_dict() if job else None

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda j: j.submitted_at, reverse=True)
        return [job.to_dict() for job in jobs[:limit]]

    def shutdown(self, wait: bool = True):
        with self._lock:
            futures = list(self._futures.values())
        for future in futures:
            future.cancel()
        self._executor.shutdown(wait=wait)


__all__ = ["BackgroundJobManager", "JobStatus", "BackgroundJob"]

