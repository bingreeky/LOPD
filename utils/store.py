
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import fcntl
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_HELD_LOCKS: dict[str, tuple[Any, int]] = {}


class RunLock:

    def __init__(self, run_dir: str | os.PathLike[str]):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / ".run.lock"
        self._key = str(self.path.resolve())
        self._acquired = False

    def acquire(self) -> "RunLock":
        if self._acquired:
            return self
        held = _HELD_LOCKS.get(self._key)
        if held is not None:
            handle, count = held
            _HELD_LOCKS[self._key] = (handle, count + 1)
            self._acquired = True
            return self

        self.run_dir.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            handle.close()
            raise RuntimeError(
                f"run directory is already active: {self.run_dir} ({owner})"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        os.fsync(handle.fileno())
        _HELD_LOCKS[self._key] = (handle, 1)
        self._acquired = True
        return self

    def release(self) -> None:
        if not self._acquired:
            return
        handle, count = _HELD_LOCKS[self._key]
        if count > 1:
            _HELD_LOCKS[self._key] = (handle, count - 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            del _HELD_LOCKS[self._key]
        self._acquired = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()


class IncrementalResultStore:

    def __init__(self, run_dir: str | os.PathLike[str], *, key_field: str = "task_id"):
        self.run_dir = Path(run_dir)
        self.key_field = key_field
        self._run_lock = RunLock(self.run_dir).acquire()
        self.records_dir = self.run_dir / "_incremental_records"
        self.failures_dir = self.run_dir / "_incremental_failures"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)
        self._records = self._load_shards(self.records_dir, require_key=True)

    def close(self) -> None:
        self._run_lock.release()

    def __enter__(self) -> "IncrementalResultStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_run_lock"):
            self.close()

    @property
    def completed_keys(self) -> frozenset[str]:
        return frozenset(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def is_completed(self, key: Any) -> bool:
        return str(key) in self._records

    def pending(self, items: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        return [item for item in items if not self.is_completed(item[self.key_field])]

    def add(self, record: Mapping[str, Any]) -> bool:
        if self.key_field not in record:
            raise KeyError(f"record is missing key field {self.key_field!r}")
        key = str(record[self.key_field])
        materialized = dict(record)
        existing = self._records.get(key)
        if existing is not None:
            if existing == materialized:
                return False
            raise ValueError(f"conflicting completed record for {self.key_field}={key!r}")

        payload = json.dumps(materialized, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        _atomic_write_bytes(self._shard_path(self.records_dir, key), payload)
        self._records[key] = materialized

        failure_path = self._shard_path(self.failures_dir, key)
        if failure_path.exists():
            failure_path.unlink()
        return True

    def record_failure(self, key: Any, error: BaseException | str, **details: Any) -> None:
        key = str(key)
        payload = {
            self.key_field: key,
            "error_type": type(error).__name__ if isinstance(error, BaseException) else "Error",
            "error": str(error),
            **details,
        }
        _atomic_write_json(self._shard_path(self.failures_dir, key), payload)

    def failure_records(self) -> list[dict[str, Any]]:
        return list(self._load_shards(self.failures_dir, require_key=False).values())

    def records(self, ordered_keys: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        if ordered_keys is None:
            return [self._records[key] for key in sorted(self._records)]
        return [self._records[str(key)] for key in ordered_keys if str(key) in self._records]

    def write_json(self, filename: str, payload: Mapping[str, Any]) -> Path:
        path = self.run_dir / filename
        _atomic_write_json(path, payload)
        return path

    def write_parquet(
        self,
        filename: str,
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> Path:
        import pandas as pd

        materialized = list(records) if records is not None else self.records()
        destination = self.run_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(fd)
        try:
            pd.DataFrame(materialized).to_parquet(temporary, index=False)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    def _load_shards(self, directory: Path, *, require_key: bool) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(f"invalid incremental shard: {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"incremental shard is not an object: {path}")
            if self.key_field not in record:
                if require_key:
                    raise ValueError(f"incremental shard lacks {self.key_field!r}: {path}")
                key = path.stem
            else:
                key = str(record[self.key_field])
            if key in records:
                raise ValueError(f"duplicate incremental key {key!r} in {directory}")
            records[key] = record
        return records

    @staticmethod
    def _shard_path(directory: Path, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return directory / f"{digest}.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, data)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
