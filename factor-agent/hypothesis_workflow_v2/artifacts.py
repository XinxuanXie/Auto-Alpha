"""Atomic artifact persistence with checksum verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from .contracts import ArtifactRef


_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact does not match its recorded checksum."""


class ArtifactRepository:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(
        self,
        run_id: str,
        kind: str,
        payload: Any,
        *,
        item_count: int | None = None,
    ) -> ArtifactRef:
        target = self._target(run_id, kind, ".json")
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._atomic_write_bytes(target, encoded)
        return self._reference(
            run_id,
            kind,
            target,
            encoded,
            self._infer_item_count(payload) if item_count is None else item_count,
        )

    def read_json(self, reference: ArtifactRef) -> Any:
        path, payload = self._read_verified(reference)
        if path.suffix != ".json":
            raise ValueError("artifact is not JSON")
        return json.loads(payload.decode("utf-8"))

    def write_parquet(
        self,
        run_id: str,
        kind: str,
        frame: pl.DataFrame,
    ) -> ArtifactRef:
        target = self._target(run_id, kind, ".parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            frame.write_parquet(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        encoded = target.read_bytes()
        return self._reference(run_id, kind, target, encoded, frame.height)

    def read_parquet(self, reference: ArtifactRef) -> pl.DataFrame:
        path, _ = self._read_verified(reference)
        if path.suffix != ".parquet":
            raise ValueError("artifact is not Parquet")
        return pl.read_parquet(path)

    def verify(self, reference: ArtifactRef) -> bool:
        self._read_verified(reference)
        return True

    def _target(self, run_id: str, kind: str, suffix: str) -> Path:
        safe_run_id = self._validate_identifier(run_id, "run_id")
        safe_kind = self._validate_identifier(kind, "kind")
        target = (self.root / safe_run_id / f"{safe_kind}{suffix}").resolve()
        if self.root not in target.parents:
            raise ValueError("artifact path escapes repository root")
        return target

    @staticmethod
    def _validate_identifier(value: str, field: str) -> str:
        normalized = value.strip()
        if _SAFE_IDENTIFIER.fullmatch(normalized) is None:
            raise ValueError(f"{field} contains unsafe characters")
        return normalized

    @staticmethod
    def _atomic_write_bytes(target: Path, payload: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def _reference(
        self,
        run_id: str,
        kind: str,
        target: Path,
        payload: bytes,
        item_count: int,
    ) -> ArtifactRef:
        if item_count < 0:
            raise ValueError("item_count cannot be negative")
        checksum = hashlib.sha256(payload).hexdigest()
        relative_path = target.relative_to(self.root).as_posix()
        return ArtifactRef(
            artifact_id=f"{run_id}:{kind}:{checksum[:12]}",
            kind=kind,
            path=relative_path,
            checksum=checksum,
            item_count=item_count,
        )

    def _read_verified(self, reference: ArtifactRef) -> tuple[Path, bytes]:
        path = (self.root / reference.path).resolve()
        if self.root not in path.parents:
            raise ArtifactIntegrityError("artifact path escapes repository root")
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != reference.checksum:
            raise ArtifactIntegrityError(
                f"checksum mismatch for artifact {reference.artifact_id}"
            )
        return path, payload

    @staticmethod
    def _infer_item_count(payload: Any) -> int:
        if isinstance(payload, (list, tuple, dict)):
            return len(payload)
        return 1
