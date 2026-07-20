"""Canonical identity primitives for CellPax components and entities."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _canonicalize(value: Any) -> Any:
    """Convert supported values to a deterministic JSON-compatible tree."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON does not permit NaN or infinity")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Canonical datetimes must be timezone-aware")
        normalized = value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Canonical JSON object keys must be strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("Unordered collections are forbidden in canonical JSON")
    raise TypeError(f"Unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value using CellPax's canonical JSON rules."""
    normalized = _canonicalize(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return a lowercase SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash a canonical JSON value with SHA-256."""
    return sha256_hex(canonical_json_bytes(value))


def membership_hash(cell_ids: Iterable[int]) -> str:
    """Hash sorted unique signed 64-bit cell identifiers."""
    normalized: set[int] = set()
    for cell_id in cell_ids:
        if isinstance(cell_id, bool) or not isinstance(cell_id, int):
            raise TypeError("cell_id values must be integers, not booleans")
        if not -(2**63) <= cell_id < 2**63:
            raise OverflowError(f"cell_id {cell_id} is outside signed Int64 range")
        normalized.add(cell_id)
    ordered = sorted(normalized)
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(ordered)))
    for cell_id in ordered:
        digest.update(struct.pack("<q", cell_id))
    return digest.hexdigest()


def ordered_records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash ordered semantic records without discarding record order."""
    return canonical_hash(list(records))


def new_entity_id() -> str:
    """Create a lowercase canonical UUIDv4 entity id."""
    return str(uuid.uuid4())


def is_sha256(value: str) -> bool:
    """Return whether a string is a canonical SHA-256 hex digest."""
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def is_uuid4(value: str) -> bool:
    """Return whether a string is a lowercase canonical UUIDv4."""
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return parsed.version == 4 and str(parsed) == value
