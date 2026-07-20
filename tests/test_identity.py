from datetime import datetime, timedelta, timezone
from enum import Enum
from uuid import UUID

import pytest

from cellpax.identity import (
    canonical_hash,
    canonical_json_bytes,
    is_sha256,
    is_uuid4,
    membership_hash,
    new_entity_id,
    ordered_records_hash,
)


def test_canonical_hash_golden_vector_is_order_independent() -> None:
    expected = "efbd0040190fb0871831e606c581f8a66db79d8e2bb836745a70051306956070"

    assert canonical_hash({"b": [2, 3], "a": 1}) == expected
    assert canonical_hash({"a": 1, "b": [2, 3]}) == expected
    assert is_sha256(expected)


def test_canonical_datetime_normalizes_to_utc() -> None:
    eastern = timezone(timedelta(hours=-4))
    value = datetime(2026, 7, 18, 12, 30, tzinfo=eastern)

    assert canonical_json_bytes(value) == b'"2026-07-18T16:30:00.000000Z"'


def test_canonical_json_supports_finite_floats_enums_uuids_and_tuples() -> None:
    class State(Enum):
        READY = "ready"

    value = (1.5, State.READY, UUID("12345678-1234-5678-1234-567812345678"))

    assert canonical_json_bytes(value) == (
        b'[1.5,"ready","12345678-1234-5678-1234-567812345678"]'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json_bytes(value)


def test_canonical_json_rejects_naive_datetimes_and_unordered_sets() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        canonical_json_bytes(datetime(2026, 7, 18))
    with pytest.raises(TypeError, match="Unordered"):
        canonical_json_bytes({"a", "b"})
    with pytest.raises(TypeError, match="keys must be strings"):
        canonical_json_bytes({1: "value"})
    with pytest.raises(TypeError, match="Unsupported"):
        canonical_json_bytes(b"bytes")


def test_membership_hash_golden_vector_sorts_and_deduplicates() -> None:
    expected = "f973b5daed54541269ece414158ab666dbc472450a1ef66f4ebd97d0880fa62b"

    assert membership_hash([3, 1, 3, -2]) == expected
    assert membership_hash([-2, 1, 3]) == expected


def test_membership_hash_rejects_invalid_ids() -> None:
    with pytest.raises(TypeError):
        membership_hash([True])
    with pytest.raises(OverflowError):
        membership_hash([2**63])


def test_ordered_records_hash_preserves_order() -> None:
    forward = ordered_records_hash([{"feature_id": "a"}, {"feature_id": "b"}])
    reverse = ordered_records_hash([{"feature_id": "b"}, {"feature_id": "a"}])

    assert forward != reverse


def test_entity_ids_are_canonical_uuid4() -> None:
    entity_id = new_entity_id()

    assert is_uuid4(entity_id)
    assert not is_uuid4(entity_id.upper())
    assert not is_uuid4("not-a-uuid")
    assert not is_sha256("A" * 64)
