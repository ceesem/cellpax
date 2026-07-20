from pathlib import Path

import pytest
from datafolio import DataFolio

from cellpax.artifacts import (
    commit_ref,
    new_item_ref,
    resolve_checksum,
    validate_item_ref,
    validate_item_refs,
)


def test_namespaced_item_refs_follow_datafolio_v2_grammar() -> None:
    ref = new_item_ref("scope-members")

    assert ref.startswith("cellpax/objects/scope-members/")
    assert validate_item_ref(ref) == ref
    assert commit_ref("a" * 64) == f"cellpax/commits/{'a' * 64}"


@pytest.mark.parametrize("ref", ["", "/bad", "bad/../escape", "bad/a b"])
def test_invalid_item_refs_fail_early(ref: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_item_ref(ref)


def test_checksum_resolution_and_integrity_use_datafolio_v2_api(
    tmp_path: Path,
) -> None:
    folio = DataFolio(tmp_path / "study")
    ref = new_item_ref("manifest")
    folio.add(ref, {"schema_version": "1.0"})

    checksum = resolve_checksum(folio, ref)

    assert checksum == folio.item_info(ref)["checksum"]
    validate_item_refs(folio, [ref, ref])


def test_refs_without_owned_payload_checksums_are_rejected(tmp_path: Path) -> None:
    external = tmp_path / "external.parquet"
    external.write_bytes(b"not read in this test")
    folio = DataFolio(tmp_path / "study")
    folio.reference_table("external", path=external)

    with pytest.raises(ValueError, match="no owned-payload checksum"):
        resolve_checksum(folio, "external")
    with pytest.raises(ValueError, match="missing"):
        validate_item_refs(folio, ["does-not-exist"])


def test_invalid_checksum_metadata_is_rejected() -> None:
    class InvalidChecksumFolio:
        def item_info(self, ref: str) -> dict[str, str]:
            return {"checksum": ""}

    with pytest.raises(ValueError, match="invalid checksum"):
        resolve_checksum(InvalidChecksumFolio(), "cellpax/objects/test/one")  # type: ignore[arg-type]
