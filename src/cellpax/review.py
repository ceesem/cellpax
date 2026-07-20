"""Strict action payloads for the append-only review ledger."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cellpax.config import canonical_json_text

_EMPTY_ACTIONS = {"merge", "exclude", "mark_ambiguous", "assign", "assign_parent_only"}
_ACTIONS = _EMPTY_ACTIONS | {"split", "rename", "attach_local_revision"}


@dataclass(frozen=True, slots=True)
class DecisionActionConfig:
    """One strictly validated action-specific decision payload."""

    action: str
    params_json: str

    @classmethod
    def resolve(
        cls, *, action: str, params: Mapping[str, Any] | None = None
    ) -> "DecisionActionConfig":
        if action not in _ACTIONS:
            raise ValueError(f"Unsupported decision action {action!r}")
        supplied = dict(params or {})
        if action in _EMPTY_ACTIONS:
            if supplied:
                raise ValueError(f"Decision action {action!r} accepts no parameters")
            resolved: dict[str, Any] = {}
        elif action == "rename":
            if (
                set(supplied) != {"name"}
                or not isinstance(supplied["name"], str)
                or not supplied["name"]
            ):
                raise ValueError("rename requires exactly one non-empty string name")
            resolved = supplied
        elif action == "attach_local_revision":
            if (
                set(supplied) != {"revision_id"}
                or not isinstance(supplied["revision_id"], str)
                or not supplied["revision_id"]
            ):
                raise ValueError(
                    "attach_local_revision requires exactly one revision_id"
                )
            resolved = supplied
        else:
            if (
                set(supplied) != {"parts"}
                or not isinstance(supplied["parts"], list)
                or not supplied["parts"]
            ):
                raise ValueError("split requires a non-empty parts list")
            parts: list[dict[str, object]] = []
            for part in supplied["parts"]:
                if not isinstance(part, Mapping) or set(part) != {
                    "target_ref",
                    "taxon_id",
                }:
                    raise ValueError("Each split part requires target_ref and taxon_id")
                target_ref = part["target_ref"]
                taxon_id = part["taxon_id"]
                if not isinstance(target_ref, str) or not target_ref:
                    raise TypeError("split target_ref must be a non-empty string")
                if (
                    not isinstance(taxon_id, int)
                    or isinstance(taxon_id, bool)
                    or taxon_id < 0
                ):
                    raise TypeError("split taxon_id must be a non-negative integer")
                parts.append({"target_ref": target_ref, "taxon_id": taxon_id})
            resolved = {"parts": parts}
        return cls(action=action, params_json=canonical_json_text(resolved))

    @property
    def params(self) -> dict[str, Any]:
        return json.loads(self.params_json)


def validate_decision_semantics(
    *,
    config: DecisionActionConfig,
    target_kind: str,
    target_count: int | None,
    taxon_id: int | None,
) -> None:
    """Validate cross-field action semantics not expressible in the table schema."""
    if target_kind not in {"candidates", "cells", "taxon"}:
        raise ValueError(f"Unsupported decision target kind {target_kind!r}")
    if config.action in {"assign", "assign_parent_only", "merge"} and taxon_id is None:
        raise ValueError(f"Decision action {config.action!r} requires taxon_id")
    if config.action == "merge":
        if target_kind != "candidates" or target_count is None or target_count < 2:
            raise ValueError("merge requires at least two candidate targets")
    if config.action == "split" and (target_kind != "candidates" or target_count != 1):
        raise ValueError("split requires exactly one candidate target")
    if config.action in {"rename"} and target_kind != "taxon":
        raise ValueError("rename requires a taxon target")
    if config.action == "attach_local_revision" and target_kind != "cells":
        raise ValueError("attach_local_revision requires cell targets")
    if taxon_id is not None and (
        not isinstance(taxon_id, int) or isinstance(taxon_id, bool) or taxon_id < 0
    ):
        raise TypeError("taxon_id must be a non-negative integer or null")
