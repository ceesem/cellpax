"""Structured validation for CellPax registry tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from cellpax.contracts.core import TableContract
from cellpax.contracts.schemas import CONTRACTS


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One actionable contract validation failure."""

    table: str
    code: str
    message: str
    columns: tuple[str, ...] = ()
    invalid_count: int | None = None

    def __str__(self) -> str:
        columns = f" [{', '.join(self.columns)}]" if self.columns else ""
        count = f" ({self.invalid_count} rows)" if self.invalid_count else ""
        return f"{self.table}.{self.code}{columns}: {self.message}{count}"


class ContractValidationError(ValueError):
    """Raised when one or more materialized contracts are violated."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        detail = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"CellPax contract validation failed:\n{detail}")


def _finish(
    issues: list[ValidationIssue], *, raise_on_error: bool
) -> tuple[ValidationIssue, ...]:
    if issues and raise_on_error:
        raise ContractValidationError(issues)
    return tuple(issues)


def validate_table(
    frame: pl.DataFrame,
    contract: TableContract,
    *,
    raise_on_error: bool = True,
) -> tuple[ValidationIssue, ...]:
    """Validate one frame against its complete local table contract.

    Referential integrity is handled by :func:`validate_registries`, because it
    requires other registry frames.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"Expected pl.DataFrame, got {type(frame).__name__}")

    issues: list[ValidationIssue] = []
    expected_columns = contract.schema.names()
    if frame.columns != expected_columns:
        missing = [name for name in expected_columns if name not in frame.columns]
        extra = [name for name in frame.columns if name not in expected_columns]
        order_only = not missing and not extra
        if order_only:
            message = "columns have the wrong order"
        else:
            message = f"column mismatch; missing={missing}, extra={extra}"
        issues.append(
            ValidationIssue(contract.name, "columns", message, tuple(frame.columns))
        )
        return _finish(issues, raise_on_error=raise_on_error)

    wrong_dtypes = [
        name
        for name, expected in contract.schema.items()
        if frame.schema[name] != expected
    ]
    if wrong_dtypes:
        details = ", ".join(
            f"{name}={frame.schema[name]!s} (expected {contract.schema[name]!s})"
            for name in wrong_dtypes
        )
        issues.append(
            ValidationIssue(
                contract.name,
                "dtypes",
                details,
                tuple(wrong_dtypes),
            )
        )

    for column in expected_columns:
        if column in contract.nullable:
            continue
        null_count = frame[column].null_count()
        if null_count:
            issues.append(
                ValidationIssue(
                    contract.name,
                    "nullability",
                    "column is non-nullable",
                    (column,),
                    null_count,
                )
            )

    keys = ([contract.primary_key] if contract.primary_key else []) + list(
        contract.unique_keys
    )
    for key in keys:
        duplicated = int(frame.select(list(key)).is_duplicated().sum())
        if duplicated:
            code = "primary_key" if key == contract.primary_key else "unique_key"
            issues.append(
                ValidationIssue(
                    contract.name,
                    code,
                    "key values must be unique",
                    key,
                    duplicated,
                )
            )

    for constraint in contract.enums:
        invalid = frame.filter(
            pl.col(constraint.column).is_not_null()
            & ~pl.col(constraint.column).is_in(list(constraint.values))
        ).height
        if invalid:
            allowed = ", ".join(sorted(str(value) for value in constraint.values))
            issues.append(
                ValidationIssue(
                    contract.name,
                    "enum",
                    f"value must be one of: {allowed}",
                    (constraint.column,),
                    invalid,
                )
            )

    for constraint in contract.ranges:
        expression = pl.lit(False)
        if constraint.minimum is not None:
            expression |= pl.col(constraint.column) < constraint.minimum
        if constraint.maximum is not None:
            expression |= pl.col(constraint.column) > constraint.maximum
        invalid = frame.filter(expression.fill_null(False)).height
        if invalid:
            bounds = f"[{constraint.minimum}, {constraint.maximum}]"
            issues.append(
                ValidationIssue(
                    contract.name,
                    "range",
                    f"value must be within inclusive bounds {bounds}",
                    (constraint.column,),
                    invalid,
                )
            )

    for constraint in contract.row_constraints:
        invalid_mask = constraint.invalid(frame)
        if invalid_mask.dtype != pl.Boolean or len(invalid_mask) != frame.height:
            raise TypeError(
                f"Row constraint {constraint.name!r} on {contract.name!r} "
                "must return one Boolean value per row"
            )
        invalid = int(invalid_mask.fill_null(True).sum())
        if invalid:
            issues.append(
                ValidationIssue(
                    contract.name,
                    constraint.name,
                    constraint.message,
                    constraint.columns,
                    invalid,
                )
            )

    return _finish(issues, raise_on_error=raise_on_error)


def validate_registries(
    frames: Mapping[str, pl.DataFrame],
    *,
    require_foreign_targets: bool = False,
    raise_on_error: bool = True,
) -> tuple[ValidationIssue, ...]:
    """Validate local contracts and direct foreign keys across registry frames."""
    issues: list[ValidationIssue] = []
    unknown = set(frames) - set(CONTRACTS)
    for name in sorted(unknown):
        issues.append(
            ValidationIssue(name, "unknown_contract", "no version 1 contract exists")
        )

    for name, frame in frames.items():
        table_contract = CONTRACTS.get(name)
        if table_contract is None:
            continue
        issues.extend(validate_table(frame, table_contract, raise_on_error=False))

    if any(issue.code in {"columns", "dtypes"} for issue in issues):
        return _finish(issues, raise_on_error=raise_on_error)

    for name, frame in frames.items():
        table_contract = CONTRACTS.get(name)
        if table_contract is None:
            continue
        for foreign_key in table_contract.foreign_keys:
            target = frames.get(foreign_key.target_table)
            if target is None:
                if require_foreign_targets:
                    issues.append(
                        ValidationIssue(
                            name,
                            "foreign_key_target_missing",
                            f"target registry {foreign_key.target_table!r} was not supplied",
                            foreign_key.columns,
                        )
                    )
                continue

            local = frame.select(list(foreign_key.columns)).drop_nulls().unique()
            if local.is_empty():
                continue
            target_keys = target.select(list(foreign_key.target_columns)).unique()
            missing = local.join(
                target_keys,
                left_on=list(foreign_key.columns),
                right_on=list(foreign_key.target_columns),
                how="anti",
            ).height
            if missing:
                issues.append(
                    ValidationIssue(
                        name,
                        "foreign_key",
                        f"values do not exist in {foreign_key.target_table}",
                        foreign_key.columns,
                        missing,
                    )
                )

    return _finish(issues, raise_on_error=raise_on_error)
