"""
Validation utilities for visibility batches.

This module provides structural validation for VisibilityBatch objects.
The validation is intentionally backend-agnostic and does not depend on Spark,
Kafka, Dask, Pandas, PyArrow or Pyralysis.
"""

from dataclasses import dataclass

from radio_pipeline.shared.contracts.errors import (
    InvalidVisibilityBatchError,
    InvalidVisibilityMetadataError,
    MissingVisibilityColumnsError,
)
from radio_pipeline.shared.contracts.schema import required_column_names
from radio_pipeline.shared.contracts.visibility import VisibilityBatch


@dataclass(frozen=True)
class ValidationResult:
    """
    Result of a visibility batch validation.
    """

    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def get_column_names(data: object) -> set[str]:
    """
    Extract column names from a table-like object.

    Supported objects include:
    - Pandas-like DataFrames with a .columns attribute.
    - Spark-like DataFrames with .schema.names.
    - Dictionary-like structures.
    """

    if data is None:
        raise InvalidVisibilityBatchError(
            "VisibilityBatch.data is required."
        )

    if hasattr(data, "columns"):
        return set(data.columns)

    if hasattr(data, "schema") and hasattr(data.schema, "names"):
        return set(data.schema.names)

    if isinstance(data, dict):
        return set(data.keys())

    raise InvalidVisibilityBatchError(
        "Unable to infer column names from visibility data object."
    )


def validate_visibility_batch(
    batch: VisibilityBatch,
    *,
    raise_on_error: bool = True,
) -> ValidationResult:
    """
    Validate a VisibilityBatch against the shared visibility data contract.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if batch is None:
        errors.append("VisibilityBatch is required.")

        result = ValidationResult(
            is_valid=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

        if raise_on_error:
            raise InvalidVisibilityBatchError("; ".join(result.errors))

        return result

    errors.extend(_validate_metadata(batch))

    try:
        columns = get_column_names(batch.data)
    except InvalidVisibilityBatchError as exc:
        errors.append(str(exc))
        columns = set()

    missing_columns = required_column_names() - columns

    if missing_columns:
        errors.append(
            "Missing required visibility columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = ValidationResult(
        is_valid=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )

    if raise_on_error and not result.is_valid:
        _raise_validation_error(result)

    return result


def _validate_metadata(batch: VisibilityBatch) -> list[str]:
    """
    Validate VisibilityBatch metadata.
    """

    errors: list[str] = []

    if batch.metadata is None:
        errors.append("VisibilityBatch.metadata is required.")
        return errors

    if not batch.metadata.batch_id:
        errors.append("VisibilityMetadata.batch_id is required.")

    if batch.metadata.n_rows < 0:
        errors.append(
            "VisibilityMetadata.n_rows must be greater than or equal to zero."
        )

    return errors


def _raise_validation_error(result: ValidationResult) -> None:
    """
    Raise the most appropriate contract error for a failed validation.
    """

    message = "; ".join(result.errors)

    if any("metadata" in error.lower() for error in result.errors):
        raise InvalidVisibilityMetadataError(message)

    if any(
        "missing required visibility columns" in error.lower()
        for error in result.errors
    ):
        raise MissingVisibilityColumnsError(message)

    raise InvalidVisibilityBatchError(message)