"""
Schema definition for visibility batches.

The schema is intentionally backend-agnostic. It should be usable by producer
and consumer code regardless of whether the underlying data object is a
Pandas DataFrame, a Spark DataFrame, a PyArrow table, or another tabular
representation.
"""

from dataclasses import dataclass

from radio_pipeline.shared.columns import VisibilityColumns as VC


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    required: bool
    description: str
    expected_shape: str | None = None
    expected_dtype: str | None = None


REQUIRED_VISIBILITY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        name=VC.TIME,
        required=True,
        expected_dtype="float",
        expected_shape="(n_rows,)",
        description=(
            "Observation time associated with each visibility row."
        ),
    ),
    ColumnSpec(
        name=VC.UVW,
        required=True,
        expected_dtype="array<float>",
        expected_shape="(n_rows, 3)",
        description=(
            "Projected baseline coordinates in meters. The three components "
            "represent the u, v and w coordinates."
        ),
    ),
    ColumnSpec(
        name=VC.ANTENNA1,
        required=True,
        expected_dtype="integer",
        expected_shape="(n_rows,)",
        description=(
            "First antenna index of the interferometric baseline."
        ),
    ),
    ColumnSpec(
        name=VC.ANTENNA2,
        required=True,
        expected_dtype="integer",
        expected_shape="(n_rows,)",
        description=(
            "Second antenna index of the interferometric baseline."
        ),
    ),
    ColumnSpec(
        name=VC.SCAN_NUMBER,
        required=True,
        expected_dtype="integer",
        expected_shape="(n_rows,)",
        description=(
            "Scan identifier used to avoid invalid aggregation across "
            "different scans."
        ),
    ),
    ColumnSpec(
        name=VC.DATA,
        required=True,
        expected_dtype="complex or array<float>",
        expected_shape="(n_rows, n_channels, n_correlations)",
        description=(
            "Complex visibility data. Backends that do not support complex "
            "numbers may represent this field using a real-imaginary encoding."
        ),
    ),
    ColumnSpec(
        name=VC.FLAG,
        required=True,
        expected_dtype="boolean",
        expected_shape="compatible with DATA",
        description=(
            "Flag information associated with the visibility data."
        ),
    ),
    ColumnSpec(
        name=VC.WEIGHT,
        required=True,
        expected_dtype="float",
        expected_shape="(n_rows, n_correlations) or compatible with DATA",
        description=(
            "Visibility weights used during averaging and imaging-related "
            "operations."
        ),
    ),
)


OPTIONAL_VISIBILITY_COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        name=VC.IMAGING_WEIGHT,
        required=False,
        expected_dtype="float",
        description="Weights used for imaging.",
    ),
    ColumnSpec(
        name=VC.EXPOSURE,
        required=False,
        expected_dtype="float",
        description="Exposure time.",
    ),
    ColumnSpec(
        name=VC.INTERVAL,
        required=False,
        expected_dtype="float",
        description="Integration interval.",
    ),
    ColumnSpec(
        name=VC.CHANNELS,
        required=False,
        expected_dtype="array<float>",
        description="Frequency channel information.",
    ),
    ColumnSpec(
        name=VC.CORRELATIONS,
        required=False,
        expected_dtype="array<string> or array<int>",
        description="Correlation labels or indexes.",
    ),
)

VISIBILITY_COLUMNS = REQUIRED_VISIBILITY_COLUMNS + OPTIONAL_VISIBILITY_COLUMNS


def required_column_names() -> set[str]:
    return {column.name for column in REQUIRED_VISIBILITY_COLUMNS}


def optional_column_names() -> set[str]:
    return {column.name for column in OPTIONAL_VISIBILITY_COLUMNS}


def known_column_names() -> set[str]:
    return {column.name for column in VISIBILITY_COLUMNS}


def get_column_spec(column_name: str) -> ColumnSpec:
    specs = {column.name: column for column in VISIBILITY_COLUMNS}

    if column_name not in specs:
        raise KeyError(f"Unknown visibility column: {column_name}")

    return specs[column_name]