"""
Visibility batch contract.

A VisibilityBatch groups tabular visibility data with metadata. The data
object is intentionally typed as generic because different execution contexts
may use Pandas, PyArrow, Spark, or native Python structures.
"""

from dataclasses import dataclass
from typing import Any

from radio_pipeline.shared.metadata import VisibilityMetadata


@dataclass
class VisibilityBatch:
    """
    Container for a batch of interferometric visibilities.

    Parameters
    ----------
    data:
        Tabular visibility data. It may be a Pandas DataFrame, Spark DataFrame,
        PyArrow Table, or another table-like object.
    metadata:
        Metadata associated with the batch.
    """

    data: Any
    metadata: VisibilityMetadata