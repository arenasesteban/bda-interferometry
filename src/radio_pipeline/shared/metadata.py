"""
Metadata models associated with visibility batches.

These metadata classes provide traceability between producer and consumer.
They do not perform simulation or processing logic.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulationMetadata:
    """
    Metadata describing the origin of the simulated visibility data.
    """

    framework: str | None = None
    array_configuration: str | None = None
    frequency_hz: float | None = None
    integration_time_s: float | None = None
    field_of_view_deg: float | None = None
    phase_center_ra: str | None = None
    phase_center_dec: str | None = None
    source_model: str | None = None


@dataclass(frozen=True)
class TransmissionMetadata:
    """
    Metadata describing how a visibility batch was transmitted.
    """

    publisher: str | None = None
    topic: str | None = None
    partition: int | None = None
    message_index: int | None = None
    chunk_index: int | None = None
    chunk_size_rows: int | None = None
    compression: str | None = None


@dataclass(frozen=True)
class VisibilityMetadata:
    """
    General metadata attached to a visibility batch.
    """

    batch_id: str
    n_rows: int

    schema_version: str = "0.1"
    producer_id: str | None = None
    created_at: str | None = None

    subms_id: int | None = None
    field_id: int | None = None
    spw_id: int | None = None
    polarization_id: int | None = None
    
    simulation: SimulationMetadata | None = None
    transmission: TransmissionMetadata | None = None

    extra: dict[str, Any] = field(default_factory=dict)