"""
Canonical column definitions for the shared visibility data contract.

This module defines the official vocabulary used by producer and consumer
subsystems to exchange interferometric visibility data.

The canonical contract is intentionally independent from the current legacy
serialization format. Legacy field names are kept in separate classes only to
support progressive migration.
"""


class VisibilityColumns:
    """
    Canonical visibility columns used by the clean shared data contract.

    These names represent the target contract between producer and consumer.
    They should be used by new code that operates on VisibilityBatch objects.
    """

    TIME = "TIME"
    UVW = "UVW"

    ANTENNA1 = "ANTENNA1"
    ANTENNA2 = "ANTENNA2"
    SCAN_NUMBER = "SCAN_NUMBER"

    DATA = "DATA"
    FLAG = "FLAG"
    WEIGHT = "WEIGHT"
    
    EXPOSURE = "EXPOSURE"
    INTERVAL = "INTERVAL"

    CHANNELS = "CHANNELS"
    CORRELATIONS = "CORRELATIONS"


class DerivedVisibilityColumns:
    """
    Derived or auxiliary columns used internally by processing stages.

    These fields are useful for BDA, grouping, gridding, metrics or debugging,
    but they are not part of the minimum canonical visibility contract.
    """

    BASELINE_KEY = "BASELINE_KEY"
    BASELINE_LENGTH = "BASELINE_LENGTH"


class VisibilityMessageFields:
    """
    Fields associated with the current message-level contract.

    These fields describe the current Kafka/message metadata. They are not
    scientific visibility columns, but they are part of the transmission layer.
    """

    SCHEMA = "schema"
    METADATA = "metadata"

    MESSAGE_ID = "message_id"
    SUBMS_ID = "subms_id"
    FIELD_ID = "field_id"
    SPW_ID = "spw_id"
    POLARIZATION_ID = "polarization_id"

    N_CHANNELS = "n_channels"
    N_CORRELATIONS = "n_correlations"

    TOTAL_BLOCKS = "total_blocks"
