"""
Custom errors for visibility contract validation.
"""


class VisibilityContractError(Exception):
    """
    Base exception for visibility contract errors.
    """


class MissingVisibilityColumnsError(VisibilityContractError):
    """
    Raised when required visibility columns are missing.
    """


class InvalidVisibilityMetadataError(VisibilityContractError):
    """
    Raised when visibility metadata is invalid.
    """


class InvalidVisibilityBatchError(VisibilityContractError):
    """
    Raised when the visibility batch does not satisfy the contract.
    """