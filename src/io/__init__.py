"""
IO package for data input/output operations.

This package provides interfaces for reading and writing various
astronomical data formats, with a focus on MS (Measurement Set) files.
"""

from .ms_reader import MSReader
from ..models import MSFile

__all__ = ['MSReader', 'MSFile']
