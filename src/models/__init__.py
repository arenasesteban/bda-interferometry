"""
Models package for MS file structures.

This package contains dataclasses that represent the different components
of an MS (Measurement Set) file used in radio astronomy.
"""

from .antenna import Antenna
from .spectral_info import SpectralInfo
from .column_info import ColumnInfo
from .visibility import Visibility
from .ms_file import MSFile

__all__ = ['Antenna', 'SpectralInfo', 'ColumnInfo', 'Visibility', 'MSFile']
