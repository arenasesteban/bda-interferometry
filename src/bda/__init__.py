"""
BDA Package - Baseline-Dependent Averaging Implementation

Contains the BDA (Baseline-Dependent Averaging) implementation for radio
interferometry data processing. Provides algorithms for optimizing data
volume while preserving scientific information content.

Modules
-------
bda : Core BDA implementation and algorithms
"""

from .bda import BDA

__all__ = ['BDA']