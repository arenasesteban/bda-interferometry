"""
Pipeline module for BDA Interferometry processing.

This module contains the components for the complete data processing pipeline:
- Dataset generation from simulated observations
- BDA processing
- Kafka streaming
- Grid organization for imaging
"""

from .dataset_generator import DatasetGenerator

__all__ = ['DatasetGenerator']
