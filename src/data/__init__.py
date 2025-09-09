"""
Data Package - Interferometry Data Operations

Contains modules for simulating and extracting radio interferometry data.
Provides comprehensive functionality for dataset generation and visibility
data extraction for streaming operations.

Modules
-------
simulation : Complete pipeline for synthetic dataset generation using Pyralysis
extraction : Visibility data extraction and chunking utilities for streaming
"""

# Simulation functions
from .simulation import (
    generate_dataset,
    load_antenna_configuration,
    configure_observation,
    simulate_dataset
)

# Extraction functions  
from .extraction import (
    stream_subms_chunks
)

__all__ = [
    # Simulation functions
    'generate_dataset',
    'load_antenna_configuration',
    'configure_observation',
    'simulate_dataset',
    
    # Extraction functions
    'stream_subms_chunks'
]
