"""
Basic Analytics - Distributed Processing for Visibility Data

Provides basic analysis and validation functions for interferometry
visibility data chunks in a distributed Spark environment.

This module implements simple analytics operations that can be
applied to streaming visibility data without complex transformations.
"""

import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class ChunkStats:
    """Statistics for a single visibility chunk."""
    chunk_id: str
    subms_id: str
    nrows: int
    n_channels: int
    n_correlations: int
    processing_time_ms: float
    timestamp: str
    
    # Visibility statistics
    vis_mean_real: float = 0.0
    vis_mean_imag: float = 0.0
    vis_std_real: float = 0.0
    vis_std_imag: float = 0.0
    
    # Data quality indicators
    nan_count: int = 0
    inf_count: int = 0
    is_valid: bool = True


class ChunkAnalyzer:
    """
    Analyzer for individual visibility chunks.
    
    Provides basic statistical analysis and validation for
    visibility data chunks in a distributed processing context.
    """
    
    def __init__(self):
        """Initialize chunk analyzer."""
        self.processed_chunks = 0
        self.total_processing_time = 0.0
    
    def analyze_chunk(self, chunk_data: Dict[str, Any]) -> ChunkStats:
        """
        Analyze a single visibility chunk.
        
        Computes basic statistics and validates data quality
        for a visibility data chunk. Handles both full chunks
        with arrays and metadata-only chunks.
        
        Parameters
        ----------
        chunk_data : Dict[str, Any]
            Dictionary containing chunk data (full or metadata-only)
            
        Returns
        -------
        ChunkStats
            Statistics and metadata for the processed chunk
        """
        # Check if this is metadata-only (from Spark processing)
        has_arrays = any(
            isinstance(chunk_data.get(key), np.ndarray) 
            for key in ['u', 'v', 'w', 'visibilities', 'time', 'antenna1', 'antenna2']
        )
        
        if not has_arrays:
            # Process as metadata-only
            return self.analyze_chunk_metadata(chunk_data)
        
        # Original full analysis with arrays
        start_time = time.time()
        
        # Extract basic metadata
        chunk_id = chunk_data.get('chunk_id', 'unknown')
        subms_id = chunk_data.get('subms_id', 'unknown')
        nrows = chunk_data.get('nrows', 0)
        n_channels = chunk_data.get('n_channels', 0)
        n_correlations = chunk_data.get('n_correlations', 0)
        
        # Initialize stats object
        stats = ChunkStats(
            chunk_id=str(chunk_id),
            subms_id=str(subms_id),
            nrows=nrows,
            n_channels=n_channels,
            n_correlations=n_correlations,
            processing_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Analyze visibility data if present
        visibilities = chunk_data.get('visibilities')
        if visibilities is not None and isinstance(visibilities, np.ndarray):
            stats = self._analyze_visibilities(visibilities, stats)
        
        # Validate data quality
        stats.is_valid = self._validate_chunk(chunk_data)
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000
        stats.processing_time_ms = processing_time
        
        # Update analyzer state
        self.processed_chunks += 1
        self.total_processing_time += processing_time
        
        return stats
    
    def _analyze_visibilities(self, visibilities: np.ndarray, stats: ChunkStats) -> ChunkStats:
        """
        Analyze visibility array statistics.
        
        Parameters
        ----------
        visibilities : np.ndarray
            Complex visibility data array
        stats : ChunkStats
            Stats object to update
            
        Returns
        -------
        ChunkStats
            Updated stats object with visibility statistics
        """
        try:
            # Handle complex data
            if np.iscomplexobj(visibilities):
                real_part = visibilities.real
                imag_part = visibilities.imag
                
                # Compute statistics for real and imaginary parts
                stats.vis_mean_real = float(np.nanmean(real_part))
                stats.vis_mean_imag = float(np.nanmean(imag_part))
                stats.vis_std_real = float(np.nanstd(real_part))
                stats.vis_std_imag = float(np.nanstd(imag_part))
                
                # Check for problematic values
                stats.nan_count = int(np.sum(np.isnan(visibilities)))
                stats.inf_count = int(np.sum(np.isinf(visibilities)))
            else:
                # Handle real-valued data
                stats.vis_mean_real = float(np.nanmean(visibilities))
                stats.vis_std_real = float(np.nanstd(visibilities))
                stats.nan_count = int(np.sum(np.isnan(visibilities)))
                stats.inf_count = int(np.sum(np.isinf(visibilities)))
                
        except Exception as e:
            print(f"Warning: Error analyzing visibilities: {e}")
            stats.is_valid = False
        
        return stats
    
    def _validate_chunk(self, chunk_data: Dict[str, Any]) -> bool:
        """
        Validate chunk data structure and content.
        
        Parameters
        ----------
        chunk_data : Dict[str, Any]
            Chunk data to validate
            
        Returns
        -------
        bool
            True if chunk is valid, False otherwise
        """
        required_fields = [
            'subms_id', 'chunk_id', 'nrows', 'n_channels', 'n_correlations'
        ]
        
        # Check required metadata fields
        for field in required_fields:
            if field not in chunk_data:
                return False
        
        # Check data consistency
        nrows = chunk_data.get('nrows', 0)
        if nrows <= 0:
            return False
        
        # Validate array dimensions if present
        for array_name in ['u', 'v', 'w', 'time', 'antenna1', 'antenna2']:
            arr = chunk_data.get(array_name)
            if arr is not None and isinstance(arr, np.ndarray):
                if len(arr.shape) > 0 and arr.shape[0] != nrows:
                    return False
        
        return True
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get analyzer performance statistics.
        
        Returns
        -------
        Dict[str, float]
            Performance metrics for the analyzer
        """
        avg_processing_time = (
            self.total_processing_time / self.processed_chunks 
            if self.processed_chunks > 0 else 0.0
        )
        
        return {
            'total_chunks_processed': self.processed_chunks,
            'total_processing_time_ms': self.total_processing_time,
            'average_processing_time_ms': avg_processing_time,
            'throughput_chunks_per_second': (
                self.processed_chunks / (self.total_processing_time / 1000.0)
                if self.total_processing_time > 0 else 0.0
            )
        }
    
    def analyze_chunk_metadata(self, chunk_metadata: Dict[str, Any]) -> ChunkStats:
        """
        Analyze chunk using only metadata (for Spark distributed processing).
        
        When processing in Spark, we often work with metadata only to avoid
        transferring large arrays between executors and the driver.
        
        Parameters
        ----------
        chunk_metadata : Dict[str, Any]
            Dictionary containing chunk metadata (not full arrays)
            
        Returns
        -------
        ChunkStats
            Statistics based on metadata information
        """
        start_time = time.time()
        
        # Extract basic metadata
        chunk_id = chunk_metadata.get('chunk_id', 'unknown')
        subms_id = chunk_metadata.get('subms_id', 'unknown')
        nrows = chunk_metadata.get('nrows', 0)
        n_channels = chunk_metadata.get('n_channels', 0)
        n_correlations = chunk_metadata.get('n_correlations', 0)
        
        # Initialize stats object
        stats = ChunkStats(
            chunk_id=str(chunk_id),
            subms_id=str(subms_id),
            nrows=nrows,
            n_channels=n_channels,
            n_correlations=n_correlations,
            processing_time_ms=0.0,
            timestamp=datetime.now().isoformat()
        )
        
        # Check if we have visibility metadata
        vis_metadata = chunk_metadata.get('visibilities')
        if vis_metadata and isinstance(vis_metadata, dict):
            # Extract array information from metadata
            shape = vis_metadata.get('shape', [])
            dtype = vis_metadata.get('dtype', 'unknown')
            is_compressed = vis_metadata.get('compressed', False)
            
            # Basic validation based on shape
            if len(shape) >= 2:
                expected_rows = shape[0] if len(shape) > 0 else 0
                if expected_rows != nrows and nrows > 0:
                    stats.is_valid = False
                    
            # Set metadata-based stats
            stats.vis_mean_real = 0.0  # Not available from metadata
            stats.vis_mean_imag = 0.0  # Not available from metadata
        
        # Basic validation
        stats.is_valid = stats.is_valid and self._validate_chunk_metadata(chunk_metadata)
        
        # Record processing time
        processing_time = (time.time() - start_time) * 1000
        stats.processing_time_ms = processing_time
        
        # Update analyzer state
        self.processed_chunks += 1
        self.total_processing_time += processing_time
        
        return stats
    
    def _validate_chunk_metadata(self, chunk_metadata: Dict[str, Any]) -> bool:
        """
        Validate chunk metadata structure.
        
        Parameters
        ----------
        chunk_metadata : Dict[str, Any]
            Chunk metadata to validate
            
        Returns
        -------
        bool
            True if metadata is valid, False otherwise
        """
        required_fields = [
            'subms_id', 'chunk_id', 'nrows', 'n_channels', 'n_correlations'
        ]
        
        # Check required metadata fields
        for field in required_fields:
            if field not in chunk_metadata:
                return False
        
        # Check data consistency
        nrows = chunk_metadata.get('nrows', 0)
        if nrows <= 0:
            return False
        
        return True

    # ...existing code...


def validate_chunk_data(chunk_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate chunk data and return detailed validation results.
    
    Parameters
    ----------
    chunk_data : Dict[str, Any]
        Chunk data dictionary to validate
        
    Returns
    -------
    Tuple[bool, List[str]]
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Required fields validation
    required_fields = [
        'subms_id', 'chunk_id', 'nrows', 'n_channels', 'n_correlations'
    ]
    
    for field in required_fields:
        if field not in chunk_data:
            errors.append(f"Missing required field: {field}")
    
    # Data type validation
    if 'nrows' in chunk_data:
        if not isinstance(chunk_data['nrows'], (int, np.integer)) or chunk_data['nrows'] <= 0:
            errors.append("Invalid nrows: must be positive integer")
    
    # Array validation
    array_fields = ['u', 'v', 'w', 'visibilities', 'time', 'antenna1', 'antenna2']
    nrows = chunk_data.get('nrows', 0)
    
    for field in array_fields:
        if field in chunk_data and chunk_data[field] is not None:
            arr = chunk_data[field]
            if not isinstance(arr, np.ndarray):
                errors.append(f"Field {field} is not a numpy array")
            elif field in ['u', 'v', 'w', 'time', 'antenna1', 'antenna2']:
                if len(arr.shape) > 0 and arr.shape[0] != nrows:
                    errors.append(f"Array {field} has incorrect length: {arr.shape[0]} != {nrows}")
    
    return len(errors) == 0, errors


def format_chunk_summary(stats: ChunkStats) -> str:
    """
    Format chunk statistics for console output.
    
    Parameters
    ----------
    stats : ChunkStats
        Chunk statistics to format
        
    Returns
    -------
    str
        Formatted string summary
    """
    status = "✓ VALID" if stats.is_valid else "✗ INVALID"
    
    return (
        f"[CHUNK] {stats.subms_id}_{stats.chunk_id} | "
        f"Rows: {stats.nrows:,} | "
        f"Channels: {stats.n_channels} | "
        f"Corrs: {stats.n_correlations} | "
        f"Time: {stats.processing_time_ms:.1f}ms | "
        f"Status: {status}"
    )
