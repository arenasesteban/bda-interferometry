"""
Extraction - Visibility Data Extraction and Chunking

Extracts and processes radio interferometry visibility data from Pyralysis datasets
for streaming transmission. Handles the conversion of complex dataset objects into
streamable data chunks while preserving scientific data integrity.

This module provides efficient chunking algorithms and data extraction utilities
optimized for real-time streaming operations via messaging systems.
"""

import numpy as np
from typing import Dict, Any, Generator


def stream_subms_chunks(dataset) -> Generator[Dict[str, Any], None, None]:
    """
    Stream chunks from SubMS objects in the dataset's ms_list.
    
    Extracts and yields visibility data chunks from all SubMS objects
    contained in the dataset for streaming transmission.
    
    Parameters
    ----------
    dataset : object
        Pyralysis dataset object containing ms_list
        
    Yields
    ------
    Dict[str, Any]
        Dictionary containing raw visibility data chunk
        
    Raises
    ------
    ValueError
        If no ms_list is found in the dataset
    """
    
    if not hasattr(dataset, 'ms_list') or dataset.ms_list is None:
        raise ValueError("No ms_list found in dataset")
    
    for subms in dataset.ms_list:
        if subms is None or subms.visibilities is None:
            continue
            
        yield from _extract_subms_chunks(subms)


def _extract_subms_chunks(subms) -> Generator[Dict[str, Any], None, None]:
    """
    Extract chunks from a single SubMS object.
    
    Processes a single SubMS partition to generate appropriately sized
    chunks for streaming based on optimal chunking strategies.
    
    Parameters
    ----------
    subms : object
        SubMS object containing VisibilitySet
        
    Yields
    ------
    Dict[str, Any]
        Dictionary containing chunk data with raw arrays
    """
    
    vis = subms.visibilities
    
    # Get basic dimensions
    nrows = vis.nrows
    n_channels = vis.data.shape[1]
    n_correlations = vis.data.shape[2]
    
    # Determine chunk size based on Dask chunks or default
    chunk_size = _get_optimal_chunk_size(vis)
    
    # Generate chunks
    chunk_id = 0
    for start_row in range(0, nrows, chunk_size):
        end_row = min(start_row + chunk_size, nrows)
        
        chunk = _extract_chunk_data(
            subms, vis, chunk_id, start_row, end_row, 
            n_channels, n_correlations
        )
        
        yield chunk
        chunk_id += 1


def _get_optimal_chunk_size(vis_set) -> int:
    """
    Determine optimal chunk size based on Dask chunking.
    
    Analyzes the Dask chunk structure of Pyralysis datasets to determine
    the most efficient chunk size for streaming operations.
    
    Parameters
    ----------
    vis_set : object
        VisibilitySet object from Pyralysis
        
    Returns
    -------
    int
        Optimal chunk size for streaming based on Dask chunks
    """
    
    chunks = vis_set.data.data.chunks[0]  # Row dimension chunks
    
    if len(chunks) > 1:
        return chunks[0]
    else:
        return min(1000, vis_set.nrows)


def _extract_chunk_data(subms, vis_set, chunk_id: int, start_row: int, end_row: int,
                       n_channels: int, n_correlations: int) -> Dict[str, Any]:
    """
    Extract raw data for a specific chunk range.
    
    Extracts raw visibility data arrays for a specified row range,
    creating a chunk dictionary with essential metadata and scientific arrays only.
    Includes frequency and timing metadata critical for BDA processing.
    
    Parameters
    ----------
    subms : object
        SubMS object containing metadata
    vis_set : object
        VisibilitySet object containing data arrays
    chunk_id : int
        Unique identifier for this chunk
    start_row : int
        Starting row index for extraction
    end_row : int
        Ending row index for extraction
    n_channels : int
        Number of frequency channels
    n_correlations : int
        Number of polarization correlations
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing chunk data with raw numpy arrays only
    """
    
    chunk_size = end_row - start_row
    
    return {
        # Metadata
        'subms_id': subms._id,
        'chunk_id': chunk_id,
        'field_id': subms.field_id,
        'spw_id': subms.spw_id,
        'polarization_id': subms.polarization_id,
        'row_start': start_row,
        'row_end': end_row,
        'nrows': chunk_size,
        'n_channels': n_channels,
        'n_correlations': n_correlations,
        
        # Timing information (available in dataset)
        'exposure': _safe_compute_slice(vis_set.dataset.EXPOSURE, start_row, end_row),   # Integration time per row (seconds)
        'interval': _safe_compute_slice(vis_set.dataset.INTERVAL, start_row, end_row),   # Time interval per row (seconds)
        'integration_time_s': _extract_integration_time(vis_set, start_row, end_row),    # Median integration time for chunk
        
        # Essential scientific data arrays
        'u': _safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=0),          # UVW coordinates per row
        'v': _safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=1),          # (critical for baseline length calc)
        'w': _safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=2),
        'visibilities': _safe_compute_slice(vis_set.data, start_row, end_row),           # Main scientific data [nrows, nchans, npols]
        'weight': _safe_compute_slice(vis_set.weight, start_row, end_row),               # Statistical weights (critical for BDA)
        'time': _safe_compute_slice(vis_set.time, start_row, end_row),                   # Individual timestamps per row (critical for windowing)
        'antenna1': _safe_compute_slice(vis_set.antenna1, start_row, end_row),           # Baseline definition (critical)
        'antenna2': _safe_compute_slice(vis_set.antenna2, start_row, end_row),           # Baseline definition (critical)  
        'scan_number': _safe_compute_slice(vis_set.scan_number, start_row, end_row),     # Observation period identifier (critical for BDA grouping)
        'flag': _safe_compute_slice(vis_set.flag, start_row, end_row),                   # Data quality flags
    }


def _safe_compute_slice(xr_array, start_row: int, end_row: int, coord_idx: int = None):
    """
    Safely compute a slice of an xarray DataArray.
    
    Handles the extraction and computation of data slices from xarray
    DataArrays with proper error handling and type conversion.
    
    Parameters
    ----------
    xr_array : xarray.DataArray
        xarray DataArray to slice
    start_row : int
        Starting row index for slicing
    end_row : int
        Ending row index for slicing
    coord_idx : int, optional
        Coordinate index for UVW extraction
        
    Returns
    -------
    numpy.ndarray or None
        Computed numpy array for the slice, or None if extraction fails
    """
    
    if xr_array is None:
        return None
        
    try:
        if coord_idx is not None:
            # For UVW coordinates (u, v, w)
            data_slice = xr_array.data[start_row:end_row, coord_idx]
        else:
            # For other arrays
            data_slice = xr_array.data[start_row:end_row]
            
        # Compute if it's a dask array
        if hasattr(data_slice, 'compute'):
            return data_slice.compute()
        else:
            return np.array(data_slice)
            
    except Exception:
        return None


def _extract_integration_time(vis_set, start_row: int, end_row: int) -> float:
    """
    Extract integration time from VisibilitySet dataset using EXPOSURE/INTERVAL.
    
    Accesses EXPOSURE and INTERVAL columns directly from the dataset.
    These fields are available in the VisibilitySet.dataset even though
    not mapped as properties in the VisibilitySet class.
    
    Parameters
    ----------
    vis_set : object
        VisibilitySet object with dataset containing EXPOSURE/INTERVAL
    start_row : int
        Starting row index
    end_row : int
        Ending row index
        
    Returns
    -------
    float
        Integration time in seconds (median value from EXPOSURE column)
    """
    
    try:
        # Access EXPOSURE column directly from dataset - this IS available!
        if hasattr(vis_set, 'dataset') and 'EXPOSURE' in vis_set.dataset.data_vars:
            exposure = vis_set.dataset.EXPOSURE
            exposure_slice = _safe_compute_slice(exposure, start_row, end_row)
            if exposure_slice is not None and len(exposure_slice) > 0:
                # Return median exposure time for this chunk
                return float(np.median(exposure_slice))
        
        # Try INTERVAL column as alternative
        if hasattr(vis_set, 'dataset') and 'INTERVAL' in vis_set.dataset.data_vars:
            interval = vis_set.dataset.INTERVAL
            interval_slice = _safe_compute_slice(interval, start_row, end_row)
            if interval_slice is not None and len(interval_slice) > 0:
                return float(np.median(interval_slice))
        
        # Calculate from time differences as fallback
        time_data = _safe_compute_slice(vis_set.time, start_row, end_row)
        if time_data is not None and len(time_data) > 1:
            time_diffs = np.diff(time_data)
            if len(time_diffs) > 0:
                integration_time = float(np.median(time_diffs))
                if integration_time > 0:
                    return integration_time
                
    except (AttributeError, IndexError, KeyError, TypeError):
        pass
    
    # Fallback to default integration time (3 minutes = 180 seconds)
    return 180.0
