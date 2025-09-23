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
    Extract raw data for a specific chunk range with temporal and frequency info.
    
    Extracts and computes visibility data arrays for a specified row range,
    creating a complete chunk dictionary with metadata, raw arrays, and
    temporal/frequency information for windowed streaming.
    
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
        Dictionary containing chunk data with raw numpy arrays and temporal info
    """
    
    chunk_size = end_row - start_row
    
    # Extract temporal information for windowing
    temporal_info = _extract_temporal_info(vis_set, start_row, end_row)
    
    # Extract frequency information  
    frequency_info = _extract_frequency_info(subms)
    
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
        
        # Temporal information for windowing  
        'time_start': temporal_info['time_start'],
        'time_end': temporal_info['time_end'],
        'dt': temporal_info['dt'],
        
        # Frequency information
        'channel_frequencies': frequency_info['channel_frequencies'],
        'nu0': frequency_info['nu0'],
        
        # Raw visibility data arrays
        'u': _safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=0),
        'v': _safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=1),
        'w': _safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=2),
        'visibilities': _safe_compute_slice(vis_set.data, start_row, end_row),
        'weight': _safe_compute_slice(vis_set.weight, start_row, end_row),
        'time': _safe_compute_slice(vis_set.time, start_row, end_row),
        'antenna1': _safe_compute_slice(vis_set.antenna1, start_row, end_row),
        'antenna2': _safe_compute_slice(vis_set.antenna2, start_row, end_row),
        'flag': _safe_compute_slice(vis_set.flag, start_row, end_row),
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


def _extract_temporal_info(vis_set, start_row: int, end_row: int) -> Dict[str, Any]:
    """
    Extract temporal information for windowed streaming.
    
    Calculates time_start, time_end, and dt from the time array slice
    for use in temporal windowing and watermarking.
    
    Parameters
    ----------
    vis_set : object
        VisibilitySet object containing time data
    start_row : int
        Starting row index
    end_row : int
        Ending row index
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with temporal information
    """
    time_array = _safe_compute_slice(vis_set.time, start_row, end_row)
    
    if time_array is None or len(time_array) == 0:
        return {
            'time_start': None,
            'time_end': None,
            'dt': None
        }
    
    time_start = float(time_array.min())
    time_end = float(time_array.max())
    
    # Calculate dt if time intervals are constant
    dt = None
    if len(time_array) > 1:
        time_diffs = np.diff(time_array)
        if np.allclose(time_diffs, time_diffs[0], rtol=1e-6):
            dt = float(time_diffs[0])
    
    return {
        'time_start': time_start,
        'time_end': time_end,
        'dt': dt
    }


def _extract_frequency_info(subms) -> Dict[str, Any]:
    """
    Extract frequency information for the SubMS.
    
    Gets channel frequencies and reference frequency from the SubMS
    for use in BDA algorithms.
    
    Parameters
    ----------
    subms : object
        SubMS object containing frequency information
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with frequency information
    """
    try:
        # Extract channel frequencies if available
        if hasattr(subms, 'chan_freq') and subms.chan_freq is not None:
            if hasattr(subms.chan_freq, 'compute'):
                channel_frequencies = subms.chan_freq.compute()
            else:
                channel_frequencies = np.array(subms.chan_freq)
        elif hasattr(subms.visibilities, 'frequency') and subms.visibilities.frequency is not None:
            freq_data = subms.visibilities.frequency
            if hasattr(freq_data, 'compute'):
                channel_frequencies = freq_data.compute()
            else:
                channel_frequencies = np.array(freq_data)
        else:
            # Fallback: create frequency array from available info
            n_channels = subms.visibilities.data.shape[1] if subms.visibilities.data is not None else 50
            channel_frequencies = np.linspace(35e9, 50e9, n_channels)  # Default ALMA Band 1 range
        
        # Reference frequency (center frequency)
        nu0 = float(channel_frequencies[len(channel_frequencies)//2])
        
        return {
            'channel_frequencies': channel_frequencies,
            'nu0': nu0
        }
        
    except Exception:
        # Fallback values
        n_channels = 50  # Default
        channel_frequencies = np.linspace(35e9, 50e9, n_channels)
        nu0 = 42.5e9  # Center of default range
        
        return {
            'channel_frequencies': channel_frequencies,
            'nu0': nu0
        }