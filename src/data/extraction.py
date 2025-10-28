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
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c

def stream_subms_chunks(dataset, longitude, latitude) -> Generator[Dict[str, Any], None, None]:
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
    
    antennas = dataset.antenna.dataset.POSITION
    
    ref_nu = dataset.spws.ref_nu
    ra_deg = dataset.field.ref_dirs.ra[0]
    dec_deg = dataset.field.ref_dirs.dec[0]

    sky_coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')

    ra = sky_coord.ra.rad
    dec = sky_coord.dec.rad

    for subms in dataset.ms_list:
        if subms is None or subms.visibilities is None:
            continue

        yield from _extract_subms_chunks(subms, antennas, longitude, latitude, ref_nu, ra, dec)


def _extract_subms_chunks(subms, antennas, longitude, latitude, ref_nu, ra, dec) -> Generator[Dict[str, Any], None, None]:
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
            subms, antennas, vis, chunk_id, start_row, end_row, 
            n_channels, n_correlations,
            longitude,
            latitude,
            ref_nu, ra, dec,
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


def to_simple_array(np_array):
    """Convert NumPy array to simple Python list for clean serialization."""
    if np_array is None:
        return []
    try:
        # Asegurar que es un array NumPy
        if not isinstance(np_array, np.ndarray):
            np_array = np.array(np_array)
        
        # Convertir a lista Python simple
        return np_array.tolist()
    except Exception as e:
        print(f"Could not convert array to list: {e}")
        return []
    

def _extract_chunk_data(subms, antennas, vis_set, chunk_id: int, start_row: int, end_row: int,
                       n_channels: int, n_correlations: int, longitude, latitude, ref_nu, ra, dec) -> Dict[str, Any]:
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

    uvw_lambda = vis_set.uvw.get_uvw_lambda(ref_nu, as_xarray=True)
    lambda_ = c.value / ref_nu

    antenna1 = _safe_compute_slice(vis_set.antenna1, start_row, end_row)
    antenna2 = _safe_compute_slice(vis_set.antenna2, start_row, end_row)

    Lx = antennas[antenna1][:, 0] - antennas[antenna2][:, 0]
    Ly = antennas[antenna1][:, 1] - antennas[antenna2][:, 1]

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
        
        'antenna1': to_simple_array(antenna1),
        'antenna2': to_simple_array(antenna2),
        'scan_number': to_simple_array(_safe_compute_slice(vis_set.scan_number, start_row, end_row)),

        'longitude': longitude,
        'latitude': latitude,

        'lambda_': lambda_,
        'ra': ra,
        'dec': dec,

        # Timing information
        'exposure': to_simple_array(_safe_compute_slice(vis_set.dataset.EXPOSURE, start_row, end_row)),
        'interval': to_simple_array(_safe_compute_slice(vis_set.dataset.INTERVAL, start_row, end_row)),
        'time': to_simple_array(_safe_compute_slice(vis_set.time, start_row, end_row)),

        'u': to_simple_array(_safe_compute_slice(uvw_lambda, start_row, end_row, coord_idx=0)),
        'v': to_simple_array(_safe_compute_slice(uvw_lambda, start_row, end_row, coord_idx=1)),
        'w': to_simple_array(_safe_compute_slice(uvw_lambda, start_row, end_row, coord_idx=2)),
        'Lx': to_simple_array(Lx),
        'Ly': to_simple_array(Ly),

        # Essential scientific data arrays
        'visibilities': _safe_compute_slice(vis_set.data, start_row, end_row),
        'weight': _safe_compute_slice(vis_set.weight, start_row, end_row),
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
