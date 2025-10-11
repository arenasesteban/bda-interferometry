"""
BDA Processor - Scientific Array Conversion for Vectorized Processing

This module handles the conversion of row-oriented data to numpy arrays optimized
for vectorized BDA scientific calculations. It integrates with BDA core algorithms
to perform averaging on pre-grouped data from the distributed processing pipeline.

The module focuses exclusively on array format conversion and BDA algorithm application,
with all grouping handled by Spark for optimal distributed performance.

Key Functions
-------------
convert_rows_to_group_arrays : Converts row dictionaries to numpy arrays
process_group_with_bda : Applies BDA algorithms to pre-grouped visibility data
"""

import logging
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

from .bda_core import apply_bda_to_group
from .bda_config import get_default_bda_config


def convert_rows_to_group_arrays(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert list of rows to numpy arrays for efficient BDA processing.
    
    Reorganizes data from individual row dictionaries to numpy arrays organized
    by field, enabling vectorized scientific processing in BDA algorithms.
    Assumes consumer has already validated and prepared all arrays.
    
    Parameters
    ----------
    rows : List[Dict[str, Any]]
        List of rows from same scientific group, with arrays already validated by consumer
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with numpy arrays organized by field for BDA_Core processing
    """
    if not rows:
        return {}
    
    n_rows = len(rows)
    
    # Validate array shapes across all rows and find consistent dimensions
    vis_shapes = [row['visibility'].shape for row in rows]
    unique_shapes = list(set(vis_shapes))
    
    if len(unique_shapes) > 1:
        # Log shape inconsistency for debugging
        logging.warning(f"Inconsistent visibility shapes in group: {unique_shapes}")
        # Use the most common shape or the first one
        from collections import Counter
        most_common_shape = Counter(vis_shapes).most_common(1)[0][0]
        n_chans, n_pols = most_common_shape
        logging.info(f"Using most common shape: {most_common_shape}")
    else:
        n_chans, n_pols = unique_shapes[0]
    
    # Pre-allocate arrays for efficient copying
    group_arrays = {
        'visibilities': np.zeros((n_rows, n_chans, n_pols), dtype=complex),
        'weights': np.zeros((n_rows, n_chans, n_pols), dtype=float),
        'flags': np.zeros((n_rows, n_chans, n_pols), dtype=bool),
        'u': np.zeros(n_rows, dtype=float),
        'v': np.zeros(n_rows, dtype=float),
        'w': np.zeros(n_rows, dtype=float),
        'time': np.zeros(n_rows, dtype=float),
        'antenna1': np.zeros(n_rows, dtype=int),
        'antenna2': np.zeros(n_rows, dtype=int),
        'scan_number': np.zeros(n_rows, dtype=int),
    }
    
    # Copy data from rows to pre-allocated arrays for vectorized processing
    valid_row_count = 0
    for i, row in enumerate(rows):
        try:
            # Check for shape compatibility before assignment
            row_vis_shape = row['visibility'].shape
            row_weight_shape = row['weight'].shape 
            row_flag_shape = row['flag'].shape
            
            if (row_vis_shape == (n_chans, n_pols) and 
                row_weight_shape == (n_chans, n_pols) and 
                row_flag_shape == (n_chans, n_pols)):
                
                group_arrays['visibilities'][valid_row_count] = row['visibility']
                group_arrays['weights'][valid_row_count] = row['weight'] 
                group_arrays['flags'][valid_row_count] = row['flag']
                group_arrays['u'][valid_row_count] = row['u']
                group_arrays['v'][valid_row_count] = row['v']
                group_arrays['w'][valid_row_count] = row['w']
                group_arrays['time'][valid_row_count] = row['time']
                group_arrays['antenna1'][valid_row_count] = row['antenna1']
                group_arrays['antenna2'][valid_row_count] = row['antenna2']
                group_arrays['scan_number'][valid_row_count] = row['scan_number']
                valid_row_count += 1
            else:
                logging.debug(f"Row {i}: incompatible shapes - vis:{row_vis_shape}, weight:{row_weight_shape}, flag:{row_flag_shape}, expected:({n_chans},{n_pols})")
                
        except Exception as shape_error:
            logging.warning(f"Row {i} processing error: {shape_error}")
            continue
    
    # Trim arrays to actual valid row count
    if valid_row_count < n_rows:
        logging.info(f"Using {valid_row_count}/{n_rows} rows with consistent shapes")
        for key in ['visibilities', 'weights', 'flags']:
            group_arrays[key] = group_arrays[key][:valid_row_count]
        for key in ['u', 'v', 'w', 'time', 'antenna1', 'antenna2', 'scan_number']:
            group_arrays[key] = group_arrays[key][:valid_row_count]
    
    return group_arrays


def process_group_with_bda(group_rows: List[Dict[str, Any]], 
                          bda_config: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Process a single pre-grouped set of visibility rows with BDA.
    
    Performs BDA processing on visibility data that has been pre-grouped by Spark.
    Converts row-oriented data to numpy arrays, applies BDA algorithms, and returns
    averaged results with metadata. Called by Spark's Pandas UDF for distributed processing.
    
    Parameters
    ----------
    group_rows : List[Dict[str, Any]]
        Visibility rows from the same baseline and scan number, pre-grouped by Spark
    bda_config : Dict[str, float], optional
        BDA configuration parameters including decorrelation factor and frequency
        
    Returns
    -------
    List[Dict[str, Any]]
        Averaged visibility data with BDA metadata and processing statistics
        
    Raises
    ------
    ValueError
        If group_rows is empty or contains invalid data structure
    """
    if bda_config is None:
        bda_config = get_default_bda_config()
    
    if not group_rows:
        return []
    
    # Extract group metadata from first row (all rows in group have same baseline+scan)
    first_row = group_rows[0]
    antenna1 = first_row['antenna1']
    antenna2 = first_row['antenna2'] 
    scan_number = first_row['scan_number']
    baseline_key = first_row.get('baseline_key', f"{min(antenna1, antenna2)}-{max(antenna1, antenna2)}")
    group_key = f"{baseline_key}_scan{scan_number}"
    
    logging.debug(f"Processing BDA group: {group_key} ({len(group_rows)} rows)")
    
    # Convert rows to numpy arrays for vectorized BDA processing
    group_arrays = convert_rows_to_group_arrays(group_rows)
    
    if not group_arrays:
        logging.warning(f"Empty arrays for group {group_key}")
        return []
    
    # Apply BDA algorithms to the group
    averaged_results = apply_bda_to_group(group_arrays, bda_config)
    
    # Add group metadata to results
    for result in averaged_results:
        result['group_key_str'] = group_key
        result['input_rows_count'] = len(group_rows)
    
    logging.debug(f"Group ({antenna1}-{antenna2}, scan {scan_number}): "
                 f"{len(group_rows)} rows → {len(averaged_results)} windows")
    
    return averaged_results

