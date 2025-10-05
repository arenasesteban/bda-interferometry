"""
BDA Processor - Practical BDA Application in Streaming Flow

Implements practical BDA application with the following operations:
1. Grouping by (baseline, scan_number)  
2. Calculate baseline length = sqrt(u²+v²)
3. Define smearing tolerance
4. Temporal windowing
5. Window averaging

This module integrates with the consumer service to apply BDA to microbatches.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from .bda_core import create_bda_config, apply_bda_to_group


def normalize_baseline_key(antenna1: int, antenna2: int) -> Tuple[int, int]:
    """
    Normalize baseline for consistency with consumer service.
    
    Parameters
    ----------
    antenna1 : int
        First antenna identifier
    antenna2 : int
        Second antenna identifier
        
    Returns
    -------
    Tuple[int, int]
        Normalized antenna pair as (min_antenna, max_antenna)
    """
    return tuple(sorted([antenna1, antenna2]))


def group_visibility_rows(rows_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group visibility rows by consistent group key.
    
    Uses same grouping logic as consumer service to ensure
    compatibility between systems.
    
    Parameters
    ----------
    rows_data : List[Dict[str, Any]]
        List of visibility rows decomposed from chunks
        
    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary grouped by string group key
    """
    groups = defaultdict(list)
    
    for row in rows_data:
        if 'baseline_key' in row:
            baseline_key = row['baseline_key']
            scan_number = row['scan_number']
            group_key = f"{baseline_key}_scan{scan_number}"
        else:
            ant1 = int(row['antenna1'])
            ant2 = int(row['antenna2'])
            scan_num = int(row['scan_number'])
            subms_id = row.get('subms_id', None)
            
            ant_min, ant_max = sorted([ant1, ant2])
            
            if subms_id:
                baseline_key = f"{ant_min}-{ant_max}-{subms_id}"
            else:
                baseline_key = f"{ant_min}-{ant_max}"
            
            group_key = f"{baseline_key}_scan{scan_num}"
        
        groups[group_key].append(row)
    
    return dict(groups)


def convert_rows_to_group_arrays(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Convert list of rows to numpy arrays for BDA processing.
    
    Reorganizes data from individual rows to numpy arrays organized
    by field, facilitating vectorized processing in BDA.
    
    Parameters
    ----------
    rows : List[Dict[str, Any]]
        List of rows from same group (baseline, scan_number)
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with numpy arrays organized by field
    """
    if not rows:
        return {}
    
    n_rows = len(rows)
    first_row = rows[0]
    
    if 'visibility' not in first_row or not hasattr(first_row['visibility'], 'shape'):
        raise ValueError("Row missing 'visibility' array - consumer may not be providing real data")
    
    vis_shape = first_row['visibility'].shape
    n_chans, n_pols = vis_shape
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
    
    for i, row in enumerate(rows):
        group_arrays['visibilities'][i] = row['visibility']
        group_arrays['weights'][i] = row['weight'] 
        group_arrays['flags'][i] = row['flag']
        group_arrays['u'][i] = row['u']
        group_arrays['v'][i] = row['v']
        group_arrays['w'][i] = row['w']
        group_arrays['time'][i] = row['time']
        group_arrays['antenna1'][i] = row['antenna1']
        group_arrays['antenna2'][i] = row['antenna2']
        group_arrays['scan_number'][i] = row['scan_number']
    
    return group_arrays


def process_microbatch_with_bda(rows_data: List[Dict[str, Any]], 
                               bda_config: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Process complete microbatch applying BDA.
    
    Implements complete BDA flow by grouping rows by (baseline, scan_number),
    converting to numpy arrays, applying BDA to each group, and returning
    averaged visibilities.
    
    Parameters
    ----------
    rows_data : List[Dict[str, Any]]
        List of visibility rows from microbatch
    bda_config : Dict[str, float], optional
        BDA configuration parameters
        
    Returns
    -------
    List[Dict[str, Any]]
        List of averaged visibilities with BDA metadata
    """
    if bda_config is None:
        bda_config = create_bda_config()
    
    groups = group_visibility_rows(rows_data)
    logging.info(f"BDA Processing: {len(rows_data)} input rows → {len(groups)} groups")
    
    all_averaged_results = []
    
    for group_key, group_rows in groups.items():
        if not group_rows:
            continue
            
        first_row = group_rows[0]
        antenna1 = first_row['antenna1']
        antenna2 = first_row['antenna2'] 
        scan_number = first_row['scan_number']
        
        group_arrays = convert_rows_to_group_arrays(group_rows)
        
        if not group_arrays:
            continue
        
        averaged_results = apply_bda_to_group(group_arrays, bda_config)
        
        for result in averaged_results:
            result['group_key_str'] = group_key
            result['input_rows_count'] = len(group_rows)
        
        all_averaged_results.extend(averaged_results)
        
        logging.debug(f"  Group ({antenna1}-{antenna2}, scan {scan_number}): "
                     f"{len(group_rows)} rows → {len(averaged_results)} windows")
    
    total_compression = len(rows_data) / len(all_averaged_results) if all_averaged_results else 1
    logging.info(f"BDA Complete: compression ratio {total_compression:.2f}:1 "
                f"({len(rows_data)} → {len(all_averaged_results)} averaged visibilities)")
    
    return all_averaged_results


def format_bda_result_for_output(bda_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format BDA result for output and storage.
    
    Converts numpy arrays to lists and adds additional metadata
    to facilitate serialization and later analysis.
    
    Parameters
    ----------
    bda_result : Dict[str, Any]
        BDA result containing numpy arrays
        
    Returns
    -------
    Dict[str, Any]
        Formatted result ready for serialization
    """
    formatted = {
        'visibility_averaged': bda_result['visibility_averaged'].tolist(),
        'weight_total': bda_result['weight_total'].tolist(),
        'flag_combined': bda_result['flag_combined'].tolist(),
        'u_avg': float(bda_result['u_avg']),
        'v_avg': float(bda_result['v_avg']),
        'w_avg': float(bda_result['w_avg']),
        'time_avg': float(bda_result['time_avg']),
        'n_input_rows': int(bda_result['n_input_rows']),
        'window_dt_s': float(bda_result['window_dt_s']),
        'baseline_length': float(bda_result['baseline_length']),
        'delta_t_max': float(bda_result['delta_t_max']),
        'antenna1': int(bda_result['antenna1']),
        'antenna2': int(bda_result['antenna2']),
        'scan_number': int(bda_result['scan_number']),
    }
    
    if 'group_key_str' in bda_result:
        formatted['group_key_str'] = bda_result['group_key_str']
    if 'input_rows_count' in bda_result:
        formatted['input_rows_count'] = int(bda_result['input_rows_count'])
    
    return formatted


def create_bda_summary_stats(bda_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create BDA processing summary statistics.
    
    Generates useful metrics for monitoring and analysis of
    BDA algorithm performance in streaming environments.
    
    Parameters
    ----------
    bda_results : List[Dict[str, Any]]
        List of BDA processing results
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary statistics and performance metrics
    """
    if not bda_results:
        return {'total_windows': 0, 'total_input_rows': 0}
    
    total_windows = len(bda_results)
    total_input_rows = sum(r.get('n_input_rows', 0) for r in bda_results)
    total_groups = len(set(r.get('group_key_str', 'unknown') for r in bda_results))
    
    window_sizes = [r.get('n_input_rows', 0) for r in bda_results]
    window_durations = [r.get('window_dt_s', 0) for r in bda_results]
    baseline_lengths = [r.get('baseline_length', 0) for r in bda_results]
    
    compression_ratio = total_input_rows / total_windows if total_windows > 0 else 1
    
    averaging_applied_rows = sum(r.get('n_input_rows', 0) for r in bda_results if r.get('n_input_rows', 0) > 1)
    single_row_groups = sum(1 for r in bda_results if r.get('n_input_rows', 0) == 1)
    
    averaging_time_distribution = []
    for r in bda_results:
        if r.get('n_input_rows', 0) > 1:
            averaging_time_distribution.append(r.get('delta_t_max', 0))
    
    stats = {
        'total_windows': total_windows,
        'total_input_rows': total_input_rows,
        'total_groups': total_groups,
        'compression_ratio': compression_ratio,
        'avg_window_size': np.mean(window_sizes) if window_sizes else 0,
        'max_window_size': max(window_sizes) if window_sizes else 0,
        'avg_window_duration_s': np.mean(window_durations) if window_durations else 0,
        'max_window_duration_s': max(window_durations) if window_durations else 0,
        'avg_baseline_length': np.mean(baseline_lengths) if baseline_lengths else 0,
        'baseline_length_range': [min(baseline_lengths), max(baseline_lengths)] if baseline_lengths else [0, 0],
        'total_output_rows': total_windows,
        'averaging_applied_rows': averaging_applied_rows,
        'single_row_groups': single_row_groups,
        'averaging_time_distribution': averaging_time_distribution,
        'baseline_statistics': {},
        'groups_processed': total_groups,
    }
    
    return stats
