"""
BDA Processor - Incremental Streaming Processing with Decorrelation Windows

This module implements memory-efficient BDA processing using incremental windows
based on physical decorrelation time following Wijnholds et al. 2018 methodology.
Instead of accumulating full baselines, it processes data online as it arrives,
maintaining constant RAM usage.

The approach uses Spark's mapPartitions for distributed processing with window
closure based on baseline-dependent decorrelation criteria using proper physics.

Key Functions
-------------
process_rows_incrementally : Main streaming BDA processor with online windows
complete_bda_window : Weighted averaging and window completion
get_decorrelation_time : Physics-based window sizing per visibility sample
"""

import numpy as np
from typing import Dict, Any, Tuple

from .bda_core import calculate_optimal_averaging_time, calculate_decorrelation_time


def get_decorrelation_time(u: float, v: float, bda_config: Dict[str, Any]) -> float:
    """
    Calculate decorrelation time for a single visibility sample using physics.
    
    Replaces heuristic baseline-length thresholds with proper Wijnholds et al. 2018
    equations accounting for frequency, declination, Earth rotation, and safety factors.
    
    Parameters
    ----------
    u : float
        U coordinate in wavelengths (assumed from consumer)
    v : float
        V coordinate in wavelengths (assumed from consumer)
    bda_config : Dict[str, Any]
        BDA configuration with physics parameters
        
    Returns
    -------
    float
        Optimal decorrelation time in seconds for this baseline
        
    Notes
    -----
    This function assumes UV coordinates are already in wavelengths from the consumer.
    If consumer provides meters, modify extract functions to convert units there.
    """
    frequency_hz = bda_config.get('frequency_hz', 700e6)
    
    decorr_time = calculate_optimal_averaging_time(
        u=u, 
        v=v,
        frequency_hz=frequency_hz,
        bda_config=bda_config,
    )
    
    return decorr_time


def adapt_decorrelation_time_streaming(baseline_key: Tuple[int, int, int], 
                                     current_decorr_time: float,
                                     new_decorr_time: float,
                                     tolerance: float = 0.2) -> bool:
    """
    Check if decorrelation time change requires window restart.
    
    In streaming mode, if the optimal decorrelation time changes significantly
    between samples (due to changing UV coordinates), we may need to close
    the current window and start fresh.
    
    Parameters
    ----------
    baseline_key : Tuple[int, int, int]
        Baseline identifier (antenna1, antenna2, scan)
    current_decorr_time : float
        Current window's decorrelation time
    new_decorr_time : float
        New sample's optimal decorrelation time
    tolerance : float, optional
        Relative tolerance for change (default: 0.2 = 20%)
        
    Returns
    -------
    bool
        True if window should be restarted due to significant Δt change
    """
    if current_decorr_time == 0:
        return False
    
    relative_change = abs(new_decorr_time - current_decorr_time) / current_decorr_time
    should_restart = relative_change > tolerance
    
    if should_restart:
        print(f"🔄 Decorr time changed {relative_change:.1%} for {baseline_key} "
              f"({current_decorr_time:.1f}s → {new_decorr_time:.1f}s) - restarting window")
    
    return should_restart


def create_bda_window(start_time: float, decorr_time: float) -> Dict[str, Any]:
    """
    Create empty BDA window dictionary for streaming processing.
    
    Parameters
    ----------
    start_time : float
        Window start time in seconds
    decorr_time : float
        Decorrelation time limit for this window
        
    Returns
    -------
    Dict[str, Any]
        Empty window dictionary ready for sample accumulation
    """
    return {
        'start_time': start_time,
        'end_time': start_time + decorr_time,
        'decorr_time': decorr_time,
        'visibilities': [],
        'weights': [],
        'flags': [],
        'u_coords': [],
        'v_coords': [],
        'w_coords': [],
        'times': [],
        'n_samples': 0
    }


def add_sample_to_window(window, vis, weight, flag, u, v, w, t):
    """Add visibility sample to window (functional - returns modified window)"""
    window['visibilities'].append(vis)
    window['weights'].append(weight) 
    window['flags'].append(flag)
    window['u_coords'].append(u)
    window['v_coords'].append(v)
    window['w_coords'].append(w)
    window['times'].append(t)
    window['n_samples'] += 1
    return window


def should_close_window(window: Dict[str, Any], current_time: float, 
                       new_decorr_time: float, baseline_key: Tuple[int, int, int] = None) -> bool:
    """
    Check if window should be closed based on multiple decorrelation criteria.
    
    Enhanced window closure logic that considers:
    1. Decorrelation time limits (physics-based)
    2. Data sparsity gaps 
    3. Memory management limits
    4. Adaptive decorrelation time changes
    
    Parameters
    ----------
    window : Dict[str, Any]
        Current window state
    current_time : float
        Current sample timestamp
    new_decorr_time : float
        Decorrelation time for current sample
    baseline_key : Tuple[int, int, int], optional
        Baseline identifier for logging
        
    Returns
    -------
    bool
        True if window should be closed and emitted
    """
    if window['n_samples'] == 0:
        return False
    
    # Criterion 1: Maximum decorrelation time exceeded
    time_since_start = current_time - window['start_time']
    if time_since_start >= new_decorr_time:
        return True
    
    # Criterion 2: Large time gap since last sample
    if window['times']:
        time_since_last = current_time - max(window['times'])
        # Use smaller of current and new decorrelation times for gap detection
        gap_threshold = min(new_decorr_time, window.get('decorr_time', new_decorr_time)) * 0.5
        if time_since_last > gap_threshold:
            return True
    
    # Criterion 3: Significant change in optimal decorrelation time
    if 'decorr_time' in window and baseline_key is not None:
        if adapt_decorrelation_time_streaming(baseline_key, window['decorr_time'], 
                                            new_decorr_time, tolerance=0.3):
            return True
    
    # Criterion 4: Window buffer too large
    max_samples_per_window = 1000
    if window['n_samples'] >= max_samples_per_window:
        return True
    
    return False


def complete_bda_window(window, baseline_key):
    """
    Complete BDA processing for a window using weighted averaging.
    
    Processes all samples in the window to produce single averaged result.
    Memory is freed after this function returns.
    """
    if window['n_samples'] == 0:
        return create_empty_bda_result(baseline_key)
    
    try:
        # Convert lists to numpy arrays for vectorized processing
        vis_array = np.array(window['visibilities'])      
        weight_array = np.array(window['weights'])        
        flag_array = np.array(window['flags'])
        
        # Weighted averaging with flag masking
        if vis_array.size > 0 and weight_array.size > 0:
            valid_mask = ~flag_array if flag_array.size > 0 else np.ones_like(vis_array, dtype=bool)
            
            target_shape = vis_array.shape
            n_samples = target_shape[0]
            
            # Reshape weight_array to match vis_array dimensions
            if weight_array.shape != target_shape:
                if len(target_shape) == 3:  # (samples, channels, correlations)
                    n_samples, n_channels, n_corr = target_shape
                    
                    if weight_array.shape == (n_samples, n_channels * n_corr):
                        # Reshape from (samples, channels*corr) to (samples, channels, corr)
                        weight_array = weight_array.reshape(n_samples, n_channels, n_corr)
                        
                    elif weight_array.shape == (n_samples, n_corr):
                        # Broadcast from (samples, corr) to (samples, channels, corr)
                        weight_array = np.broadcast_to(weight_array[:, np.newaxis, :], target_shape)
                        
                    elif weight_array.shape == (n_samples,):
                        # Broadcast from (samples,) to (samples, channels, corr)
                        weight_array = np.broadcast_to(weight_array[:, np.newaxis, np.newaxis], target_shape)
                        
                    else:
                        try:
                            weight_array = np.broadcast_to(weight_array, target_shape)
                        except ValueError:
                            weight_array = np.ones_like(vis_array, dtype=float)
                            
                elif len(target_shape) == 2:
                    n_samples, n_corr = target_shape
                    
                    if weight_array.shape == (n_samples,):
                        weight_array = np.broadcast_to(weight_array[:, np.newaxis], target_shape)
                    else:
                        weight_array = np.broadcast_to(weight_array, target_shape)
            
            # Ensure flag_array also matches
            if flag_array.size > 0 and flag_array.shape != target_shape:
                try:
                    flag_array = np.broadcast_to(flag_array, target_shape)
                except ValueError:
                    # Use no flags if broadcasting fails
                    flag_array = np.zeros_like(vis_array, dtype=bool)
                    valid_mask = np.ones_like(vis_array, dtype=bool)
            
            # Now all arrays should have compatible shapes
            weighted_vis = vis_array * weight_array * valid_mask
            total_weight = np.sum(weight_array * valid_mask, axis=0)
            
            # Avoid division by zero
            vis_averaged = np.divide(
                np.sum(weighted_vis, axis=0),
                total_weight, 
                out=np.zeros_like(total_weight, dtype=complex),
                where=total_weight != 0
            )
            
            weight_total = total_weight
            flag_combined = np.all(flag_array, axis=0) if flag_array.size > 0 else np.zeros_like(vis_averaged, dtype=bool)
            
        else:
            vis_averaged = np.array([])
            weight_total = np.array([])
            flag_combined = np.array([])
            
        # Average coordinates
        u_avg = np.mean(window['u_coords']) if window['u_coords'] else 0.0
        v_avg = np.mean(window['v_coords']) if window['v_coords'] else 0.0
        w_avg = np.mean(window['w_coords']) if window['w_coords'] else 0.0
        time_avg = np.mean(window['times']) if window['times'] else 0.0
        
        # Calculate effective window duration (actual vs planned)
        planned_duration = window.get('decorr_time', window['end_time'] - window['start_time'])
        actual_duration = window['end_time'] - window['start_time']
        
        return {
            'group_key': baseline_key,
            'visibility_averaged': vis_averaged,
            'weight_total': weight_total,
            'flag_combined': flag_combined,
            'time_avg': time_avg,
            'u_avg': u_avg,
            'v_avg': v_avg,
            'w_avg': w_avg,
            'n_input_rows': window['n_samples'],
            'window_duration_s': actual_duration,
            'compression_ratio': float(window['n_samples']),
            'window_id': f"{baseline_key}_{int(window['start_time'])}",
            'decorrelation_time_used': actual_duration,
            'decorrelation_time_planned': planned_duration,
            'window_efficiency': actual_duration / planned_duration if planned_duration > 0 else 1.0
        }
        
    except Exception as e:
        print(f"⚠️ Error completing BDA window: {e}")
        import traceback
        traceback.print_exc()
        return create_empty_bda_result(baseline_key)


def create_empty_bda_result(baseline_key):
    """Create empty BDA result for failed/empty windows"""
    return {
        'group_key': baseline_key,
        'visibility_averaged': np.array([]),
        'weight_total': np.array([]),
        'flag_combined': np.array([]),
        'time_avg': 0.0,
        'u_avg': 0.0,
        'v_avg': 0.0,
        'w_avg': 0.0,
        'n_input_rows': 0,
        'window_duration_s': 0.0,
        'compression_ratio': 1.0
    }


def extract_visibility_array(row):
    """
    Extract and reshape visibility array from Spark row.
    
    EXPECTED ROW FIELDS (must be provided by consumer service):
    - row.visibilities: List of floats in interleaved real/imag format [r1,i1,r2,i2,...]
    - row.n_channels: Integer number of frequency channels
    - row.n_correlations: Integer number of polarization correlations
    
    Parameters
    ----------
    row : pyspark.sql.Row
        Spark row containing visibility data from consumer service
        
    Returns
    -------
    np.ndarray
        Complex visibility array shaped (n_channels, n_correlations)
        Empty array if extraction fails
        
    Notes
    -----
    If consumer service changes field names, update field access here.
    This function centralizes consumer data format dependencies.
    """
    try:
        # Extract interleaved real/imag list (consumer contract)
        vis_list = getattr(row, 'visibilities', None)
        if not vis_list:
            return np.array([], dtype=np.complex64)
        
        # Convert interleaved real/imag back to complex
        vis_flat = np.array(vis_list, dtype=np.float32)
        if len(vis_flat) % 2 != 0:
            vis_flat = vis_flat[:-1]  # Remove odd element for safety
            
        vis_complex = vis_flat[::2] + 1j * vis_flat[1::2]
        
        # Get dimensions from consumer metadata
        n_channels = int(getattr(row, 'n_channels', 1))
        n_correlations = int(getattr(row, 'n_correlations', 1))
        
        # Reshape to expected format
        expected_size = n_channels * n_correlations
        if vis_complex.size == expected_size:
            return vis_complex.reshape(n_channels, n_correlations)
        else:
            # Fallback for size mismatch - log warning in production
            print(f"⚠️ Visibility size mismatch: got {vis_complex.size}, expected {expected_size}")
            return vis_complex.reshape(-1, max(1, n_correlations))
            
    except Exception as e:
        print(f"⚠️ Error extracting visibilities: {e}")
        return np.array([], dtype=np.complex64)


def extract_weight_array(row):
    """
    Extract weight array from Spark row.
    
    EXPECTED ROW FIELDS:
    - row.weight: List of floats representing visibility weights
    
    Parameters
    ----------
    row : pyspark.sql.Row
        Spark row from consumer service
        
    Returns
    -------
    np.ndarray
        Weight array (float32), empty if extraction fails
    """
    try:
        weight_list = getattr(row, 'weight', None)
        if not weight_list:
            return np.array([], dtype=np.float32)
        return np.array(weight_list, dtype=np.float32)
    except Exception as e:
        print(f"⚠️ Error extracting weights: {e}")
        return np.array([], dtype=np.float32)


def extract_flag_array(row):
    """
    Extract flag array from Spark row.
    
    EXPECTED ROW FIELDS:
    - row.flag: List of booleans/integers representing data flags
    
    Parameters
    ----------
    row : pyspark.sql.Row
        Spark row from consumer service
        
    Returns
    -------
    np.ndarray
        Boolean flag array, empty if extraction fails
    """
    try:
        flag_list = getattr(row, 'flag', None)
        if not flag_list:
            return np.array([], dtype=bool)
        
        # Handle both boolean and integer flags
        flag_array = np.array(flag_list)
        return flag_array.astype(bool)
        
    except Exception as e:
        print(f"⚠️ Error extracting flags: {e}")
        return np.array([], dtype=bool)


def process_rows(row_iterator, frecuency_hz, decorr_factor, field_offset_deg):
    """
    Process visibility rows incrementally using decorrelation windows.
    
    CORE CONTRACT: This is the main incremental BDA engine called by bda_integration
    within mapPartitions. Expects rows sorted by time and maintains active windows
    per (antenna1, antenna2, scan). Emits results as windows complete.
    
    Memory-efficient streaming BDA - processes one window at a time.
    RAM usage is constant regardless of baseline size.
    
    MODULE CONTRACT:
    - INPUT: Iterator[Row] from Spark partition (pre-sorted by time)
    - OUTPUT: Generator[Dict] of completed BDA windows
    - CALLED BY: bda_integration.incremental_bda_partition()
    - CONVERTED BY: convert_bda_result_to_spark_tuple()
    
    Parameters
    ----------
    row_iterator : Iterator
        Iterator over Spark Row objects containing visibility data (time-sorted)
    bda_config : dict
        BDA configuration including decorrelation time settings
        
    Yields
    ------
    dict
        Completed BDA window results with keys: 'group_key', 'visibility_averaged',
        'weight_total', 'flag_combined', 'time_avg', 'u_avg', 'v_avg', 'w_avg',
        'n_input_rows', 'window_duration_s', 'compression_ratio'
    """
    active_windows = {}  # baseline_key -> window
    
    for row in row_iterator:
        # Extract baseline info
        baseline_key = row.baseline_key
        current_time = float(row.time)
        
        # Extract UV coordinates
        u, v, w = float(row.u), float(row.v), float(row.w)
        u_lambda, v_lambda, w_lambda = u / frecuency_hz, v / frecuency_hz, w / frecuency_hz

        
        
        # Calculate physics-based decorrelation time using corrected equations
        decorr_time = calculate_decorrelation_time()
        
        # Check if existing window should be closed
        if baseline_key in active_windows:
            window = active_windows[baseline_key]
            if should_close_window(window, current_time, decorr_time, baseline_key):
                # Complete and yield window
                result = complete_bda_window(window, baseline_key)
                yield result

                # Remove completed window
                del active_windows[baseline_key]
        
        # Create new window if needed
        if baseline_key not in active_windows:
            active_windows[baseline_key] = create_bda_window(current_time, decorr_time)
        
        # Extract arrays from row
        vis_array = extract_visibility_array(row)
        weight_array = extract_weight_array(row)
        flag_array = extract_flag_array(row)
        
        # Add sample to active window
        window = active_windows[baseline_key]
        add_sample_to_window(window, vis_array, weight_array, flag_array, u, v, w, current_time)
    
    # Flush remaining windows at end of partition
    for baseline_key, window in active_windows.items():
        if window['n_samples'] > 0:
            result = complete_bda_window(window, baseline_key)
            yield result
