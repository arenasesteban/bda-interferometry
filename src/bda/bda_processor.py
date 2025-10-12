"""
BDA Processor - Incremental Streaming Processing with Decorrelation Windows

This module implements memory-efficient BDA processing using incremental windows
based on physical decorrelation time. Instead of accumulating full baselines,
it processes data online as it arrives, maintaining constant RAM usage.

The approach uses Spark's mapPartitions for distributed processing with window
closure based on baseline-dependent decorrelation criteria.

Key Functions
-------------
process_rows_incrementally : Main streaming BDA processor with online windows
complete_bda_window : Weighted averaging and window completion
calculate_decorrelation_time : Physics-based window sizing
"""

import numpy as np


def calculate_decorrelation_time(baseline_length_m, base_time_s=10.0):
    """
    Calculate decorrelation time based on baseline length.
    
    Shorter baselines decorrelate slower (longer windows),
    longer baselines decorrelate faster (shorter windows).
    
    Parameters
    ----------
    baseline_length_m : float
        Baseline length in meters
    base_time_s : float
        Base decorrelation time in seconds
        
    Returns
    -------
    float
        Appropriate decorrelation time for this baseline
    """
    if baseline_length_m < 100:
        return base_time_s * 2.0    # Short baselines: longer windows
    elif baseline_length_m < 1000:
        return base_time_s * 1.0    # Medium baselines: standard windows  
    else:
        return base_time_s * 0.5    # Long baselines: shorter windows


def create_bda_window(start_time, decorr_time):
    """Create empty BDA window dictionary"""
    return {
        'start_time': start_time,
        'end_time': start_time + decorr_time,
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


def should_close_window(window, current_time, decorr_time):
    """
    Check if window should be closed based on multiple decorrelation criteria.
    
    Online window closure logic - emit results as soon as criteria are met.
    """
    if window['n_samples'] == 0:
        return False
    
    # Criterion 1: Maximum decorrelation time exceeded
    time_since_start = current_time - window['start_time']
    if time_since_start >= decorr_time:
        return True
    
    # Criterion 2: Large time gap since last sample (data sparsity)
    if window['times']:
        time_since_last = current_time - max(window['times'])
        if time_since_last > decorr_time * 0.5:
            return True
    
    # Criterion 3: Window buffer too large (memory management)
    max_samples_per_window = 1000  # Configurable limit
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
        
        # Debug: Log array shapes for troubleshooting
        print(f"🔍 Debug shapes - vis: {vis_array.shape}, weight: {weight_array.shape}, flag: {flag_array.shape}")
        
        # Weighted averaging with flag masking
        if vis_array.size > 0 and weight_array.size > 0:
            valid_mask = ~flag_array if flag_array.size > 0 else np.ones_like(vis_array, dtype=bool)
            
            # ROBUST SHAPE HANDLING
            target_shape = vis_array.shape
            n_samples = target_shape[0]
            
            # Reshape weight_array to match vis_array dimensions
            if weight_array.shape != target_shape:
                if len(target_shape) == 3:  # (samples, channels, correlations)
                    n_samples, n_channels, n_corr = target_shape
                    
                    if weight_array.shape == (n_samples, n_channels * n_corr):
                        # Reshape from (samples, channels*corr) to (samples, channels, corr)
                        weight_array = weight_array.reshape(n_samples, n_channels, n_corr)
                        print(f"✅ Reshaped weights from (samples, flat) to (samples, channels, corr)")
                        
                    elif weight_array.shape == (n_samples, n_corr):
                        # Broadcast from (samples, corr) to (samples, channels, corr)
                        weight_array = np.broadcast_to(weight_array[:, np.newaxis, :], target_shape)
                        print(f"✅ Broadcasted weights from (samples, corr) to (samples, channels, corr)")
                        
                    elif weight_array.shape == (n_samples,):
                        # Broadcast from (samples,) to (samples, channels, corr)
                        weight_array = np.broadcast_to(weight_array[:, np.newaxis, np.newaxis], target_shape)
                        print(f"✅ Broadcasted weights from (samples,) to (samples, channels, corr)")
                        
                    else:
                        # Fallback: try direct broadcasting
                        try:
                            weight_array = np.broadcast_to(weight_array, target_shape)
                            print(f"✅ Direct broadcast successful")
                        except ValueError:
                            # Last resort: use ones with same shape
                            print(f"⚠️ Broadcasting failed, using uniform weights")
                            weight_array = np.ones_like(vis_array, dtype=float)
                            
                elif len(target_shape) == 2:  # (samples, correlations)
                    n_samples, n_corr = target_shape
                    
                    if weight_array.shape == (n_samples,):
                        weight_array = np.broadcast_to(weight_array[:, np.newaxis], target_shape)
                        print(f"✅ Broadcasted weights 1D to 2D")
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
            
            print(f"✅ BDA window completed: {n_samples} samples → 1 averaged result")
            
        else:
            vis_averaged = np.array([])
            weight_total = np.array([])
            flag_combined = np.array([])
            
        # Average coordinates
        u_avg = np.mean(window['u_coords']) if window['u_coords'] else 0.0
        v_avg = np.mean(window['v_coords']) if window['v_coords'] else 0.0
        w_avg = np.mean(window['w_coords']) if window['w_coords'] else 0.0
        time_avg = np.mean(window['times']) if window['times'] else 0.0
        
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
            'window_duration_s': window['end_time'] - window['start_time'],
            'compression_ratio': float(window['n_samples']),  # n_samples -> 1 result (per window)
            'window_id': f"{baseline_key}_{int(window['start_time'])}",
            'decorrelation_time_used': window['end_time'] - window['start_time']
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
    """Extract and reshape visibility array from Spark row"""
    try:
        vis_list = row.visibilities
        if not vis_list:
            return np.array([])
        
        # Convert interleaved real/imag back to complex
        vis_flat = np.array(vis_list, dtype=np.float32)
        if len(vis_flat) % 2 != 0:
            vis_flat = vis_flat[:-1]  # Remove odd element
            
        vis_complex = vis_flat[::2] + 1j * vis_flat[1::2]
        
        # Reshape to (n_channels, n_correlations)
        n_channels = int(row.n_channels)
        n_correlations = int(row.n_correlations)
        
        if vis_complex.size == n_channels * n_correlations:
            return vis_complex.reshape(n_channels, n_correlations)
        else:
            # Fallback for size mismatch
            return vis_complex.reshape(-1, 1)
            
    except Exception:
        return np.array([])


def extract_weight_array(row):
    """Extract weight array from Spark row"""
    try:
        weight_list = row.weight
        if not weight_list:
            return np.array([])
        return np.array(weight_list, dtype=np.float32)
    except Exception:
        return np.array([])


def extract_flag_array(row):
    """Extract flag array from Spark row"""
    try:
        flag_list = row.flag
        if not flag_list:
            return np.array([])
        return np.array(flag_list, dtype=np.bool_)
    except Exception:
        return np.array([])


def process_rows_incrementally(row_iterator, bda_config):
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
    base_decorr_time = bda_config.get('decorrelation_time_s', 10.0)
    
    for row in row_iterator:
        # Extract baseline info
        baseline_key = (int(row.antenna1), int(row.antenna2), int(row.scan_number))
        current_time = float(row.time)
        
        # Calculate baseline length for decorrelation time
        u, v, w = float(row.u), float(row.v), float(row.w)
        baseline_length = np.sqrt(u*u + v*v + w*w)
        decorr_time = calculate_decorrelation_time(baseline_length, base_decorr_time)
        
        # Check if existing window should be closed
        if baseline_key in active_windows:
            window = active_windows[baseline_key]
            if should_close_window(window, current_time, decorr_time):
                # Complete and yield window
                result = complete_bda_window(window, baseline_key)
                
                # Log window completion for monitoring
                if window['n_samples'] > 0:
                    compression = window['n_samples'] / 1.0  # n samples -> 1 BDA result
                    print(f"🔄 Window closed: {baseline_key} | {window['n_samples']} samples → 1 result (compression: {compression:.1f}:1)")
                
                yield result
                # Remove completed window (frees memory)
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

