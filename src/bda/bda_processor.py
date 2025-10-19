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
complete_window : Weighted averaging and window completion
get_decorrelation_time : Physics-based window sizing per visibility sample
"""

import numpy as np
from typing import Dict, Any
import traceback

from .bda_core import calculate_decorrelation_time, calculate_uv_rate, average_visibilities, average_fields


def create_window(row, start_time, decorr_time) -> Dict[str, Any]:
    try:
        return {
            'start_time': start_time,
            'end_time': start_time + decorr_time,
            'decorr_time': decorr_time,
            'nrows': 0,
            'subms_id': row.subms_id,
            'field_id': row.field_id,
            'spw_id': row.spw_id,
            'polarization_id': row.polarization_id,
            'n_channels': row.n_channels,
            'n_correlations': row.n_correlations,
            'antenna1': row.antenna1,
            'antenna2': row.antenna2,
            'scan_number': row.scan_number,
            'longitude': row.longitude,
            'time': [],
            'u': [],
            'v': [],
            'w': [],
            'visibilities': [],
            'weight': [],
            'flag': []
        }
    
    except Exception as e:
        print(f"Error creating window: {e}")
        traceback.print_exc()
        raise


def add_window(window, time, u, v, w, visibilities, weight, flag):
    try:
        window["nrows"] += 1
        window["time"].append(time)
        window["u"].append(u)
        window["v"].append(v)
        window["w"].append(w)

        vs = np.array(visibilities, dtype=np.float64)
        ws = np.array(weight, dtype=np.float64)
        fs = np.array(flag, dtype=np.bool_)

        window["visibilities"].append(vs)
        window["weight"].append(ws)
        window['flag'].append(fs)

        return window

    except Exception as e:
        print(f"Error adding sample to window: {e}")
        traceback.print_exc()
        raise


def should_close_window(window, time) -> bool:
    try:
        # Criterion: Maximum decorrelation time exceeded
        time_since_start = time - window['start_time']
        if time_since_start >= window['decorr_time']:
            return True

        return False
    
    except Exception as e:
        print(f"Error checking if window should close: {e}")
        traceback.print_exc()
        raise


def complete_window(window, baseline_key):
    try:
        if window["nrows"] == 0:
            return None

        visibilities, weights, flags = window["visibilities"], window["weight"], window["flag"]
        avg_vis, avg_weight, flag_avg = average_visibilities(visibilities, weights, flags)
        u_avg, v_avg, w_avg, time_avg = average_fields(window["u"], window["v"], window["w"], window["time"], flags)

        # Return completed window with averaged values
        return {
            'nrows': window['nrows'],
            'time_start': window['start_time'],
            'time_end': window['end_time'],
            'decorr_time': window['decorr_time'],

            'subms_id': window['subms_id'],
            'field_id': window['field_id'],
            'spw_id': window['spw_id'],
            'polarization_id': window['polarization_id'],
            'n_channels': window['n_channels'],
            'n_correlations': window['n_correlations'],
            'antenna1': window['antenna1'],
            'antenna2': window['antenna2'],
            'scan_number': window['scan_number'],
            'baseline_key': baseline_key,
            'time': time_avg,
            'u': u_avg,
            'v': v_avg,
            'w': w_avg,
            'visibilities': avg_vis,
            'weight': avg_weight,
            'flag': flag_avg
        }
    
    except Exception as e:
        print(f"Error completing window: {e}")
        traceback.print_exc()
        raise

def process_rows(row_iterator, bda_config):
    active_windows = {}  # baseline_key -> window
    
    try:

        decorr_factor = bda_config.get('decorr_factor', 0.95)
        field_offset_deg = bda_config.get('field_offset_deg', 2.0)
        
        for row in row_iterator:
            baseline_key = row.baseline_key
            time = row.time

            u, v, w = row.u, row.v, row.w
            lambda_ref = row.lambda_ref 
            longitude = row.longitude
            dec, ra = row.dec, row.ra

            # Calculates uv rates
            u_dot, v_dot = calculate_uv_rate(time, u, v, lambda_ref, dec, longitude, ra)

            # Calculate decorrelation time
            decorr_time = calculate_decorrelation_time(u_dot, v_dot, decorr_factor, field_offset_deg)

            # Create new window if needed
            if baseline_key not in active_windows:
                active_windows[baseline_key] = create_window(row, time, decorr_time)

            else:
                window = active_windows[baseline_key]
                window["decorr_time"] = min(window["decorr_time"], decorr_time)

            # Check if existing window should be closed
            if should_close_window(window, time):
                # Complete and yield window
                result = complete_window(window, baseline_key)
                yield result

                # Remove completed window
                del active_windows[baseline_key]
                    
            visibilities, weight, flag = row.visibilities, row.weight, row.flag
            
            # Add sample to active window
            window = active_windows[baseline_key]
            add_window(window, time, u, v, w, visibilities, weight, flag)

        # Flush remaining windows at end of partition
        for baseline_key, window in active_windows.items():
            if window["nrows"] > 0:
                result = complete_window(window, baseline_key)
                yield result
    
    except Exception as e:
        print(f"Error processing rows: {e}")
        traceback.print_exc()
        raise
