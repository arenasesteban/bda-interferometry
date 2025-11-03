import numpy as np
import traceback

from .bda_core import calculate_decorrelation_time, calculate_uv_rate, average_visibilities, average_uv


def create_window(row, start_time, decorr_time):
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
            'interval': [],
            'exposure': [],
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


def add_window(window, interval, exposure, time, u, v, visibilities, weight, flag):
    try:
        window['nrows'] += 1
        window['interval'].append(interval)
        window['exposure'].append(exposure)
        window['time'].append(time)
        window['u'].append(u)
        window['v'].append(v)

        vs = np.array(visibilities, dtype=np.float64)
        ws = np.array(weight, dtype=np.float64)
        fs = np.array(flag, dtype=np.bool_)

        window['visibilities'].append(vs)
        window['weight'].append(ws)
        window['flag'].append(fs)

        return window

    except Exception as e:
        print(f"Error adding sample to window: {e}")
        traceback.print_exc()
        raise


def calculate_window_duration(times, intervals):
    try:
        if len(times) == 0:
            return 0.0
        
        if len(times) == 1:
            return intervals[0]

        MJD_TO_SECONDS = 86400.0

        times_sec = np.array(times) * MJD_TO_SECONDS
        intervals_sec = np.array(intervals)

        t_start = times_sec[0] - (intervals_sec[0] / 2.0)
        t_end = times_sec[-1] + (intervals_sec[-1] / 2.0)

        return t_end - t_start

    except Exception as e:
        print(f"Error calculating window duration: {e}")
        traceback.print_exc()
        raise


def should_close_window(window, current_time, current_interval):
    try:
        if window['nrows'] == 0:
            return False

        times = window['time'] + [current_time]
        intervals = window['interval'] + [current_interval]

        # Criterion: Maximum decorrelation time exceeded
        return calculate_window_duration(times, intervals) > window['decorr_time']
    
    except Exception as e:
        print(f"Error checking if window should close: {e}")
        traceback.print_exc()
        raise


def complete_window(window, baseline_key):
    try:
        if window['nrows'] == 0:
            return None

        visibilities, weights, flags = window['visibilities'], window['weight'], window['flag']
        avg_vis, avg_weight, flag_avg = average_visibilities(visibilities, weights, flags)
        u_avg, v_avg = average_uv(window['u'], window['v'])

        interval_avg = float(calculate_window_duration(window['time'], window['interval']))
        exposures = np.array(window['exposure'])
        exposure_avg = float(exposures.sum())

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
            'interval': interval_avg,
            'exposure': exposure_avg,
            'time': window['start_time'],
            'u': u_avg,
            'v': v_avg,
            'visibilities': avg_vis,
            'weight': avg_weight,
            'flag': flag_avg
        }
    
    except Exception as e:
        print(f"Error completing window: {e}")
        traceback.print_exc()
        raise


def process_rows(iterator, bda_config):
    active_windows = {}  # baseline_key -> window
    baseline_stats = {}  # baseline_key -> stats dict
    
    try:
        decorr_factor = bda_config.get('decorr_factor', 0.95)
        field_offset_deg = bda_config.get('field_offset_deg', 2.0)
        
        for row in iterator:
            baseline_key = row.baseline_key

            time = row.time
            interval = row.interval
            exposure = row.exposure

            u, v = row.u, row.v
            Lx, Ly = row.Lx, row.Ly
            
            lambda_ = row.lambda_
            
            # Calculates uv rates
            u_dot, v_dot = calculate_uv_rate(time, Lx, Ly, lambda_, row.dec, row.ra, row.longitude, row.latitude)

            if baseline_key not in baseline_stats:
                baseline_stats[baseline_key] = {'rows_in': 0, 'windows_out': 0, 'decorr_times': []}
            baseline_stats[baseline_key]['rows_in'] += 1
            
            # Calculate decorrelation time
            decorr_time = calculate_decorrelation_time(u_dot, v_dot, decorr_factor, field_offset_deg)
            
            baseline_stats[baseline_key]['decorr_times'].append(decorr_time)
            
            # Create new window if needed
            if baseline_key not in active_windows:
                active_windows[baseline_key] = create_window(row, time, decorr_time)
            else:
                window = active_windows[baseline_key]

                # Check if existing window should be closed
                if should_close_window(window, time, interval):
                    result = complete_window(window, baseline_key)
                    baseline_stats[baseline_key]['windows_out'] += 1
                    yield result

                    #Remove completed window
                    del active_windows[baseline_key]

                    # Create new window for current row
                    active_windows[baseline_key] = create_window(row, time, decorr_time)

            window = active_windows[baseline_key]
            visibilities, weight, flag = row.visibilities, row.weight, row.flag

            # Add sample to active window
            add_window(window, interval, exposure, time, u, v, visibilities, weight, flag)

        # Flush remaining windows at end of partition
        for baseline_key, window in active_windows.items():
            if window['nrows'] > 0:
                result = complete_window(window, baseline_key)
                baseline_stats[baseline_key]['windows_out'] += 1
                yield result
        
        # Print baseline statistics
        print("=== BDA Baseline Statistics ===")
        for bl_key, stats in sorted(baseline_stats.items()):
            avg_decorr = np.mean(stats['decorr_times'])
            print(f"{bl_key}: {stats['rows_in']} rows → {stats['windows_out']} windows "
                  f"(decorr_avg={avg_decorr:.1f}s, compression={100*(1-stats['windows_out']/stats['rows_in']):.1f}%)")
        print("===========================\n")

    except Exception as e:
        print(f"Error processing rows for bda: {e}")
        traceback.print_exc()
        raise
