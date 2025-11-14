import numpy as np
import traceback

from .bda_core import calculate_phase_rate, calculate_uv_rate, average_visibilities, average_uv


def create_window(row, start_time):
    try:
        return {
            'start_time': start_time,
            'nrows': 0,
            'x_acc': 0.0,
            'phi_dot': [],
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


def add_window(window, phi_dot, interval, exposure, time, u, v, visibilities, weight, flag):
    try:
        window['nrows'] += 1
        window['x_acc'] += 0.5 * abs(phi_dot) * exposure
        window['phi_dot'].append(phi_dot)
        
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


def should_close_window(window, phi_dot, x, exposure, baseline_key):
    try:
        if window['nrows'] == 0:
            return False

        x_acc = window['x_acc']
        x_inc = 0.5 * abs(phi_dot) * exposure

        if baseline_key == "0-1":
            print(f"[BDA] x_acc: {x_acc}, x_inc: {x_inc}, x: {x}")
            print(f"[BDA] Should close window? {(x_acc + x_inc) > x}\n")

        return (x_acc + x_inc) > x

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
        u_avg, v_avg = average_uv(window['u'], window['v'], window['exposure'])

        intervals, exposures = np.array(window['interval']), np.array(window['exposure'])
        interval_avg, exposure_avg = float(np.sum(intervals)) , float(np.sum(exposures))

        if baseline_key == "0-1":
            print("[BDA] Baseline 0-1")
            print(f"nrows: {window['nrows']}")
            print(f"x_acc: {window['x_acc']}")
            print(f"time: {window['start_time']}")
            print(f"interval: {interval_avg}")
            print(f"exposure: {exposure_avg}")
            print(f"coords: ({u_avg}, {v_avg})")
            print(f"weight: {avg_weight}")
            print(f"flags: {flag_avg}")
            print(f"visibilities: {avg_vis}\n")
            print("-" * 60 + "\n")

        # Return completed window with averaged values
        return {
            'nrows': window['nrows'],            
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
        x = bda_config.get('x', 0.0)
        min_diameter = bda_config.get('min_diameter', 0.0)

        for row in iterator:
            baseline_key = row.baseline_key
            scan_number = row.scan_number

            Lx, Ly = row.Lx, row.Ly
            lambda_ = row.lambda_
            time = row.time
            interval = row.interval
            exposure = row.exposure

            u, v = row.u, row.v
            visibilities, weight, flag = row.visibilities, row.weight, row.flag

            if baseline_key == "0-1":
                print("[-] Baseline 0-1")
                print(f"time: {time}")
                print(f"interval: {interval}")
                print(f"exposure: {exposure}")
                print(f"coords: ({u}, {v})")
                print(f"weight: {weight}")
                print(f"flags: {flag}")
                print(f"visibilities: {visibilities}\n")
                print("-" * 60 + "\n")

            if baseline_key not in baseline_stats:
                baseline_stats[baseline_key] = {'rows_in': 0, 'windows_out': 0, 'decorr_times': []}
            baseline_stats[baseline_key]['rows_in'] += 1

            # Calculates uv rates
            u_dot, v_dot = calculate_uv_rate(time, Lx, Ly, lambda_, row.dec, row.ra, row.longitude, row.latitude)

            # Calculate phase rate
            phi_dot = calculate_phase_rate(u_dot, v_dot, lambda_, min_diameter)

            if baseline_key == "0-1":
                print(f"[BDA] phi_dot: {phi_dot}")

            # Create new window if needed
            if baseline_key not in active_windows:
                active_windows[baseline_key] = {}
            
            if scan_number not in active_windows[baseline_key]:
                active_windows[baseline_key][scan_number] = create_window(row, time)
            else:
                window = active_windows[baseline_key][scan_number]

                # Check if existing window should be closed
                if should_close_window(window, phi_dot, x, exposure, baseline_key):
                    result = complete_window(window, baseline_key)
                    baseline_stats[baseline_key]['windows_out'] += 1
                    yield result

                    #Remove completed window
                    del active_windows[baseline_key][scan_number]

                    # Create new window for current row
                    active_windows[baseline_key][scan_number] = create_window(row, time)

            window = active_windows[baseline_key][scan_number]  

            # Add sample to active window
            add_window(window, phi_dot, interval, exposure, time, u, v, visibilities, weight, flag)

        # Flush remaining windows at end of partition
        for baseline_key, windows in active_windows.items():
            for scan_number, window in windows.items():
                if window['nrows'] > 0:
                    result = complete_window(window, baseline_key)
                    baseline_stats[baseline_key]['windows_out'] += 1
                    yield result
        
        """ # Print baseline statistics
        print("=== BDA Baseline Statistics ===")
        for bl_key, stats in sorted(baseline_stats.items()):
            avg_decorr = np.mean(stats['decorr_times'])
            print(f"{bl_key}: {stats['rows_in']} rows → {stats['windows_out']} windows "
                  f"(decorr_avg={avg_decorr:.1f}s, compression={100*(1-stats['windows_out']/stats['rows_in']):.1f}%)")
        print("===========================\n") """

    except Exception as e:
        print(f"Error processing rows for bda: {e}")
        traceback.print_exc()
        raise
