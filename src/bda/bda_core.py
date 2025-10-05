"""
BDA Core - Core Scientific Algorithms

Implementation of fundamental scientific algorithms for Baseline-Dependent Averaging
based on Wijnholds et al. 2018. Includes decorrelation time calculations, fringe rates
and BDA parameters optimized for radio interferometry.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from astropy import constants as const


def create_bda_config(decorr_factor: float = 0.95,
                     frequency_hz: float = 42.5e9,
                     declination_deg: float = -45.0,
                     safety_factor: float = 0.8) -> Dict[str, float]:
    """
    Create BDA configuration using simple dictionary.
    
    Parameters
    ----------
    decorr_factor : float
        Decorrelation factor R (default: 0.95)
    frequency_hz : float
        Observation frequency in Hz (default: 42.5e9)
    declination_deg : float
        Source declination in degrees (default: -45.0)
    safety_factor : float
        Conservative safety factor (default: 0.8)
        
    Returns
    -------
    Dict[str, float]
        BDA configuration as dictionary
    """
    return {
        'decorr_factor': decorr_factor,
        'frequency_hz': frequency_hz,
        'declination_deg': declination_deg,
        'safety_factor': safety_factor
    }


def calculate_baseline_length(u: float, v: float, w: float = 0.0, in_wavelengths: bool = True) -> float:
    """
    Calculate baseline length from UVW coordinates using euclidean distance in UV plane.
    
    Parameters
    ----------
    u : float
        U coordinate in wavelengths or meters
    v : float
        V coordinate in wavelengths or meters  
    w : float, optional
        W coordinate (not used in calculation, default: 0.0)
    in_wavelengths : bool, optional
        If True, coordinates are in wavelengths, if False in meters (default: True)
        
    Returns
    -------
    float
        Baseline length in same units as input coordinates
    """
    baseline_length = np.sqrt(u**2 + v**2)
    return baseline_length


def calculate_fringe_rate(baseline_length_lambda: float, 
                         declination_deg: float = -45.0,
                         hour_angle_deg: float = 0.0,
                         baseline_east_fraction: float = 1.0,
                         baseline_north_fraction: float = 0.0) -> float:
    """
    Calculates total fringe rate based on UV coordinate derivatives.
    
    Implements equations 42-44 from Wijnholds et al. 2018:
    - ∂u/∂t = (1/λ) * (Lx*cos(H) - Ly*sin(H)) * ωE * cos(δ)
    - ∂v/∂t = (1/λ) * (Lx*sin(H)*sin(δ) + Ly*cos(H)*sin(δ)) * ωE  
    - ∂w/∂t = (1/λ) * (Lx*sin(H)*cos(δ) + Ly*cos(H)*cos(δ)) * ωE
    
    Parameters
    ----------
    baseline_length_lambda : float
        Baseline length in wavelengths
    declination_deg : float, optional
        Source declination in degrees (default: -45.0)
    hour_angle_deg : float, optional
        Hour angle in degrees (default: 0.0 for zenith)
    baseline_east_fraction : float, optional
        Fraction of baseline in East direction (default: 1.0)
    baseline_north_fraction : float, optional
        Fraction of baseline in North direction (default: 0.0)
        
    Returns
    -------
    float
        Total fringe rate in rad/s
    """
    # Physical constants
    omega_earth = 7.2925e-5  # Earth angular velocity in rad/s
    
    # Convert angles to radians
    dec_rad = np.radians(declination_deg)
    hour_angle_rad = np.radians(hour_angle_deg)
    
    # Baseline components in wavelengths
    Lx_lambda = baseline_length_lambda * baseline_east_fraction
    Ly_lambda = baseline_length_lambda * baseline_north_fraction
    
    # Calculate UV derivatives according to equations 42-43
    du_dt = (Lx_lambda * np.cos(hour_angle_rad) - 
             Ly_lambda * np.sin(hour_angle_rad)) * omega_earth * np.cos(dec_rad)
    
    dv_dt = (Lx_lambda * np.sin(hour_angle_rad) * np.sin(dec_rad) + 
             Ly_lambda * np.cos(hour_angle_rad) * np.sin(dec_rad)) * omega_earth
    
    # W derivative (equation 44)
    dw_dt = (Lx_lambda * np.sin(hour_angle_rad) * np.cos(dec_rad) + 
             Ly_lambda * np.cos(hour_angle_rad) * np.cos(dec_rad)) * omega_earth
    
    # Total fringe rate with conservative weight for w-term
    w_factor = 0.1  # Conservative value
    
    total_fringe_rate = np.sqrt(du_dt**2 + dv_dt**2 + (w_factor * dw_dt)**2)
    
    return total_fringe_rate


def calculate_fringe_rate_conservative(baseline_length_lambda: float,
                                     declination_deg: float = -45.0,
                                     safety_margin: float = 2.0) -> float:
    """
    Calculates fringe rate using conservative orientation-independent approximation.
    
    Parameters
    ----------
    baseline_length_lambda : float
        Baseline length in wavelengths
    declination_deg : float, optional
        Source declination in degrees (default: -45.0)
    safety_margin : float, optional
        Conservative safety factor (default: 2.0 for worst case)
        
    Returns
    -------
    float
        Conservative fringe rate in rad/s
    """
    # Physical constants
    omega_earth = 7.2925e-5  # Earth angular velocity in rad/s
    
    # Convert declination to radians
    dec_rad = np.radians(declination_deg)
    
    # Conservative approximation: worst case for any orientation
    max_du_dt = baseline_length_lambda * omega_earth * np.cos(dec_rad)
    max_dv_dt = baseline_length_lambda * omega_earth * abs(np.sin(dec_rad))
    
    # Conservative fringe rate (worst case scenario)
    conservative_fringe_rate = safety_margin * np.sqrt(max_du_dt**2 + max_dv_dt**2)
    
    return conservative_fringe_rate


def calculate_decorrelation_time(baseline_length_meters: float,
                               frequency_hz: float,
                               config: Dict[str, float] = None) -> float:
    """
    Calculates decorrelation time for a specific baseline.
    
    Implements equation 41 from Wijnholds et al. 2018:
    T_decorr = sqrt(1 - R²) / |fringe_rate_total|
    
    Parameters
    ----------
    baseline_length_meters : float
        Baseline length in meters
    frequency_hz : float
        Observation frequency in Hz
    config : Dict[str, float], optional
        BDA configuration. If None, uses default values
        
    Returns
    -------
    float
        Decorrelation time in seconds, limited by min/max_averaging_time
    """
    if config is None:
        config = create_bda_config()
    
    # Convert baseline to wavelengths
    wavelength_m = const.c.value / frequency_hz
    baseline_length_lambda = baseline_length_meters / wavelength_m
    
    # Calculate total fringe rate using conservative version
    fringe_rate = calculate_fringe_rate_conservative(
        baseline_length_lambda=baseline_length_lambda,
        declination_deg=config['declination_deg'],
        safety_margin=2.0  # Conservative factor
    )
    
    # Calculate decorrelation time (equation 41)
    if fringe_rate > 0:
        # sqrt(1 - R²) / |fringe_rate| 
        numerator = np.sqrt(1 - config['decorr_factor']**2)
        decorr_time = numerator / fringe_rate
    else:
        # Fallback for very short baselines
        decorr_time = 180.0  # Default max averaging time
    
    # Apply safety factor
    decorr_time *= config['safety_factor']
    
    # Apply limits (1s min, 180s max)
    decorr_time = np.clip(decorr_time, 1.0, 180.0)
    
    return decorr_time


def calculate_optimal_averaging_time(u: float, v: float, 
                                   frequency_hz: float,
                                   config: Dict[str, float] = None,
                                   input_units: str = 'auto') -> float:
    """
    Calculates optimal averaging time for a baseline given by UV coordinates.
    
    Parameters
    ----------
    u : float
        U coordinate (units according to input_units)
    v : float  
        V coordinate (units according to input_units)
    frequency_hz : float
        Observation frequency in Hz
    config : Dict[str, float], optional
        BDA configuration. If None, uses default values
    input_units : str, optional
        Input units: 'meters', 'wavelengths', 'auto' (default: 'auto')
        
    Returns
    -------
    float
        Optimal averaging time in seconds
    """
    if config is None:
        config = create_bda_config()
    
    # Ensure units are in wavelengths
    u_lambda, v_lambda = ensure_baseline_units_wavelengths(u, v, frequency_hz, input_units)
    
    # Calculate baseline length in wavelengths  
    baseline_length_lambda = calculate_baseline_length(u_lambda, v_lambda)
    
    # Convert to meters for decorrelation time
    wavelength_m = const.c.value / frequency_hz
    baseline_length_meters = baseline_length_lambda * wavelength_m
    
    # Calculate decorrelation time
    averaging_time = calculate_decorrelation_time(
        baseline_length_meters=baseline_length_meters,
        frequency_hz=frequency_hz,
        config=config
    )
    
    return averaging_time


def validate_bda_config(config: Dict[str, float]) -> Tuple[bool, str]:
    """
    Validates BDA configuration to ensure physically reasonable values.
    
    Parameters
    ----------
    config : Dict[str, float]
        BDA configuration to validate
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if not (0.0 < config.get('decorr_factor', 0.95) < 1.0):
        return False, f"decorr_factor debe estar entre 0 y 1, got {config.get('decorr_factor')}"
    
    if config.get('frequency_hz', 42.5e9) <= 0:
        return False, f"frequency_hz debe ser positiva, got {config.get('frequency_hz')}"
        
    declination = config.get('declination_deg', -45.0)
    if not (-90.0 <= declination <= 90.0):
        return False, f"declination_deg debe estar entre -90 y 90, got {declination}"
    
    safety_factor = config.get('safety_factor', 0.8)
    if not (0.0 < safety_factor <= 1.0):
        return False, f"safety_factor debe estar entre 0 y 1, got {safety_factor}"
    
    return True, "Valid configuration"


def get_baseline_classification(baseline_length_lambda: float) -> str:
    """
    Classifies a baseline according to its length for analysis.
    
    Parameters
    ----------
    baseline_length_lambda : float
        Baseline length in wavelengths
        
    Returns
    -------
    str
        Baseline classification
    """
    if baseline_length_lambda < 100:
        return "short"
    elif baseline_length_lambda < 1000:
        return "medium"  
    elif baseline_length_lambda < 10000:
        return "long"
    else:
        return "very_long"


def estimate_compression_ratio(baseline_length_lambda: float, 
                             integration_time_sec: float,
                             averaging_time_sec: float) -> float:
    """
    Estimates expected compression ratio for a baseline.
    
    Parameters
    ----------
    baseline_length_lambda : float
        Baseline length in wavelengths
    integration_time_sec : float
        Original integration time in seconds
    averaging_time_sec : float
        BDA averaging time in seconds
        
    Returns
    -------
    float
        Estimated compression ratio (1 - output_size/input_size)
    """
    if averaging_time_sec <= integration_time_sec:
        return 0.0  # No compression
    
    compression_factor = averaging_time_sec / integration_time_sec
    compression_ratio = 1.0 - (1.0 / compression_factor)
    
    return compression_ratio


def create_bda_windows(times: np.ndarray, 
                      delta_t_max: float,
                      baseline_length: float = None,
                      frequency_hz: float = None) -> List[Tuple[int, int]]:
    """
    Divides observations into temporal windows according to smearing limits.
    
    Parameters
    ----------
    times : np.ndarray
        Array of timestamps ordered by time
    delta_t_max : float
        Maximum decorrelation time for this baseline
    baseline_length : float, optional
        Baseline length (for logging/debug)
    frequency_hz : float, optional
        Observation frequency (for logging/debug)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) for each window
    """
    if len(times) == 0:
        return []
    
    windows = []
    window_start = 0
    
    for i in range(1, len(times)):
        # Calculate cumulative window width if we include this point
        window_span = times[i] - times[window_start]
        
        # If total width would exceed Δt_max, close current window
        if window_span > delta_t_max:
            windows.append((window_start, i))
            window_start = i
    
    # Add last window
    windows.append((window_start, len(times)))
    
    return windows


def average_visibility_window(visibilities: np.ndarray,
                            weights: np.ndarray,
                            u_coords: np.ndarray,
                            v_coords: np.ndarray,
                            w_coords: np.ndarray,
                            times: np.ndarray,
                            flags: np.ndarray,
                            window_indices: Tuple[int, int]) -> Dict[str, Any]:
    """
    Averages visibilities within a temporal window.
    
    Calculates weighted average visibility (V̄ = Σ w_i V_i / Σ w_i).
    
    Parameters
    ----------
    visibilities : np.ndarray
        Array of visibilities [nrows, nchans, npols]
    weights : np.ndarray
        Array of weights [nrows, nchans, npols] or [nrows, npols]
    u_coords : np.ndarray
        U coordinates [nrows]
    v_coords : np.ndarray
        V coordinates [nrows]
    w_coords : np.ndarray
        W coordinates [nrows]
    times : np.ndarray
        Timestamps [nrows]
    flags : np.ndarray
        Flags [nrows, nchans, npols]
    window_indices : Tuple[int, int]
        (start_idx, end_idx) of the window
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with averaged visibility and metadata
    """
    start_idx, end_idx = window_indices
    n_input_rows = end_idx - start_idx
    
    if n_input_rows == 0:
        return None
    
    # Extract window data
    vis_window = visibilities[start_idx:end_idx]
    weights_window = weights[start_idx:end_idx]
    u_window = u_coords[start_idx:end_idx]
    v_window = v_coords[start_idx:end_idx]
    w_window = w_coords[start_idx:end_idx]
    times_window = times[start_idx:end_idx]
    flags_window = flags[start_idx:end_idx]
    
    # Expand weights if necessary
    if weights_window.ndim == 2:  # [nrows, npols]
        weights_window = np.expand_dims(weights_window, axis=1)  # [nrows, 1, npols]
        weights_window = np.broadcast_to(weights_window, vis_window.shape)
    
    # Combine flags - where any sample is flagged
    flag_combined = np.any(flags_window, axis=0)  # [nchans, npols]
    
    # Apply flags to weights (set weight 0 where flagged)
    weights_masked = weights_window.copy()
    weights_masked[flags_window] = 0.0
    
    # Calculate weight sum
    weight_total = np.sum(weights_masked, axis=0)  # [nchans, npols]
    
    # Avoid division by zero
    safe_weights = np.where(weight_total > 0, weight_total, 1.0)
    
    # Weighted average of visibilities: V̄ = Σ w_i V_i / Σ w_i
    vis_weighted_sum = np.sum(weights_masked * vis_window, axis=0)  # [nchans, npols]
    vis_averaged = vis_weighted_sum / safe_weights
    
    # Where no valid data, mark as flagged
    vis_averaged = np.where(weight_total > 0, vis_averaged, 0.0 + 0.0j)
    flag_combined = np.logical_or(flag_combined, weight_total == 0)
    
    # Average UVW coordinates and time
    u_avg = np.mean(u_window)
    v_avg = np.mean(v_window)
    w_avg = np.mean(w_window)
    time_avg = np.mean(times_window)
    
    # Calculate window metadata
    window_dt_s = times_window[-1] - times_window[0] if n_input_rows > 1 else 0.0
    
    return {
        'visibility_averaged': vis_averaged,  # [nchans, npols]
        'weight_total': weight_total,         # [nchans, npols]
        'flag_combined': flag_combined,       # [nchans, npols]
        'u_avg': u_avg,                       # scalar
        'v_avg': v_avg,                       # scalar
        'w_avg': w_avg,                       # scalar
        'time_avg': time_avg,                 # scalar
        'n_input_rows': n_input_rows,         # scalar
        'window_dt_s': window_dt_s,           # scalar
    }


def apply_bda_to_group(group_data: Dict[str, np.ndarray],
                      config: Dict[str, float] = None) -> List[Dict[str, Any]]:
    """
    Applies complete BDA to a data group (baseline, scan_number).
    
    Implements the complete flow:
    1. Calculate baseline_length = sqrt(u²+v²) per row
    2. Define smearing tolerance using decorrelation formulas
    3. Divide into temporal windows
    4. Average each window
    
    Parameters
    ----------
    group_data : Dict[str, np.ndarray]
        Dictionary with group data arrays:
        - 'visibilities': [nrows, nchans, npols]
        - 'weights': [nrows, nchans, npols] or [nrows, npols]
        - 'u', 'v', 'w': [nrows]
        - 'time': [nrows]
        - 'flags': [nrows, nchans, npols]
        - 'antenna1', 'antenna2': [nrows]
        - 'scan_number': [nrows]
    config : Dict[str, float], optional
        BDA configuration
        
    Returns
    -------
    List[Dict[str, Any]]
        List of averaged visibilities per window
    """
    if config is None:
        config = create_bda_config()
    
    # Extract data
    times = group_data['time']
    u_coords = group_data['u']  
    v_coords = group_data['v']
    w_coords = group_data['w']
    visibilities = group_data['visibilities']
    weights = group_data['weights']
    flags = group_data['flags']
    
    if len(times) == 0:
        return []
    
    # Sort by time
    time_order = np.argsort(times)
    times_sorted = times[time_order]
    u_sorted = u_coords[time_order]
    v_sorted = v_coords[time_order]
    w_sorted = w_coords[time_order]
    vis_sorted = visibilities[time_order]
    weights_sorted = weights[time_order]
    flags_sorted = flags[time_order]
    
    # Calculate average baseline length for the group
    baseline_lengths = np.sqrt(u_sorted**2 + v_sorted**2)
    baseline_length_avg = np.mean(baseline_lengths)
    
    # Calculate maximum decorrelation time
    delta_t_max = calculate_optimal_averaging_time(
        u=np.mean(u_sorted), 
        v=np.mean(v_sorted),
        frequency_hz=config['frequency_hz'],
        config=config
    )
    
    # Create temporal windows
    windows = create_bda_windows(
        times=times_sorted,
        delta_t_max=delta_t_max,
        baseline_length=baseline_length_avg,
        frequency_hz=config['frequency_hz']
    )
    
    # Average each window
    averaged_results = []
    for window in windows:
        result = average_visibility_window(
            visibilities=vis_sorted,
            weights=weights_sorted,
            u_coords=u_sorted,
            v_coords=v_sorted,
            w_coords=w_sorted,
            times=times_sorted,
            flags=flags_sorted,
            window_indices=window
        )
        
        if result is not None:
            # Add original group metadata
            result['baseline_length'] = baseline_length_avg
            result['delta_t_max'] = delta_t_max
            result['antenna1'] = group_data['antenna1'][0]  # Same for entire group
            result['antenna2'] = group_data['antenna2'][0]  # Same for entire group
            result['scan_number'] = group_data['scan_number'][0]  # Same for entire group
            averaged_results.append(result)
    
    return averaged_results


def ensure_baseline_units_wavelengths(u: float, v: float, 
                                    frequency_hz: float,
                                    input_units: str = 'auto') -> Tuple[float, float]:
    """
    Ensures that u,v coordinates are in wavelengths.
    
    Parameters
    ----------
    u : float
        U coordinate
    v : float
        V coordinate  
    frequency_hz : float
        Observation frequency in Hz
    input_units : str, optional
        Input units: 'meters', 'wavelengths', 'auto' (default: 'auto')
        
    Returns
    -------
    Tuple[float, float]
        (u_wavelengths, v_wavelengths)
    """
    # Auto-detect units if not specified
    if input_units == 'auto':
        baseline_magnitude = np.sqrt(u**2 + v**2)
        # If magnitude > 10000, probably in meters
        # If between 1-10000, probably in wavelengths
        if baseline_magnitude > 10000:
            input_units = 'meters'
        else:
            input_units = 'wavelengths'
    
    if input_units == 'meters':
        # Convert meters to wavelengths
        wavelength_m = const.c.value / frequency_hz
        u_lambda = u / wavelength_m
        v_lambda = v / wavelength_m
        return u_lambda, v_lambda
    elif input_units == 'wavelengths':
        # Already in wavelengths
        return u, v
    else:
        raise ValueError(f"Unsupported units: {input_units}")
