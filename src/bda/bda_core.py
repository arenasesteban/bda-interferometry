"""
BDA Core Scientific Algorithms

Implementation of fundamental scientific algorithms for baseline-dependent averaging
in radio interferometry data processing. Provides mathematical computations for
decorrelation time calculations, fringe rate analysis, temporal windowing, and
weighted visibility averaging based on Wijnholds et al. 2018 methodology.

Functions handle interferometry-specific calculations including baseline length
determination, optimal averaging time computation, and vectorized visibility
processing for distributed scientific computing environments.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, List
from astropy import constants as const

# Import configuration from dedicated config module
from .bda_config import get_default_bda_config


def calculate_baseline_length(u: float, v: float, w: float = 0.0, 
                            mode: str = 'uv_plane', in_wavelengths: bool = True) -> float:
    """
    Calculate baseline length from UVW coordinates with consistent methodology.
    
    For BDA temporal averaging, UV-plane length is typically sufficient since
    decorrelation is dominated by Earth rotation effects in the UV plane.
    Full 3D length may be used for specific geometric applications.
    
    Parameters
    ----------
    u : float
        U coordinate in wavelengths or meters
    v : float
        V coordinate in wavelengths or meters  
    w : float, optional
        W coordinate in same units (default: 0.0)
    mode : str, optional
        Calculation mode: 'uv_plane' (√(u²+v²)) or '3d_full' (√(u²+v²+w²))
        For BDA temporal averaging, 'uv_plane' is recommended (default: 'uv_plane')
    in_wavelengths : bool, optional
        If True, coordinates are in wavelengths, if False in meters (default: True)
        
    Returns
    -------
    float
        Baseline length in same units as input coordinates
        
    Notes
    -----
    Following Wijnholds et al. 2018, BDA decorrelation calculations use UV-plane
    baseline length since temporal smearing is dominated by du/dt and dv/dt terms.
    """
    if mode == 'uv_plane':
        baseline_length = np.sqrt(u**2 + v**2)
    elif mode == '3d_full':
        baseline_length = np.sqrt(u**2 + v**2 + w**2)
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'uv_plane' or '3d_full'")
    
    return baseline_length


def calculate_fringe_rate_conservative(baseline_length_lambda: float,
                                     declination_deg: float = -45.0,
                                     safety_margin: float = 2.0) -> float:
    """
    Calculate conservative fringe rate using geometric upper-bound approximation.
    
    This is a fast, orientation-independent estimate suitable for streaming BDA.
    Provides upper bound by assuming worst-case baseline orientation and source position.
    For exact calculations, use calculate_fringe_rate_exact() with hour angle.
    
    Parameters
    ----------
    baseline_length_lambda : float
        Baseline length in wavelengths (UV-plane)
    declination_deg : float, optional
        Source declination in degrees (default: -45.0)
    safety_margin : float, optional
        Multiplicative margin for conservative estimation (≥1.0, typically 1.5-2.0)
        Higher values increase estimated fringe rate → shorter averaging windows
        
    Returns
    -------
    float
        Conservative fringe rate in rad/s (upper bound)
        
    Notes
    -----
    The safety_margin here is different from safety_factor in decorrelation calculation:
    - safety_margin (≥1): Multiplies fringe rate estimate (higher = more conservative)
    - safety_factor (<1): Applied to decorrelation tolerance (lower = more conservative)
    
    This function OVERESTIMATES fringe rates for conservative BDA. The actual
    safety control is applied in calculate_decorrelation_time() via safety_factor.
    """
    # Physical constants
    omega_earth = 7.2925e-5  # Earth angular velocity in rad/s
    
    # Convert declination to radians
    dec_rad = np.radians(declination_deg)
    
    # Conservative approximation: worst case for any baseline orientation
    # Assumes maximum possible du/dt and dv/dt components
    max_du_dt = baseline_length_lambda * omega_earth * abs(np.cos(dec_rad))
    max_dv_dt = baseline_length_lambda * omega_earth * abs(np.sin(dec_rad))
    
    # Conservative fringe rate (geometric upper bound)
    conservative_fringe_rate = safety_margin * np.sqrt(max_du_dt**2 + max_dv_dt**2)
    
    return conservative_fringe_rate


def calculate_fringe_rate_exact(baseline_u_lambda: float, baseline_v_lambda: float,
                               declination_deg: float, hour_angle_deg: float,
                               site_latitude_deg: float = -30.7) -> float:
    """
    Calculate exact fringe rate using Wijnholds et al. 2018 equations (42-43).
    
    Implements precise du/dt and dv/dt calculations accounting for baseline
    orientation, source position, and Earth rotation geometry.
    
    Parameters
    ----------
    baseline_u_lambda : float
        Baseline U component in wavelengths
    baseline_v_lambda : float
        Baseline V component in wavelengths
    declination_deg : float
        Source declination in degrees
    hour_angle_deg : float
        Source hour angle in degrees
    site_latitude_deg : float, optional
        Observatory latitude in degrees (default: -30.7 for SKA-mid)
        
    Returns
    -------
    float
        Exact fringe rate in rad/s
        
    Notes
    -----
    Implements equations (42-43) from Wijnholds et al. 2018:
    du/dt = (1/λ) * (Lx*cos(H) - Ly*sin(H)) * ωE
    dv/dt = (1/λ) * (Lx*sin(δ)*sin(H) + Ly*sin(δ)*cos(H)) * ωE
    
    where Lx, Ly are baseline components in ITRF coordinates.
    """
    # Physical constants
    omega_earth = 7.2925e-5  # Earth angular velocity in rad/s
    
    # Convert to radians
    dec_rad = np.radians(declination_deg)
    ha_rad = np.radians(hour_angle_deg)
    lat_rad = np.radians(site_latitude_deg)
    
    # Transform UV baseline to approximate ITRF components
    # This is simplified - exact transformation requires full coordinate system
    # For now, assume U ≈ East-West, V ≈ North-South component
    Lx_lambda = baseline_u_lambda  # Approximate East-West component
    Ly_lambda = baseline_v_lambda  # Approximate North-South component
    
    # Calculate du/dt and dv/dt (equations 42-43)
    du_dt = (np.cos(ha_rad) * Lx_lambda - np.sin(ha_rad) * Ly_lambda) * omega_earth
    dv_dt = (np.sin(dec_rad) * np.sin(ha_rad) * Lx_lambda + 
             np.sin(dec_rad) * np.cos(ha_rad) * Ly_lambda) * omega_earth
    
    # Total fringe rate
    fringe_rate = np.sqrt(du_dt**2 + dv_dt**2)
    
    return fringe_rate


def calculate_decorrelation_time(baseline_length_meters: float,
                               frequency_hz: float,
                               config: Dict[str, float] = None,
                               mode: str = 'conservative') -> float:
    """
    Calculate decorrelation time for a specific baseline.
    
    Implements equation (41) from Wijnholds et al. 2018:
    T_decorr = sqrt(1 - R²) / |fringe_rate_total|
    
    The safety_factor is applied to the decorrelation tolerance (numerator):
    T_decorr = safety_factor * sqrt(1 - R²) / |fringe_rate_total|
    
    where safety_factor < 1 makes BDA more conservative (shorter averaging times).
    
    Parameters
    ----------
    baseline_length_meters : float
        Baseline length in meters (physical or UV-plane)
    frequency_hz : float
        Observation frequency in Hz
    config : Dict[str, float], optional
        BDA configuration. If None, uses default values
    mode : str, optional
        Calculation mode: 'conservative' (fast) or 'exact' (requires hour angle)
        
    Returns
    -------
    float
        Decorrelation time in seconds, clipped to configured limits
        
    Notes
    -----
    Safety factor application:
    - safety_factor < 1.0: More conservative (shorter averaging times)
    - safety_factor = 1.0: Use theoretical limit exactly
    - safety_factor > 1.0: Less conservative (longer averaging times, riskier)
    
    Correlator dump time is subtracted from the final result since dump
    integration also contributes to decorrelation (Section 4.2 of paper).
    """
    if config is None:
        config = get_default_bda_config()
    
    # Convert baseline to wavelengths for fringe rate calculation
    wavelength_m = const.c.value / frequency_hz
    baseline_length_lambda = baseline_length_meters / wavelength_m
    
    # Calculate fringe rate based on mode (WITHOUT safety factor - applied later)
    if mode == 'conservative':
        fringe_rate = calculate_fringe_rate_conservative(
            baseline_length_lambda=baseline_length_lambda,
            declination_deg=config['declination_deg'],
            safety_margin=2.0  # Fixed conservative margin for upper bound estimation
        )
    elif mode == 'exact':
        # Requires hour angle - for now, use conservative with warning
        logging.warning("Exact mode requires hour angle, falling back to conservative")
        fringe_rate = calculate_fringe_rate_conservative(
            baseline_length_lambda=baseline_length_lambda,
            declination_deg=config['declination_deg'],
            safety_margin=2.0  # Fixed conservative margin
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'conservative' or 'exact'")
    
    # Calculate decorrelation time (equation 41) with safety factor applied correctly
    if fringe_rate > 0:
        # sqrt(1 - R²) / |fringe_rate| with safety factor reducing allowed decorrelation
        decorr_factor = config['decorr_factor']
        safety_factor = config.get('safety_factor', 0.8)  # <1 reduces decorr time (more conservative)
        
        # Apply safety factor to the decorrelation tolerance (more conservative)
        effective_numerator = np.sqrt(1 - decorr_factor**2) * safety_factor
        decorr_time = effective_numerator / fringe_rate
    else:
        # Fallback for very short baselines (near zero fringe rate)
        decorr_time = config.get('max_window_s', 180.0)
    
    # Account for correlator dump time effect
    # Effective decorrelation budget must include dump time
    correlator_dump = config.get('correlator_dump_s', 0.14)
    if decorr_time > correlator_dump:
        # Reserve some decorrelation budget for correlator dump
        decorr_time = decorr_time - correlator_dump
    
    # Apply configured limits
    min_time = config.get('min_window_s', 1.0)
    max_time = config.get('max_window_s', 180.0)
    decorr_time = np.clip(decorr_time, min_time, max_time)
    
    return decorr_time


def calculate_optimal_averaging_time(u: float, v: float, 
                                   frequency_hz: float,
                                   config: Dict[str, float] = None,
                                   input_units: str = 'wavelengths') -> float:
    """
    Calculate optimal averaging time for a single baseline sample.
    
    This function is designed for per-sample/per-row calculations in streaming
    BDA processing. For group-based processing, use this function on individual
    samples rather than group averages.
    
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
        Input units: 'meters' or 'wavelengths' (default: 'wavelengths')
        Explicit units required - no auto-detection to avoid errors
        
    Returns
    -------
    float
        Optimal averaging time in seconds for this baseline
        
    Notes
    -----
    This function can be vectorized for batch processing of multiple baselines.
    For streaming applications, call this per visibility sample to get
    appropriate window sizes.
    """
    if config is None:
        config = get_default_bda_config()
    
    # Ensure units are in wavelengths (no auto-detection for reliability)
    if input_units not in ['meters', 'wavelengths']:
        raise ValueError(f"input_units must be 'meters' or 'wavelengths', got '{input_units}'")
    
    u_lambda, v_lambda = ensure_baseline_units_wavelengths(u, v, frequency_hz, input_units)
    
    # Calculate UV-plane baseline length (consistent with BDA theory)
    baseline_length_lambda = calculate_baseline_length(u_lambda, v_lambda, mode='uv_plane')
    
    # Convert to meters for decorrelation time calculation
    wavelength_m = const.c.value / frequency_hz
    baseline_length_meters = baseline_length_lambda * wavelength_m
    
    # Calculate decorrelation time using configured method
    averaging_time = calculate_decorrelation_time(
        baseline_length_meters=baseline_length_meters,
        frequency_hz=frequency_hz,
        config=config,
        mode='conservative'  # Use conservative mode for streaming
    )
    
    return averaging_time


def calculate_optimal_averaging_time_vectorized(u_array: np.ndarray, v_array: np.ndarray,
                                              frequency_hz: float,
                                              config: Dict[str, float] = None,
                                              input_units: str = 'wavelengths') -> np.ndarray:
    """
    Vectorized calculation of optimal averaging times for multiple baselines.
    
    Efficient batch processing for arrays of UV coordinates, useful for
    preprocessing entire datasets or partition-level processing.
    
    Parameters
    ----------
    u_array : np.ndarray
        Array of U coordinates
    v_array : np.ndarray
        Array of V coordinates (same shape as u_array)
    frequency_hz : float
        Observation frequency in Hz
    config : Dict[str, float], optional
        BDA configuration
    input_units : str, optional
        Input units for coordinates (default: 'wavelengths')
        
    Returns
    -------
    np.ndarray
        Array of optimal averaging times in seconds (same shape as input)
    """
    if config is None:
        config = get_default_bda_config()
    
    # Vectorized unit conversion
    if input_units == 'meters':
        wavelength_m = const.c.value / frequency_hz
        u_lambda = u_array / wavelength_m
        v_lambda = v_array / wavelength_m
    else:
        u_lambda = u_array
        v_lambda = v_array
    
    # Vectorized baseline length calculation
    baseline_lengths_lambda = np.sqrt(u_lambda**2 + v_lambda**2)
    baseline_lengths_meters = baseline_lengths_lambda * (const.c.value / frequency_hz)
    
    # Vectorized decorrelation time calculation
    # This could be optimized further for large arrays
    averaging_times = np.array([
        calculate_decorrelation_time(bl_m, frequency_hz, config, mode='conservative')
        for bl_m in baseline_lengths_meters
    ])
    
    return averaging_times


def create_bda_windows(times: np.ndarray, 
                      delta_t_max: float,
                      baseline_length: float = None,
                      frequency_hz: float = None,
                      min_samples_per_window: int = 1) -> List[Tuple[int, int]]:
    """
    Divide observations into temporal windows according to decorrelation limits.
    
    Creates windows that respect the maximum decorrelation time while avoiding
    single-sample windows when possible (unless data is naturally sparse).
    
    Parameters
    ----------
    times : np.ndarray
        Array of timestamps ordered by time (assumed sorted)
    delta_t_max : float
        Maximum decorrelation time for this baseline in seconds
    baseline_length : float, optional
        Baseline length for logging/debug (not used in calculation)
    frequency_hz : float, optional
        Observation frequency for logging/debug (not used in calculation)
    min_samples_per_window : int, optional
        Minimum samples per window when possible (default: 1)
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (start_idx, end_idx) tuples defining each window.
        end_idx is exclusive (Python slice convention).
        
    Notes
    -----
    Window closure uses >= comparison for robustness with floating point times.
    Single-sample windows are created when necessary due to large time gaps.
    """
    if len(times) == 0:
        return []
    
    if len(times) == 1:
        return [(0, 1)]
    
    windows = []
    window_start = 0
    
    for i in range(1, len(times)):
        # Calculate cumulative window span including current sample
        window_span = times[i] - times[window_start]
        
        # Close window if span exceeds decorrelation limit
        # Use >= for robustness with floating-point comparisons
        if window_span >= delta_t_max:
            # Close previous window (don't include current sample)
            if i > window_start:
                windows.append((window_start, i))
                window_start = i
            else:
                # Edge case: single sample exceeds limit (shouldn't happen normally)
                windows.append((window_start, i))
                window_start = i
    
    # Add final window if it contains any samples
    if window_start < len(times):
        windows.append((window_start, len(times)))
    
    # Optional: log statistics for monitoring
    if baseline_length is not None and len(windows) > 0:
        avg_samples = np.mean([end - start for start, end in windows])
        logging.debug(f"BDA windows: {len(windows)} windows, "
                     f"avg {avg_samples:.1f} samples/window, "
                     f"baseline {baseline_length:.1f}m")
    
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
        config = get_default_bda_config()
    
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
                                    input_units: str) -> Tuple[float, float]:
    """
    Convert baseline coordinates to wavelengths with explicit unit specification.
    
    Requires explicit input_units to avoid auto-detection errors that can occur
    with real observational data where magnitudes vary significantly.
    
    Parameters
    ----------
    u : float
        U coordinate
    v : float
        V coordinate  
    frequency_hz : float
        Observation frequency in Hz
    input_units : str
        Input units: 'meters' or 'wavelengths' (explicit specification required)
        
    Returns
    -------
    Tuple[float, float]
        (u_wavelengths, v_wavelengths)
        
    Raises
    ------
    ValueError
        If input_units is not 'meters' or 'wavelengths'
        
    Notes
    -----
    Auto-detection is removed to prevent errors with real data where baseline
    magnitudes can vary significantly. Consumer services should specify units
    explicitly based on their data format knowledge.
    """
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
        raise ValueError(f"input_units must be 'meters' or 'wavelengths', got '{input_units}'. "
                        "Auto-detection removed for reliability with real data.")
