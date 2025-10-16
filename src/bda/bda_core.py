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

import numpy as np
from typing import Dict, Tuple
from astropy import constants as const


def calculate_decorrelation_time(u_dot, v_dot, w_dot, decorr_factor, field_offset_deg):
    """
    Calcula el tiempo de decorrelación T (en segundos) para una fila, usando el modelo uv_rate_fov.

    Parámetros
    ----------
    u_dot, v_dot, w_dot : float
        Tasas de cambio de u, v, w en longitudes de onda por segundo.
    decorr_factor : float
        R_T objetivo (p.ej. 0.99) que indica la mínima correlación permitida.
    field_offset_deg : float
        Desplazamiento angular desde el centro del FoV (borde del FoV, en grados).

    Retorna
    -------
    T_eff : float
        Tiempo máximo de promediado para la fila, garantizando la pérdida de correlación ≤ 1-R_T.
    """
    # Convertimos el ángulo del borde del FoV a radianes
    theta = np.radians(field_offset_deg)

    # Definimos la dirección de la fuente en coordenadas de imagen
    l = np.sin(theta)        # dirección horizontal (borde)
    m = 0.0                  # dirección vertical
    n = np.sqrt(max(0.0, 1.0 - l**2 - m**2))  # componente perpendicular al plano

    # Calculamos la tasa de fase efectiva para esta fila
    phi_dot = np.pi * (u_dot * l + v_dot * m + w_dot * (n - 1))

    # Aproximación de pequeña pérdida (Taylor de sinc) para resolver T
    T_eff = np.sqrt(6 * (1.0 - decorr_factor)) / abs(phi_dot)

    return T_eff

def calculate_baseline_length(u: float, v: float, w: float) -> float:
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
    
    return np.sqrt(u**2 + v**2 + w**2)


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


def calculate_decorrelation_time(baseline_length_lambda: float,
                               bda_config: Dict[str, float] = None) -> float:
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
    # Calculate fringe rate
    fringe_rate = calculate_fringe_rate_exact()
     
    # Calculate decorrelation time with safety factor
    if fringe_rate > 0:
        # sqrt(1 - R²) / |fringe_rate| with safety factor reducing allowed decorrelation
        decorr_factor = bda_config['decorr_factor']
        safety_factor = bda_config.get('safety_factor', 0.8)  # <1 reduces decorr time
        
        # Apply safety factor to the decorrelation tolerance
        effective_numerator = np.sqrt(1 - decorr_factor**2) * safety_factor
        decorr_time = effective_numerator / fringe_rate
    else:
        # Fallback for very short baselines (near zero fringe rate)
        decorr_time = bda_config.get('max_window_s', 180.0)
    
    # Account for correlator dump time effect
    # Effective decorrelation budget must include dump time
    correlator_dump = bda_config.get('correlator_dump_s', 0.14)
    if decorr_time > correlator_dump:
        # Reserve some decorrelation budget for correlator dump
        decorr_time = decorr_time - correlator_dump
    
    # Apply configured limits
    min_time = bda_config.get('min_window_s', 1.0)
    max_time = bda_config.get('max_window_s', 180.0)
    decorr_time = np.clip(decorr_time, min_time, max_time)
    
    return decorr_time


def calculate_optimal_averaging_time(u: float, v: float, 
                                   frequency_hz: float,
                                   bda_config: Dict[str, float] = None) -> float:
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
    
    u_lambda, v_lambda = ensure_baseline_units_wavelengths(u, v, frequency_hz)
    
    # Calculate UV-plane baseline length
    baseline_length_lambda = calculate_baseline_length(u_lambda, v_lambda)
    
    # Calculate decorrelation time using configured method
    averaging_time = calculate_decorrelation_time(
        baseline_length_lambda=baseline_length_lambda,
        bda_config=bda_config,
    )
    
    return averaging_time


def ensure_baseline_units_wavelengths(u: float, v: float, 
                                    frequency_hz: float) -> Tuple[float, float]:
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
    wavelength_m = const.c.value / frequency_hz
    u_lambda = u / wavelength_m
    v_lambda = v / wavelength_m
    return u_lambda, v_lambda