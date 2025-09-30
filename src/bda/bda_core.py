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
    Calculate baseline length from UVW coordinates.
    
    La longitud del baseline se calcula como la distancia euclidiana en el plano UV,
    ya que W representa la componente perpendicular a la esfera celeste y no contribuye
    a la longitud proyectada del baseline para el cálculo de fringe rates.
    
    Parameters
    ----------
    u : float
        Coordenada U en wavelengths o metros
    v : float
        Coordenada V en wavelengths o metros  
    w : float, optional
        Coordenada W (no se usa en el cálculo, default: 0.0)
    in_wavelengths : bool, optional
        Si True, las coordenadas están en wavelengths, si False en metros (default: True)
        
    Returns
    -------
    float
        Longitud del baseline en la misma unidad que las coordenadas de entrada
    """
    baseline_length = np.sqrt(u**2 + v**2)
    return baseline_length


def calculate_fringe_rate(baseline_length_lambda: float, 
                         declination_deg: float = -45.0,
                         hour_angle_deg: float = 0.0,
                         baseline_east_fraction: float = 1.0,
                         baseline_north_fraction: float = 0.0) -> float:
    """
    Calcula la fringe rate total basada en las derivadas de coordenadas UV.
    
    Implementa las ecuaciones 42-44 de Wijnholds et al. 2018:
    - ∂u/∂t = (1/λ) * (Lx*cos(H) - Ly*sin(H)) * ωE * cos(δ)
    - ∂v/∂t = (1/λ) * (Lx*sin(H)*sin(δ) + Ly*cos(H)*sin(δ)) * ωE  
    - ∂w/∂t = (1/λ) * (Lx*sin(H)*cos(δ) + Ly*cos(H)*cos(δ)) * ωE
    
    Parameters
    ----------
    baseline_length_lambda : float
        Longitud del baseline en wavelengths
    declination_deg : float, optional
        Declinación de la fuente en grados (default: -45.0)
    hour_angle_deg : float, optional
        Ángulo horario en grados (default: 0.0 para cenit)
    baseline_east_fraction : float, optional
        Fracción del baseline en dirección Este (default: 1.0)
    baseline_north_fraction : float, optional
        Fracción del baseline en dirección Norte (default: 0.0)
        
    Returns
    -------
    float
        Fringe rate total en rad/s
    """
    # Constantes físicas
    omega_earth = 7.2925e-5  # Velocidad angular de la Tierra en rad/s
    
    # Convertir ángulos a radianes
    dec_rad = np.radians(declination_deg)
    hour_angle_rad = np.radians(hour_angle_deg)
    
    # Componentes del baseline en wavelengths
    Lx_lambda = baseline_length_lambda * baseline_east_fraction
    Ly_lambda = baseline_length_lambda * baseline_north_fraction
    
    # Calcular derivadas UV según ecuaciones 42-43 del paper
    du_dt = (Lx_lambda * np.cos(hour_angle_rad) - 
             Ly_lambda * np.sin(hour_angle_rad)) * omega_earth * np.cos(dec_rad)
    
    dv_dt = (Lx_lambda * np.sin(hour_angle_rad) * np.sin(dec_rad) + 
             Ly_lambda * np.cos(hour_angle_rad) * np.sin(dec_rad)) * omega_earth
    
    # Derivada w (ecuación 44)
    dw_dt = (Lx_lambda * np.sin(hour_angle_rad) * np.cos(dec_rad) + 
             Ly_lambda * np.cos(hour_angle_rad) * np.cos(dec_rad)) * omega_earth
    
    # Fringe rate total con peso conservador para w-term
    # Para natural weighting, w_factor ≈ 0; para uniform weighting, mayor peso
    w_factor = 0.1  # Valor conservador
    
    total_fringe_rate = np.sqrt(du_dt**2 + dv_dt**2 + (w_factor * dw_dt)**2)
    
    return total_fringe_rate


def calculate_fringe_rate_conservative(baseline_length_lambda: float,
                                     declination_deg: float = -45.0,
                                     safety_margin: float = 2.0) -> float:
    """
    Calcula fringe rate usando aproximación conservadora independiente de orientación.
    
    CORREGIDO: Usa aproximación conservadora que no depende de orientación específica
    del baseline ni de ángulo horario fijo. Evita sesgos por orientación arbitraria.
    
    Parameters
    ----------
    baseline_length_lambda : float
        Longitud del baseline en wavelengths
    declination_deg : float, optional
        Declinación de la fuente en grados (default: -45.0)
    safety_margin : float, optional
        Factor de seguridad conservador (default: 2.0 para worst case)
        
    Returns
    -------
    float
        Fringe rate conservador en rad/s
    """
    # Constantes físicas
    omega_earth = 7.2925e-5  # Velocidad angular de la Tierra en rad/s
    
    # Convertir declinación a radianes
    dec_rad = np.radians(declination_deg)
    
    # Aproximación conservadora: worst case para cualquier orientación
    # Máximo fringe rate posible para este baseline y declinación
    max_du_dt = baseline_length_lambda * omega_earth * np.cos(dec_rad)
    max_dv_dt = baseline_length_lambda * omega_earth * abs(np.sin(dec_rad))
    
    # Fringe rate conservador (worst case scenario)
    conservative_fringe_rate = safety_margin * np.sqrt(max_du_dt**2 + max_dv_dt**2)
    
    return conservative_fringe_rate


def calculate_decorrelation_time(baseline_length_meters: float,
                               frequency_hz: float,
                               config: Dict[str, float] = None) -> float:
    """
    Calcula el tiempo de decorrelación para un baseline específico.
    
    Implementa la ecuación 41 de Wijnholds et al. 2018:
    T_decorr = sqrt(1 - R²) / |fringe_rate_total|
    
    Donde R es el factor de decorrelación y fringe_rate_total incluye las contribuciones
    de las derivadas temporales de las coordenadas UV debidas a la rotación terrestre.
    
    Parameters
    ----------
    baseline_length_meters : float
        Longitud del baseline en metros
    frequency_hz : float
        Frecuencia de observación en Hz
    params : BDAParameters, optional
        Parámetros BDA. Si None, usa valores por defecto
        
    Returns
    -------
    float
        Tiempo de decorrelación en segundos, limitado por min/max_averaging_time
    """
    if config is None:
        config = create_bda_config()
    
    # Convertir baseline a wavelengths
    wavelength_m = const.c.value / frequency_hz
    baseline_length_lambda = baseline_length_meters / wavelength_m
    
    # Calcular fringe rate total usando versión conservadora
    fringe_rate = calculate_fringe_rate_conservative(
        baseline_length_lambda=baseline_length_lambda,
        declination_deg=config['declination_deg'],
        safety_margin=2.0  # Factor conservador
    )
    
    # Calcular tiempo de decorrelación (ecuación 41)
    if fringe_rate > 0:
        # sqrt(1 - R²) / |fringe_rate| 
        numerator = np.sqrt(1 - config['decorr_factor']**2)
        decorr_time = numerator / fringe_rate
    else:
        # Fallback para baselines muy cortos
        decorr_time = 180.0  # Default max averaging time
    
    # Aplicar factor de seguridad
    decorr_time *= config['safety_factor']
    
    # Aplicar límites (1s min, 180s max)
    decorr_time = np.clip(decorr_time, 1.0, 180.0)
    
    return decorr_time


def calculate_optimal_averaging_time(u: float, v: float, 
                                   frequency_hz: float,
                                   config: Dict[str, float] = None,
                                   input_units: str = 'auto') -> float:
    """
    Calcula el tiempo de averaging óptimo para un baseline dado por coordenadas UV.
    
    CORREGIDO: Maneja unidades explícitamente y usa fringe rate conservador.
    
    Parameters
    ----------
    u : float
        Coordenada U (unidades según input_units)
    v : float  
        Coordenada V (unidades según input_units)
    frequency_hz : float
        Frecuencia de observación en Hz
    config : Dict[str, float], optional
        Configuración BDA. Si None, usa valores por defecto
    input_units : str, optional
        Unidades de entrada: 'meters', 'wavelengths', 'auto' (default: 'auto')
        
    Returns
    -------
    float
        Tiempo de averaging óptimo en segundos
    """
    if config is None:
        config = create_bda_config()
    
    # Garantizar unidades en wavelengths
    u_lambda, v_lambda = ensure_baseline_units_wavelengths(u, v, frequency_hz, input_units)
    
    # Calcular longitud del baseline en wavelengths  
    baseline_length_lambda = calculate_baseline_length(u_lambda, v_lambda)
    
    # Convertir a metros para decorrelation time
    wavelength_m = const.c.value / frequency_hz
    baseline_length_meters = baseline_length_lambda * wavelength_m
    
    # Calcular tiempo de decorrelación
    averaging_time = calculate_decorrelation_time(
        baseline_length_meters=baseline_length_meters,
        frequency_hz=frequency_hz,
        config=config
    )
    
    return averaging_time


def validate_bda_config(config: Dict[str, float]) -> Tuple[bool, str]:
    """
    Valida la configuración BDA para asegurar valores físicamente razonables.
    
    Parameters
    ----------
    config : Dict[str, float]
        Configuración BDA a validar
        
    Returns
    -------
    Tuple[bool, str]
        (es_válido, mensaje_error)
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
    
    return True, "Configuración válida"


# Funciones de utilidad para debugging y análisis
def get_baseline_classification(baseline_length_lambda: float) -> str:
    """
    Clasifica un baseline según su longitud para análisis.
    
    Parameters
    ----------
    baseline_length_lambda : float
        Longitud del baseline en wavelengths
        
    Returns
    -------
    str
        Clasificación del baseline
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
    Estima el ratio de compresión esperado para un baseline.
    
    Parameters
    ----------
    baseline_length_lambda : float
        Longitud del baseline en wavelengths
    integration_time_sec : float
        Tiempo de integración original en segundos
    averaging_time_sec : float
        Tiempo de averaging BDA en segundos
        
    Returns
    -------
    float
        Ratio de compresión estimado (1 - output_size/input_size)
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
    Divide las observaciones en ventanas temporales según límites de smearing.
    
    CORREGIDO: Limita el ancho acumulado total de cada ventana, no solo gaps individuales.
    Una ventana se cierra cuando su duración total (t_last - t_first) excedería Δt_max.
    
    Parameters
    ----------
    times : np.ndarray
        Array de timestamps ordenados por tiempo
    delta_t_max : float
        Tiempo máximo de decorrelación para este baseline
    baseline_length : float, optional
        Longitud del baseline (para logging/debug)
    frequency_hz : float, optional
        Frecuencia de observación (para logging/debug)
        
    Returns
    -------
    List[Tuple[int, int]]
        Lista de (start_idx, end_idx) para cada ventana
    """
    if len(times) == 0:
        return []
    
    windows = []
    window_start = 0
    
    for i in range(1, len(times)):
        # Calcular ancho acumulado de la ventana si incluimos este punto
        window_span = times[i] - times[window_start]
        
        # Si el ancho total excedería Δt_max, cerrar ventana actual
        if window_span > delta_t_max:
            windows.append((window_start, i))
            window_start = i
    
    # Agregar última ventana
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
    Promedia las visibilidades dentro de una ventana temporal.
    
    Calcula visibilidad promedio con pesos (V̄ = Σ w_i V_i / Σ w_i).
    Guarda ū,v̄,w̄, t̄, weight_total, flag_combined.
    
    Parameters
    ----------
    visibilities : np.ndarray
        Array de visibilidades [nrows, nchans, npols]
    weights : np.ndarray
        Array de pesos [nrows, nchans, npols] o [nrows, npols]
    u_coords : np.ndarray
        Coordenadas U [nrows]
    v_coords : np.ndarray
        Coordenadas V [nrows]
    w_coords : np.ndarray
        Coordenadas W [nrows]
    times : np.ndarray
        Timestamps [nrows]
    flags : np.ndarray
        Flags [nrows, nchans, npols]
    window_indices : Tuple[int, int]
        (start_idx, end_idx) de la ventana
        
    Returns
    -------
    Dict[str, Any]
        Diccionario con visibilidad promediada y metadatos
    """
    start_idx, end_idx = window_indices
    n_input_rows = end_idx - start_idx
    
    if n_input_rows == 0:
        return None
    
    # Extraer datos de la ventana
    vis_window = visibilities[start_idx:end_idx]
    weights_window = weights[start_idx:end_idx]
    u_window = u_coords[start_idx:end_idx]
    v_window = v_coords[start_idx:end_idx]
    w_window = w_coords[start_idx:end_idx]
    times_window = times[start_idx:end_idx]
    flags_window = flags[start_idx:end_idx]
    
    # Expandir weights si es necesario
    if weights_window.ndim == 2:  # [nrows, npols]
        weights_window = np.expand_dims(weights_window, axis=1)  # [nrows, 1, npols]
        weights_window = np.broadcast_to(weights_window, vis_window.shape)
    
    # Combinar flags - donde cualquier sample está flagged
    flag_combined = np.any(flags_window, axis=0)  # [nchans, npols]
    
    # Aplicar flags a los pesos (poner peso 0 donde hay flags)
    weights_masked = weights_window.copy()
    weights_masked[flags_window] = 0.0
    
    # Calcular suma de pesos
    weight_total = np.sum(weights_masked, axis=0)  # [nchans, npols]
    
    # Evitar división por cero
    safe_weights = np.where(weight_total > 0, weight_total, 1.0)
    
    # Promedio ponderado de visibilidades: V̄ = Σ w_i V_i / Σ w_i
    vis_weighted_sum = np.sum(weights_masked * vis_window, axis=0)  # [nchans, npols]
    vis_averaged = vis_weighted_sum / safe_weights
    
    # Donde no hay datos válidos, marcar como flagged
    vis_averaged = np.where(weight_total > 0, vis_averaged, 0.0 + 0.0j)
    flag_combined = np.logical_or(flag_combined, weight_total == 0)
    
    # Promediar coordenadas UVW y tiempo
    u_avg = np.mean(u_window)
    v_avg = np.mean(v_window)
    w_avg = np.mean(w_window)
    time_avg = np.mean(times_window)
    
    # Calcular metadatos de ventana
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
    Aplica BDA completo a un grupo de datos (baseline, scan_number).
    
    Implementa el flujo completo:
    1. Calcular baseline_length = sqrt(u²+v²) por fila
    2. Definir tolerancia de smearing usando fórmulas de decorrelation
    3. Dividir en ventanas temporales
    4. Promediar cada ventana
    
    Parameters
    ----------
    group_data : Dict[str, np.ndarray]
        Diccionario con arrays de datos del grupo:
        - 'visibilities': [nrows, nchans, npols]
        - 'weights': [nrows, nchans, npols] o [nrows, npols]
        - 'u', 'v', 'w': [nrows]
        - 'time': [nrows]
        - 'flags': [nrows, nchans, npols]
        - 'antenna1', 'antenna2': [nrows]
        - 'scan_number': [nrows]
    config : Dict[str, float], optional
        Configuración BDA
        
    Returns
    -------
    List[Dict[str, Any]]
        Lista de visibilidades promediadas por ventana
    """
    if config is None:
        config = create_bda_config()
    
    # Extraer datos
    times = group_data['time']
    u_coords = group_data['u']  
    v_coords = group_data['v']
    w_coords = group_data['w']
    visibilities = group_data['visibilities']
    weights = group_data['weights']
    flags = group_data['flags']
    
    if len(times) == 0:
        return []
    
    # Ordenar por tiempo
    time_order = np.argsort(times)
    times_sorted = times[time_order]
    u_sorted = u_coords[time_order]
    v_sorted = v_coords[time_order]
    w_sorted = w_coords[time_order]
    vis_sorted = visibilities[time_order]
    weights_sorted = weights[time_order]
    flags_sorted = flags[time_order]
    
    # Calcular baseline length promedio para el grupo
    baseline_lengths = np.sqrt(u_sorted**2 + v_sorted**2)
    baseline_length_avg = np.mean(baseline_lengths)
    
    # Calcular tiempo máximo de decorrelación
    delta_t_max = calculate_optimal_averaging_time(
        u=np.mean(u_sorted), 
        v=np.mean(v_sorted),
        frequency_hz=config['frequency_hz'],
        config=config
    )
    
    # Crear ventanas temporales
    windows = create_bda_windows(
        times=times_sorted,
        delta_t_max=delta_t_max,
        baseline_length=baseline_length_avg,
        frequency_hz=config['frequency_hz']
    )
    
    # Promediar cada ventana
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
            # Agregar metadatos del grupo original
            result['baseline_length'] = baseline_length_avg
            result['delta_t_max'] = delta_t_max
            result['antenna1'] = group_data['antenna1'][0]  # Mismo para todo el grupo
            result['antenna2'] = group_data['antenna2'][0]  # Mismo para todo el grupo
            result['scan_number'] = group_data['scan_number'][0]  # Mismo para todo el grupo
            averaged_results.append(result)
    
    return averaged_results


def ensure_baseline_units_wavelengths(u: float, v: float, 
                                    frequency_hz: float,
                                    input_units: str = 'auto') -> Tuple[float, float]:
    """
    Garantiza que coordenadas u,v estén en wavelengths.
    
    CORREGIDO: Maneja conversión de unidades explícitamente.
    
    Parameters
    ----------
    u : float
        Coordenada U
    v : float
        Coordenada V  
    frequency_hz : float
        Frecuencia de observación en Hz
    input_units : str, optional
        Unidades de entrada: 'meters', 'wavelengths', 'auto' (default: 'auto')
        
    Returns
    -------
    Tuple[float, float]
        (u_wavelengths, v_wavelengths)
    """
    # Detectar unidades automáticamente si no se especifica
    if input_units == 'auto':
        baseline_magnitude = np.sqrt(u**2 + v**2)
        # Si la magnitud es > 10000, probablemente está en metros
        # Si está entre 1-10000, probablemente en wavelengths
        if baseline_magnitude > 10000:
            input_units = 'meters'
        else:
            input_units = 'wavelengths'
    
    if input_units == 'meters':
        # Convertir metros a wavelengths
        wavelength_m = const.c.value / frequency_hz
        u_lambda = u / wavelength_m
        v_lambda = v / wavelength_m
        return u_lambda, v_lambda
    elif input_units == 'wavelengths':
        # Ya están en wavelengths
        return u, v
    else:
        raise ValueError(f"Unidades no soportadas: {input_units}")
