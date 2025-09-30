"""
BDA Utils - Utility and Support Functions

Utility functions for validation, debugging, configuration and analysis
of BDA processing. Includes tools for performance analysis,
data validation and report generation.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from .bda_core import create_bda_config, validate_bda_config


def load_bda_config(config_path: str) -> Dict[str, float]:
    """
    Carga configuración BDA desde archivo JSON.
    
    Parameters
    ----------
    config_path : str
        Ruta al archivo de configuración JSON
        
    Returns
    -------
    Dict[str, float]
        Parámetros BDA cargados desde archivo
        
    Raises
    ------
    FileNotFoundError
        Si el archivo no existe
    ValueError
        Si la configuración no es válida
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"BDA config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Si es configuración nueva (flat)
        if 'decorr_factor' in config_data:
            config = config_data
        else:
            # Si es configuración legacy con bda_parameters
            config = config_data.get('bda_parameters', {})
        
        # Crear configuración con valores por defecto
        bda_config = create_bda_config(
            decorr_factor=config.get('decorr_factor', 0.95),
            frequency_hz=config.get('frequency_hz', 42.5e9),
            declination_deg=config.get('declination_deg', -45.0),
            safety_factor=config.get('safety_factor', 0.8)
        )
        
        # Validar configuración
        is_valid, error_msg = validate_bda_config(bda_config)
        if not is_valid:
            raise ValueError(f"Invalid BDA configuration: {error_msg}")
        
        return bda_config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in BDA config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading BDA config: {e}")


def save_bda_config(params: Dict[str, float], config_path: str) -> None:
    """
    Guarda configuración BDA a archivo JSON.
    
    Parameters
    ----------
    params : Dict[str, float]
        Parámetros BDA a guardar
    config_path : str
        Ruta donde guardar el archivo
    """
    config_data = {
        "bda_parameters": {
            "decorr_factor": params['decorr_factor'],
            "frequency_hz": params['frequency_hz'],
            "declination_deg": params['declination_deg'],
            "safety_factor": params['safety_factor']
        },
        "metadata": {
            "created_by": "bda_utils",
            "description": "BDA configuration parameters"
        }
    }
    
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)


def analyze_baseline_distribution(groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analiza la distribución de baselines en los datos para optimización BDA.
    
    Parameters
    ----------
    groups : Dict[str, List[Dict[str, Any]]]
        Grupos de filas por baseline+scan
        
    Returns
    -------
    Dict[str, Any]
        Análisis de distribución de baselines
    """
    baseline_analysis = {
        'unique_baselines': set(),
        'baseline_lengths': {},
        'length_distribution': {
            'short': 0,      # < 100 λ
            'medium': 0,     # 100-1000 λ  
            'long': 0,       # 1000-10000 λ
            'very_long': 0,  # > 10000 λ
        },
        'rows_per_baseline': {},
        'scans_per_baseline': {}
    }
    
    for group_key, rows in groups.items():
        if not rows:
            continue
            
        # Extraer información del primer row del grupo
        first_row = rows[0]
        baseline_key = first_row.get('baseline_key', 'unknown')
        scan_number = first_row.get('scan_number', -1)
        u_coord = first_row.get('u', 0.0)
        v_coord = first_row.get('v', 0.0)
        
        baseline_analysis['unique_baselines'].add(baseline_key)
        
        # Calcular longitud de baseline
        baseline_length = np.sqrt(u_coord**2 + v_coord**2)
        baseline_analysis['baseline_lengths'][baseline_key] = baseline_length
        
        # Clasificar por longitud
        if baseline_length < 100:
            baseline_analysis['length_distribution']['short'] += 1
        elif baseline_length < 1000:
            baseline_analysis['length_distribution']['medium'] += 1
        elif baseline_length < 10000:
            baseline_analysis['length_distribution']['long'] += 1
        else:
            baseline_analysis['length_distribution']['very_long'] += 1
        
        # Contar filas por baseline
        if baseline_key not in baseline_analysis['rows_per_baseline']:
            baseline_analysis['rows_per_baseline'][baseline_key] = 0
        baseline_analysis['rows_per_baseline'][baseline_key] += len(rows)
        
        # Contar scans por baseline
        if baseline_key not in baseline_analysis['scans_per_baseline']:
            baseline_analysis['scans_per_baseline'][baseline_key] = set()
        baseline_analysis['scans_per_baseline'][baseline_key].add(scan_number)
    
    # Convertir sets a counts
    baseline_analysis['unique_baselines'] = len(baseline_analysis['unique_baselines'])
    for baseline_key in baseline_analysis['scans_per_baseline']:
        baseline_analysis['scans_per_baseline'][baseline_key] = len(
            baseline_analysis['scans_per_baseline'][baseline_key]
        )
    
    return baseline_analysis


def estimate_bda_potential(groups: Dict[str, List[Dict[str, Any]]], 
                         params: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Estima el potencial de compresión BDA sin aplicar el processing.
    
    Parameters
    ----------
    groups : Dict[str, List[Dict[str, Any]]]
        Grupos de filas
    params : Dict[str, float], optional
        Parámetros BDA
        
    Returns
    -------
    Dict[str, Any]
        Estimación de potencial BDA
    """
    if params is None:
        params = create_bda_config()
    
    from .bda_core import calculate_optimal_averaging_time
    
    estimation = {
        'total_input_rows': 0,
        'estimated_output_rows': 0,
        'estimated_compression': 0.0,
        'baseline_estimates': {},
        'integration_time_assumed': 180.0  # segundos, valor típico
    }
    
    for group_key, rows in groups.items():
        if not rows:
            continue
            
        estimation['total_input_rows'] += len(rows)
        
        # Calcular tiempo de averaging para este baseline
        first_row = rows[0]
        u_coord = first_row.get('u', 0.0)
        v_coord = first_row.get('v', 0.0)
        
        optimal_avg_time = calculate_optimal_averaging_time(
            u=u_coord, v=v_coord,
            frequency_hz=params['frequency_hz'],
            config=params
        )
        
        # Estimar reducción basada en averaging time vs integration time
        if optimal_avg_time > estimation['integration_time_assumed']:
            compression_factor = optimal_avg_time / estimation['integration_time_assumed']
            estimated_output = max(1, len(rows) // int(compression_factor))
        else:
            estimated_output = len(rows)  # No compression
        
        estimation['estimated_output_rows'] += estimated_output
        
        # Guardar estimación por baseline
        baseline_key = first_row.get('baseline_key', 'unknown')
        estimation['baseline_estimates'][baseline_key] = {
            'input_rows': len(rows),
            'estimated_output_rows': estimated_output,
            'optimal_averaging_time': optimal_avg_time,
            'compression_factor': len(rows) / estimated_output if estimated_output > 0 else 1.0
        }
    
    # Calcular compresión total estimada
    if estimation['total_input_rows'] > 0:
        estimation['estimated_compression'] = (
            1.0 - estimation['estimated_output_rows'] / estimation['total_input_rows']
        )
    
    return estimation


def format_bda_report(bda_stats: Dict[str, Any], 
                     baseline_analysis: Dict[str, Any] = None) -> str:
    """
    Genera un reporte formateado de resultados BDA.
    
    Parameters
    ----------
    bda_stats : Dict[str, Any]
        Estadísticas BDA
    baseline_analysis : Dict[str, Any], optional
        Análisis de baseline distribution
        
    Returns
    -------
    str
        Reporte formateado
    """
    lines = []
    lines.append("=" * 60)
    lines.append("                 BDA PROCESSING REPORT")
    lines.append("=" * 60)
    
    # Estadísticas principales
    lines.append(f"Input rows:           {bda_stats.get('total_input_rows', 0):,}")
    lines.append(f"Output rows:          {bda_stats.get('total_output_rows', 0):,}")
    lines.append(f"Compression ratio:    {bda_stats.get('compression_ratio', 0.0):.1%}")
    lines.append(f"Processing time:      {bda_stats.get('processing_time_ms', 0.0):.1f} ms")
    lines.append(f"Groups processed:     {bda_stats.get('groups_processed', 0)}")
    lines.append(f"Averaging applied:    {bda_stats.get('averaging_applied_rows', 0)} rows")
    
    # Distribución de tiempos de averaging
    if 'averaging_time_distribution' in bda_stats and bda_stats['averaging_time_distribution']:
        avg_times = bda_stats['averaging_time_distribution']
        lines.append(f"\nAveraging Times:")
        lines.append(f"  Mean:               {np.mean(avg_times):.1f}s")
        lines.append(f"  Range:              {np.min(avg_times):.1f}s - {np.max(avg_times):.1f}s")
        lines.append(f"  Std deviation:      {np.std(avg_times):.1f}s")
    
    # Análisis de baselines si está disponible
    if baseline_analysis:
        lines.append(f"\nBaseline Analysis:")
        lines.append(f"  Unique baselines:   {baseline_analysis.get('unique_baselines', 0)}")
        
        dist = baseline_analysis.get('length_distribution', {})
        lines.append(f"  Length distribution:")
        lines.append(f"    Short (< 100λ):   {dist.get('short', 0)}")
        lines.append(f"    Medium (100-1kλ): {dist.get('medium', 0)}")
        lines.append(f"    Long (1k-10kλ):   {dist.get('long', 0)}")
        lines.append(f"    Very long (>10kλ): {dist.get('very_long', 0)}")
    
    # Top baselines por compresión
    baseline_stats = bda_stats.get('baseline_statistics', {})
    if baseline_stats:
        lines.append(f"\nTop Baselines by Compression:")
        sorted_baselines = sorted(
            baseline_stats.items(),
            key=lambda x: x[1].get('compression_ratio', 0.0),
            reverse=True
        )[:5]
        
        for baseline_key, stats in sorted_baselines:
            comp_ratio = stats.get('compression_ratio', 0.0)
            input_rows = stats.get('input_rows', 0)
            output_rows = stats.get('output_rows', 0)
            classification = stats.get('classification', 'unknown')
            
            lines.append(f"  {baseline_key[:20]:20} {comp_ratio:6.1%} "
                        f"({input_rows}→{output_rows}, {classification})")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def validate_row_structure(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida que una fila tenga la estructura requerida para BDA.
    
    Parameters
    ----------
    row : Dict[str, Any]
        Fila a validar
        
    Returns
    -------
    Tuple[bool, List[str]]
        (es_válida, lista_de_errores)
    """
    errors = []
    
    # Campos requeridos (subms_id es opcional)
    required_fields = [
        'baseline_key', 'scan_number', 'antenna1', 'antenna2',
        'time', 'u', 'v', 'w'
    ]
    
    # Campos opcionales (no causan error si faltan)
    optional_fields = ['subms_id']
    
    for field in required_fields:
        if field not in row:
            errors.append(f"Missing required field: {field}")
    
    # Validar campos opcionales si están presentes
    for field in optional_fields:
        if field in row and row[field] is None:
            errors.append(f"Optional field {field} is None")
    
    # Validar tipos de datos
    if 'time' in row:
        try:
            float(row['time'])
        except (ValueError, TypeError):
            errors.append(f"Invalid time value: {row['time']}")
    
    if 'u' in row and 'v' in row:
        try:
            float(row['u'])
            float(row['v'])
        except (ValueError, TypeError):
            errors.append(f"Invalid UV coordinates: u={row.get('u')}, v={row.get('v')}")
    
    # Validar valores lógicos
    if 'antenna1' in row and 'antenna2' in row:
        try:
            ant1 = int(row['antenna1'])
            ant2 = int(row['antenna2'])
            if ant1 < 0 or ant2 < 0:
                errors.append(f"Invalid antenna IDs: {ant1}, {ant2}")
        except (ValueError, TypeError):
            errors.append(f"Non-integer antenna IDs: {row.get('antenna1')}, {row.get('antenna2')}")
    
    return len(errors) == 0, errors


def create_default_bda_config() -> Dict[str, Any]:
    """
    Crea configuración BDA por defecto para usar como template.
    
    Returns
    -------
    Dict[str, Any]
        Configuración BDA por defecto en formato JSON
    """
    return {
        "bda_parameters": {
            "decorr_factor": 0.95,
            "max_averaging_time": 180.0,
            "min_averaging_time": 1.0,
            "safety_factor": 0.8,
            "frequency_hz": 42.5e9,
            "declination_deg": -45.0
        },
        "streaming_config": {
            "enable_bda": True,
            "enable_validation": True,
            "enable_detailed_logging": False,
            "fallback_on_error": True
        },
        "performance_config": {
            "max_processing_time_ms": 5000,
            "min_compression_ratio": 0.1,
            "enable_performance_monitoring": True
        },
        "metadata": {
            "description": "Default BDA configuration for interferometry streaming",
            "version": "1.0",
            "created_by": "bda_utils"
        }
    }
