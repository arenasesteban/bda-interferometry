"""
BDA Config - Simple Configuration Loading

Functions for loading BDA configuration from JSON using pure functional programming.
No classes, no state, only functions that transform data.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_bda_config(config_path: str) -> Dict[str, float]:
    """
    Load BDA configuration from simple JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to JSON configuration file
        
    Returns
    -------
    Dict[str, float]
        BDA configuration as simple dictionary
        
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        Si el JSON no es válido
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"BDA config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validar que tiene los campos mínimos requeridos
        required_fields = ['decorr_factor', 'frequency_hz', 'declination_deg']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in BDA config")
        
        # Agregar safety_factor si no existe
        if 'safety_factor' not in config:
            config['safety_factor'] = 0.8
        
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in BDA config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading BDA config: {e}")


def get_default_bda_config() -> Dict[str, float]:
    """
    Retorna configuración BDA por defecto para ALMA Band 1.
    
    Returns
    -------
    Dict[str, float]
        Configuración por defecto
    """
    return {
        'decorr_factor': 0.95,
        'frequency_hz': 42.5e9,  # ALMA Band 1 center
        'declination_deg': -45.0,  # Southern hemisphere
        'safety_factor': 0.8
    }


def load_bda_config_with_fallback(config_path: str) -> Dict[str, float]:
    """
    Carga configuración BDA con fallback a valores por defecto.
    
    Parameters
    ----------
    config_path : str
        Ruta al archivo de configuración
        
    Returns
    -------
    Dict[str, float]
        Configuración cargada o por defecto si hay error
    """
    try:
        return load_bda_config(config_path)
    except Exception as e:
        print(f"⚠️  Could not load BDA config from {config_path}: {e}")
        print("🔄 Using default BDA configuration...")
        return get_default_bda_config()
