"""
BDA Configuration Management

Provides configuration loading and validation functions for baseline-dependent averaging
parameters. Handles JSON file parsing, default value provision, and fallback mechanisms
for robust configuration management in streaming interferometry processing.
"""

import json
import logging
from pathlib import Path
from typing import Dict


def load_bda_config(config_path: str) -> Dict[str, float]:
    """
    Load baseline-dependent averaging configuration from JSON file.
    
    Reads JSON configuration file, validates required scientific parameters,
    and ensures all necessary fields are present for BDA algorithm execution.
    Automatically adds missing optional parameters with default values.
    
    Parameters
    ----------
    config_path : str
        Absolute or relative path to JSON configuration file
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing validated BDA configuration parameters
        
    Raises
    ------
    FileNotFoundError
        When the specified configuration file does not exist
    ValueError
        When JSON format is invalid or required fields are missing
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"BDA config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check presence of required scientific parameters
        required_fields = ['decorr_factor', 'frequency_hz', 'declination_deg']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in BDA config")
        
        return validate_bda_config(config)
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in BDA config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading BDA config: {e}")


def validate_bda_config(config: Dict[str, float]) -> Dict[str, float]:
    """
    Validate and sanitize BDA configuration parameters.
    
    Performs scientific validation of BDA parameters following Wijnholds et al. 2018
    constraints and ensures all values are within physically reasonable ranges.
    
    Parameters
    ----------
    config : Dict[str, float]
        Raw BDA configuration dictionary
        
    Returns
    -------
    Dict[str, float]
        Validated and sanitized configuration
        
    Raises
    ------
    ValueError
        If any parameter is outside scientifically valid range
    """
    validated = config.copy()
    
    # Validate decorrelation factor (0 < R ≤ 1)
    if not (0.0 < validated['decorr_factor'] <= 1.0):
        raise ValueError(f"decorr_factor must be in (0,1], got {validated['decorr_factor']}")
    
    # Validate frequency (reasonable radio astronomy range)
    if not (10e6 <= validated['frequency_hz'] <= 100e9):  # 10 MHz to 100 GHz
        raise ValueError(f"frequency_hz outside reasonable range: {validated['frequency_hz']} Hz")
    
    # Validate declination (-90° to +90°)
    if not (-90.0 <= validated['declination_deg'] <= 90.0):
        raise ValueError(f"declination_deg must be in [-90,90], got {validated['declination_deg']}")
    
    # Validate safety_factor range and warn about non-standard values
    safety_factor = validated.get('safety_factor', 0.8)
    if not (0.1 <= safety_factor <= 2.0):
        raise ValueError(f"safety_factor should be in [0.1, 2.0], got {safety_factor}")
    
    if safety_factor > 1.0:
        logging.warning(f"safety_factor > 1.0 ({safety_factor}) makes BDA less conservative - "
                       f"averaging times will be longer than theoretical optimum") 
    elif safety_factor < 0.5:
        logging.warning(f"safety_factor < 0.5 ({safety_factor}) is very conservative - "
                       f"may significantly reduce data compression benefits")

    return validated