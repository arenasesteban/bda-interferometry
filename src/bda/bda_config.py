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
        
        # Apply default value for optional safety factor parameter
        if 'safety_factor' not in config:
            config['safety_factor'] = 0.8
        
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in BDA config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading BDA config: {e}")


def get_default_bda_config() -> Dict[str, float]:
    """
    Provide default baseline-dependent averaging configuration parameters.
    
    Returns scientifically validated default values suitable for ALMA Band 1
    interferometry observations in the southern hemisphere. Values are based
    on typical observing parameters and conservative safety margins.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing default BDA configuration with decorrelation factor,
        observation frequency, source declination, and safety factor parameters
    """
    return {
        'decorr_factor': 0.95,        # Decorrelation factor for visibility coherence
        'frequency_hz': 42.5e9,   # ALMA Band 1 center frequency
        'declination_deg': -45.0, # Southern hemisphere source declination
        'safety_factor': 0.8       # Conservative safety margin for averaging
    }


def load_bda_config_with_fallback(config_path: str) -> Dict[str, float]:
    """
    Load BDA configuration with automatic fallback to default values.
    
    Attempts to load configuration from specified file path. If loading fails
    due to file not found, invalid JSON, or missing parameters, automatically
    falls back to scientifically validated default values to ensure system
    continues operation.
    
    Parameters
    ----------
    config_path : str
        Path to JSON configuration file to attempt loading
        
    Returns
    -------
    Dict[str, float]
        Successfully loaded configuration from file, or default configuration
        if loading fails for any reason
    """
    try:
        return load_bda_config(config_path)
    except Exception as e:
        logging.warning(f"Could not load BDA config from {config_path}: {e}")
        logging.info("Using default BDA configuration...")
        return get_default_bda_config()
