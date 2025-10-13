"""
BDA Configuration Management

Provides configuration loading and validation functions for baseline-dependent averaging
parameters. Handles JSON file parsing, default value provision, and fallback mechanisms
for robust configuration management in streaming interferometry processing.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Union


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
        
        # Apply defaults for optional parameters following Wijnholds et al. 2018
        optional_defaults = {
            'safety_factor': 0.8,                    # Conservative safety margin
            'min_window_s': 1.0,                     # Minimum averaging window (hard limit)
            'max_window_s': 180.0,                   # Maximum averaging window (hard limit)
            'correlator_dump_s': 0.14,               # SKA1 correlator dump time from paper
            'site_lat_deg': -30.7,                   # SKA-mid latitude (Karoo, South Africa)
            'avg_scheme': 'scheme1_bins',             # Default to Scheme 1 from paper
            'max_multiplier': 512,                   # Maximum averaging multiplier for Scheme 2
            'field_offset_deg': 2.0,                 # Field-of-view edge offset for decorr calculation
            'source_declination_deg': None           # Override for specific source declination
        }
        
        for field, default_value in optional_defaults.items():
            if field not in config:
                config[field] = default_value
        
        # Validate decorr_factor is the target decorrelation factor at field edge
        if not (0.0 < config['decorr_factor'] <= 1.0):
            raise ValueError(f"decorr_factor must be between 0 and 1, got {config['decorr_factor']}")
        
        # Validate averaging scheme
        valid_schemes = ['scheme1_bins', 'scheme2_cap', 'custom']
        if config['avg_scheme'] not in valid_schemes:
            raise ValueError(f"avg_scheme must be one of {valid_schemes}, got {config['avg_scheme']}")
        
        # Validate the complete configuration
        validated_config = validate_bda_config(config)
        
        return validated_config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in BDA config file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading BDA config: {e}")


def get_default_bda_config() -> Dict[str, float]:
    """
    Provide default baseline-dependent averaging configuration parameters.
    
    Returns scientifically validated default values suitable for SKA1-mid
    interferometry observations based on Wijnholds et al. 2018 methodology.
    Values are optimized for southern hemisphere observations with conservative
    safety margins to ensure decorrelation stays below 1% at field edge.
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing complete BDA configuration following paper specifications:
        - decorr_factor: Target decorrelation at field edge (R in equations 41-43)
        - frequency_hz: Observation frequency for wavelength conversions
        - declination_deg: Source declination for Earth rotation effects
        - safety_factor: Conservative factor applied to decorrelation tolerance
          (<1 shortens averaging time for more conservative BDA, >1 lengthens time)
        - Window limits and correlator parameters from SKA1 specifications
        - Averaging scheme selection (Scheme 1 or 2 from paper)
    """
    return {
        # === CORE DECORRELATION PARAMETERS (Wijnholds et al. 2018) ===
        'decorr_factor': 0.95,              # Target R at 2° field offset (eq. 41)
        'frequency_hz': 700e6,               # SKA1-mid L-band center (700 MHz)
        'declination_deg': -45.0,            # Southern hemisphere typical declination
        'safety_factor': 0.8,                # Conservative factor (<1 shortens averaging time)
        
        # === TEMPORAL WINDOW CONSTRAINTS ===
        'min_window_s': 1.0,                 # Hard minimum averaging time (seconds)
        'max_window_s': 180.0,               # Hard maximum averaging time (seconds)
        'correlator_dump_s': 0.14,           # SKA1 correlator dump time (paper Sec. 4.2)
        
        # === SITE AND OBSERVING PARAMETERS ===
        'site_lat_deg': -30.7,               # SKA-mid latitude (Karoo, South Africa)
        'field_offset_deg': 2.0,             # Field-of-view edge for decorr calculation
        'source_declination_deg': None,      # Override for specific source (if different)
        
        # === BDA AVERAGING SCHEMES (Paper Sec. 4.3-4.5) ===
        'avg_scheme': 'scheme1_bins',         # 'scheme1_bins' | 'scheme2_cap' | 'custom'
        'max_multiplier': 512,               # Maximum averaging factor (Scheme 2)
        
        # === BASELINE ZONING (Scheme 1 from paper) ===
        'baseline_zones_km': [80, 40, 30, 20, 15, 10, 7.5, 5, 3.75, 2.5, 1.875, 1.25, 
                              0.9375, 0.625, 0.5625, 0.375, 0.28125],
        'averaging_multipliers': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 
                                 192, 256, 384, 512]
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
        default_config = get_default_bda_config()
        return validate_bda_config(default_config)


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
    
    # Validate window constraints
    if validated['min_window_s'] >= validated['max_window_s']:
        raise ValueError(f"min_window_s ({validated['min_window_s']}) >= max_window_s ({validated['max_window_s']})")
    
    # Validate correlator dump time
    if not (0.001 <= validated['correlator_dump_s'] <= 10.0):  # 1ms to 10s
        raise ValueError(f"correlator_dump_s outside reasonable range: {validated['correlator_dump_s']} s")
    
    # Validate averaging scheme
    valid_schemes = ['scheme1_bins', 'scheme2_cap', 'custom']
    if validated['avg_scheme'] not in valid_schemes:
        raise ValueError(f"avg_scheme must be one of {valid_schemes}")
    
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
    
    # Ensure baseline zones and multipliers have same length (for Scheme 1)
    if validated['avg_scheme'] == 'scheme1_bins':
        zones = validated.get('baseline_zones_km', [])
        multipliers = validated.get('averaging_multipliers', [])
        if len(zones) != len(multipliers):
            logging.warning(f"Baseline zones ({len(zones)}) and multipliers ({len(multipliers)}) length mismatch")
    
    return validated


def get_averaging_time_for_baseline(baseline_length_km: float, config: Dict[str, float]) -> float:
    """
    Get averaging time for specific baseline using configured scheme.
    
    Implements the baseline-dependent averaging schemes from Wijnholds et al. 2018
    (Scheme 1: binned zones, Scheme 2: continuous with cap).
    
    Parameters
    ----------
    baseline_length_km : float
        Baseline length in kilometers
    config : Dict[str, float]
        Validated BDA configuration
        
    Returns
    -------
    float
        Averaging time in seconds for this baseline
    """
    scheme = config['avg_scheme']
    correlator_dump = config['correlator_dump_s']
    min_time = config['min_window_s']
    max_time = config['max_window_s']
    
    if scheme == 'scheme1_bins':
        # Scheme 1: Discrete baseline zones with fixed multipliers
        zones = config.get('baseline_zones_km', [])
        multipliers = config.get('averaging_multipliers', [])
        
        if not zones or not multipliers:
            # Fallback to simple inverse relationship
            multiplier = max(1, int(100.0 / max(baseline_length_km, 1.0)))
        else:
            # Find appropriate zone
            multiplier = 1
            for i, zone_limit in enumerate(zones):
                if baseline_length_km >= zone_limit:
                    multiplier = multipliers[i] if i < len(multipliers) else multipliers[-1]
                    break
            else:
                # Shorter than shortest zone - use maximum multiplier
                multiplier = multipliers[-1] if multipliers else 512
        
        averaging_time = correlator_dump * multiplier
        
    elif scheme == 'scheme2_cap':
        # Scheme 2: Continuous scaling with maximum cap
        max_multiplier = config.get('max_multiplier', 512)
        reference_length = 80.0  # km - longest baselines get multiplier 1
        
        if baseline_length_km >= reference_length:
            multiplier = 1
        else:
            # Continuous scaling: shorter baselines get higher multipliers
            multiplier = min(max_multiplier, int(reference_length / max(baseline_length_km, 0.1)))
        
        averaging_time = correlator_dump * multiplier
        
    else:  # custom
        # Simple inverse relationship
        multiplier = max(1, int(100.0 / max(baseline_length_km, 1.0)))
        averaging_time = correlator_dump * multiplier
    
    # Apply hard limits
    averaging_time = np.clip(averaging_time, min_time, max_time)
    
    return averaging_time


def create_example_bda_config(output_path: str, telescope: str = 'ska1_mid') -> None:
    """
    Create example BDA configuration file for different telescopes.
    
    Generates scientifically validated JSON configuration files with parameters
    optimized for specific telescope arrays following Wijnholds et al. 2018.
    
    Parameters
    ----------
    output_path : str
        Path where to save the example configuration file
    telescope : str, optional
        Telescope configuration: 'ska1_mid', 'alma', 'vla' (default: 'ska1_mid')
    """
    telescope_configs = {
        'ska1_mid': {
            'decorr_factor': 0.95,
            'frequency_hz': 700e6,              # L-band
            'declination_deg': -45.0,
            'safety_factor': 0.8,
            'min_window_s': 1.0,
            'max_window_s': 180.0,
            'correlator_dump_s': 0.14,          # From paper
            'site_lat_deg': -30.7,              # Karoo
            'field_offset_deg': 2.0,
            'avg_scheme': 'scheme1_bins',
            'max_multiplier': 512
        },
        'alma': {
            'decorr_factor': 0.98,
            'frequency_hz': 100e9,              # Band 3
            'declination_deg': -23.0,           # Atacama
            'safety_factor': 0.9,
            'min_window_s': 0.5,
            'max_window_s': 60.0,
            'correlator_dump_s': 0.1,
            'site_lat_deg': -24.6,
            'field_offset_deg': 1.0,
            'avg_scheme': 'scheme2_cap',
            'max_multiplier': 256
        },
        'vla': {
            'decorr_factor': 0.95,
            'frequency_hz': 1.4e9,              # L-band
            'declination_deg': 34.0,            # New Mexico
            'safety_factor': 0.8,
            'min_window_s': 1.0,
            'max_window_s': 120.0,
            'correlator_dump_s': 0.1,
            'site_lat_deg': 34.1,
            'field_offset_deg': 2.0,
            'avg_scheme': 'scheme1_bins',
            'max_multiplier': 256
        }
    }
    
    if telescope not in telescope_configs:
        raise ValueError(f"Unknown telescope: {telescope}. Available: {list(telescope_configs.keys())}")
    
    config = telescope_configs[telescope]
    
    # Add metadata
    config_with_metadata = {
        "_metadata": {
            "description": f"BDA configuration for {telescope.upper()}",
            "reference": "Wijnholds et al. 2018 - Baseline Dependent Averaging",
            "telescope": telescope,
            "created_by": "bda_config.create_example_bda_config()"
        },
        **config
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(config_with_metadata, f, indent=2)
    
    logging.info(f"Example BDA config for {telescope} saved to: {output_path}")
