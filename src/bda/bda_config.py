"""
BDA Configuration Management

Provides configuration loading and validation functions for baseline-dependent averaging
parameters. Handles JSON file parsing, default value provision, and fallback mechanisms
for robust configuration management in streaming interferometry processing.
"""

import json
from pathlib import Path
from typing import Dict
import traceback


def load_bda_config(config_path: str) -> Dict[str, float]:
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"BDA config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check presence of required scientific parameters
        required_fields = ['decorr_factor', 'field_offset_deg']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in BDA config")
        
        return validate_bda_config(config)
        
    except Exception as e:
        print(f"Error loading BDA config: {e}")
        traceback.print_exc()
        raise


def validate_bda_config(config: Dict[str, float]) -> Dict[str, float]:
    try:
        validated = config.copy()
        
        # Validate decorrelation factor (0 < R ≤ 1)
        if not (0.0 < validated['decorr_factor'] <= 1.0):
            raise ValueError(f"decorr_factor must be in (0,1], got {validated['decorr_factor']}")
        
        # Validate field offset in degrees (0 < θ ≤ 90)
        if not (0.0 < validated['field_offset_deg'] <= 90.0):
            raise ValueError(f"field_offset_deg must be in (0,90], got {validated['field_offset_deg']}")

        return validated

    except Exception as e:
        print(f"Error validating BDA config: {e}")
        traceback.print_exc()
        raise
