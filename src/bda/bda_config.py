import json
import traceback
from pathlib import Path


def load_bda_config(config_path):
    """
    Load and validate BDA configuration from a JSON file.
    
    Parameters
    ----------
    config_path : str
        Path to the BDA configuration JSON file.
    
    Returns
    -------
    dict
        Validated BDA configuration.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"BDA config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check presence of required scientific parameters
        required_fields = ['decorr_factor', 'fov']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in BDA config")

        return validate_bda_config(config)
        
    except Exception as e:
        print(f"Error loading BDA config: {e}")
        traceback.print_exc()
        raise


def validate_bda_config(config):
    """
    Validate BDA configuration parameters.
    
    Parameters
    ----------
    config : dict
        BDA configuration dictionary.

    Returns
    -------
    dict
        Validated BDA configuration.
    """
    try:
        validated = config.copy()

        if not (0.0 < validated['decorr_factor'] <= 1.0):
            raise ValueError("decorr_factor must be in the range (0.0, 1.0]")

        return validated

    except Exception as e:
        print(f"Error validating BDA config: {e}")
        traceback.print_exc()
        raise
