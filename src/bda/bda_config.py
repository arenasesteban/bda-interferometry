import json
from pathlib import Path
import traceback

from .bda_core import calculate_amplitude_loss, calculate_loss_exact, calculate_threshold_loss 


def load_bda_config(config_path):
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"BDA config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check presence of required scientific parameters
        required_fields = ['decorr_limit', 'max_time_window']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in BDA config")
        
        validated  = validate_bda_config(config)
        updated = update_bda_config(validated)

        with open(config_file, 'w') as f:
            json.dump(updated, f, indent=4)

        return updated
        
    except Exception as e:
        print(f"Error loading BDA config: {e}")
        traceback.print_exc()
        raise


def validate_bda_config(config):
    try:
        validated = config.copy()

        # Validate decorrelation limit (0 < δ ≤ 1)
        if not (0.0 < validated['decorr_limit'] <= 1.0):
            raise ValueError(f"decorr_limit must be in (0,1], got {validated['decorr_limit']}")

        # Validate max time window (0 < τ ≤ 3600)
        if not (0.0 < validated['max_time_window'] <= 3600.0):
            raise ValueError(f"max_time_window must be in (0,3600], got {validated['max_time_window']}")

        return validated

    except Exception as e:
        print(f"Error validating BDA config: {e}")
        traceback.print_exc()
        raise


def update_bda_config(config):
    try:
        x_exact = calculate_loss_exact(config['decorr_limit'])
        x_adjusted = calculate_threshold_loss(x_exact)
        x = calculate_amplitude_loss(x_adjusted)

        updated = config.copy()
        updated['x'] = x

        return updated

    except Exception as e:
        print(f"Error updating BDA config: {e}")
        traceback.print_exc()
        raise