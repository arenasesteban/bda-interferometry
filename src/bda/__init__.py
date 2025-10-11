"""
BDA Package - Baseline-Dependent Averaging Implementation

Baseline-Dependent Averaging implementation for radio interferometry data processing
in streaming environments. Based on Wijnholds et al. 2018.
"""

try:
    from .bda_core import apply_bda_to_group
    from .bda_processor import process_group_with_bda
    from .bda_config import (
        load_bda_config_with_fallback,
        get_default_bda_config
    )
    
except ImportError as e:
    import logging
    logging.warning(f"BDA module imports failed: {e}")
    
    def process_group_with_bda(*args, **kwargs):
        return []
    
    def load_bda_config_with_fallback(*args, **kwargs):
        return {'decorr_factor': 0.95, 'frequency_hz': 42.5e9, 'declination_deg': -45.0}

__all__ = [
    # Core BDA functions
    'apply_bda_to_group',
    
    # Processing functions
    'process_group_with_bda',
    
    # Configuration functions
    'load_bda_config_with_fallback',
    'get_default_bda_config'
]
