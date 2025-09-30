"""
BDA Package - Baseline-Dependent Averaging Implementation

Baseline-Dependent Averaging implementation for radio interferometry data processing
in streaming environments. Based on Wijnholds et al. 2018.
"""

try:
    from .bda_core import (
        create_bda_config,
        apply_bda_to_group
    )
    from .bda_processor import (
        process_microbatch_with_bda,
        format_bda_result_for_output,
        create_bda_summary_stats
    )
    from .bda_config import (
        load_bda_config_with_fallback,
        get_default_bda_config
    )
    
except ImportError as e:
    print(f"Warning: BDA module imports failed: {e}")
    
    def process_microbatch_with_bda(*args, **kwargs):
        return []
    
    def load_bda_config_with_fallback(*args, **kwargs):
        return {'decorr_factor': 0.95, 'frequency_hz': 42.5e9, 'declination_deg': -45.0}

__all__ = [
    'create_bda_config',
    'apply_bda_to_group',
    'process_microbatch_with_bda',
    'format_bda_result_for_output',
    'create_bda_summary_stats',
    'load_bda_config_with_fallback',
    'get_default_bda_config'
]
