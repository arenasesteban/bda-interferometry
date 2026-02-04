import traceback

from .amplitude import amplitude, calculate_amplitude_error
from .rms import rms, calculate_rms_measure
from .baseline import baseline_dependency, validate_baseline_dependency
from .coberture import coberture_uv, calculate_coberture_uv


def calculate_metrics(df_scientific, df_averaging, num_partitions):
    try:
        df_scientific = df_scientific.repartition(num_partitions * 3, 'baseline_key', 'window_id')
        df_averaging = df_averaging.repartition(num_partitions * 3, 'baseline_key', 'window_id')

        df_coberture_uv = coberture_uv(df_scientific, df_averaging)

        df_baseline_dependency = baseline_dependency(df_averaging)
                
        df_amplitude = amplitude(df_scientific, df_averaging)

        df_rms = rms(df_scientific, df_averaging)
        
        return df_amplitude, df_rms, df_baseline_dependency, df_coberture_uv

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        traceback.print_exc()
        raise


def consolidate_metrics(df_amplitude, df_rms, df_baseline_dependency, df_coberture_uv, bda_config):
    try:
        calculate_amplitude_error(df_amplitude, bda_config)

        calculate_rms_measure(df_rms, bda_config)

        validate_baseline_dependency(df_baseline_dependency, bda_config)

        calculate_coberture_uv(df_coberture_uv)

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        traceback.print_exc()
        raise