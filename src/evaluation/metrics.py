import traceback

from .amplitude import amplitude, calculate_amplitude_error
from .rms import rms, calculate_rms_measure
from .baseline import validate_baseline_dependency
from .coverage import calculate_coverage_uv


def calculate_metrics(df_scientific, df_averaging, bda_config, slurm_job_id):
    try:
        df_scientific, df_averaging = prepare_dataframes(df_scientific, df_averaging)

        df_amplitude = amplitude(df_scientific, df_averaging)
        df_rms = rms(df_scientific, df_averaging)

        output_metrics = f"./output/metrics_{slurm_job_id}.txt"
        output_coverage = f"./output/coverage_uv_{slurm_job_id}.png"

        print(f"[Evaluation] Calculating amplitude error")
        calculate_amplitude_error(df_amplitude, bda_config, output_metrics)

        print(f"[Evaluation] Calculating RMS measure")
        calculate_rms_measure(df_rms, bda_config, output_metrics)

        print(f"[Evaluation] Validating baseline dependency")
        validate_baseline_dependency(df_scientific, df_averaging, bda_config, output_metrics)

        print(f"[Evaluation] Calculating UV coverage")
        calculate_coverage_uv(df_scientific, df_averaging, output_coverage)

    except Exception as e:
        print(f"[Evaluation] Error calculating metrics: {e}")
        traceback.print_exc()
        raise

    finally:
        df_scientific.unpersist()
        df_averaging.unpersist()


def prepare_dataframes(df_scientific, df_averaging):
    try:
        cols = ["baseline_key", "window_id", "u", "v", "visibility", "flag"]
        df_scientific = df_scientific.select(cols)
        df_averaging = df_averaging.select(cols)

        df_scientific.persist()
        df_averaging.persist()

        df_scientific.count()
        df_averaging.count()

        return df_scientific, df_averaging

    except Exception as e:
        print(f"[Evaluation] Error preparing dataframes: {e}")
        traceback.print_exc()
        raise
