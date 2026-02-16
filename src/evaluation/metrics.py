import traceback

from evaluation.coverage import coverage_uv

from .amplitude import amplitude, calculate_amplitude_error
from .rms import rms, calculate_rms_measure
from .baseline import baseline_dependency, validate_baseline_dependency
from .coverage import coverage_uv, calculate_coverage_uv


def calculate_metrics(df_scientific, df_averaging, num_partitions):
    try:
        df_scientific = df_scientific.repartition(num_partitions * 2, 'baseline_key')
        df_averaging = df_averaging.repartition(num_partitions * 2, 'baseline_key')

        df_scientific, df_averaging = prepare_metrics(df_scientific, df_averaging)

        df_coverage_uv = coverage_uv(df_scientific, df_averaging)
        df_baseline_dependency = baseline_dependency(df_averaging)
        df_amplitude = amplitude(df_scientific, df_averaging)
        df_rms = rms(df_scientific, df_averaging)

        return df_amplitude, df_rms, df_baseline_dependency, df_coverage_uv, df_scientific, df_averaging

    except Exception as e:
        print(f"[Metrics] Error calculating metrics: {e}")
        traceback.print_exc()
        raise


def prepare_metrics(df_scientific, df_averaging):
    try:
        cols = ["baseline_key", "window_id", "u", "v", "visibility", "flag"]

        df_scientific = df_scientific.select(cols).persist()
        df_averaging = df_averaging.select(cols).persist()
        
        print("=" * 60)
        print("BDA reduction dataset")
        print(f"Rows (before BDA): {df_scientific.count()}")
        print(f"Rows (after BDA): {df_averaging.count()}")
        print("=" * 60)

        return df_scientific, df_averaging

    except Exception as e:
        print(f"[Metrics] Error preparing metrics: {e}")
        traceback.print_exc()
        raise


def consolidate_metrics(df_amplitude, df_rms, df_baseline_dependency, df_coverage_uv, bda_config, df_scientific, df_averaging, slurm_job_id):
    try:
        output_metrics = f"./output/metrics_{slurm_job_id}.txt"
        output_coverage = f"./output/coverage_uv_{slurm_job_id}.png"

        calculate_amplitude_error(df_amplitude, bda_config, output_metrics)
        calculate_rms_measure(df_rms, bda_config, output_metrics)
        validate_baseline_dependency(df_baseline_dependency, bda_config, output_metrics)
        calculate_coverage_uv(df_coverage_uv, output_coverage)

    except Exception as e:
        print(f"[Metrics] Error calculating metrics: {e}")
        traceback.print_exc()
        raise

    finally:
        try:
            if df_scientific is not None:
                df_scientific.unpersist()
            if df_averaging is not None:
                df_averaging.unpersist()

        except Exception as e:
            pass
