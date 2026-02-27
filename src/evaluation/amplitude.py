import traceback
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType


def amplitude(df_scientific, df_averaging):
    try:
        df_amplitude_scientific = calculate_amplitude(df_scientific)
        df_amplitude_averaging = calculate_amplitude(df_averaging)

        df_union_amplitude = union_amplitude(df_amplitude_scientific, df_amplitude_averaging)

        return df_union_amplitude

    except Exception as e:
        print(f"Error calculating amplitude: {e}")
        traceback.print_exc()
        raise


def calculate_amplitude(df_scientific):
    try:
        schema = StructType([
            StructField("baseline_key",   StringType(), False),
            StructField("window_id",      LongType(),   False),
            StructField("sum_amplitude",  DoubleType(), False),
            StructField("n_valid",        LongType(),   False),
        ])

        def compute_amplitude(pdf):
            baseline_key = pdf['baseline_key'].iloc[0]
            window_id = pdf['window_id'].iloc[0]
            sum_amplitude = 0.0
            n_valid = 0

            for vs_data, fs_data in zip(pdf["visibility"], pdf["flag"]):
                vs = np.array([[corr for corr in chan] for chan in vs_data], dtype=np.float64)  # (C, P, 2)
                fs = np.array([[flag for flag in chan] for chan in fs_data], dtype=np.bool_)    # (C, P)

                valid = ~fs
                real = vs[..., 0]
                imag = vs[..., 1]

                amp_per_sample = np.sqrt(real**2 + imag**2)  # (C, P)
                
                sum_amplitude += float(amp_per_sample[valid].sum())
                n_valid += int(valid.sum())

            return pd.DataFrame([{
                "baseline_key": baseline_key,
                "window_id": window_id,
                "sum_amplitude": sum_amplitude,
                "n_valid": n_valid,
            }])
        
        df_amplitude = df_scientific.groupBy('baseline_key', 'window_id').applyInPandas(
            compute_amplitude, schema=schema
        )

        return df_amplitude

    except Exception as e:
        print(f"Error calculating amplitude: {e}")
        traceback.print_exc()
        raise


def union_amplitude(df_amplitude_scientific, df_amplitude_averaging):
    try:
        df_scientific_marked = df_amplitude_scientific.select(
            F.lit("scientific").alias("type"),
            F.col("sum_amplitude"),
            F.col("n_valid"),
        )
        
        df_averaging_marked = df_amplitude_averaging.select(
            F.lit("averaging").alias("type"),
            F.col("sum_amplitude"),
            F.col("n_valid"),
        )
        
        df_union = df_scientific_marked.unionByName(df_averaging_marked)

        df_aggregated = df_union.groupBy("type").agg(
            F.sum("sum_amplitude").alias("sum_amplitude"),
            F.sum("n_valid").alias("n_valid"),
        )
        
        df_pivoted = df_aggregated.groupBy().pivot("type").agg(
            F.first("sum_amplitude").alias("sum_amplitude"),
            F.first("n_valid").alias("n_valid"),
        )
        
        df_result = df_pivoted.select(
            F.col("scientific_sum_amplitude").alias("sum_amplitude_scientific"),
            F.col("scientific_n_valid").alias("n_valid_scientific"),
            F.col("averaging_sum_amplitude").alias("sum_amplitude_averaging"),
            F.col("averaging_n_valid").alias("n_valid_averaging"),
        )

        return df_result
    
    except Exception as e:
        print(f"Error unioning amplitudes: {e}")
        traceback.print_exc()
        raise


def sum_amplitudes(df_amplitude):
    try:
        results = df_amplitude.collect()[0]
        
        return (
            results['sum_amplitude_scientific'],
            results['n_valid_scientific'],
            results['sum_amplitude_averaging'],
            results['n_valid_averaging'],
        )
    except Exception as e:
        print(f"Error summing amplitudes: {e}")
        traceback.print_exc()
        raise


def write_amplitude_results(
    results_amplitude,
    mean_amp_scientific,
    mean_amp_averaging,
    relative_error,
    relative_error_percent,
    tolerance,
    passed,
    output_file,
):
    try:
        (sum_amp_sci, n_valid_sci,
         sum_amp_avg, n_valid_avg) = results_amplitude

        with open(output_file, "a") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"Amplitude Error (BDA)\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Equation:\n")
            f.write(f"  |<A_BDA> - <A_orig>| / |<A_orig>|\n")
            f.write(f"  donde <A> = (Σ |V_i|) / N  (amplitud media)\n")
            f.write(f"\n")
            f.write(f"Original Data:\n")
            f.write(f"  Σ |V_i|:         {sum_amp_sci:.6e}\n")
            f.write(f"  N valid:         {n_valid_sci:,}\n")
            f.write(f"  Mean Amplitude:  {mean_amp_scientific:.6e}\n")
            f.write(f"\n")
            f.write(f"Data with BDA:\n")
            f.write(f"  Σ |V_i|:         {sum_amp_avg:.6e}\n")
            f.write(f"  N valid:         {n_valid_avg:,}\n")
            f.write(f"  Mean Amplitude:  {mean_amp_averaging:.6e}\n")
            f.write(f"\n")
            f.write(f"Error Metric:\n")
            f.write(f"  Relative Error:  {relative_error:.6f} ({relative_error_percent:.4f}%)\n")
            f.write(f"  Tolerance:       {tolerance:.6f} ({tolerance * 100.0:.4f}%)\n")
            f.write(f"  Status:          {'✓ PASSED' if passed else '✗ FAILED'}\n")
            f.write(f"{'=' * 80}\n")
        
    except Exception as e:
        print(f"Error writing amplitude results: {e}")
        traceback.print_exc()
        raise


def calculate_amplitude_error(df_amplitude, bda_config, output_file):
    try:
        tolerance = bda_config.get('amplitude_tolerance', 0.1)
        results_amplitude = sum_amplitudes(df_amplitude)

        (sum_amp_sci, n_valid_sci,
         sum_amp_avg, n_valid_avg) = results_amplitude
        
        mean_amp_scientific = sum_amp_sci / n_valid_sci
        mean_amp_averaging = sum_amp_avg / n_valid_avg

        relative_error = abs(mean_amp_averaging - mean_amp_scientific) / mean_amp_scientific
        relative_error_percent = relative_error * 100.0
        passed = relative_error <= tolerance

        write_amplitude_results(
            results_amplitude,
            mean_amp_scientific,
            mean_amp_averaging,
            relative_error,
            relative_error_percent,
            tolerance,
            passed,
            output_file,
        )

        print("[Evaluation] ✓ Amplitude Error completed successfully.")

    except Exception as e:
        print(f"Error comparing amplitudes: {e}")
        traceback.print_exc()
        raise