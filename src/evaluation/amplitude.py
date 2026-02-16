import traceback
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType
import pandas as pd


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
            StructField("baseline_key", StringType(), False),
            StructField("window_id", LongType(), False),
            StructField("sum_real", DoubleType(), False),
            StructField("sum_imag", DoubleType(), False),
            StructField("n_valid", LongType(), False)
        ])

        def compute_amplitude(pdf):
            baseline_key = pdf['baseline_key'].iloc[0]
            window_id = pdf['window_id'].iloc[0]
            sum_real = 0.0
            sum_imag = 0.0
            n_valid = 0

            for vs_data, fs_data in zip(pdf["visibility"], pdf["flag"]):
                vs = np.array([[corr for corr in chan] for chan in vs_data], dtype=np.float64)     # (C, P, 2)
                fs = np.array([[flag for flag in chan] for chan in fs_data], dtype=np.bool_)       # (C, P)

                valid = ~fs

                real = vs[..., 0]
                imag = vs[..., 1]
                
                sum_real += float(real[valid].sum())
                sum_imag += float(imag[valid].sum())
                n_valid += int(valid.sum())

            return pd.DataFrame([{
                "baseline_key": baseline_key,
                "window_id": window_id,
                "sum_real": sum_real,
                "sum_imag": sum_imag,
                "n_valid": n_valid
            }])
        
        df_amplitude = df_scientific.groupBy('baseline_key', 'window_id').applyInPandas(
            compute_amplitude, 
            schema=schema
        )

        return df_amplitude

    except Exception as e:
        print(f"Error calculating amplitude: {e}")
        traceback.print_exc()
        raise


def union_amplitude(df_amplitude_scientific, df_amplitude_averaging):
    """
    Combina las sumas de componentes complejas de datos científicos y promediados.
    
    Resultado: Un DataFrame con:
        - sum_real_scientific, sum_imag_scientific, n_valid_scientific
        - sum_real_averaging, sum_imag_averaging, n_valid_averaging
    """
    try:
        df_scientific_marked = df_amplitude_scientific.select(
            F.lit("scientific").alias("type"),
            F.col("sum_real"),
            F.col("sum_imag"),
            F.col("n_valid")
        )
        
        df_averaging_marked = df_amplitude_averaging.select(
            F.lit("averaging").alias("type"),
            F.col("sum_real"),
            F.col("sum_imag"),
            F.col("n_valid")
        )
        
        df_union = df_scientific_marked.unionByName(df_averaging_marked)
        df_aggregated = df_union.groupBy("type").agg(
            F.sum("sum_real").alias("sum_real"),
            F.sum("sum_imag").alias("sum_imag"),
            F.sum("n_valid").alias("n_valid")
        )
        
        df_pivoted = df_aggregated.groupBy().pivot("type").agg(
            F.first("sum_real").alias("sum_real"),
            F.first("sum_imag").alias("sum_imag"),
            F.first("n_valid").alias("n_valid")
        )
        
        df_result = df_pivoted.select(
            F.col("scientific_sum_real").alias("sum_real_scientific"),
            F.col("scientific_sum_imag").alias("sum_imag_scientific"),
            F.col("scientific_n_valid").alias("n_valid_scientific"),
            F.col("averaging_sum_real").alias("sum_real_averaging"),
            F.col("averaging_sum_imag").alias("sum_imag_averaging"),
            F.col("averaging_n_valid").alias("n_valid_averaging")
        )

        return df_result
    
    except Exception as e:
        print(f"Error unioning amplitudes: {e}")
        traceback.print_exc()
        raise


def sum_amplitudes(df_amplitude):
    try:
        results = df_amplitude.collect()[0]
        
        return (results['sum_real_scientific'], results['sum_imag_scientific'], results['n_valid_scientific'],
                results['sum_real_averaging'], results['sum_imag_averaging'], results['n_valid_averaging'])

    except Exception as e:
        print(f"Error summing amplitudes: {e}")
        traceback.print_exc()
        raise


def write_amplitude_results(
    results_amplitude, 
    flux_scientific, flux_averaging, 
    relative_error, relative_error_percent, 
    tolerance, passed, 
    output_file
):
    try:
        with open(output_file, "a") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"Amplitude Error\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Equation:\n")
            f.write(f"  |A_BDA - A_orig| / |A_orig|\n")
            f.write(f"\n")
            f.write(f"Original Data:\n")
            f.write(f"  Σ Re:                {results_amplitude[0]:.6e}\n")
            f.write(f"  Σ Im:                {results_amplitude[1]:.6e}\n")
            f.write(f"  Total Flux:          {flux_scientific:.6e}\n")
            f.write(f"  N valid:             {results_amplitude[2]:,}\n")
            f.write(f"\n")
            f.write(f"Data with BDA:\n")
            f.write(f"  Σ Re:                {results_amplitude[3]:.6e}\n")
            f.write(f"  Σ Im:                {results_amplitude[4]:.6e}\n")
            f.write(f"  Total Flux:          {flux_averaging:.6e}\n")
            f.write(f"  N valid:             {results_amplitude[5]:,}\n")
            f.write(f"\n")
            f.write(f"Error Metric:\n")
            f.write(f"  Relative Error:      {relative_error:.6f} ({relative_error_percent:.4f}%)\n")
            f.write(f"  Tolerance:           {tolerance:.6f} ({tolerance * 100.0:.4f}%)\n")
            f.write(f"  Status:              {'✓ PASSED' if passed else '✗ FAILED'}\n")
            f.write(f"{'=' * 80}\n")

    except Exception as e:
        print(f"Error writing amplitude results: {e}")
        traceback.print_exc()
        raise


def calculate_amplitude_error(df_amplitude, bda_config, output_file):
    try:
        tolerance = bda_config.get('amplitude_tolerance', 0.1)

        results_amplitude = sum_amplitudes(df_amplitude)

        (sum_real_sci, sum_imag_sci, n_valid_sci,
         sum_real_avg, sum_imag_avg, n_valid_avg) = results_amplitude

        flux_scientific = np.sqrt(sum_real_sci**2 + sum_imag_sci**2)
        flux_averaging = np.sqrt(sum_real_avg**2 + sum_imag_avg**2)
        
        relative_error = abs(flux_averaging - flux_scientific) / abs(flux_scientific)
        relative_error_percent = relative_error * 100.0

        passed = relative_error <= tolerance

        write_amplitude_results(
            results_amplitude, 
            flux_scientific, flux_averaging, 
            relative_error, relative_error_percent, 
            tolerance, passed, 
            output_file
        )

        print("[Metrics] ✓ Amplitude Error completed successfully.")

    except Exception as e:
        print(f"Error comparing amplitudes: {e}")
        traceback.print_exc()
        raise