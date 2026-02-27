import traceback
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DoubleType, LongType, StringType,
)


def rms(df_scientific, df_averaging):
    try:
        df_combined = combine_for_rms(df_scientific, df_averaging)

        df_rms = calculate_rms_error(df_combined)
        
        return df_rms
    
    except Exception as e:
        print(f"Error calculating RMS: {e}")
        traceback.print_exc()
        raise


def combine_for_rms(df_scientific, df_averaging):
    try:
        df_sci = df_scientific.select(
            "baseline_key", "window_id",
            df_scientific["visibility"].alias("visibility_scientific"),
            df_scientific["flag"].alias("flag_scientific"),
        )

        df_avg = df_averaging.select(
            "baseline_key", "window_id",
            df_averaging["visibility"].alias("visibility_averaging"),
            df_averaging["flag"].alias("flag_averaging"),
        )

        df_combined = df_sci.join(
            df_avg,
            on=["baseline_key", "window_id"],
            how="inner",
        )

        return df_combined

    except Exception as e:
        print(f"Error combining dataframes for RMS: {e}")
        traceback.print_exc()
        raise


def calculate_rms_error(df_combined):
    try:
        schema = StructType([
            StructField("baseline_key",       StringType(), False),
            StructField("window_id",          LongType(),   False),
            StructField("squared_difference", DoubleType(), False),
            StructField("squared_scientific", DoubleType(), False),
            StructField("denominator",        LongType(),   False),
        ])

        def compute_rms(pdf):
            baseline_key = pdf["baseline_key"].iloc[0]
            window_id = pdf["window_id"].iloc[0]
            squared_difference = 0.0
            squared_scientific = 0.0
            denominator = 0

            vs_avg = np.array([[corr for corr in chan] for chan in pdf["visibility_averaging"].iloc[0]], dtype=np.float64)  # (C, P, 2)
            fs_avg = np.array([[flag for flag in chan] for chan in pdf["flag_averaging"].iloc[0]], dtype=np.bool_)          # (C, P)

            for vs_sci_data, flag_sci_data in zip(
                pdf["visibility_scientific"],
                pdf["flag_scientific"],
            ):
                vs_sci = np.array([[corr for corr in chan] for chan in vs_sci_data], dtype=np.float64)  # (C, P, 2)
                fs_sci= np.array([[flag for flag in chan] for chan in flag_sci_data], dtype=np.bool_)   # (C, P)

                fs_combined = fs_sci | fs_avg
                valid = ~fs_combined

                if not valid.any():
                    continue

                diff = vs_avg - vs_sci

                diff_sq = diff[..., 0] ** 2 + diff[..., 1] ** 2
                sci_sq  = vs_sci[..., 0] ** 2 + vs_sci[..., 1] ** 2

                squared_difference += float(diff_sq[valid].sum())
                squared_scientific += float(sci_sq[valid].sum())
                denominator += int(valid.sum())

            return pd.DataFrame([{
                "baseline_key": baseline_key,
                "window_id": window_id,
                "squared_difference": squared_difference,
                "squared_scientific": squared_scientific,
                "denominator": denominator,
            }])

        df_rms = df_combined.groupBy("baseline_key", "window_id").applyInPandas(
            compute_rms, schema=schema
        )

        return df_rms

    except Exception as e:
        print(f"Error calculating RMS: {e}")
        traceback.print_exc()
        raise


def sum_rms(df_rms):
    try:
        results = df_rms.agg(
            F.sum("squared_difference").alias("squared_difference"),
            F.sum("squared_scientific").alias("squared_scientific"),
            F.sum("denominator").alias("denominator"),
        ).collect()[0]

        return (
            results["squared_difference"],
            results["squared_scientific"],
            results["denominator"],
        )

    except Exception as e:
        print(f"Error summing RMS values: {e}")
        traceback.print_exc()
        raise


def write_rms_results(
    squared_difference, squared_scientific, denominator,
    rms_absolute, rms_relative, rms_relative_percent,
    tolerance, passed,
    output_file,
):
    try:
        with open(output_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"RMS Error in Visibility Domain (BDA)\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Equations:\n")
            f.write(f"  RMS_abs = sqrt( Σ_i |V_BDA(w_i) - V_i|² / N_valid )\n")
            f.write(f"  RMS_rel = sqrt( Σ_i |V_BDA(w_i) - V_i|² / Σ_i |V_i|² )\n")
            f.write(f"  V_i     = visibilidad científica individual\n")
            f.write(f"  V_BDA   = promedio ponderado de la ventana que contiene V_i\n")
            f.write(f"  (suma sobre visibilidades válidas en ambos datasets)\n")
            f.write(f"\nAggregated Values:\n")
            f.write(f"  Σ|V_BDA - V_i|²:     {squared_difference:.6e}\n")
            f.write(f"  Σ|V_i|²:             {squared_scientific:.6e}\n")
            f.write(f"  N valid:             {denominator:,}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"  Absolute RMS:        {rms_absolute:.6e}\n")
            f.write(f"    → desviación media por visibilidad respecto al promedio BDA\n")
            f.write(f"  Relative RMS:        {rms_relative:.6f} ({rms_relative_percent:.4f}%)\n")
            f.write(f"    → fracción de la potencia de señal perdida por el promediado\n")
            f.write(f"  Tolerance:           {tolerance:.6f} ({tolerance * 100:.4f}%)\n")
            f.write(f"  Status:              {'✓ PASSED' if passed else '✗ FAILED'}\n")
            f.write(f"{'=' * 80}\n")

    except Exception as e:
        print(f"Error writing RMS results: {e}")
        traceback.print_exc()
        raise


def calculate_rms_measure(df_rms, bda_config, output_file):
    try:
        tolerance = bda_config.get("rms_tolerance", 0.01)  # 1% por defecto

        squared_difference, squared_scientific, denominator = sum_rms(df_rms)

        if denominator == 0:
            raise ValueError(
                "denominator=0: no hay visibilidades válidas para calcular el RMS. "
                "Revisa los flags de entrada y la alineación de window_id."
            )
        if squared_scientific == 0.0:
            raise ValueError(
                "squared_scientific=0: la potencia de referencia es cero; "
                "el RMS relativo no está definido."
            )

        rms_absolute = np.sqrt(squared_difference / denominator)
        rms_relative = np.sqrt(squared_difference / squared_scientific)
        rms_relative_percent = rms_relative * 100.0
        passed = rms_relative <= tolerance

        write_rms_results(
            squared_difference, squared_scientific, denominator,
            rms_absolute, rms_relative, rms_relative_percent,
            tolerance, passed,
            output_file,
        )

        print("[Evaluation] ✓ RMS Error completed successfully.")

    except Exception as e:
        print(f"Error calculating RMS measure: {e}")
        traceback.print_exc()
        raise