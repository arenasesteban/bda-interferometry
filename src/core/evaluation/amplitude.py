import traceback
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType


def amplitude(df_scientific, df_averaging):
    """
    Calcula el error de amplitud introducido por BDA comparando cada
    visibilidad científica individual contra la visibilidad promediada
    de su ventana.

    Mide exclusivamente la pérdida de amplitud escalar (módulo),
    ignorando la fase. Complementa al RMS, que opera en el dominio
    complejo e incluye variaciones de fase.

    Interpretación conjunta con RMS:
      AE_rel ≈ RMS_rel  →  el error es principalmente de amplitud
      RMS_rel >> AE_rel →  el error es principalmente de fase
    """
    try:
        df_combined  = combine_for_amplitude(df_scientific, df_averaging)
        df_amplitude = calculate_amplitude_error_paired(df_combined)
        return df_amplitude
    except Exception as e:
        print(f"Error calculating amplitude: {e}")
        traceback.print_exc()
        raise


def combine_for_amplitude(df_scientific, df_averaging):
    """
    Une cada visibilidad científica individual con la visibilidad BDA
    de su ventana. Join 1-a-muchos: una fila BDA se repite para cada
    integración científica de su ventana.

    Claves de join: (baseline_key, window_id).
    """
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

        n_sci = df_sci.count()
        n_avg = df_avg.count()

        df_combined = df_sci.join(
            df_avg,
            on=["baseline_key", "window_id"],
            how="inner",
        )

        n_joined = df_combined.count()

        if n_joined < n_sci:
            print(
                f"[WARNING] combine_for_amplitude: filas científicas={n_sci}, "
                f"ventanas BDA={n_avg}, filas matcheadas={n_joined}. "
                f"Puede haber desalineación de window_id entre datasets."
            )

        return df_combined

    except Exception as e:
        print(f"Error combining dataframes for amplitude: {e}")
        traceback.print_exc()
        raise


def calculate_amplitude_error_paired(df_combined):
    """
    Por cada (baseline_key, window_id) calcula:

      sum_abs_diff : Σ_i Σ_{c,p} | |V_BDA(c,p)| - |V_i(c,p)| |
                     suma de diferencias absolutas de amplitud,
                     solo para (c,p) válidos en ambos datasets.

      sum_amp_sci  : Σ_i Σ_{c,p} |V_i(c,p)|
                     suma de amplitudes científicas, misma máscara.

      n_valid      : número de pares (integración, canal, pol) válidos.
    """
    try:
        schema = StructType([
            StructField("baseline_key",  StringType(), False),
            StructField("window_id",     LongType(),   False),
            StructField("sum_abs_diff",  DoubleType(), False),
            StructField("sum_amp_sci",   DoubleType(), False),
            StructField("n_valid",       LongType(),   False),
        ])

        def compute_amplitude(pdf):
            baseline_key = pdf["baseline_key"].iloc[0]
            window_id    = pdf["window_id"].iloc[0]
            sum_abs_diff = 0.0
            sum_amp_sci  = 0.0
            n_valid      = 0

            # V_BDA es igual para todas las filas del grupo (una por ventana)
            vs_avg = np.array([[corr for corr in chan] for chan in pdf["visibility_averaging"].iloc[0]], dtype=np.float64)  # (C, P, 2)
            fs_avg = np.array([[flag for flag in chan] for chan in pdf["flag_averaging"].iloc[0]], dtype=np.bool_)          # (C, P)

            if vs_avg.ndim != 3 or vs_avg.shape[2] != 2:
                raise ValueError(
                    f"Shape inesperado para visibility_averaging: {vs_avg.shape}; "
                    f"se esperaba (C, P, 2)."
                )

            # Amplitud de la visibilidad BDA: |V_BDA| = sqrt(Re² + Im²)
            amp_avg = np.sqrt(vs_avg[..., 0] ** 2 + vs_avg[..., 1] ** 2)  # (C, P)

            # Iterar sobre cada integración científica de la ventana
            for vs_sci_data, flag_sci_data in zip(
                pdf["visibility_scientific"],
                pdf["flag_scientific"],
            ):
                vs_sci   = np.array([[corr for corr in chan] for chan in vs_sci_data], dtype=np.float64)  # (C, P, 2)
                flag_sci = np.array([[flag for flag in chan] for chan in flag_sci_data], dtype=bool)         # (C, P)

                if vs_sci.ndim != 3 or vs_sci.shape[2] != 2:
                    raise ValueError(
                        f"Shape inesperado para visibility_scientific: {vs_sci.shape}; "
                        f"se esperaba (C, P, 2)."
                    )
                if vs_sci.shape[:2] != vs_avg.shape[:2]:
                    raise ValueError(
                        f"Shape (C, P) no coincide: científico {vs_sci.shape[:2]} "
                        f"vs BDA {vs_avg.shape[:2]}."
                    )

                # Excluir si flaggeado en la integración científica O en BDA
                fs_combined = flag_sci | fs_avg  # (C, P)
                valid         = ~fs_combined        # (C, P)

                if not valid.any():
                    continue

                # Amplitud de cada visibilidad científica individual
                amp_sci = np.sqrt(vs_sci[..., 0] ** 2 + vs_sci[..., 1] ** 2)  # (C, P)

                # Diferencia absoluta de amplitud: | |V_BDA| - |V_i| |
                abs_diff = np.abs(amp_avg - amp_sci)  # (C, P)

                sum_abs_diff += float(abs_diff[valid].sum())
                sum_amp_sci  += float(amp_sci[valid].sum())
                n_valid      += int(valid.sum())

            return pd.DataFrame([{
                "baseline_key": baseline_key,
                "window_id":    window_id,
                "sum_abs_diff": sum_abs_diff,
                "sum_amp_sci":  sum_amp_sci,
                "n_valid":      n_valid,
            }])

        df_amplitude = df_combined.groupBy("baseline_key", "window_id").applyInPandas(
            compute_amplitude, schema=schema
        )
        return df_amplitude

    except Exception as e:
        print(f"Error calculating paired amplitude error: {e}")
        traceback.print_exc()
        raise


def sum_amplitudes(df_amplitude):
    try:
        results = df_amplitude.agg(
            F.sum("sum_abs_diff").alias("sum_abs_diff"),
            F.sum("sum_amp_sci").alias("sum_amp_sci"),
            F.sum("n_valid").alias("n_valid"),
        ).collect()[0]

        return (
            results["sum_abs_diff"],
            results["sum_amp_sci"],
            results["n_valid"],
        )

    except Exception as e:
        print(f"Error summing amplitudes: {e}")
        traceback.print_exc()
        raise


def write_amplitude_results(
    sum_abs_diff,
    sum_amp_sci,
    n_valid,
    mean_abs_error,
    relative_error,
    relative_error_percent,
    tolerance,
    passed,
    output_file,
):
    try:
        with open(output_file, "a") as f:
            f.write(f"{'=' * 80}\n")
            f.write(f"Amplitude Error\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Equations:\n")
            f.write(f"  AE_abs = ( Σ_i ||V_BDA(w_i)| - |V_i|| ) / N_valid\n")
            f.write(f"  AE_rel = Σ_i ||V_BDA(w_i)| - |V_i|| / Σ_i |V_i|\n")
            f.write(f"  V_i    = visibilidad científica individual\n")
            f.write(f"  V_BDA  = promedio ponderado de la ventana que contiene V_i\n")
            f.write(f"  (suma sobre visibilidades válidas en ambos datasets)\n")
            f.write(f"\n")
            f.write(f"Aggregated Values:\n")
            f.write(f"  Σ ||V_BDA| - |V_i||:  {sum_abs_diff:.6e}\n")
            f.write(f"  Σ |V_i|:              {sum_amp_sci:.6e}\n")
            f.write(f"  N valid:              {n_valid:,}\n")
            f.write(f"\n")
            f.write(f"Metrics:\n")
            f.write(f"  Absolute AE:          {mean_abs_error:.6e}\n")
            f.write(f"    → diferencia media de amplitud por visibilidad\n")
            f.write(f"  Relative AE:          {relative_error:.6f} ({relative_error_percent:.4f}%)\n")
            f.write(f"    → fracción de la amplitud de señal perdida por el promediado\n")
            f.write(f"  Tolerance:            {tolerance:.6f} ({tolerance * 100.0:.4f}%)\n")
            f.write(f"  Status:               {'✓ PASSED' if passed else '✗ FAILED'}\n")
            f.write(f"{'=' * 80}\n")

    except Exception as e:
        print(f"Error writing amplitude results: {e}")
        traceback.print_exc()
        raise


def calculate_amplitude_error(df_amplitude, bda_config, output_file):
    try:
        tolerance = bda_config.get("amplitude_tolerance", 0.01)  # 1% por defecto

        sum_abs_diff, sum_amp_sci, n_valid = sum_amplitudes(df_amplitude)

        if n_valid == 0:
            raise ValueError(
                "n_valid=0: no hay visibilidades válidas para calcular el error "
                "de amplitud. Revisa los flags y la alineación de window_id."
            )
        if sum_amp_sci == 0.0:
            raise ValueError(
                "sum_amp_sci=0: la amplitud de referencia es cero; "
                "el error relativo no está definido."
            )

        mean_abs_error         = sum_abs_diff / n_valid
        relative_error         = sum_abs_diff / sum_amp_sci
        relative_error_percent = relative_error * 100.0
        passed                 = relative_error <= tolerance

        write_amplitude_results(
            sum_abs_diff,
            sum_amp_sci,
            n_valid,
            mean_abs_error,
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