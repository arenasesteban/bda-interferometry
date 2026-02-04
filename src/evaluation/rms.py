import traceback
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType, ArrayType
import pandas as pd


def rms(df_scientific, df_averaging):
    try:
        df_reference = aggregate_reference_visibilities(df_scientific)
        
        df_combined = combine_for_rms(df_reference, df_averaging)
        
        df_rms = calculate_rms_error(df_combined)
        
        return df_rms
    except Exception as e:
        print(f"Error calculating RMS: {e}")
        traceback.print_exc()
        raise


def aggregate_reference_visibilities(df_scientific):
    try:
        schema = StructType([
            StructField("baseline_key", StringType(), False),
            StructField("window_id", LongType(), False),
            StructField("visibilities", ArrayType(ArrayType(ArrayType(DoubleType()))), False),
        ])

        def compute_reference(pdf):
            baseline_key = pdf['baseline_key'].iloc[0]
            window_id = pdf['window_id'].iloc[0]

            vs_list = []

            for vs_data, fs_data in zip(pdf['visibilities'], pdf['flag']):
                vs = np.array([[corr for corr in chan] for chan in vs_data], dtype=np.float64)
                fs = np.array([[f for f in chan] for chan in fs_data], dtype=np.bool_)
                
                vs_masked = vs.copy()
                vs_masked[fs] = np.nan

                vs_list.append(vs_masked)
            
            vs_stacked = np.stack(vs_list, axis=0)
            visibilities_ref = np.nanmean(vs_stacked, axis=0)
            
            visibilities_ref = np.nan_to_num(visibilities_ref, nan=0.0)

            return pd.DataFrame([{
                'baseline_key': baseline_key,
                'window_id': window_id,
                'visibilities': visibilities_ref.tolist()
            }])

        
        df_reference = df_scientific.groupBy('baseline_key', 'window_id').applyInPandas(compute_reference, schema=schema)

        return df_reference

    except Exception as e:
        print(f"Error aggregating reference visibilities: {e}")
        traceback.print_exc()
        raise


def combine_for_rms(df_reference, df_averaging):
    try:
        df_reference = df_reference.select(
            "baseline_key", "window_id",
            df_reference["visibilities"].alias("visibilities_scientific"),
        )

        df_averaging = df_averaging.select(
            "baseline_key", "window_id",
            df_averaging["visibilities"].alias("visibilities_averaging"),
        )

        df_combined = df_reference.join(
            df_averaging, 
            on=["baseline_key", "window_id"], 
            how="inner"
        )
        
        return df_combined

    except Exception as e:
        print(f"Error combining dataframes for RMS: {e}")
        traceback.print_exc()
        raise


def calculate_rms_error(df_combined):
    try:
        schema = StructType([
            StructField("baseline_key", StringType(), False),
            StructField("window_id", LongType(), False),
            StructField("squared_difference", DoubleType(), False),
            StructField("squared_scientific", DoubleType(), False),
            StructField("denominator", LongType(), False)
        ])

        def compute_rms(pdf):
            baseline_key = pdf['baseline_key'].iloc[0]
            window_id = pdf['window_id'].iloc[0]
            squared_difference = 0.0
            squared_scientific = 0.0
            denominator = 0

            for vs_scientific_data, vs_averaging_data in zip(pdf["visibilities_scientific"], pdf["visibilities_averaging"]):
                vs_scientific = np.array([[corr for corr in chan] for chan in vs_scientific_data], dtype=np.float64)
                vs_averaging = np.array([[corr for corr in chan] for chan in vs_averaging_data], dtype=np.float64)

                diff = vs_averaging - vs_scientific

                squared_difference += np.sum(diff ** 2)
                squared_scientific += np.sum(vs_scientific ** 2)
                denominator += vs_scientific.size

            return pd.DataFrame([{
                'baseline_key': baseline_key,
                'window_id': window_id,
                'squared_difference': float(squared_difference), 
                'squared_scientific': float(squared_scientific), 
                'denominator': int(denominator)
            }])

        df_rms = df_combined.groupBy('baseline_key', 'window_id').applyInPandas(
            compute_rms, 
            schema=schema
        )

        return df_rms

    except Exception as e:
        print(f"Error calculating RMS: {e}")
        traceback.print_exc()
        raise


def sum_rms(df_rms):
    try:
        results = df_rms.select("squared_difference", "squared_scientific", "denominator").agg(
            F.sum("squared_difference").alias("squared_difference"),
            F.sum("squared_scientific").alias("squared_scientific"),
            F.sum("denominator").alias("denominator")
        ).collect()[0]

        return results['squared_difference'], results['squared_scientific'], results['denominator']

    except Exception as e:
        print(f"Error summing RMS values: {e}")
        traceback.print_exc()
        raise


def calculate_rms_measure(df_rms, bda_config):
    try:
        tolerance = bda_config.get('rms_tolerance', 0.1)

        squared_difference, squared_scientific, denominator = sum_rms(df_rms)

        rms_absolute = np.sqrt(squared_difference / denominator)
        
        rms_relative = np.sqrt(squared_difference / squared_scientific)
        rms_relative_percent = rms_relative * 100
        
        passed = rms_relative <= tolerance

        print(f"\n{'=' * 80}")
        print(f"RMS Error in Visibility Domain")
        print(f"{'=' * 80}")
        print(f"Equation:")
        print(f"  Absolute RMS:  sqrt( Σ|V_BDA - V_ref|² / N )")
        print(f"  Relative RMS:  sqrt( Σ|V_BDA - V_ref|² / Σ|V_ref|² )")
        print(f"")
        print(f"Results:")
        print(f"  Σ|V_BDA - V_ref|²:   {squared_difference:.6e}")
        print(f"  Σ|V_ref|²:           {squared_scientific:.6e}")
        print(f"  N:     {denominator:,}")
        print(f"")
        print(f"Metrics:")
        print(f"  Absolute RMS:        {rms_absolute:.6e}")
        print(f"  Relative RMS:        {rms_relative:.6f} ({rms_relative_percent:.4f}%)")
        print(f"  Tolerance:          {tolerance:.6f} ({tolerance * 100:.4f}%)")
        print(f"  Status:              {'✓ PASSED' if passed else '✗ FAILED'}")
        print(f"{'=' * 80}\n")
        
    except Exception as e:
        print(f"Error calculating RMS measure: {e}")
        traceback.print_exc()
        raise