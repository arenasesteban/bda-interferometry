"""
BDA Integration - Incremental Streaming BDA Pipeline Orchestration

This module orchestrates the incremental BDA (Baseline-Dependent Averaging)        results_data = [{
            'result_type': 'incremental_streaming_bda',
            'processing_status': 'bda_completed_streaming',
            'total_bda_windows': total_windows,
            'total_input_rows': total_input_rows,
            'unique_groups_processed': min(unique_groups, 10),  # Estimation without collect()
            'sample_baseline_keys': min(unique_groups, 10),  # Estimation without collect()
            'sample_avg_compression': avg_compression_ratio,  # Use already calculated value
            'processing_mode': 'incremental_streaming_windows'
        }]using Apache Spark's mapPartitions. It processes visibility data in decorrelation-time
windows for memory-efficient streaming with constant RAM usage.

The module uses online window closure based on physical decorrelation criteria,
eliminating the need to accumulate full baselines in memory.

Key Functions
-------------
apply_distributed_bda_pipeline : Main orchestration function for incremental streaming BDA
convert_bda_result_to_spark_tuple : Converts streaming window results to Spark format
"""

import traceback

from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, ArrayType
)

from .bda_processor import process_rows


def apply_bda(df, bda_config):
    try:        
        # Partition by baseline for streaming
        scientific_df = df.repartition(4, col("antenna1"), col("antenna2"), col("scan_number"))
        
        # Sort within partitions by time
        sorted_df = scientific_df.sortWithinPartitions("time")
        
        # Define incremental BDA processing function
        def apply_bda_to_partition(iterator):
            results = []
            
            for window in process_rows(iterator, bda_config):
                result = convert_window_to_tuple(window)
                results.append(result)

            return iter(results)
        
        bda_results_rdd = sorted_df.rdd.mapPartitions(apply_bda_to_partition)
        bda_results_df = df.sparkSession.createDataFrame(bda_results_rdd, define_bda_schema())

        return bda_results_df
        
    except Exception as e:
        print(f"BDA processing failed: {e}")
        traceback.print_exc()
        raise


def convert_window_to_tuple(window):
    try:
        return (
            window["subms_id"],
            window["field_id"],
            window["spw_id"],
            window["polarization_id"],
            window["n_channels"],
            window["n_correlations"],
            window["antenna1"],
            window["antenna2"],
            window["scan_number"],
            window["baseline_key"],
            window['time'],
            window['u'],
            window['v'],
            window['w'],
            window['visibilities'],
            window['weight'],
            window['flag']
        )

    except Exception as e:
        print(f"Error converting window to tuple: {e}")
        traceback.print_exc()
        raise


def define_bda_schema():
    return StructType([
        StructField("subms_id", IntegerType(), True),
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
        StructField("polarization_id", IntegerType(), True),

        StructField("n_channels", IntegerType(), True),
        StructField("n_correlations", IntegerType(), True),

        StructField("antenna1", IntegerType(), True),
        StructField("antenna2", IntegerType(), True),
        StructField("scan_number", IntegerType(), True),
        StructField("baseline_key", StringType(), True),

        StructField('time', DoubleType(), True),
        StructField('u', DoubleType(), True),
        StructField('v', DoubleType(), True),
        StructField('w', DoubleType(), True),

        StructField('visibilities', ArrayType(ArrayType(DoubleType())), True),
        StructField('weight', ArrayType(DoubleType()), True),
        StructField('flag', ArrayType(IntegerType()), True)
    ])
