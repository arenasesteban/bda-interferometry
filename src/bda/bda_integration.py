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

import logging
import numpy as np
from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, max as spark_max, countDistinct
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, 
    DoubleType, ArrayType
)

from .bda_processor import process_rows


def apply_bda(df, bda_config):
    """
    Execute incremental streaming BDA pipeline with decorrelation windows.
    
    Memory-efficient approach using online window processing with Spark mapPartitions.
    Processes visibility data in small windows based on decorrelation time rather 
    than accumulating entire baselines in memory. Maintains constant RAM usage.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Spark DataFrame containing visibility data prepared by consumer service
    bda_config : dict
        BDA configuration parameters including decorrelation time settings
    enable_event_time : bool, optional
        Enable watermarking for streaming data processing (default: False)
    watermark_duration : str, optional
        Duration for late data watermarking (default: "2 minutes")
    config_file_path : str, optional
        Path to configuration file (currently unused)
        
    Returns
    -------
    dict
        Incremental BDA processing results with streaming window statistics
        Contains 'results', 'kpis', 'config' keys and streaming metadata
        
    Raises
    ------
    Exception
        If incremental processing fails, returns safe fallback with empty results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Use scientific data prepared by consumer service
        scientific_df = df
        
        # Partition by baseline for streaming
        scientific_df_partitioned = scientific_df.repartition(4, col("antenna1"), col("antenna2"), col("scan_number"))
        
        # Sort within partitions by time
        sorted_df = scientific_df_partitioned.sortWithinPartitions("time")
        
        # Define incremental BDA processing function
        def apply_bda_to_partition(partition_iterator):
            """Process partition with incremental BDA windows"""
            results = []
            
            for bda_result in process_rows(partition_iterator, bda_config):
                result_tuple = convert_bda_result_to_spark_tuple(bda_result)
                results.append(result_tuple)
            
            return iter(results)
        
        bda_results_rdd = sorted_df.rdd.mapPartitions(apply_bda_to_partition)
        
        # Create DataFrame from RDD with proper schema
        spark_session = df.sparkSession
        bda_results_df = spark_session.createDataFrame(bda_results_rdd, get_bda_result_schema())
        
        # Calculate distributed statistics from BDA results
        final_kpis = (bda_results_df
                     .agg(
                         count("*").alias("total_bda_windows"),
                         spark_sum(col("n_input_rows").cast("integer")).alias("total_input_rows"),
                         avg(col("compression_ratio").cast("double")).alias("avg_compression_ratio"),
                         spark_max(col("compression_ratio").cast("double")).alias("max_compression_ratio"),
                         countDistinct("group_key").alias("unique_groups_processed"),
                         avg(col("window_duration_s").cast("double")).alias("avg_window_duration_s"),
                         spark_max("processing_timestamp").alias("last_processing_time")
                     )
                     .collect()[0])
        
        # Extract final metrics
        total_windows = int(final_kpis["total_bda_windows"] or 0)
        total_input_rows = int(final_kpis["total_input_rows"] or 0)
        avg_compression_ratio = float(final_kpis["avg_compression_ratio"] or 1.0)
        max_compression_ratio = float(final_kpis["max_compression_ratio"] or 1.0)
        unique_groups = int(final_kpis["unique_groups_processed"] or 0)
        avg_window_duration = float(final_kpis["avg_window_duration_s"] or 0.0)
        
        # Create results without expensive collect() operations
        results_data = [{
            'result_type': 'fully_distributed_bda',
            'processing_status': 'bda_completed_distributed',
            'total_bda_windows': total_windows,
            'total_input_rows': total_input_rows,
            'unique_groups_processed': unique_groups,
            'sample_group_count': min(unique_groups, 10),    # Representative group count sample
            'avg_compression_ratio': avg_compression_ratio,  # Average compression achieved
            'processing_mode': 'incremental_streaming_windows'
        }]
        
        spark_session = df.sparkSession
        results_df = spark_session.createDataFrame(results_data, get_final_bda_results_schema())
        
        # Final distributed KPIs
        kpis_data = [{
            'total_input_rows': int(total_input_rows),
            'total_windows': int(total_windows),
            'compression_factor': float(avg_compression_ratio),
            'compression_ratio_pct': float((1 - 1/avg_compression_ratio) * 100) if avg_compression_ratio > 1 else 0.0
        }]
        
        kpis_df = spark_session.createDataFrame(kpis_data, get_kpis_schema())
        
        # Validate BDA results
        validation_results = validate_bda_results(bda_results_df, logger)
        
        return {
            'results': bda_results_df,
            'kpis': kpis_df,
            'config': bda_config,
            'spark_config': {
                'mode': 'incremental_streaming_bda',
                'total_input_rows': int(total_input_rows),
                'total_bda_windows': int(total_windows),
                'unique_groups_processed': int(unique_groups),
                'avg_compression_ratio': float(avg_compression_ratio),
                'max_compression_ratio': float(max_compression_ratio),
                'pipeline_status': 'complete_incremental_bda_pipeline',
                'architecture': 'spark_mappartitions_decorrelation_windows',
            }
        }
        
    except Exception as e:        
        # Safe fallback
        spark_session = df.sparkSession
        empty_results_df = spark_session.createDataFrame([], get_empty_results_schema())
        empty_kpis_df = spark_session.createDataFrame([{
            'total_input_rows': int(0),
            'total_windows': int(0), 
            'compression_factor': float(1.0),
            'compression_ratio_pct': float(0.0)
        }], get_kpis_schema())
        
        return {
            'results': empty_results_df,
            'kpis': empty_kpis_df,
            'config': bda_config,
            'spark_config': {'mode': 'incremental_streaming_error', 'error': str(e)}
        }


def get_kpis_schema():
    """
    Define Spark schema for BDA key performance indicators.
    
    Creates schema for DataFrame containing BDA processing statistics including
    compression ratios, row counts, and processing efficiency metrics.
    
    Returns
    -------
    StructType
        Spark schema for BDA performance indicators DataFrame
    """
    
    return StructType([
        StructField("total_input_rows", IntegerType(), True),
        StructField("total_windows", IntegerType(), True),
        StructField("compression_factor", FloatType(), True), 
        StructField("compression_ratio_pct", FloatType(), True)
    ])


def get_empty_results_schema():
    """
    Define minimal schema for empty result DataFrames.
    
    Creates fallback schema used when BDA processing returns no data
    or encounters processing errors requiring empty results.
    
    Returns
    -------
    StructType
        Minimal Spark schema for empty result DataFrames
    """
    
    return StructType([
        StructField("empty_result", StringType(), True)
    ])


def get_bda_result_schema():
    """
    Define comprehensive schema for BDA processing results.
    
    Creates detailed schema for DataFrames containing averaged visibility data,
    baseline metadata, processing statistics, and BDA performance metrics
    following Wijnholds et al. 2018 methodology.
    
    Returns
    -------
    StructType
        Complete Spark schema for BDA result DataFrames with fields:
        - Baseline identifiers and averaged coordinates (UV-plane consistent)
        - Physics-based decorrelation timing metrics
        - Window efficiency and compression statistics
        - Scientific data arrays (visibilities, weights, flags)
        
    Notes
    -----
    Schema updated to include decorrelation time analysis fields:
    - decorrelation_time_used: Actual window duration
    - decorrelation_time_planned: Optimal duration from physics
    - window_efficiency: Ratio of actual vs planned duration
    """
    
    return StructType([
        # === IDENTIFIERS ===
        StructField("group_key", StringType(), False),
        StructField("baseline_key", StringType(), False),
        StructField("scan_number", IntegerType(), False), 
        StructField("antenna1", IntegerType(), False),
        StructField("antenna2", IntegerType(), False),
        
        # === AVERAGED BDA DATA ===
        StructField("time_avg", DoubleType(), False),
        StructField("u_avg", DoubleType(), False),
        StructField("v_avg", DoubleType(), False),
        StructField("w_avg", DoubleType(), False),
        StructField("baseline_length", DoubleType(), False),
        
        # === AVERAGED VISIBILITY (Native Spark Types) ===
        StructField("visibility_avg", ArrayType(DoubleType()), False),  # Flattened [real, imag, real, imag, ...]
        StructField("weight_total", ArrayType(DoubleType()), False),    # Flattened weights
        StructField("flag_combined", ArrayType(IntegerType()), False),  # Flattened flags
        
        # === BDA STATISTICS ===
        StructField("n_input_rows", IntegerType(), False),
        StructField("n_windows_created", IntegerType(), False),
        StructField("window_duration_s", DoubleType(), False),
        StructField("decorrelation_time_used", DoubleType(), False),
        StructField("decorrelation_time_planned", DoubleType(), False),
        StructField("window_efficiency", DoubleType(), False),
        StructField("compression_ratio", DoubleType(), False),
        
        # === METADATA ===
        StructField("processing_timestamp", DoubleType(), False),
        StructField("bda_config_hash", StringType(), True),
    ])


def get_final_bda_results_schema():
    """
    Define schema for final BDA pipeline summary results.
    
    Creates schema for DataFrame containing overall pipeline statistics,
    processing status, and high-level performance metrics from the complete
    distributed BDA execution.
    
    Returns
    -------
    StructType
        Spark schema for final pipeline summary DataFrame
    """
    
    return StructType([
        StructField("result_type", StringType(), False),
        StructField("processing_status", StringType(), False),
        StructField("total_bda_windows", IntegerType(), False),
        StructField("total_input_rows", IntegerType(), False),
        StructField("unique_groups_processed", IntegerType(), False),
        StructField("sample_group_count", IntegerType(), False),
        StructField("avg_compression_ratio", FloatType(), False),
        StructField("processing_mode", StringType(), False)
    ])


def convert_bda_result_to_spark_tuple(bda_result):
    """
    Convert BDA result dictionary to Spark DataFrame tuple.
    
    Transforms incremental BDA window results to format compatible
    with Spark schema for distributed DataFrame creation.
    """
    import time
    
    # Convert complex arrays to interleaved real/imag format for Spark
    vis_avg = bda_result.get('visibility_averaged', np.array([]))
    if vis_avg.size > 0:
        vis_flat = vis_avg.flatten()
        vis_real_imag = np.stack([vis_flat.real, vis_flat.imag], axis=-1)
        vis_list = vis_real_imag.flatten().tolist()
    else:
        vis_list = []
    
    weight_list = bda_result.get('weight_total', np.array([])).flatten().tolist()
    flag_list = bda_result.get('flag_combined', np.array([])).flatten().astype(int).tolist()
    
    baseline_key = bda_result.get('group_key', (0, 0, 0))
    
    return (
        str(baseline_key),                                    # group_key
        f"{baseline_key[0]}-{baseline_key[1]}",              # baseline_key
        int(baseline_key[2]),                                # scan_number
        int(baseline_key[0]),                                # antenna1
        int(baseline_key[1]),                                # antenna2
        
        float(bda_result.get('time_avg', 0.0)),              # time_avg
        float(bda_result.get('u_avg', 0.0)),                 # u_avg
        float(bda_result.get('v_avg', 0.0)),                 # v_avg
        float(bda_result.get('w_avg', 0.0)),                 # w_avg
        float(np.sqrt(bda_result.get('u_avg', 0.0)**2 + 
                     bda_result.get('v_avg', 0.0)**2)),      # baseline_length (UV-plane, consistent with BDA theory)
        
        vis_list,                                            # visibility_avg
        weight_list,                                         # weight_total
        flag_list,                                           # flag_combined
        
        int(bda_result.get('n_input_rows', 0)),              # n_input_rows
        1,                                                   # n_windows_created
        float(bda_result.get('window_duration_s', 0.0)),     # window_duration_s
        float(bda_result.get('decorrelation_time_used', 0.0)),        # decorrelation_time_used
        float(bda_result.get('decorrelation_time_planned', 0.0)),     # decorrelation_time_planned
        float(bda_result.get('window_efficiency', 1.0)),     # window_efficiency
        float(bda_result.get('compression_ratio', 1.0)),     # compression_ratio
        float(time.time()),                                  # processing_timestamp
        'incremental_v1'                                     # bda_config_hash
    )


def validate_bda_results(bda_results_df, logger=None):
    """
    Validate BDA processing results for consistency and quality.
    
    Performs sanity checks on BDA output to detect potential issues with
    physics calculations, data integrity, or processing anomalies.
    
    Parameters
    ----------
    bda_results_df : pyspark.sql.DataFrame
        DataFrame containing BDA processing results
    logger : logging.Logger, optional
        Logger for validation messages
        
    Returns
    -------
    dict
        Validation results with statistics and warnings
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    try:
        # Basic counts and statistics
        total_results = bda_results_df.count()
        if total_results == 0:
            logger.warning("BDA results DataFrame is empty")
            return {'status': 'empty', 'total_results': 0}
        
        # Check for negative compression ratios (should never happen)
        negative_compression = bda_results_df.filter(col("compression_ratio") <= 0).count()
        if negative_compression > 0:
            logger.error(f"Found {negative_compression} results with non-positive compression ratios")
        
        # Check window efficiency distribution
        efficiency_stats = (bda_results_df
                           .select(avg("window_efficiency").alias("avg_efficiency"),
                                  spark_max("window_efficiency").alias("max_efficiency"),
                                  count("*").alias("total_windows"))
                           .collect()[0])
        
        avg_efficiency = float(efficiency_stats["avg_efficiency"] or 0.0)
        max_efficiency = float(efficiency_stats["max_efficiency"] or 0.0)
        
        # Validate efficiency ranges
        if avg_efficiency < 0.1:
            logger.warning(f"Very low average window efficiency: {avg_efficiency:.2%}")
        elif avg_efficiency > 1.2:
            logger.warning(f"Suspiciously high average window efficiency: {avg_efficiency:.2%}")
        
        # Check for baseline length consistency (should be UV-plane)
        zero_baselines = bda_results_df.filter(col("baseline_length") <= 0).count()
        if zero_baselines > 0:
            logger.warning(f"Found {zero_baselines} results with zero/negative baseline lengths")
        
        logger.info(f"BDA validation completed: {total_results} results, "
                   f"avg efficiency: {avg_efficiency:.1%}, max efficiency: {max_efficiency:.1%}")
        
        return {
            'status': 'validated',
            'total_results': total_results,
            'avg_window_efficiency': avg_efficiency,
            'max_window_efficiency': max_efficiency,
            'negative_compression_count': negative_compression,
            'zero_baseline_count': zero_baselines
        }
        
    except Exception as e:
        logger.error(f"BDA validation failed: {e}")
        return {'status': 'validation_failed', 'error': str(e)}
