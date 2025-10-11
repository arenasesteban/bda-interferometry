"""
BDA Integration - Distributed BDA Pipeline Orchestration

This module orchestrates the distributed BDA (Baseline-Dependent Averaging) pipeline
using Apache Spark. It coordinates distributed grouping, applies BDA processing through
Pandas UDFs, and aggregates results with comprehensive performance metrics.

The module works with data already prepared by the consumer service, focusing exclusively
on distributed processing orchestration and scientific result aggregation.

Key Functions
-------------
apply_distributed_bda_pipeline : Main orchestration function for distributed BDA processing
create_distributed_bda_group_processor : Creates Pandas UDF for group-level BDA processing
"""

import time
import logging
import traceback
import numpy as np
import pandas as pd

from pyspark.sql.functions import (
    explode, col, count, sum as spark_sum, avg, max as spark_max, 
    pandas_udf, udf
)
from pyspark.sql.pandas.functions import PandasUDFType
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, 
    DoubleType, ArrayType
)

from .bda_processor import process_group_with_bda


def apply_distributed_bda_pipeline(df, bda_config, enable_event_time=False, 
                                  watermark_duration="2 minutes", config_file_path=None):
    """
    Execute distributed BDA pipeline with complete scientific processing orchestration.
    
    Coordinates the full distributed BDA workflow using Spark for optimal performance.
    Groups visibility data by baseline and scan, applies BDA algorithms through Pandas UDFs,
    and generates comprehensive processing statistics and results.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Spark DataFrame containing visibility data prepared by consumer service
    bda_config : dict
        BDA configuration parameters including decorrelation factors and frequency settings
    enable_event_time : bool, optional
        Enable watermarking for streaming data processing (default: False)
    watermark_duration : str, optional
        Duration for late data watermarking (default: "2 minutes")
    config_file_path : str, optional
        Path to configuration file (currently unused)
        
    Returns
    -------
    dict
        Complete BDA processing results with DataFrames, KPIs, and configuration metadata
        Contains 'results', 'kpis', 'config', and 'spark_config' keys
        
    Raises
    ------
    Exception
        If distributed processing fails, returns safe fallback with empty results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting distributed BDA orchestration pipeline")
    logger.info("Using data already prepared by consumer service")
    logger.info("Applying distributed BDA processing with Pandas UDFs") 
    logger.info("Executing Spark pipeline: groupBy → applyInPandas")
    logger.info("Computing final distributed aggregations")
    
    try:
        # Use scientific data prepared by consumer service
        logger.info("Using scientific data already prepared by consumer service...")
        
        # DataFrame contains prepared columns: visibility, weight, flag, u, v, w, time, etc.
        scientific_df = df
        
        logger.info("Ready to process scientific data from consumer")
        
        # Watermarking for streaming
        if enable_event_time:
            logger.info(f"Applying watermark for late data: {watermark_duration}")
            scientific_df = scientific_df.withWatermark("time", watermark_duration)
        else:
            logger.info("Watermarking disabled")
        
        # Distributed grouping
        logger.info("Distributed grouping by (antenna1, antenna2, scan_number)...")
        
        # Group data by baseline (antenna pair) and scan number for distributed processing
        grouped_df = scientific_df.groupBy("antenna1", "antenna2", "scan_number")
        
        logger.info("Grouping by baseline+scan configured")
        
        # Distributed BDA per group
        logger.info("Applying distributed BDA with Pandas UDFs...")
        
        # Create distributed BDA UDF
        bda_group_udf = create_distributed_bda_group_processor(bda_config)
        
        # Apply distributed BDA to each group
        bda_results_df = grouped_df.apply(bda_group_udf)
        
        logger.info("Distributed BDA applied to all groups")
        
        # Final distributed aggregations
        logger.info("Calculating final distributed KPIs...")
        
        # Calculate distributed statistics from BDA results
        final_kpis = (bda_results_df
                     .agg(
                         count("*").alias("total_bda_windows"),
                         spark_sum(col("n_input_rows").cast("integer")).alias("total_input_rows"),
                         avg(col("compression_ratio").cast("double")).alias("avg_compression_ratio"),
                         spark_max(col("compression_ratio").cast("double")).alias("max_compression_ratio"),
                         count("group_key").alias("unique_groups_processed"),
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
        
        # Generate final results
        logger.info("Generating final distributed BDA results...")
        
        # Create results without expensive collect() operations
        results_data = [{
            'result_type': 'fully_distributed_bda',
            'processing_status': 'bda_completed_distributed',
            'total_bda_windows': total_windows,
            'total_input_rows': total_input_rows,
            'unique_groups_processed': unique_groups,
            'sample_baseline_keys': min(unique_groups, 10),  # Estimation without collect()
            'sample_avg_compression': avg_compression_ratio,  # Use already calculated value
            'processing_mode': 'distributed_pandas_udf_bda'
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
        
        # Final result
        logger.info(f"DISTRIBUTED BDA PIPELINE COMPLETED")
        logger.info(f"   Input rows: {total_input_rows}")
        logger.info(f"   BDA windows: {total_windows}")
        logger.info(f"   Average compression: {avg_compression_ratio:.2f}:1")
        logger.info(f"   Maximum compression: {max_compression_ratio:.2f}:1")
        logger.info(f"   Groups processed: {unique_groups}")
        logger.info(f"   Average window duration: {avg_window_duration:.2f}s")
        logger.info(f"   Mode: FULLY DISTRIBUTED")
        
        return {
            'results': bda_results_df,  # Complete DataFrame with BDA results
            'kpis': kpis_df,
            'config': bda_config,
            'spark_config': {
                'mode': 'fully_distributed_bda',
                'total_input_rows': int(total_input_rows),
                'total_bda_windows': int(total_windows),
                'unique_groups_processed': int(unique_groups),
                'avg_compression_ratio': float(avg_compression_ratio),
                'max_compression_ratio': float(max_compression_ratio),
                'pipeline_status': 'complete_distributed_bda_pipeline',
                'architecture': 'spark_pandas_udf_groupby_applyinpandas'
            }
        }
        
    except Exception as e:
        logger.error(f"ERROR IN DISTRIBUTED PIPELINE")
        logger.error(f"   Error: {e}")
        logger.error(traceback.format_exc())
        
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
            'spark_config': {'mode': 'error_fallback', 'error': str(e)}
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
    returned by distributed processing UDFs.
    
    Returns
    -------
    StructType
        Complete Spark schema for BDA result DataFrames
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
        StructField("max_averaging_time_s", DoubleType(), False),
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
        StructField("sample_baseline_keys", IntegerType(), False),
        StructField("sample_avg_compression", FloatType(), False),
        StructField("processing_mode", StringType(), False)
    ])


def create_distributed_bda_group_processor(bda_config: dict):
    """
    Create Pandas UDF for distributed BDA processing by baseline groups.
    
    Generates a Spark Pandas UDF that applies BDA algorithms to individual
    baseline-scan groups on distributed workers. Each group is processed
    independently using scientific BDA algorithms.
    
    Parameters
    ----------
    bda_config : dict
        BDA configuration parameters including frequency and decorrelation settings
        
    Returns
    -------
    function
        Pandas UDF function for distributed group-level BDA processing
    """
    
    @pandas_udf(returnType=get_bda_result_schema(), functionType=PandasUDFType.GROUPED_MAP)
    def bda_group_processor(group_df):
        """
        Processes a group (baseline, scan) with BDA on distributed worker.
        
        This function executes on each Spark worker and applies BDA to a specific
        group of visibilities using existing scientific functions.
        
        Parameters
        ----------
        group_df : pd.DataFrame
            Pandas DataFrame with all group rows
            
        Returns
        -------
        pd.DataFrame
            DataFrame with BDA results for the group
        """
        start_time = time.time()
        
        try:
            if group_df.empty:
                return create_empty_bda_result()
            
            # Extract group information
            first_row = group_df.iloc[0]
            baseline_key = first_row['baseline_key']
            scan_number = first_row['scan_number']
            antenna1 = first_row['antenna1']
            antenna2 = first_row['antenna2']
            
            # Create group_key from baseline and scan information
            group_key = f"{baseline_key}_scan{scan_number}"
            
            # Logging moved to debug level to avoid spam in production
            logging.debug(f"Processing BDA group: {group_key} ({len(group_df)} rows)")
            
            # Convert pandas DataFrame to BDA-compatible format
            visibility_rows = convert_pandas_to_bda_format(group_df)
            
            # Apply existing BDA logic
            try:
                # Use existing BDA processor function (now works on pre-grouped data)
                bda_results = process_group_with_bda(visibility_rows, bda_config)
                
                if not bda_results:
                    # Fallback: create result without averaging
                    return create_fallback_bda_result_df(group_key, baseline_key, scan_number, antenna1, antenna2, len(group_df))
                
            except Exception as e:
                logging.warning(f"BDA processing failed for {group_key}: {e}")
                return create_fallback_bda_result_df(group_key, baseline_key, scan_number, antenna1, antenna2, len(group_df))
            
            # Convert BDA results to Spark format
            spark_results = convert_bda_results_to_spark(
                bda_results, group_key, baseline_key, scan_number, 
                antenna1, antenna2, len(group_df), start_time
            )
            
            processing_time = (time.time() - start_time) * 1000
            logging.debug(f"BDA group {group_key} completed in {processing_time:.1f}ms")
            
            return spark_results
            
        except Exception as e:
            logging.error(f"Error processing BDA group: {e}")
            return create_error_bda_result(e, group_key if 'group_key' in locals() else 'unknown')
    
    return bda_group_processor


def convert_pandas_to_bda_format(group_df) -> list:
    """
    Convert pandas DataFrame to BDA processor compatible format.
    
    Transforms visibility data from pandas DataFrame rows to the dictionary
    format required by BDA processing functions. Handles conversion of
    Spark arrays to numpy arrays and ensures proper data types.
    
    Parameters
    ----------
    group_df : pd.DataFrame
        Pandas DataFrame containing visibility data for a single group
        
    Returns
    -------
    list of dict
        List of dictionaries with numpy arrays compatible with BDA processor
        
    Raises
    ------
    Exception
        If data conversion fails or required columns are missing
    """
    rows = []
    
    try:
        for idx, row in group_df.iterrows():
            # Extract visibility data arrays from Spark DataFrame rows
            # Convert Spark list format to numpy arrays for BDA processing
            vis_list = row.get('visibilities', [])  # Note: column name is 'visibilities' (plural)
            weight_list = row.get('weight', [])
            flag_list = row.get('flag', [])
            
            # Get expected dimensions from DataFrame metadata
            n_channels = int(row.get('n_channels', 1))
            n_correlations = int(row.get('n_correlations', 1))
            
            # Handle None values and ensure we have valid lists
            if vis_list is None:
                vis_list = []
            if weight_list is None:
                weight_list = []
            if flag_list is None:
                flag_list = []
                
            # Ensure we have valid data types (convert from numpy if needed)
            if hasattr(vis_list, 'tolist'):
                vis_list = vis_list.tolist()
            if hasattr(weight_list, 'tolist'):
                weight_list = weight_list.tolist()
            if hasattr(flag_list, 'tolist'):
                flag_list = flag_list.tolist()
            
            # Convert list data to numpy arrays with appropriate types
            # Complex visibility data expected as [real, imag, real, imag, ...] format
            try:
                vis_array = np.array(vis_list, dtype=complex) if len(vis_list) > 0 else np.array([0+0j])
                weight_array = np.array(weight_list, dtype=float) if len(weight_list) > 0 else np.array([1.0])
                flag_array = np.array(flag_list, dtype=bool) if len(flag_list) > 0 else np.array([False])
            except Exception as array_error:
                logging.warning(f"Array conversion error: {array_error}, using defaults")
                vis_array = np.array([0+0j])
                weight_array = np.array([1.0])
                flag_array = np.array([False])
            
            # Reshape arrays to [nchans, npols] format using metadata dimensions
            # This ensures consistent shapes across all rows in the group
            try:
                expected_shape = (n_channels, n_correlations)
                
                # Reshape or pad/truncate arrays to expected dimensions
                vis_array = np.resize(vis_array, expected_shape).astype(complex)
                weight_array = np.resize(weight_array, expected_shape).astype(float)
                flag_array = np.resize(flag_array, expected_shape).astype(bool)
                
            except Exception as reshape_error:
                logging.warning(f"Array reshape error: {reshape_error}, using fallback reshape")
                # Fallback to simple reshape
                if vis_array.ndim == 1:
                    vis_array = vis_array.reshape(-1, 1)
                if weight_array.ndim == 1:
                    weight_array = weight_array.reshape(-1, 1)
                if flag_array.ndim == 1:
                    flag_array = flag_array.reshape(-1, 1)
            
            # Create row dictionary with format expected by BDA processor
            bda_row = {
                'antenna1': int(row.get('antenna1', 0)),
                'antenna2': int(row.get('antenna2', 0)),
                'scan_number': int(row.get('scan_number', 0)),
                'time': float(row.get('time', 0.0)),
                'u': float(row.get('u', 0.0)),
                'v': float(row.get('v', 0.0)),
                'w': float(row.get('w', 0.0)),
                'visibility': vis_array,
                'weight': weight_array,
                'flag': flag_array,
            }
            
            rows.append(bda_row)
            
    except Exception as e:
        logging.error(f"Error converting pandas to BDA format: {e}")
        
    return rows


def convert_bda_results_to_spark(bda_results, group_key, baseline_key, scan_number, 
                                antenna1, antenna2, input_rows, start_time):
    """
    Convert BDA processing results to Spark-compatible DataFrame format.
    
    Transforms BDA algorithm outputs to pandas DataFrame with proper schema
    for Spark distributed processing. Handles complex visibility data conversion
    and computes processing statistics.
    
    Parameters  
    ----------
    bda_results : list
        BDA processing results containing averaged visibilities and metadata
    group_key : str
        Unique identifier for the baseline-scan group
    baseline_key : str
        Baseline identifier (antenna pair)
    scan_number : int
        Observation scan number
    antenna1, antenna2 : int
        Baseline antenna pair identifiers
    input_rows : int
        Number of input visibility rows processed
    start_time : float
        Processing start timestamp for performance metrics
        
    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with BDA results compatible with Spark schema
        
    Raises
    ------
    Exception
        If result conversion fails, returns fallback DataFrame
    """
    
    try:
        if not bda_results:
            return create_fallback_bda_result_df(
                group_key, baseline_key, scan_number, antenna1, antenna2, input_rows
            )
        
        # Take first BDA result (may have multiple windows)
        result = bda_results[0] if isinstance(bda_results, list) else bda_results
        
        # Calculate statistics
        n_windows = len(bda_results) if isinstance(bda_results, list) else 1
        compression_ratio = input_rows / n_windows if n_windows > 0 else 1.0
        processing_time = (time.time() - start_time) * 1000
        
        # Extract averaged arrays in native format
        vis_avg = result.get('visibility_averaged', np.array([]))
        weight_total = result.get('weight_total', np.array([]))
        flag_combined = result.get('flag_combined', np.array([]))
        
        # Convert to flat lists for Spark compatibility
        # Complex visibility data needs to be converted to [real, imag, real, imag, ...] format
        if hasattr(vis_avg, 'flatten'):
            vis_complex = vis_avg.flatten()
            # Convert complex numbers to interleaved real/imag pairs
            vis_flat = []
            for complex_val in vis_complex:
                if isinstance(complex_val, complex):
                    vis_flat.extend([float(complex_val.real), float(complex_val.imag)])
                else:
                    # Handle case where it's already a real number
                    vis_flat.extend([float(complex_val), 0.0])
        else:
            vis_flat = []
            
        weight_flat = weight_total.flatten().tolist() if hasattr(weight_total, 'flatten') else []
        flag_flat = flag_combined.flatten().astype(int).tolist() if hasattr(flag_combined, 'flatten') else []
        
        # Create result DataFrame
        result_data = {
            'group_key': [group_key],
            'baseline_key': [baseline_key],
            'scan_number': [scan_number],
            'antenna1': [antenna1],
            'antenna2': [antenna2],
            
            # Averaged coordinates
            'time_avg': [result.get('time_avg', result.get('time', 0.0))],
            'u_avg': [result.get('u_avg', result.get('u', 0.0))],
            'v_avg': [result.get('v_avg', result.get('v', 0.0))],
            'w_avg': [result.get('w_avg', result.get('w', 0.0))],
            'baseline_length': [result.get('baseline_length', 0.0)],
            
            # Scientific arrays (flattened native format)
            'visibility_avg': [vis_flat],
            'weight_total': [weight_flat],
            'flag_combined': [flag_flat],
            
            # BDA statistics
            'n_input_rows': [int(input_rows)],
            'n_windows_created': [n_windows],
            'window_duration_s': [result.get('window_dt_s', 0.0)],
            'max_averaging_time_s': [result.get('delta_t_max', 0.0)],
            'compression_ratio': [compression_ratio],
            
            # Metadata
            'processing_timestamp': [time.time()],
            'bda_config_hash': ['distributed_v1'],
        }
        
        return pd.DataFrame(result_data)
        
    except Exception as e:
        logging.warning(f"Error converting BDA results: {e}")
        return create_fallback_bda_result_df(
            group_key, baseline_key, scan_number, antenna1, antenna2, input_rows
        )


def create_fallback_bda_result_df(group_key, baseline_key, scan_number, antenna1, antenna2, input_rows):
    """
    Create fallback DataFrame when BDA processing encounters errors.
    
    Generates default result DataFrame with minimal valid data when
    BDA processing fails or encounters exceptions. Ensures pipeline
    continuity with safe default values.
    
    Parameters
    ----------
    group_key : str
        Unique identifier for the failed group
    baseline_key : str
        Baseline identifier for the failed processing
    scan_number : int
        Scan number of the failed processing
    antenna1, antenna2 : int
        Antenna pair identifiers
    input_rows : int
        Number of input rows that failed processing
        
    Returns
    -------
    pd.DataFrame
        DataFrame with safe default values matching BDA result schema
    """
    
    return pd.DataFrame({
        'group_key': [group_key],
        'baseline_key': [baseline_key], 
        'scan_number': [scan_number],
        'antenna1': [antenna1],
        'antenna2': [antenna2],
        'time_avg': [0.0],
        'u_avg': [0.0],
        'v_avg': [0.0],
        'w_avg': [0.0],
        'baseline_length': [0.0],
        'visibility_avg': [[0.0]],  # Flattened format
        'weight_total': [[1.0]],    # Flattened format
        'flag_combined': [[0]],     # Flattened format
        'n_input_rows': [int(input_rows)],
        'n_windows_created': [1],
        'window_duration_s': [0.0],
        'max_averaging_time_s': [0.0],
        'compression_ratio': [1.0],
        'processing_timestamp': [time.time()],
        'bda_config_hash': ['fallback'],
    })


def create_empty_bda_result():
    """
    Create empty result DataFrame for groups without input data.
    
    Generates minimal DataFrame with default values when a processing
    group contains no visibility data. Maintains schema consistency
    for empty data scenarios.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with empty group default values matching BDA result schema
    """
    
    return pd.DataFrame({
        'group_key': ['empty'],
        'baseline_key': ['empty'],
        'scan_number': [0],
        'antenna1': [0],
        'antenna2': [0],
        'time_avg': [0.0],
        'u_avg': [0.0],
        'v_avg': [0.0],
        'w_avg': [0.0],
        'baseline_length': [0.0],
        'visibility_avg': [[0.0]],  # Flattened format
        'weight_total': [[1.0]],    # Flattened format
        'flag_combined': [[0]],     # Flattened format
        'n_input_rows': [0],
        'n_windows_created': [0],
        'window_duration_s': [0.0],
        'max_averaging_time_s': [0.0],
        'compression_ratio': [1.0],
        'processing_timestamp': [time.time()],
        'bda_config_hash': ['empty'],
    })


def create_error_bda_result(error, group_key):
    """
    Create error result DataFrame for debugging failed group processing.
    
    Generates DataFrame containing error information when BDA group
    processing encounters exceptions. Includes error context in the
    configuration hash field for debugging purposes.
    
    Parameters
    ----------
    error : Exception
        Exception that occurred during processing
    group_key : str
        Identifier of the group that failed processing
        
    Returns
    -------
    pd.DataFrame
        DataFrame with error information matching BDA result schema
    """
    
    return pd.DataFrame({
        'group_key': [group_key],
        'baseline_key': ['error'],
        'scan_number': [0],
        'antenna1': [0],
        'antenna2': [0],
        'time_avg': [0.0],
        'u_avg': [0.0],
        'v_avg': [0.0],
        'w_avg': [0.0],
        'baseline_length': [0.0],
        'visibility_avg': [[0.0]],  # Flattened format
        'weight_total': [[1.0]],    # Flattened format
        'flag_combined': [[0]],     # Flattened format
        'n_input_rows': [0],
        'n_windows_created': [0],
        'window_duration_s': [0.0],
        'max_averaging_time_s': [0.0],
        'compression_ratio': [1.0],
        'processing_timestamp': [time.time()],
        'bda_config_hash': [f'error:{str(error)[:50]}'],
    })
