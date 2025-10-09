"""
Spark Session Management for Interferometry Data Processing

Provides Spark session creation and configuration management for distributed
interferometry data streaming with Kafka integration and optimized settings
for BDA processing workflows.
"""

import json
from pathlib import Path
from typing import Dict, Any

import findspark
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, ArrayType, BooleanType, LongType
)

# Initialize findspark at module level
findspark.init()


def create_spark_session(config_path: str = None) -> SparkSession:
    """
    Creates optimized Spark session for interferometry data streaming.
    
    Configures Spark with Kafka integration, adaptive query execution,
    and optimized settings for distributed BDA processing workflows.
    
    Parameters
    ----------
    config_path : str, optional
        Path to JSON configuration file for custom settings
        
    Returns
    -------
    SparkSession
        Configured Spark session with streaming optimizations
    """
    config = load_config(config_path)
    
    # Create session with optimized configurations (OPCIÓN A - Memory conservative)
    spark = SparkSession.builder \
        .appName(config["spark"]["app_name"]) \
        .master(config["spark"]["master"]) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.driver.memory", "4g") \
        .config("spark.python.worker.reuse", "true") \
        .getOrCreate()
    
    # Reduce logging verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads Spark and Kafka configuration from JSON file or defaults.
    
    Merges user-provided configuration with default settings for
    Spark session creation and Kafka streaming parameters.
    
    Parameters
    ----------
    config_path : str, optional
        Path to JSON configuration file
        
    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary with Spark and Kafka settings
    """
    # Default configuration
    default_config = {
        "spark": {
            "app_name": "BDA-Interferometry-Consumer",
            "master": "local[*]"
        },
        "kafka": {
            "servers": "localhost:9092",
            "topic": "visibility-stream",
            "group_id": "bda-spark-consumers"
        }
    }
    
    # Load from file if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Update with user config
            for section, values in user_config.items():
                if section in default_config:
                    default_config[section].update(values)
        except Exception as e:
            print(f"Warning: Could not load config {config_path}: {e}")
    
    return default_config


def get_kafka_options(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts Kafka connection options for Spark streaming configuration.
    
    Converts configuration dictionary into Spark-compatible Kafka options
    for structured streaming data source configuration.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing Kafka settings
        
    Returns
    -------
    Dict[str, str]
        Kafka options formatted for Spark streaming
    """
    kafka_config = config["kafka"]
    
    return {
        "kafka.bootstrap.servers": kafka_config["servers"],
        "subscribe": kafka_config["topic"],
        "startingOffsets": "latest"
    }


def get_visibility_row_schema() -> StructType:
    """
    Defines Spark schema for individual visibility rows after chunk expansion.
    
    Creates structured schema matching expanded visibility data format
    with baseline identifiers, coordinates, scientific arrays, and metadata
    for distributed BDA processing compatibility.
    
    Returns
    -------
    StructType
        Spark schema definition for visibility row structure
    """
    return StructType([
        # Grouping identifiers
        StructField("baseline_key", StringType(), False),
        StructField("scan_number", IntegerType(), False),
        StructField("antenna1", IntegerType(), False),
        StructField("antenna2", IntegerType(), False),
        StructField("subms_id", StringType(), True),
        
        # Temporal and spatial metadata  
        StructField("time", FloatType(), False),
        StructField("u", FloatType(), False),
        StructField("v", FloatType(), False),
        StructField("w", FloatType(), False),
        
        # Scientific data arrays (as JSON strings for now, will be parsed in UDFs)
        StructField("visibility_json", StringType(), False),
        StructField("weight_json", StringType(), False),
        StructField("flag_json", StringType(), False),
        
        # Original chunk metadata
        StructField("original_chunk_id", IntegerType(), True),
        StructField("row_index_in_chunk", IntegerType(), False),
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
    ])


def get_grouped_visibility_schema() -> StructType:
    """
    Defines Spark schema for grouped visibility data by baseline and scan.
    
    Creates schema structure for visibility data grouped by baseline-scan
    combinations, containing group metadata and arrays of visibility rows
    for distributed BDA processing operations.
    
    Returns
    -------
    StructType
        Spark schema definition for grouped visibility structure
    """
    return StructType([
        StructField("group_key", StringType(), False),
        StructField("baseline_key", StringType(), False),
        StructField("scan_number", IntegerType(), False),
        StructField("antenna1", IntegerType(), False),
        StructField("antenna2", IntegerType(), False),
        StructField("row_count", IntegerType(), False),
        StructField("visibility_rows", ArrayType(get_visibility_row_schema()), False),
    ])


def get_bda_result_schema() -> StructType:
    """
    Defines Spark schema for BDA processing output results.
    
    Creates schema structure for averaged visibility data output from
    BDA algorithms, including averaged arrays, coordinates, timing
    metadata, and compression statistics for result consistency.
    
    Returns
    -------
    StructType
        Spark schema definition for BDA processing results
    """
    return StructType([
        StructField("visibility_averaged_json", StringType(), False),
        StructField("weight_total_json", StringType(), False),
        StructField("flag_combined_json", StringType(), False),
        StructField("u_avg", FloatType(), False),
        StructField("v_avg", FloatType(), False),
        StructField("w_avg", FloatType(), False),
        StructField("time_avg", FloatType(), False),
        StructField("n_input_rows", IntegerType(), False),
        StructField("window_dt_s", FloatType(), False),
        StructField("baseline_length", FloatType(), False),
        StructField("delta_t_max", FloatType(), False),
        StructField("antenna1", IntegerType(), False),
        StructField("antenna2", IntegerType(), False),
        StructField("scan_number", IntegerType(), False),
        StructField("group_key_str", StringType(), True),
        StructField("input_rows_count", IntegerType(), True),
    ])
