"""
Spark Session - Simple Session Management

Minimal Spark session setup for interferometry data streaming.
Focused on essential functionality for Kafka integration.
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


def create_spark_session(config_path: str = None) -> SparkSession:
    """
    Create a simple Spark session for streaming.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration file. Uses defaults if None.
        
    Returns
    -------
    SparkSession
        Configured Spark session
    """
    # Initialize findspark
    findspark.init()
    
    # Load configuration
    config = load_config(config_path)
    
    # Create session
    spark = SparkSession.builder \
        .appName(config["spark"]["app_name"]) \
        .master(config["spark"]["master"]) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()
    
    # Reduce logging verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load Spark configuration.
    
    Parameters
    ----------
    config_path : str, optional
        Path to config file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
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
    Get Kafka options for Spark streaming.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Dict[str, str]
        Kafka options for Spark
    """
    kafka_config = config["kafka"]
    
    return {
        "kafka.bootstrap.servers": kafka_config["servers"],
        "subscribe": kafka_config["topic"],
        "startingOffsets": "latest"
    }


def get_visibility_row_schema() -> StructType:
    """
    Define Spark schema for individual visibility rows after chunk expansion.
    
    This schema matches the structure created by decompose_chunk_to_rows()
    to ensure compatibility with existing BDA processing.
    
    Returns
    -------
    StructType
        Spark schema for visibility rows
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
    Define Spark schema for grouped visibility data (baseline + scan).
    
    This schema is used after groupBy operations for BDA processing.
    
    Returns
    -------
    StructType
        Spark schema for grouped visibility data
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
    Define Spark schema for BDA processing results.
    
    This schema matches the output structure from BDA algorithms
    for consistent distributed processing.
    
    Returns
    -------
    StructType
        Spark schema for BDA results
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
