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
    
    # Create session with optimized configurations - MEMORY & PERFORMANCE OPTIMIZED
    spark = SparkSession.builder \
        .appName(config["spark"]["app_name"]) \
        .master(config["spark"]["master"]) \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.driver.memory", "3g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.python.worker.reuse", "true") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "false") \
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
