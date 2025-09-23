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
