"""
Processing Module

Distributed processing utilities for interferometry data analysis.
Provides basic analytics and Spark session management for streaming.
"""

from .spark_session import create_spark_session, load_config, get_kafka_options

__all__ = [
    'create_spark_session',
    'load_config',
    'get_kafka_options'
]
