#!/usr/bin/env python3
"""
Producer Service - Interferometry Data Streaming Microservice

Independent microservice that generates Pyralysis datasets and transmits
chunks via Kafka using functional programming and lossless compression.

This service creates simulated interferometry datasets and streams them
as compressed chunks to Kafka topics for real-time data transmission.
"""

import sys
import os
import msgpack
import numpy as np
import zlib
import json
import time
from pathlib import Path
import argparse
from typing import Dict, Any

# Kafka imports
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Configuration constants
DEFAULT_KAFKA_SERVERS = ['localhost:9092']
DEFAULT_TOPIC = 'visibility-stream'

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from data.simulation import generate_dataset
from data.extraction import stream_subms_chunks


def load_simulation_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load simulation configuration from JSON file or return defaults.
    
    Parameters
    ----------
    config_path : str, optional
        Path to JSON configuration file. If None, returns default configuration.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing simulation parameters
    """
    
    default_config = {
        "freq_start": 35.0,
        "freq_end": 50.0,
        "n_frequencies": 50,
        "date_string": "2002-05-10",
        "observation_time": "1h",
        "declination": "-45d00m00s",
        "integration_time": 180.0,
        "n_point_sources": 15,
        "point_flux_density": 1.0,
        "point_spectral_index": 3.0,
        "include_gaussian": True,
        "gaussian_flux_density": 10.0,
        "gaussian_position": [0, 0],
        "gaussian_minor_radius": 20.0,
        "gaussian_major_radius": 30.0,
        "gaussian_theta_angle": 60.0
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge user config with defaults
            default_config.update(user_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    
    return default_config


def serialize_chunk(chunk: Dict[str, Any]) -> bytes:
    """
    Serialize visibility chunk using optimized MessagePack with lossless compression.
    
    Applies zlib compression to large arrays while preserving complete
    scientific data integrity of the original data.
    
    Parameters
    ----------
    chunk : Dict[str, Any]
        Dictionary containing visibility chunk data
        
    Returns
    -------
    bytes
        Serialized and compressed data with MessagePack
    """
    
    msgpack_chunk = {}
    
    for key, value in chunk.items():
        if isinstance(value, np.ndarray):
            # Serialize array without modifying data types
            array_bytes = value.tobytes()
            
            # Apply zlib compression for large arrays (>10KB)
            if len(array_bytes) > 10240:
                compressed_data = zlib.compress(array_bytes, level=6)
                
                msgpack_chunk[key] = {
                    'type': 'ndarray_compressed',
                    'data': compressed_data,
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            else:
                msgpack_chunk[key] = {
                    'type': 'ndarray',
                    'data': array_bytes,
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
        else:
            msgpack_chunk[key] = value
    
    return msgpack.packb(msgpack_chunk, use_bin_type=True, strict_types=False)


def create_kafka_producer(kafka_servers=None):
    """
    Create Kafka producer with optimized configuration for large messages.
    
    Configured to handle large scientific chunks with multiple compression layers:
    - zlib compression at array level
    - gzip compression at Kafka level
    - Extended limits for large messages
    
    Parameters
    ----------
    kafka_servers : list of str, optional
        List of Kafka server addresses. If None, uses default servers.
        
    Returns
    -------
    KafkaProducer
        Configured producer for large message transmission
    """
    
    if kafka_servers is None:
        kafka_servers = DEFAULT_KAFKA_SERVERS
    
    return KafkaProducer(
        bootstrap_servers=kafka_servers,
        compression_type='gzip',
        batch_size=32768,
        linger_ms=50,
        max_request_size=5242880,  # 5MB maximum per message
        buffer_memory=67108864,    # 64MB buffer
        retries=3,
        value_serializer=serialize_chunk,
        key_serializer=lambda x: str(x).encode('utf-8') if x is not None else None
    )


def stream_chunks_to_kafka(dataset, producer, topic: str, streaming_delay: float = 0.1) -> int:
    """
    Stream dataset chunks to Kafka with true streaming simulation.
    
    Transmits each chunk individually with configurable delay to simulate
    continuous data acquisition from radio interferometer.
    
    Parameters
    ----------
    dataset : object
        Pyralysis dataset object containing visibility data
    producer : KafkaProducer
        Configured Kafka producer instance
    topic : str
        Kafka topic name for streaming
    streaming_delay : float, optional
        Delay between chunk transmissions in seconds (default: 0.1s)
        
    Returns
    -------
    int
        Number of chunks successfully transmitted
        
    Raises
    ------
    Exception
        If streaming encounters errors during transmission
    """
    
    chunks_sent = 0
    
    try:
        for chunk in stream_subms_chunks(dataset):
            key = f"{chunk['subms_id']}_{chunk['chunk_id']}"
            
            try:
                future = producer.send(topic, value=chunk, key=key)
                future.get(timeout=30)
                chunks_sent += 1
                
                # Simulate streaming delay
                time.sleep(streaming_delay)
                    
            except KafkaError:
                continue
                
    except Exception:
        raise
        
    finally:
        producer.flush()
    
    return chunks_sent


def run_producer_service(antenna_config_path: str, 
                        simulation_config_path: str = None,
                        topic: str = None) -> Dict[str, Any]:
    """
    Execute complete production service.
    
    Generates simulated visibility data and transmits it via Kafka
    with comprehensive error handling and result reporting.
    
    Parameters
    ----------
    antenna_config_path : str
        Path to antenna configuration file
    simulation_config_path : str, optional
        Path to simulation configuration JSON file. If None, uses defaults.
    topic : str, optional
        Kafka topic for visibility streaming. If None, uses default topic.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing transmission statistics and success status
    """
    
    producer = None
    
    if topic is None:
        topic = DEFAULT_TOPIC
    
    try:
        # Load simulation configuration
        sim_config = load_simulation_config(simulation_config_path)
        
        # Generate dataset with loaded configuration
        dataset = generate_dataset(
            antenna_config_path=antenna_config_path,
            freq_start=sim_config["freq_start"],
            freq_end=sim_config["freq_end"],
            n_frequencies=sim_config["n_frequencies"],
            date_string=sim_config["date_string"],
            observation_time=sim_config["observation_time"],
            declination=sim_config["declination"],
            integration_time=sim_config["integration_time"],
            n_point_sources=sim_config["n_point_sources"],
            point_flux_density=sim_config["point_flux_density"],
            point_spectral_index=sim_config["point_spectral_index"],
            include_gaussian=sim_config["include_gaussian"],
            gaussian_flux_density=sim_config["gaussian_flux_density"],
            gaussian_position=tuple(sim_config["gaussian_position"]),
            gaussian_minor_radius=sim_config["gaussian_minor_radius"],
            gaussian_major_radius=sim_config["gaussian_major_radius"],
            gaussian_theta_angle=sim_config["gaussian_theta_angle"]
        )

        # Create producer and stream chunks
        producer = create_kafka_producer()
        chunks_sent = stream_chunks_to_kafka(dataset, producer, topic)
        
        return {
            'success': True,
            'chunks_sent': chunks_sent
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'chunks_sent': 0
        }
        
    finally:
        if producer:
            producer.flush()
            producer.close()


def main():
    """Main entry point for producer microservice."""
    
    parser = argparse.ArgumentParser(description="BDA Interferometry Producer Service")
    
    parser.add_argument(
        "antenna_config", 
        help="Path to antenna configuration file (required)"
    )
    parser.add_argument(
        "--simulation-config", 
        help="Path to simulation configuration JSON file (optional, uses defaults if not provided)"
    )
    parser.add_argument(
        "--topic", 
        help=f"Kafka topic for visibility streaming (default: {DEFAULT_TOPIC})"
    )
    
    args = parser.parse_args()
    
    # Execute service
    result = run_producer_service(
        antenna_config_path=args.antenna_config,
        simulation_config_path=args.simulation_config,
        topic=args.topic
    )
    
    # Print result summary
    if result['success']:
        print(f"✓ Successfully sent {result['chunks_sent']} chunks to Kafka")
    else:
        print(f"✗ Error: {result['error']}")
    
    # Exit based on result
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
