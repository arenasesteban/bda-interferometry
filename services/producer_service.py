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
from pathlib import Path
import argparse
from typing import Dict, Any

# Kafka imports
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from data.simulation import generate_dataset
from data.extraction import stream_subms_chunks


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


def create_kafka_producer(kafka_servers=['localhost:9092']):
    """
    Create Kafka producer with optimized configuration for large messages.
    
    Configured to handle large scientific chunks with multiple compression layers:
    - zlib compression at array level
    - gzip compression at Kafka level
    - Extended limits for large messages
    
    Parameters
    ----------
    kafka_servers : list of str, optional
        List of Kafka server addresses (default: ['localhost:9092'])
        
    Returns
    -------
    KafkaProducer
        Configured producer for large message transmission
    """
    
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


def stream_chunks_to_kafka(dataset, producer, topic: str) -> int:
    """
    Stream all dataset chunks to Kafka topic.
    
    Transmits each chunk from the dataset as individual Kafka messages
    with automatic serialization and compression.
    
    Parameters
    ----------
    dataset : object
        Pyralysis dataset object containing visibility data
    producer : KafkaProducer
        Configured Kafka producer instance
    topic : str
        Kafka topic name for streaming
        
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
                    
            except KafkaError:
                continue
                
    except Exception:
        raise
        
    finally:
        producer.flush()
    
    return chunks_sent


def run_producer_service(antenna_config_path: str, 
                        kafka_servers=['localhost:9092'], 
                        topic='visibility-stream') -> Dict[str, Any]:
    """
    Execute complete production service.
    
    Generates simulated visibility data and transmits it via Kafka
    with comprehensive error handling and result reporting.
    
    Parameters
    ----------
    antenna_config_path : str
        Path to antenna configuration file
    kafka_servers : list of str, optional
        List of Kafka server addresses (default: ['localhost:9092'])
    topic : str, optional
        Kafka topic for visibility streaming (default: 'visibility-stream')
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing transmission statistics and success status
    """
    
    producer = None
    
    try:
        # Generate dataset
        dataset = generate_dataset(
            antenna_config_path=antenna_config_path,
            freq_start=35.0,
            freq_end=50.0,
            n_frequencies=50,
            date_string="2002-05-10",
            observation_time="1h",
            declination="-45d00m00s",
            integration_time=180.0,
            n_point_sources=15,
            point_flux_density=1.0,
            point_spectral_index=3.0,
            include_gaussian=True,
            gaussian_flux_density=10.0,
            gaussian_position=(0, 0),
            gaussian_minor_radius=20.0,
            gaussian_major_radius=30.0,
            gaussian_theta_angle=60.0
        )

        # Create producer and stream chunks
        producer = create_kafka_producer(kafka_servers)
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
        "--antenna-config", 
        default="./antenna_configs/alma.cycle10.1.cfg",
        help="Path to antenna configuration file"
    )
    parser.add_argument(
        "--kafka-servers", 
        default="localhost:9092",
        help="Kafka servers (comma-separated)"
    )
    parser.add_argument(
        "--topic", 
        default="visibility-stream",
        help="Kafka topic for visibility streaming"
    )
    
    args = parser.parse_args()
    
    # Parse Kafka servers
    kafka_servers = [s.strip() for s in args.kafka_servers.split(',')]
    
    # Execute service
    result = run_producer_service(
        antenna_config_path=args.antenna_config,
        kafka_servers=kafka_servers,
        topic=args.topic
    )
    
    # Exit based on result
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
