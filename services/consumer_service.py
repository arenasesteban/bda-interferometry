#!/usr/bin/env python3
"""
Consumer Service - Interferometry Data Reception Microservice

Independent microservice that receives and processes visibility chunks
from Kafka topics with lossless decompression and validation.

This service consumes compressed visibility data chunks from Kafka topics,
deserializes them with complete data integrity preservation, and processes
them for scientific analysis.
"""

import sys
import os
import msgpack
import numpy as np
import zlib
from pathlib import Path
import argparse
from typing import Dict, Any, Generator

# Kafka imports
from kafka import KafkaConsumer
from kafka.errors import KafkaError

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))


def deserialize_chunk(raw_data: bytes) -> Dict[str, Any]:
    """
    Deserialize visibility chunk from MessagePack with lossless decompression.
    
    Handles both compressed and uncompressed arrays, restoring original
    data exactly as transmitted without any data loss.
    
    Parameters
    ----------
    raw_data : bytes
        MessagePack serialized data
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing deserialized and decompressed chunk data
    """
    
    try:
        msgpack_chunk = msgpack.unpackb(raw_data, raw=False)
        chunk = {}
        
        for key, value in msgpack_chunk.items():
            if isinstance(value, dict):
                if value.get('type') == 'ndarray_compressed':
                    # Decompress and reconstruct array with original type
                    compressed_data = value['data']
                    decompressed_data = zlib.decompress(compressed_data)
                    
                    # Reconstruct array with original data type
                    array = np.frombuffer(
                        decompressed_data, 
                        dtype=value['dtype']
                    ).reshape(value['shape'])
                    
                    chunk[key] = array
                    
                elif value.get('type') == 'ndarray':
                    # Uncompressed array
                    array = np.frombuffer(
                        value['data'], 
                        dtype=value['dtype']
                    ).reshape(value['shape'])
                    
                    chunk[key] = array
                else:
                    chunk[key] = value
            else:
                chunk[key] = value
        
        return chunk
        
    except Exception:
        return {}


def create_kafka_consumer(kafka_servers=['localhost:9092'], 
                         topic='visibility-stream', 
                         group_id='bda-consumers'):
    """
    Create Kafka consumer with optimized configuration.
    
    Initializes consumer instance with settings optimized for visibility
    data streaming with automatic deserialization.
    
    Parameters
    ----------
    kafka_servers : list of str, optional
        List of Kafka server addresses (default: ['localhost:9092'])
    topic : str, optional
        Kafka topic name (default: 'visibility-stream')
    group_id : str, optional
        Consumer group identifier (default: 'bda-consumers')
        
    Returns
    -------
    KafkaConsumer
        Configured Kafka consumer instance
    """
    
    return KafkaConsumer(
        topic,
        bootstrap_servers=kafka_servers,
        group_id=group_id,
        value_deserializer=deserialize_chunk,
        key_deserializer=lambda x: x.decode('utf-8') if x else None,
        enable_auto_commit=True,
        auto_offset_reset='latest'
    )


def validate_chunk_structure(chunk: Dict[str, Any]) -> bool:
    """
    Validate basic structure of visibility chunk.
    
    Performs comprehensive validation of chunk metadata and data arrays
    to ensure proper structure and consistency for processing.
    
    Parameters
    ----------
    chunk : Dict[str, Any]
        Chunk dictionary to validate
        
    Returns
    -------
    bool
        True if chunk structure is valid, False otherwise
    """
    
    required_metadata = [
        'subms_id', 'chunk_id', 'field_id', 'spw_id', 'polarization_id',
        'row_start', 'row_end', 'nrows', 'n_channels', 'n_correlations'
    ]
    
    required_arrays = [
        'u', 'v', 'w', 'visibilities', 'weight', 'time', 
        'antenna1', 'antenna2', 'flag'
    ]
    
    for field in required_metadata:
        if field not in chunk:
            return False
    
    for array_name in required_arrays:
        if array_name not in chunk:
            return False
        
        arr = chunk[array_name]
        if arr is not None and not isinstance(arr, np.ndarray):
            return False
    
    nrows = chunk.get('nrows', 0)
    if nrows > 0:
        for array_name in ['u', 'v', 'w', 'time', 'antenna1', 'antenna2']:
            arr = chunk.get(array_name)
            if arr is not None and len(arr.shape) > 0 and arr.shape[0] != nrows:
                return False
    
    return True


def consume_chunks_from_kafka(consumer, max_chunks=None) -> Generator[Dict[str, Any], None, None]:
    """
    Consume chunks from Kafka continuously.
    
    Iterates over Kafka messages to extract and yield deserialized
    visibility chunks until specified limit is reached.
    
    Parameters
    ----------
    consumer : KafkaConsumer
        Configured Kafka consumer instance
    max_chunks : int, optional
        Maximum number of chunks to process (None for unlimited)
        
    Yields
    ------
    Dict[str, Any]
        Deserialized visibility chunks
        
    Raises
    ------
    Exception
        If consumption encounters errors during processing
    """
    
    chunks_consumed = 0
    
    try:
        for message in consumer:
            if message.value:
                yield message.value
                chunks_consumed += 1
                
                if max_chunks and chunks_consumed >= max_chunks:
                    break
                    
    except Exception:
        raise


def process_chunk(chunk: Dict[str, Any]) -> bool:
    """
    Process individual visibility chunk.
    
    Validates and processes a single chunk of visibility data.
    This is where scientific processing logic would be implemented.
    
    Parameters
    ----------
    chunk : Dict[str, Any]
        Deserialized chunk dictionary
        
    Returns
    -------
    bool
        True if processing was successful, False otherwise
    """
    
    try:
        if not validate_chunk_structure(chunk):
            return False
        
        # Scientific processing logic would go here
        return True
        
    except Exception:
        return False


def run_consumer_service(kafka_servers=['localhost:9092'], 
                        topic='visibility-stream', 
                        group_id='bda-consumers',
                        max_chunks=None) -> Dict[str, Any]:
    """
    Execute complete consumer service.
    
    Main entry point for consumer service. Sets up Kafka consumer,
    processes chunks, and returns processing statistics.
    
    Parameters
    ----------
    kafka_servers : list of str, optional
        List of Kafka server addresses (default: ['localhost:9092'])
    topic : str, optional
        Kafka topic name (default: 'visibility-stream')
    group_id : str, optional
        Consumer group identifier (default: 'bda-consumers')
    max_chunks : int, optional
        Maximum number of chunks to process (None for unlimited)
        
    Returns
    -------
    Dict[str, Any]
        Processing statistics including success status and counts
    """
    
    consumer = None
    
    try:
        consumer = create_kafka_consumer(kafka_servers, topic, group_id)
        
        chunks_received = 0
        chunks_processed = 0
        
        for chunk in consume_chunks_from_kafka(consumer, max_chunks):
            chunks_received += 1
            
            if process_chunk(chunk):
                chunks_processed += 1
        
        return {
            'success': True,
            'chunks_received': chunks_received,
            'chunks_processed': chunks_processed
        }
        
    except KeyboardInterrupt:
        return {
            'success': True,
            'chunks_received': chunks_received if 'chunks_received' in locals() else 0,
            'chunks_processed': chunks_processed if 'chunks_processed' in locals() else 0,
            'interrupted': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'chunks_received': 0,
            'chunks_processed': 0
        }
        
    finally:
        if consumer:
            consumer.close()


def main():
    """Main entry point for consumer microservice."""
    
    parser = argparse.ArgumentParser(description="BDA Interferometry Consumer Service")
    
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
    parser.add_argument(
        "--group-id", 
        default="bda-consumers",
        help="Consumer group identifier"
    )
    parser.add_argument(
        "--max-chunks", 
        type=int,
        help="Maximum number of chunks to process (unlimited by default)"
    )
    
    args = parser.parse_args()
    
    # Parse Kafka servers
    kafka_servers = [s.strip() for s in args.kafka_servers.split(',')]
    
    # Execute service
    result = run_consumer_service(
        kafka_servers=kafka_servers,
        topic=args.topic,
        group_id=args.group_id,
        max_chunks=args.max_chunks
    )
    
    # Exit based on result
    sys.exit(0 if result['success'] else 1)


if __name__ == "__main__":
    main()
