"""
Services Package - Kafka Producer and Consumer Microservices

Contains the core microservices for streaming radio interferometry visibility data
through Apache Kafka. Implements producer service for dataset chunking and streaming,
and consumer service for chunk processing and validation.

Modules
-------
producer_service : Producer microservice for streaming visibility chunks
consumer_service : Consumer microservice for processing received chunks
"""

# Producer functions
from .producer_service import (
    run_producer_service,
    create_kafka_producer,
    stream_chunks_to_kafka,
    serialize_chunk
)

# Consumer functions  
from .consumer_service import (
    define_visibility_schema,
    process_chunk,
    deserialize_chunk_to_rows,
    process_streaming_batch,
    normalize_baseline_key,
    run_consumer,
)

__all__ = [
    # Producer functions
    'run_producer_service',
    'create_kafka_producer',
    'stream_chunks_to_kafka',
    'serialize_chunk',
    
    # Consumer functions
    'define_visibility_schema',
    'process_chunk',
    'deserialize_chunk_to_rows',
    'process_streaming_batch',
    'normalize_baseline_key',
    'run_consumer',
]
