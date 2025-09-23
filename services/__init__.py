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
    run_consumer_service,
    create_kafka_consumer,
    consume_chunks_from_kafka,
    validate_chunk_structure,
    process_chunk,
    deserialize_chunk
)

__all__ = [
    # Producer functions
    'run_producer_service',
    'create_kafka_producer',
    'stream_chunks_to_kafka',
    'serialize_chunk',
    
    # Consumer functions
    'run_consumer_service',
    'process_chunks_with_validation',
    'create_kafka_consumer',
    'consume_chunks_from_kafka', 
    'validate_chunk_structure',
    'process_chunk',
    'deserialize_chunk'
]
