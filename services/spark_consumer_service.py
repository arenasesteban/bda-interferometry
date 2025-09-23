#!/usr/bin/env python3
"""
Spark Consumer Service - Kafka to Spark Streaming Integration

Consumes interferometry visibility data from Kafka using Spark Structured Streaming.
Deserializes chunks and provides basic distributed processing with console output.

This service replaces the traditional Kafka consumer with Spark streaming capabilities
for distributed processing of large-scale interferometry datasets.

Usage:
    python spark_consumer_service.py [--kafka-servers SERVERS] [--topic TOPIC]

Examples:
    # Basic usage
    python spark_consumer_service.py
    
    # Custom Kafka configuration
    python spark_consumer_service.py --kafka-servers localhost:9092 --topic my-stream
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from processing.spark_session import create_spark_session, load_config, get_kafka_options
from processing.basic_analytics import ChunkAnalyzer, format_chunk_summary


def ensure_kafka_topic_exists(kafka_servers: str, topic: str, num_partitions: int = 4, max_retries: int = 30) -> bool:
    """
    Ensure that a Kafka topic exists, creating it if necessary with robust retry logic.
    
    This function checks if the specified topic exists in Kafka. If it doesn't exist,
    it creates the topic with the specified number of partitions and waits for it to
    be available. This is essential for the streaming architecture to handle topic
    creation before consumers connect.
    
    Parameters
    ----------
    kafka_servers : str
        Kafka bootstrap servers
    topic : str
        Topic name to create/verify
    num_partitions : int
        Number of partitions for the topic (default: 4 for distributed processing)
    max_retries : int
        Maximum number of retries to wait for topic availability (default: 30)
        
    Returns
    -------
    bool
        True if topic exists or was created successfully
    """
    import time
    
    try:
        from kafka.admin import KafkaAdminClient, NewTopic
        from kafka.errors import TopicAlreadyExistsError
        
        # Create admin client with longer timeout
        admin_client = KafkaAdminClient(
            bootstrap_servers=kafka_servers.split(','),
            client_id='spark_consumer_admin',
            request_timeout_ms=10000,
            connections_max_idle_ms=60000
        )
        
        # Retry loop to ensure topic is available
        for attempt in range(max_retries):
            try:
                # Check if topic exists
                existing_topics = admin_client.list_topics()
                if topic in existing_topics:
                    print(f"✅ Topic '{topic}' is available (attempt {attempt + 1})")
                    return True
                
                # If this is the first attempt, try to create the topic
                if attempt == 0:
                    topic_config = NewTopic(
                        name=topic,
                        num_partitions=num_partitions,
                        replication_factor=1
                    )
                    
                    print(f"🔧 Creating topic '{topic}' with {num_partitions} partitions...")
                    admin_client.create_topics([topic_config])
                    print(f"✅ Topic '{topic}' creation initiated")
                
                # Wait before next check
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting for topic '{topic}' to be available... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                    
            except TopicAlreadyExistsError:
                print(f"✅ Topic '{topic}' already exists (concurrent creation)")
                # Still need to verify it's available
                time.sleep(0.5)
                continue
                
            except Exception as e:
                if attempt == 0:
                    print(f"⚠️  Could not create topic '{topic}': {e}")
                    print(f"🔄 Will wait for auto-create or manual creation...")
                time.sleep(1)
                continue
        
        print(f"⚠️  Topic '{topic}' not confirmed available after {max_retries} attempts")
        print(f"🔄 Proceeding with streaming (relying on Kafka auto-create)")
        return False
        
    except Exception as e:
        print(f"❌ Error managing topic '{topic}': {e}")
        print(f"🔄 Will rely on Kafka auto-create during streaming")
        return False


def wait_for_kafka_ready(kafka_servers: str, timeout: int = 60) -> bool:
    """
    Wait for Kafka to be ready and responsive.
    
    Parameters
    ----------
    kafka_servers : str
        Kafka bootstrap servers
    timeout : int
        Maximum time to wait in seconds
        
    Returns
    -------
    bool
        True if Kafka is ready
    """
    import time
    from datetime import datetime, timedelta
    
    print(f"🔍 Checking Kafka connectivity to {kafka_servers}...")
    
    try:
        from kafka.admin import KafkaAdminClient
        
        end_time = datetime.now() + timedelta(seconds=timeout)
        
        while datetime.now() < end_time:
            try:
                admin_client = KafkaAdminClient(
                    bootstrap_servers=kafka_servers.split(','),
                    client_id='kafka_health_check',
                    request_timeout_ms=5000
                )
                
                # Try to list topics as a connectivity test
                topics = admin_client.list_topics()
                print(f"✅ Kafka is ready! Found {len(topics)} topics")
                return True
                
            except Exception as e:
                remaining = int((end_time - datetime.now()).total_seconds())
                if remaining > 0:
                    print(f"⏳ Kafka not ready yet... retrying in 2s (timeout: {remaining}s)")
                    time.sleep(2)
                else:
                    break
        
        print(f"❌ Kafka not ready after {timeout}s timeout")
        return False
        
    except Exception as e:
        print(f"❌ Cannot check Kafka connectivity: {e}")
        return False


def deserialize_chunk_udf():
    """
    Create UDF for deserializing MessagePack chunks in Spark.
    
    Returns
    -------
    function
        Spark UDF for chunk deserialization
    """
    import msgpack
    import numpy as np
    import zlib
    import json
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    
    def deserialize_chunk_data(raw_data_bytes):
        """
        Deserialize MessagePack chunk data.
        
        Parameters
        ----------
        raw_data_bytes : bytes
            Raw MessagePack serialized data
            
        Returns
        -------
        str
            JSON string of deserialized chunk metadata
        """
        try:
            # Unpack MessagePack data
            msgpack_chunk = msgpack.unpackb(raw_data_bytes, raw=False)
            
            # Extract metadata only (not full arrays for performance)
            metadata = {}
            
            for key, value in msgpack_chunk.items():
                if isinstance(value, dict) and value.get('type') in ['ndarray', 'ndarray_compressed']:
                    # For arrays, just extract shape and dtype info
                    metadata[key] = {
                        'shape': value.get('shape', []),
                        'dtype': value.get('dtype', 'unknown'),
                        'compressed': value.get('type') == 'ndarray_compressed'
                    }
                else:
                    # Keep simple values as-is
                    metadata[key] = value
            
            return json.dumps(metadata)
            
        except Exception as e:
            return json.dumps({'error': str(e), 'chunk_id': 'unknown'})
    
    return udf(deserialize_chunk_data, StringType())


def process_streaming_batch(df, epoch_id):
    """
    Process streaming batch with enhanced debugging and windowing information.
    
    Provides detailed feedback about microbatch status, timing, and data flow
    to help diagnose streaming pipeline behavior.
    """
    import time
    import json
    from datetime import datetime
    from pyspark.sql.functions import spark_partition_id, col, count, min as spark_min, max as spark_max
    
    # Enhanced batch header with timestamp
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n🚀 [MICROBATCH] Epoch {epoch_id} | {current_time}")
    print("=" * 70)
    
    # Get Spark context for core information
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    sc = spark.sparkContext
    
    start_time = time.time()
    
    # First check if DataFrame has any data at all
    try:
        # Get basic stats about the DataFrame without collecting
        row_count = df.count()
        
        print(f"📊 Batch Statistics:")
        print(f"   🔢 Row count: {row_count}")
        print(f"   🖥️  Available cores: {sc.defaultParallelism}")
        print(f"   ⚡ Trigger time: {current_time}")
        
        if row_count == 0:
            print(f"   ⏳ Status: EMPTY MICROBATCH")
            print(f"   💡 This is normal - waiting for data from producer")
            print(f"   🔍 Consumer is ready and listening...")
            processing_time = (time.time() - start_time) * 1000
            print(f"   ⏱️  Processing time: {processing_time:.1f}ms")
            print("-" * 70)
            return
        
        # If we have data, show timing information
        kafka_timestamps = df.select("kafka_timestamp").collect()
        if kafka_timestamps:
            timestamps = [row.kafka_timestamp for row in kafka_timestamps]
            min_ts = min(timestamps) if timestamps else None
            max_ts = max(timestamps) if timestamps else None
            
            print(f"   ⏰ Data timespan:")
            print(f"      📅 Earliest: {min_ts}")
            print(f"      📅 Latest: {max_ts}")
            
            # Calculate latency
            if max_ts:
                import pandas as pd
                latest_data_time = pd.to_datetime(max_ts)
                current_system_time = pd.Timestamp.now()
                latency = (current_system_time - latest_data_time).total_seconds()
                print(f"      ⚡ Latency: {latency:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Error getting batch stats: {e}")
        row_count = 0
    
    # Process data if we have any
    if row_count > 0:
        # Add partition information to DataFrame for core tracking
        df_with_partitions = df.withColumn("partition_id", spark_partition_id())
        
        # Collect data with partition info
        rows = df_with_partitions.collect()
        
        print(f"\n🔄 Processing {len(rows)} message(s):")
        print("-" * 40)
        
        for i, row in enumerate(rows):
            try:
                # Parse chunk metadata
                chunk_metadata = json.loads(row.chunk_data)
                
                if 'error' in chunk_metadata:
                    print(f"❌ [ERROR] Message {i}: {chunk_metadata.get('error', 'Unknown error')}")
                    continue
                
                chunk_id = chunk_metadata.get('chunk_id', f'msg_{i}')
                partition_id = row.partition_id
                kafka_timestamp = row.kafka_timestamp
                message_key = getattr(row, 'message_key', 'unknown')
                
                # Simulate distributed processing work
                processing_start = time.time()
                
                # Simulate computational work (replace with actual BDA later)
                simulated_work_time = simulate_distributed_processing(chunk_metadata, partition_id)
                
                processing_time = (time.time() - processing_start) * 1000  # ms
                
                # Core assignment (partition maps to core)
                assigned_core = partition_id % sc.defaultParallelism
                
                # Extract key metrics from chunk
                chunk_size = chunk_metadata.get('uvw', {}).get('shape', [0])[0] if 'uvw' in chunk_metadata else 0
                baseline = chunk_metadata.get('baseline', 'unknown')
                
                # Output distributed processing validation
                print(f"✅ Message {i+1}: {chunk_id}")
                print(f"   🔑 Key: {message_key}")
                print(f"   📡 Baseline: {baseline}")
                print(f"   🎯 Core: {assigned_core} | Partition: {partition_id}")
                print(f"   📏 Size: {chunk_size:,} rows")
                print(f"   🕐 Kafka time: {kafka_timestamp}")
                print(f"   ⏱️  Processing: {processing_time:.1f}ms")
                
            except Exception as e:
                print(f"❌ [PROCESSING ERROR] Message {i}: {e}")
    
    total_time = (time.time() - start_time) * 1000
    
    print("-" * 70)
    print(f"✅ [BATCH COMPLETE] Total time: {total_time:.1f}ms | Messages: {row_count}")
    print(f"🌐 Spark Web UI: http://localhost:4040")
    print()
    
    print(f"\n🚀 [TRUE STREAMING] Batch {epoch_id} - Processing {len(rows)} chunk(s)")
    print(f"📊 Available cores: {sc.defaultParallelism}")
    print("-" * 60)
    
    for i, row in enumerate(rows):
        try:
            # Parse chunk metadata
            chunk_metadata = json.loads(row.chunk_data)
            
            if 'error' in chunk_metadata:
                print(f"❌ [CHUNK ERROR] {chunk_metadata.get('chunk_id', 'unknown')}: {chunk_metadata['error']}")
                continue
            
            chunk_id = chunk_metadata.get('chunk_id', f'chunk_{i}')
            partition_id = row.partition_id
            kafka_timestamp = row.kafka_timestamp
            
            # Simulate distributed processing work
            processing_start = time.time()
            
            # Simulate computational work (replace with actual BDA later)
            simulated_work_time = simulate_distributed_processing(chunk_metadata, partition_id)
            
            processing_time = (time.time() - processing_start) * 1000  # ms
            
            # Core assignment (partition maps to core)
            assigned_core = partition_id % sc.defaultParallelism
            
            # Extract key metrics from chunk
            chunk_size = chunk_metadata.get('uvw', {}).get('shape', [0])[0] if 'uvw' in chunk_metadata else 0
            
            # Output distributed processing validation
            print(f"✅ [DISTRIBUTED] Chunk: {chunk_id}")
            print(f"   🎯 Core: {assigned_core} | Partition: {partition_id}")
            print(f"   📏 Size: {chunk_size:,} rows")
            print(f"   ⏱️  Processing: {processing_time:.1f}ms")
            print(f"   🕐 Kafka Time: {kafka_timestamp}")
            print(f"   ⚡ Simulated Work: {simulated_work_time:.1f}ms")
            
        except Exception as e:
            print(f"❌ [PROCESSING ERROR] Chunk {i}: {e}")
    
    total_time = (time.time() - start_time) * 1000
    
    print("-" * 60)
    print(f"� [BATCH SUMMARY] Total time: {total_time:.1f}ms | Chunks: {len(rows)}")
    print(f"🌐 Spark Web UI: http://localhost:4040")
    print()


def simulate_distributed_processing(chunk_metadata: dict, partition_id: int) -> float:
    """
    Simulate distributed processing work for validation.
    
    This function simulates the computational work that would be done
    in real BDA processing, distributed across different cores.
    
    Parameters
    ----------
    chunk_metadata : dict
        Chunk metadata containing array shapes and properties
    partition_id : int
        Spark partition ID (maps to core)
        
    Returns
    -------
    float
        Simulated processing time in milliseconds
    """
    import time
    import random
    
    # Simulate work based on data size and partition
    data_points = 1
    for key in ['uvw', 'vis', 'weight']:
        if key in chunk_metadata and 'shape' in chunk_metadata[key]:
            shape = chunk_metadata[key]['shape']
            data_points *= max(shape) if shape else 1
    
    # Simulate processing time (varies by core/partition)
    base_time = min(data_points / 100000, 0.1)  # Scale with data size
    core_variation = (partition_id + 1) * 0.01    # Small variation per core
    
    simulated_time = base_time + core_variation + random.uniform(-0.01, 0.01)
    
    # Actually consume some CPU time for realism
    time.sleep(max(simulated_time, 0.001))
    
    return simulated_time * 1000  # Return in milliseconds


def run_spark_consumer(kafka_servers: str = "localhost:9092", 
                      topic: str = "visibility-stream",
                      config_path: str = None) -> None:
    """
    Run the Spark streaming consumer with enhanced debugging and real-time feedback.
    """
    from datetime import datetime
    
    start_time = datetime.now().strftime("%H:%M:%S")
    
    print("🚀 BDA Interferometry Spark Streaming Consumer")
    print("🎯 VERSION: Enhanced Debugging with Real-time Feedback")
    print("=" * 60)
    print(f"📡 Kafka Servers: {kafka_servers}")
    print(f"📻 Topic: {topic}")
    print(f"🕐 Started: {start_time}")
    print(f"⚡ Microbatch Trigger: 2 seconds")
    print(f"� Watermark: 10 seconds")
    print(f"🪟 Window: 2 seconds")
    print("=" * 60)
    
    # Check Kafka connectivity first
    print("🔍 Phase 1: Checking Kafka connectivity...")
    if not wait_for_kafka_ready(kafka_servers, timeout=60):
        print("❌ Kafka is not ready. Please check Kafka service status.")
        print("💡 Try: docker-compose up -d")
        return
    print("✅ Kafka is ready and responsive")
    
    # Ensure Kafka topic exists before connecting
    print(f"\n🔍 Phase 2: Verifying topic '{topic}'...")
    topic_ready = ensure_kafka_topic_exists(kafka_servers, topic, num_partitions=4, max_retries=30)
    
    if topic_ready:
        print(f"✅ Topic '{topic}' is ready for streaming")
    else:
        print(f"⚠️  Topic '{topic}' not confirmed, proceeding with auto-create")
    
    # Add a small delay to ensure topic propagation
    print("⏳ Allowing time for topic propagation...")
    import time
    time.sleep(2)
    
    # Create Spark session
    print("\n� Phase 3: Initializing Spark session...")
    spark = create_spark_session(config_path)
    print(f"✅ Spark session created with {spark.sparkContext.defaultParallelism} cores")
    
    try:
        # Create deserializer UDF
        deserialize_udf = deserialize_chunk_udf()
        print("✅ Deserializer UDF configured")
        
        # Read from Kafka with enhanced configuration for robustness
        print("\n� Phase 4: Connecting to Kafka stream...")
        kafka_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "earliest") \
            .option("failOnDataLoss", "false") \
            .option("kafka.auto.create.topics.enable", "true") \
            .option("kafka.num.partitions", "4") \
            .option("kafka.metadata.max.age.ms", "5000") \
            .option("kafka.session.timeout.ms", "30000") \
            .option("kafka.request.timeout.ms", "40000") \
            .option("kafka.retry.backoff.ms", "1000") \
            .option("maxOffsetsPerTrigger", "1000") \
            .load()
        
        print("✅ Connected to Kafka stream")
        
        # Add event_time column and apply windowing/watermarking
        from pyspark.sql.functions import from_unixtime, to_timestamp, window, col
        
        # Deserialize and add timing information
        processed_df = kafka_df.select(
            kafka_df.key.cast("string").alias("message_key"),
            deserialize_udf(kafka_df.value).alias("chunk_data"),
            kafka_df.timestamp.alias("kafka_timestamp"),
            to_timestamp(from_unixtime(kafka_df.timestamp.cast("long") / 1000)).alias("event_time")
        )
        
        # Apply watermarking and windowing
        windowed_df = processed_df \
            .withWatermark("event_time", "10 seconds") \
            .groupBy(
                window(col("event_time"), "2 seconds"),
                col("message_key")
            ) \
            .count() \
            .select("window", "message_key", "count")
        
        print("✅ Windowing and watermarking configured")
        print("   🪟 Window: 2 seconds")
        print("   💧 Watermark: 10 seconds")
        print("   🔑 Grouping: by baseline (message key)")
        
        # Configure Spark for optimal performance
        spark.conf.set("spark.sql.shuffle.partitions", "4")
        spark.conf.set("spark.sql.adaptive.enabled", "false")
        
        # Start streaming query with detailed monitoring
        print("\n🔍 Phase 5: Starting streaming query...")
        query = processed_df.writeStream \
            .foreachBatch(process_streaming_batch) \
            .outputMode("append") \
            .trigger(processingTime='2 seconds') \
            .option("checkpointLocation", "/tmp/spark-streaming-checkpoint") \
            .start()
        
        current_time = datetime.now().strftime("%H:%M:%S")
        print("✅ Streaming query started successfully!")
        print("=" * 60)
        print("🎯 CONSUMER STATUS: READY AND LISTENING")
        print(f"⏰ Waiting for data from producer...")
        print(f"🔄 Microbatches will trigger every 2 seconds")
        print(f"📊 Each empty batch means: 'ready and waiting'")
        print(f"🌐 Monitor Spark UI: http://localhost:4040")
        print(f"🛑 Press Ctrl+C to stop")
        print("=" * 60)
        
        # Wait for termination with periodic status updates
        try:
            while query.isActive:
                time.sleep(10)
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"💓 [{current_time}] Consumer heartbeat - still listening...")
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received...")
        
        query.awaitTermination()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping consumer...")
        
    except Exception as e:
        print(f"❌ Error in streaming: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🔒 Shutting down Spark session...")
        spark.stop()
        print("✅ Consumer stopped successfully")


def main():
    """Main entry point for Spark consumer service."""
    
    parser = argparse.ArgumentParser(
        description="BDA Interferometry Spark Consumer Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python spark_consumer_service.py
  python spark_consumer_service.py --kafka-servers localhost:9092
  python spark_consumer_service.py --topic my-visibility-stream
        """
    )
    
    parser.add_argument(
        "--kafka-servers",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)"
    )
    
    parser.add_argument(
        "--topic",
        default="visibility-stream", 
        help="Kafka topic name (default: visibility-stream)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    try:
        run_spark_consumer(
            kafka_servers=args.kafka_servers,
            topic=args.topic,
            config_path=args.config
        )
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
