"""
Consumer Service - Interferometry Data Streaming Microservice

Consumes interferometry visibility data from Kafka using Spark Structured Streaming.
Deserializes chunks and provides basic distributed processing with console output.

This service replaces the traditional Kafka consumer with Spark streaming capabilities
for distributed processing of large-scale interferometry datasets.
"""

import sys
import argparse
from pathlib import Path
import time
import traceback
from datetime import datetime, timedelta
import msgpack
import numpy as np
import zlib
import json
import random

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, count, avg, sum as spark_sum, countDistinct
from pyspark.sql.types import StringType
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from processing.spark_session import create_spark_session
from bda.bda_processor import process_microbatch_with_bda, format_bda_result_for_output, create_bda_summary_stats
from bda.bda_config import load_bda_config_with_fallback, get_default_bda_config
from bda.bda_integration import apply_distributed_bda_pipeline


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
    
    try:
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
    
    print(f"🔍 Checking Kafka connectivity to {kafka_servers}...")
    
    try:        
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


def deserialize_chunk_data(raw_data_bytes: bytes) -> dict:
    """
    Deserialize MessagePack chunk data from Kafka.
    
    Deserializes complete chunk with all arrays for row decomposition.
    
    Parameters
    ----------
    raw_data_bytes : bytes
        Raw MessagePack serialized data from Kafka
        
    Returns
    -------
    dict
        Deserialized chunk with numpy arrays reconstructed
    """
    
    try:
        # Deserialize MessagePack
        msgpack_data = msgpack.unpackb(raw_data_bytes, raw=False, strict_map_key=False)
        
        chunk = {}
        for key, value in msgpack_data.items():
            if isinstance(value, dict) and 'type' in value:
                if value['type'] == 'ndarray_compressed':
                    # Decompress and reconstruct array
                    decompressed = zlib.decompress(value['data'])
                    array = np.frombuffer(decompressed, dtype=value['dtype'])
                    chunk[key] = array.reshape(value['shape'])
                elif value['type'] == 'ndarray':
                    # Reconstruct array directly
                    array = np.frombuffer(value['data'], dtype=value['dtype'])
                    chunk[key] = array.reshape(value['shape'])
                else:
                    chunk[key] = value
            else:
                chunk[key] = value
                
        return chunk
        
    except Exception as e:
        print(f"❌ Error deserializing chunk: {e}")
        return {}


def decompose_chunk_to_rows(chunk: dict) -> list:
    """
    Decompose chunk into individual rows for BDA processing.
    
    Converts a chunk containing multiple visibility rows into a list of
    individual records, each representing one row with its baseline_key
    and scan_number.
    
    Parameters
    ----------
    chunk : dict
        Deserialized chunk with scientific arrays
        
    Returns
    -------
    list
        List of individual rows with baseline_key and metadata
    """

    rows = []
    
    try:
        nrows = chunk.get('nrows', 0)
        subms_id = chunk.get('subms_id', 'unknown')
        
        if nrows == 0:
            return rows
            
        # Extract main arrays
        antenna1 = chunk.get('antenna1', np.array([]))
        antenna2 = chunk.get('antenna2', np.array([]))
        scan_number = chunk.get('scan_number', np.array([]))
        time_array = chunk.get('time', np.array([]))
        u_array = chunk.get('u', np.array([]))
        v_array = chunk.get('v', np.array([]))
        w_array = chunk.get('w', np.array([]))
        visibilities = chunk.get('visibilities', np.array([]))
        weight = chunk.get('weight', np.array([]))
        flag = chunk.get('flag', np.array([]))
        
        # Validate dimensions
        if len(antenna1) != nrows or len(antenna2) != nrows:
            print(f"⚠️  Dimension mismatch in chunk {chunk.get('chunk_id', 'unknown')}")
            return rows
            
        # Decompose row by row
        for row_idx in range(nrows):
            ant1 = int(antenna1[row_idx]) if len(antenna1) > row_idx else -1
            ant2 = int(antenna2[row_idx]) if len(antenna2) > row_idx else -1
            scan_num = int(scan_number[row_idx]) if len(scan_number) > row_idx else -1
            
            # Create normalized baseline_key
            baseline_key = normalize_baseline_key(ant1, ant2, subms_id)
            
            # Create individual row record
            row_record = {
                # Identifiers for grouping
                'baseline_key': baseline_key,
                'scan_number': scan_num,
                'antenna1': ant1,
                'antenna2': ant2,
                'subms_id': subms_id,
                
                # Temporal and spatial metadata
                'time': float(time_array[row_idx]) if len(time_array) > row_idx else 0.0,
                'u': float(u_array[row_idx]) if len(u_array) > row_idx else 0.0,
                'v': float(v_array[row_idx]) if len(v_array) > row_idx else 0.0,
                'w': float(w_array[row_idx]) if len(w_array) > row_idx else 0.0,
                
                # Real scientific data for BDA
                'visibility': visibilities[row_idx] if len(visibilities) > row_idx else np.array([]),
                'weight': weight[row_idx] if len(weight) > row_idx else np.array([]),
                'flag': flag[row_idx] if len(flag) > row_idx else np.array([]),
                
                # Original chunk metadata
                'original_chunk_id': chunk.get('chunk_id', -1),
                'row_index_in_chunk': row_idx,
                'field_id': chunk.get('field_id', -1),
                'spw_id': chunk.get('spw_id', -1),
            }
            
            rows.append(row_record)
            
    except Exception as e:
        print(f"❌ Error decomposing chunk: {e}")
        
    return rows


def group_rows_by_baseline_scan(rows: list) -> dict:
    """
    Group individual rows by baseline_key + scan_number.
    
    Parameters
    ----------
    rows : list
        List of individual rows decomposed from chunks
        
    Returns
    -------
    dict
        Dictionary with group keys (baseline_key, scan_number)
    """
    groups = {}
    
    for row in rows:
        baseline_key = row.get('baseline_key', 'unknown')
        scan_number = row.get('scan_number', -1)
        
        # Create group key using consistent function
        # Extract antenna1, antenna2 from baseline_key if possible
        if 'antenna1' in row and 'antenna2' in row:
            ant1 = row['antenna1']
            ant2 = row['antenna2']
            subms_id = row.get('subms_id', None)
            group_key = create_group_key(ant1, ant2, scan_number, subms_id)
        else:
            # Fallback to previous method
            group_key = f"{baseline_key}_scan{scan_number}"
        
        if group_key not in groups:
            groups[group_key] = []
            
        groups[group_key].append(row)
        
    return groups


def log_chunk_summary(chunk_metadata: dict, rows: list) -> None:
    """
    Log concise summary with essential chunk information.
    
    Parameters
    ----------
    chunk_metadata : dict
        Chunk metadata information
    rows : list
        List of processed rows
    """
    chunk_id = chunk_metadata.get('chunk_id', 'unknown')
    subms_id = chunk_metadata.get('subms_id', 'unknown')
    
    # Calculate essential statistics
    unique_baselines = set([row.get('baseline_key', 'unknown') for row in rows])
    unique_scans = set([row.get('scan_number', -1) for row in rows])
    times = [row.get('time', 0.0) for row in rows]
    avg_time = sum(times) / len(times) if times else 0.0
    
    print(f"📦 Chunk {subms_id}_{chunk_id}: {len(rows)} rows | {len(unique_baselines)} baselines | {len(unique_scans)} scans | time={avg_time:.1f}")


def log_groups_summary(groups: dict) -> None:
    """
    Log concise summary of groups by baseline + scan.
    
    Parameters
    ----------
    groups : dict
        Dictionary of grouped rows by baseline and scan
    """
    total_rows = sum(len(group_rows) for group_rows in groups.values())
    print(f"� Groups: {len(groups)} | Total rows: {total_rows}")


def deserialize_chunk_udf():
    """
    Create UDF for deserializing MessagePack chunks in Spark.
    
    Returns
    -------
    function
        Spark UDF for chunk deserialization
    """
    
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


def process_streaming_batch_distributed(df, epoch_id):
    """
    Process streaming batch with distributed BDA processing.
    
    Implements complete distributed processing pipeline including distributed
    deserialization and row expansion, distributed grouping by (baseline, scan),
    distributed BDA processing on Spark workers, and distributed final aggregation.
    
    Parameters
    ----------
    df : DataFrame
        Spark DataFrame with Kafka messages
    epoch_id : int
        Microbatch epoch ID
    """

    
    current_time = datetime.now().strftime("%H:%M:%S")
    start_time = time.time()
    
    try:
        # Check if we have data to process
        row_count = df.count()
        
        if row_count == 0:
            print(f"⏰ [{current_time}] Microbatch {epoch_id}: 0 chunks (waiting...)")
            return
            
        print(f"\n🚀✨ [{current_time}] Microbatch {epoch_id}: {row_count} chunks - DISTRIBUTED BDA PROCESSING")
        print("=" * 80)
        
        # Load BDA configuration
        config_path = str(project_root / "configs" / "bda_config.json")
        bda_config = load_bda_config_with_fallback(config_path)
        
        print(f"🔧 BDA Config: freq={bda_config['frequency_hz']/1e9:.1f}GHz, "
              f"decorr={bda_config['decorr_factor']}, safety={bda_config['safety_factor']}")
        
        # Apply complete distributed BDA pipeline with enhancements
        try:
            # Enhanced pipeline with KPIs and production optimizations
            pipeline_result = apply_distributed_bda_pipeline(
                df, 
                bda_config,
                enable_event_time=True,
                watermark_duration="2 minutes",
                config_file_path="configs/bda_config.json"
            )
            
            # Extract results and KPIs
            final_results_df = pipeline_result['results']
            kpi_summary_df = pipeline_result['kpis']
            pipeline_config = pipeline_result['config']
            spark_config = pipeline_result['spark_config']
            
            # Collect ONLY distributed KPIs (not raw data) - optimized with limit(1)
            print("📊 Collecting distributed BDA KPIs...")
            bda_kpis_rows = kpi_summary_df.limit(1).collect()  # Limit to 1 row to be safe
            if bda_kpis_rows:
                bda_kpis = bda_kpis_rows[0].asDict()  # Convert PySpark Row to dictionary for .get() access
            else:
                # Fallback if no KPIs collected
                bda_kpis = {
                    'total_input_rows': 0,
                    'total_windows': 0,
                    'compression_factor': 1.0,
                    'compression_ratio_pct': 0.0
                }
            
            # Enhanced KPI reporting with distributed metrics
            total_input_rows = bda_kpis["total_input_rows"] or 0
            total_windows = bda_kpis["total_windows"] or 0
            compression_factor = bda_kpis["compression_factor"] or 1.0
            compression_ratio_pct = bda_kpis["compression_ratio_pct"] or 0.0
            
            # Enhanced logging with distributed processing info
            processing_time = (time.time() - start_time) * 1000
            
            print(f"✅ DISTRIBUTED BDA PROCESSING SUMMARY:")
            print(f"   📦 Input chunks: {row_count}")
            print(f"   📊 Total input rows: {total_input_rows}")
            print(f"   🎯 BDA windows generated: {total_windows}")
            print(f"   � Compression factor: {compression_factor:.1f}:1")
            print(f"   📊 Compression ratio: {compression_ratio_pct:.1f}%")
            print(f"   🔗 Unique (baseline,scan) groups processed distributedly")
            print(f"   ⏱️  Window duration p50/p90/p99: {bda_kpis.get('window_dt_p50', 0):.3f}s / {bda_kpis.get('window_dt_p90', 0):.3f}s / {bda_kpis.get('window_dt_p99', 0):.3f}s")
            print(f"   ⚡ Baseline length avg: {bda_kpis.get('baseline_length_avg', 0):.1f} wavelengths")
            print(f"   🔧 Spark config: {spark_config.get('spark.sql.shuffle.partitions', 'default')} partitions")
            print(f"   ⏱️  Processing time: {processing_time:.0f}ms")
            print(f"   🎯✨ BDA successfully applied with production optimizations")
            
            # Sample results collection removed to eliminate collect() calls
            print("\n📋 DISTRIBUTED BDA Processing completed successfully")
            print("   Sample results collection disabled for pure distributed processing")
                      
        except Exception as e:
            print(f"❌ DISTRIBUTED BDA processing error: {e}")
            traceback.print_exc()
            
            # No fallback - distributed processing only
            print("❌ Distributed processing failed - no fallback available")
            return
        
        print("=" * 80)
        print(f"✅ Microbatch {epoch_id} completed with DISTRIBUTED processing")
        
    except Exception as e:
        print(f"❌ Distributed microbatch {epoch_id} error: {e}")
        traceback.print_exc()
        
    print()  # Separator line


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
    Run the Spark streaming consumer with distributed BDA processing.
    
    This consumer uses distributed processing with Spark UDFs including
    optimized Kafka to Spark pipeline, distributed deserialization UDF,
    distributed grouping and aggregation, with watermarks and event time
    for streaming robustness.
    
    Parameters
    ----------
    kafka_servers : str
        Kafka bootstrap servers
    topic : str
        Kafka topic name
    config_path : str, optional
        Path to configuration file
    """
    
    start_time = datetime.now().strftime("%H:%M:%S")
    
    print("🚀 BDA Interferometry Spark Consumer - DISTRIBUTED PROCESSING")
    print("🎯 FEATURES: Distributed BDA + Event Time + Watermarks")
    print("=" * 70)
    print(f"📡 Kafka Servers: {kafka_servers}")
    print(f"📻 Topic: {topic}")
    print(f"🕐 Started: {start_time}")
    print(f"⚡ Microbatch Trigger: 10 seconds")
    print("✨ Pipeline: Kafka → Deserialize UDF → Group → BDA UDF → Aggregate")
    print("🌊 Watermarks: 2 minutes for late data handling")
    print("=" * 70)
    
    # Check Kafka connectivity first
    print("🔍 Checking Kafka connectivity...")
    if not wait_for_kafka_ready(kafka_servers, timeout=60):
        print("❌ Kafka is not ready. Please check Kafka service status.")
        print("💡 Try: docker-compose up -d")
        return
    print("✅ Kafka is ready and responsive")
    
    # Ensure Kafka topic exists before connecting
    print(f"\n🔍 Verifying topic '{topic}'...")
    topic_ready = ensure_kafka_topic_exists(kafka_servers, topic, num_partitions=4, max_retries=30)
    
    if topic_ready:
        print(f"✅ Topic '{topic}' is ready for streaming")
    else:
        print(f"⚠️  Topic '{topic}' not confirmed, proceeding with auto-create")
    
    # Add a small delay to ensure topic propagation
    print("⏳ Allowing time for topic propagation...")
    time.sleep(2)
    
    # Create Spark session
    print("\n🔧 Initializing Spark session...")
    spark = create_spark_session(config_path)
    print(f"✅ Spark session created with {spark.sparkContext.defaultParallelism} cores")
    
    try:
        # Create deserializer UDF
        deserialize_udf = deserialize_chunk_udf()
        print("✅ Deserializer UDF configured")
        
        # Read from Kafka with enhanced configuration for robustness
        print("\n🔗 Connecting to Kafka stream...")
        kafka_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("kafka.auto.create.topics.enable", "true") \
            .option("kafka.num.partitions", "8") \
            .option("kafka.metadata.max.age.ms", "5000") \
            .option("kafka.session.timeout.ms", "30000") \
            .option("kafka.request.timeout.ms", "40000") \
            .option("kafka.retry.backoff.ms", "1000") \
            .option("maxOffsetsPerTrigger", "5000") \
            .load()
        
        print("✅ Connected to Kafka stream with optimizations")
        
        # Process with event time and watermarks for robustness
        processed_df = kafka_df.select(
            kafka_df.key.cast("string").alias("message_key"),
            kafka_df.value.alias("chunk_data"),
            kafka_df.timestamp.alias("kafka_timestamp"),
            kafka_df.timestamp.alias("event_time")  # Add event time for watermarks
        )
        
        # Add watermark for late data handling
        watermarked_df = processed_df.withWatermark("event_time", "2 minutes")
        
        print("✅ Watermarks configured: 2 minutes for late data tolerance")
        
        print("✅ Spark optimized for distributed processing (configured at session creation)")
        
        # Start streaming query with appropriate processing mode
        print("\n🔍 Starting streaming query...")
        
        # Use only distributed processing
        print("✨ Using DISTRIBUTED BDA processing with watermarks")
        
        query = watermarked_df.writeStream \
            .foreachBatch(process_streaming_batch_distributed) \
            .outputMode("append") \
            .trigger(processingTime='10 seconds') \
            .option("checkpointLocation", "/tmp/spark-bda-distributed-checkpoint") \
            .start()
        
        print("✅ Consumer ready - waiting for data...")
        print(f"🔄 Microbatches every 10s | Spark UI: http://localhost:4040 | Ctrl+C to stop")
        print("=" * 60)
        
        # Wait for termination with simplified status updates
        try:
            while query.isActive:
                time.sleep(30)  # Less frequent heartbeats
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"💓 [{current_time}] Listening...")
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        
        query.awaitTermination()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping consumer...")
        
    except Exception as e:
        print(f"❌ Error in streaming: {e}")
        traceback.print_exc()
        
    finally:
        print("\n🔒 Shutting down Spark session...")
        spark.stop()
        print("✅ Consumer stopped successfully")


def normalize_baseline_key(antenna1: int, antenna2: int, subms_id: str = None) -> str:
    """
    Normalize baseline key for consistency between consumer and BDA.
    
    Ensures that (1,3) and (3,1) generate the same key, and optionally
    includes subms_id to distinguish different SubMS.
    
    Parameters
    ----------
    antenna1 : int
        First antenna of the baseline
    antenna2 : int
        Second antenna of the baseline  
    subms_id : str, optional
        SubMS ID to distinguish between different partitions
        
    Returns
    -------
    str
        Normalized baseline key
    """
    # Normalize order: always min-max
    ant_min, ant_max = sorted([antenna1, antenna2])
    
    if subms_id:
        return f"{ant_min}-{ant_max}-{subms_id}"
    else:
        return f"{ant_min}-{ant_max}"


def create_group_key(antenna1: int, antenna2: int, scan_number: int, subms_id: str = None) -> str:
    """
    Create consistent group key for (baseline, scan_number).
    
    Parameters
    ----------
    antenna1 : int
        First antenna
    antenna2 : int
        Second antenna
    scan_number : int
        Scan number
    subms_id : str, optional
        SubMS ID
        
    Returns
    -------
    str
        Normalized group key
    """
    baseline_key = normalize_baseline_key(antenna1, antenna2, subms_id)
    return f"{baseline_key}_scan{scan_number}"


def main():
    """Main entry point for Spark consumer service."""
    
    parser = argparse.ArgumentParser(
        description="BDA Interferometry Spark Consumer Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python consumer_service.py                              # Distributed BDA processing
  python consumer_service.py --kafka-servers localhost:9092
  python consumer_service.py --topic my-visibility-stream
  python consumer_service.py --config /path/to/bda_config.json
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
