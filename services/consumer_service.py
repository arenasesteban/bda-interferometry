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
from pyspark.sql.functions import udf, count, avg, sum as spark_sum, countDistinct, explode, col, struct, array, lit
from pyspark.sql.types import (
    StringType, StructType, StructField, IntegerType, FloatType, 
    DoubleType, ArrayType, BinaryType
)
# Kafka admin imports removed - consumer only processes streams, doesn't manage topics

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from processing.spark_session import create_spark_session
from bda.bda_processor import process_microbatch_with_bda, format_bda_result_for_output, create_bda_summary_stats
from bda.bda_config import load_bda_config_with_fallback, get_default_bda_config
from bda.bda_integration import apply_distributed_bda_pipeline


# Topic management functions removed - now handled by bootstrap script or producer service
# This keeps the consumer lightweight and focused on pure streaming processing


def deserialize_chunk_data(raw_data_bytes: bytes) -> dict:
    """
    Deserialize MessagePack chunk data from Kafka with enhanced logging.
    
    Compatible with producer's serialize_chunk() function using matching
    MessagePack parameters and comprehensive array reconstruction.
    
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
        # Match producer MessagePack parameters: use_bin_type=True, strict_types=False
        # Consumer equivalent: raw=False, strict_map_key=False  
        msgpack_data = msgpack.unpackb(raw_data_bytes, raw=False, strict_map_key=False)
        
        chunk = {}
        arrays_processed = 0
        total_data_size = 0
        
        for key, value in msgpack_data.items():
            if isinstance(value, dict) and 'type' in value:
                if value['type'] == 'ndarray_compressed':
                    # Decompress and reconstruct array
                    try:
                        decompressed = zlib.decompress(value['data'])
                        array = np.frombuffer(decompressed, dtype=value['dtype'])
                        chunk[key] = array.reshape(value['shape'])
                        arrays_processed += 1
                        total_data_size += array.nbytes
                    except Exception as e:
                        print(f"⚠️  Failed to decompress array '{key}': {e}")
                        chunk[key] = None
                        
                elif value['type'] == 'ndarray':
                    # Reconstruct array directly
                    try:
                        array = np.frombuffer(value['data'], dtype=value['dtype'])
                        chunk[key] = array.reshape(value['shape'])
                        arrays_processed += 1 
                        total_data_size += array.nbytes
                    except Exception as e:
                        print(f"⚠️  Failed to reconstruct array '{key}': {e}")
                        chunk[key] = None
                else:
                    # Unknown array type
                    print(f"⚠️  Unknown array type for '{key}': {value.get('type', 'unknown')}")
                    chunk[key] = value
            else:
                # Non-array data (metadata)
                chunk[key] = value
        
        # Log successful deserialization with details
        chunk_id = chunk.get('chunk_id', 'unknown')
        subms_id = chunk.get('subms_id', 'unknown')
        nrows = chunk.get('nrows', 0)
        
        print(f"✅ Deserialized chunk {subms_id}_{chunk_id}: {nrows} rows, {arrays_processed} arrays, {total_data_size/1024:.1f}KB")
        
        return chunk
        
    except Exception as e:
        print(f"❌ Error deserializing chunk: {e}")
        print(f"   Raw data size: {len(raw_data_bytes)} bytes")
        traceback.print_exc()
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


def create_deserialize_to_structs_udf():
    """
    Create lightweight UDF that deserializes MessagePack to Spark structs (not pandas).
    
    This is a "cheap" deserialization optimized for memory efficiency:
    - No pandas DataFrames in the critical path
    - Returns native Spark Row objects
    - Minimal memory footprint per chunk
    
    Returns
    -------
    function
        Spark UDF that returns struct with visibility rows
    """
    
    # Define the output schema matching producer's chunk structure
    visibility_row_schema = StructType([
        # Metadata fields (match producer exactly)
        StructField("chunk_id", IntegerType(), True),
        StructField("subms_id", StringType(), True), 
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
        StructField("polarization_id", IntegerType(), True),
        
        # Baseline and scan identifiers (critical for BDA grouping)
        StructField("antenna1", IntegerType(), True),
        StructField("antenna2", IntegerType(), True),
        StructField("scan_number", IntegerType(), True),
        StructField("baseline_key", StringType(), True),
        
        # Temporal and spatial metadata
        StructField("time", DoubleType(), True),
        StructField("u", DoubleType(), True),
        StructField("v", DoubleType(), True),
        StructField("w", DoubleType(), True),
        
        # Integration time metadata (critical for BDA)
        StructField("exposure", DoubleType(), True),
        StructField("interval", DoubleType(), True),
        StructField("integration_time_s", DoubleType(), True),
        
        # Scientific data arrays (match producer field names exactly)
        StructField("visibilities", BinaryType(), True),  # Changed from 'visibility_data'
        StructField("weight", BinaryType(), True),        # Changed from 'weight_data' 
        StructField("flag", BinaryType(), True)           # Changed from 'flag_data'
    ])
    
    rows_array_schema = ArrayType(visibility_row_schema)
    
    def deserialize_chunk_to_rows(raw_data_bytes):
        """
        Deserialize MessagePack chunk directly to list of Row structs.
        
        Enhanced deserialization with detailed logging and field name matching.
        Compatible with producer's serialize_chunk() output structure.
        """
        try:
            # Use matching MessagePack parameters with producer
            chunk = msgpack.unpackb(raw_data_bytes, raw=False, strict_map_key=False)
            
            # Extract metadata (match producer field names exactly)
            chunk_id = chunk.get('chunk_id', -1)
            subms_id = chunk.get('subms_id', 'unknown')
            field_id = chunk.get('field_id', -1)
            spw_id = chunk.get('spw_id', -1) 
            polarization_id = chunk.get('polarization_id', -1)
            nrows = chunk.get('nrows', 0)
            
            # Log deserialization attempt
            print(f"🔍 Deserializing chunk {subms_id}_{chunk_id}: {nrows} rows expected")
            
            if nrows == 0:
                print(f"⚠️  Empty chunk - no rows to process")
                return []
            
            # Enhanced array extraction with detailed logging
            def extract_array(key, expected_length=None):
                if key not in chunk:
                    print(f"⚠️  Missing array '{key}' in chunk")
                    return np.array([])
                    
                value = chunk[key]
                try:
                    if isinstance(value, dict) and value.get('type') == 'ndarray_compressed':
                        decompressed = zlib.decompress(value['data'])
                        array = np.frombuffer(decompressed, dtype=value['dtype'])
                        reconstructed = array.reshape(value['shape'])
                        print(f"   ✅ Decompressed '{key}': {reconstructed.shape} {reconstructed.dtype}")
                        return reconstructed
                        
                    elif isinstance(value, dict) and value.get('type') == 'ndarray':
                        array = np.frombuffer(value['data'], dtype=value['dtype'])
                        reconstructed = array.reshape(value['shape'])
                        print(f"   ✅ Reconstructed '{key}': {reconstructed.shape} {reconstructed.dtype}")
                        return reconstructed
                        
                    elif isinstance(value, (list, np.ndarray)):
                        # Direct array data
                        array = np.array(value)
                        print(f"   ✅ Direct array '{key}': {array.shape} {array.dtype}")
                        return array
                        
                    else:
                        print(f"   ⚠️  Unknown format for '{key}': {type(value)}")
                        return np.array(value) if hasattr(value, '__iter__') else np.array([])
                        
                except Exception as e:
                    print(f"   ❌ Failed to extract '{key}': {e}")
                    return np.array([])
            
            # Extract arrays using producer field names
            antenna1 = extract_array('antenna1', nrows)
            antenna2 = extract_array('antenna2', nrows)
            scan_number = extract_array('scan_number', nrows)
            time_array = extract_array('time', nrows)
            u_array = extract_array('u', nrows)
            v_array = extract_array('v', nrows)
            w_array = extract_array('w', nrows)
            
            # Extract timing metadata
            exposure = extract_array('exposure', nrows)
            interval = extract_array('interval', nrows)
            integration_time_s = chunk.get('integration_time_s', 180.0)  # Scalar value
            
            # Extract scientific data arrays
            visibilities = extract_array('visibilities')  # Shape: [nrows, nchans, npols]
            weight = extract_array('weight')              # Shape: [nrows, npols] or [nrows, nchans, npols]
            flag = extract_array('flag')                  # Shape: [nrows, nchans, npols]
            
            # Validation
            actual_rows = min(len(antenna1), len(antenna2)) if len(antenna1) > 0 and len(antenna2) > 0 else 0
            if actual_rows != nrows:
                print(f"⚠️  Row count mismatch: expected {nrows}, got {actual_rows}")
                nrows = actual_rows
            
            if nrows == 0:
                print(f"❌ No valid rows after extraction")
                return []
            
            # Create rows efficiently
            rows = []
            for i in range(nrows):
                try:
                    ant1 = int(antenna1[i]) if i < len(antenna1) else -1
                    ant2 = int(antenna2[i]) if i < len(antenna2) else -1
                    
                    # Create normalized baseline key (consistent with consumer functions)
                    baseline_key = normalize_baseline_key(ant1, ant2, subms_id)
                    
                    # Extract row-specific scientific data
                    vis_bytes = visibilities[i].tobytes() if i < len(visibilities) and len(visibilities[i]) > 0 else b''
                    weight_bytes = weight[i].tobytes() if i < len(weight) and len(weight[i]) > 0 else b''
                    flag_bytes = flag[i].tobytes() if i < len(flag) and len(flag[i]) > 0 else b''
                    
                    # Create row tuple matching updated schema
                    row = (
                        chunk_id,
                        subms_id,
                        field_id,
                        spw_id,
                        polarization_id,
                        ant1,
                        ant2,
                        int(scan_number[i]) if i < len(scan_number) else -1,
                        baseline_key,
                        float(time_array[i]) if i < len(time_array) else 0.0,
                        float(u_array[i]) if i < len(u_array) else 0.0,
                        float(v_array[i]) if i < len(v_array) else 0.0,
                        float(w_array[i]) if i < len(w_array) else 0.0,
                        float(exposure[i]) if i < len(exposure) else 180.0,
                        float(interval[i]) if i < len(interval) else 180.0,
                        float(integration_time_s),
                        vis_bytes,      # Match schema: 'visibilities'
                        weight_bytes,   # Match schema: 'weight'
                        flag_bytes      # Match schema: 'flag'
                    )
                    rows.append(row)
                    
                except Exception as e:
                    print(f"   ⚠️  Error creating row {i}: {e}")
                    continue
            
            print(f"✅ Successfully created {len(rows)} rows from chunk {subms_id}_{chunk_id}")
            return rows
            
        except Exception as e:
            print(f"❌ Critical error in deserialization: {e}")
            print(f"   Raw data size: {len(raw_data_bytes)} bytes")
            traceback.print_exc()
            return []
    
    return udf(deserialize_chunk_to_rows, rows_array_schema)


def process_streaming_batch_optimized(df, epoch_id, enable_event_time=True, watermark_duration="20 seconds"):
    """
    ULTRA-OPTIMIZED foreachBatch implementation with clean, structured logging.
    
    Most heavy lifting (deserialization, explode, repartitioning) happens OUTSIDE
    in the DataFrame pipeline where Spark can optimize it properly.
    
    This function now only handles:
    - BDA processing on pre-processed rows
    - Clean, structured logging
    - Error handling
    
    Parameters
    ----------
    df : DataFrame
        Pre-processed DataFrame with exploded visibility rows, already partitioned by baseline
    epoch_id : int
        Microbatch epoch ID
    enable_event_time : bool
        Whether event-time processing is enabled
    watermark_duration : str
        Watermark duration configuration
    """
    
    current_time = datetime.now().strftime("%H:%M:%S")
    start_time = time.time()
    
    try:
        # CLEAN LOGGING: Structured microbatch header
        print(f"\n{'='*60}")
        print(f"🔄 MICROBATCH {epoch_id:02d} - {current_time}")
        print(f"{'='*60}")
        
        # Load BDA configuration once
        config_path = str(project_root / "configs" / "bda_config.json")
        bda_config = load_bda_config_with_fallback(config_path)
        
        # Enhanced logging: Get microbatch statistics
        try:
            # Get row count and unique chunks (lightweight Spark actions)
            total_rows = df.count()
            
            if total_rows == 0:
                print("📭 Empty microbatch - no data received")
                print(f"{'─'*60}")
                return
                
            chunk_info = df.select("chunk_id", "subms_id").distinct().collect()
            unique_baselines = df.select("baseline_key").distinct().count()
            unique_scans = df.select("scan_number").distinct().count()
            
            print(f"📊 Microbatch statistics:")
            print(f"   • Total rows: {total_rows}")
            print(f"   • Unique chunks: {len(chunk_info)}")
            print(f"   • Unique baselines: {unique_baselines}")
            print(f"   • Unique scans: {unique_scans}")
            
            if chunk_info and len(chunk_info) <= 10:  # Only show details for reasonable number of chunks
                print(f"📦 Chunks in this microbatch:")
                for i, chunk in enumerate(chunk_info, 1):
                    chunk_id = chunk['chunk_id']
                    subms_id = chunk['subms_id']
                    print(f"   Chunk {i:02d}: {subms_id}_{chunk_id}")
            elif len(chunk_info) > 10:
                print(f"� Large microbatch: {len(chunk_info)} chunks (details omitted)")
                
            print(f"{'─'*60}")
            
        except Exception as e:
            print(f"⚠️  Error getting microbatch statistics: {e}")
            print(f"📦 Processing microbatch (details unavailable)")
            print(f"{'─'*60}")
            # Continue processing despite logging issues
        
        # Apply BDA to pre-processed, partitioned rows
        try:
            # DataFrame at this point contains:
            # - Deserialized visibility rows (no pandas, memory efficient)
            # - Exploded from chunks to individual rows  
            # - Partitioned by baseline_key for balanced load
            # - Ready for BDA processing (empty batches handled naturally)
            
            print("🔧 Applying BDA pipeline...")
            
            # Apply BDA pipeline - optimized for throughput
            pipeline_result = apply_distributed_bda_pipeline(
                df, 
                bda_config,
                enable_event_time=enable_event_time,
                watermark_duration=watermark_duration,
                config_file_path="configs/bda_config.json"
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Clean completion logging
            print(f"✅ BDA processing completed successfully")
            print(f"⏱️  Processing time: {processing_time:.0f}ms")
                      
        except Exception as e:
            print(f"❌ BDA processing failed: {e}")
            traceback.print_exc()
            return
        
        # Microbatch completion
        total_time = (time.time() - start_time) * 1000
        print(f"{'─'*60}")
        print(f"🎯 Microbatch {epoch_id:02d} COMPLETED - Total time: {total_time:.0f}ms")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Microbatch {epoch_id:02d} ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()


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
                      config_path: str = None,
                      enable_event_time: bool = True,
                      watermark_duration: str = "20 seconds",
                      max_offsets_per_trigger: int = 200) -> None:
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
    
    print("🚀 BDA Interferometry Consumer - PRODUCTION OPTIMIZED")
    print("=" * 60)
    print(f"📡 Kafka: {kafka_servers} | Topic: {topic}")
    print(f"⚡ MaxOffsets: {max_offsets_per_trigger} | Event-time: {'ON' if enable_event_time else 'OFF'}")
    print(f"🕐 Started: {start_time}")
    print("=" * 60)
    
    # Skip Kafka admin - assumes topic exists (managed by producer/bootstrap)
    print("� Skipping Kafka admin checks - topic managed externally")
    print("� Ensure topic exists: kafka-topics --create --topic visibility-stream --partitions 4")
    
    # Create Spark session
    print("🔧 Initializing...")
    spark = create_spark_session(config_path)
    
    try:
        # Create lightweight deserializer UDF (no pandas!)
        deserialize_to_structs_udf = create_deserialize_to_structs_udf()
        kafka_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", str(max_offsets_per_trigger)) \
            .load()
        
        # Step 1: Basic DataFrame with event-time if needed
        if enable_event_time:
            kafka_processed = kafka_df.select(
                kafka_df.key.cast("string").alias("message_key"),
                kafka_df.value.alias("chunk_data"),
                kafka_df.timestamp.alias("kafka_timestamp"),
                kafka_df.timestamp.alias("event_time")
            )
        else:
            kafka_processed = kafka_df.select(
                kafka_df.key.cast("string").alias("message_key"),
                kafka_df.value.alias("chunk_data"),
                kafka_df.timestamp.alias("kafka_timestamp")
            )
        
        # Step 2: Deserialize chunks to rows
        chunks_with_rows = kafka_processed.withColumn(
            "visibility_rows", 
            deserialize_to_structs_udf(col("chunk_data"))
        )
        
        # Step 3: Explode chunks to individual rows
        # Step 3: Explode chunks to individual rows with updated field names
        rows_df = chunks_with_rows.select(
            col("message_key"),
            col("kafka_timestamp"),
            *([col("event_time")] if enable_event_time else []),
            explode(col("visibility_rows")).alias("row")
        ).select(
            col("message_key"),
            col("kafka_timestamp"),
            *([col("event_time")] if enable_event_time else []),
            # Metadata fields
            col("row.chunk_id").alias("chunk_id"),
            col("row.subms_id").alias("subms_id"),
            col("row.field_id").alias("field_id"),
            col("row.spw_id").alias("spw_id"),
            col("row.polarization_id").alias("polarization_id"),
            # Baseline and scan identifiers
            col("row.antenna1").alias("antenna1"),
            col("row.antenna2").alias("antenna2"),
            col("row.scan_number").alias("scan_number"),
            col("row.baseline_key").alias("baseline_key"),
            # Temporal and spatial metadata
            col("row.time").alias("vis_time"),
            col("row.u").alias("u"),
            col("row.v").alias("v"), 
            col("row.w").alias("w"),
            # Integration time metadata
            col("row.exposure").alias("exposure"),
            col("row.interval").alias("interval"),
            col("row.integration_time_s").alias("integration_time_s"),
            # Scientific data (corrected field names)
            col("row.visibilities").alias("visibilities"),  # Corrected from 'visibility_data'
            col("row.weight").alias("weight"),              # Corrected from 'weight_data'
            col("row.flag").alias("flag")                   # Corrected from 'flag_data'
        )
        
        # Step 4: Partition by baseline for optimal processing
        processed_df = rows_df.repartition(4, col("baseline_key"))
        
        # Create processing function with configuration
        def process_batch_with_config(df, epoch_id):
            return process_streaming_batch_optimized(
                df, epoch_id, enable_event_time, watermark_duration
            )
        
        # Generate unique checkpoint per execution to avoid state conflicts when changing config
        import uuid
        checkpoint_path = f"/tmp/spark-bda-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        
        # Start streaming query
        query = processed_df.writeStream \
            .foreachBatch(process_batch_with_config) \
            .outputMode("append") \
            .trigger(processingTime='10 seconds') \
            .option("checkpointLocation", checkpoint_path) \
            .start()
        
        print("✅ Consumer ready - waiting for data...")
        print("🔄 Microbatches every 10s | Ctrl+C to stop")
        print("=" * 60)
        
        # Wait for termination (clean monitoring without interfering with microbatch logs)
        try:
            while query.isActive:
                time.sleep(10)  # Sleep for microbatch intervals
                    
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
    
    parser.add_argument(
        "--no-event-time",
        action="store_true",
        help="Disable event-time processing and watermarks (lab mode)"
    )
    
    parser.add_argument(
        "--watermark",
        default="20 seconds",
        help="Watermark duration for late data tolerance (default: 20 seconds)"
    )
    
    parser.add_argument(
        "--max-offsets",
        type=int,
        default=200,
        help="Max offsets per trigger for throughput control (default: 200, test: 10, prod: 500+)"
    )
    
    args = parser.parse_args()
    
    try:
        run_spark_consumer(
            kafka_servers=args.kafka_servers,
            topic=args.topic,
            config_path=args.config,
            enable_event_time=not args.no_event_time,
            watermark_duration=args.watermark,
            max_offsets_per_trigger=args.max_offsets
        )
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
