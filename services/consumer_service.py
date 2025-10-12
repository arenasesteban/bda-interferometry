"""
Consumer Service - Interferometry Data Streaming Microservice

Processes interferometry visibility data from Kafka streams using Spark Structured Streaming.
Handles MessagePack deserialization, data validation, baseline normalization, and coordinates
distributed BDA processing through integration with the BDA pipeline.

Functions provide complete streaming pipeline from Kafka consumption through scientific
data preparation and distributed baseline-dependent averaging.
"""

import sys
import argparse
from pathlib import Path
import time
import traceback
from datetime import datetime
import msgpack
import numpy as np
import zlib

from pyspark.sql.functions import udf, explode, col, from_unixtime, to_timestamp
from pyspark.sql.types import (
    StringType, StructType, StructField, IntegerType, 
    DoubleType, ArrayType
)
# Consumer operates on existing Kafka topics without administrative functions

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from processing.spark_session import create_spark_session
from bda.bda_config import load_bda_config_with_fallback
from bda.bda_integration import apply_distributed_bda_pipeline


def create_deserialize_to_structs_udf():
    """
    Create Spark UDF for MessagePack deserialization to native Spark structures.
    
    Creates a User Defined Function that deserializes compressed MessagePack chunks
    into Spark-native Row objects with proper schema validation. Handles scientific
    data arrays conversion from binary formats to Spark-compatible types for
    downstream BDA processing.
    
    Returns
    -------
    pyspark.sql.functions.udf
        Spark UDF function that deserializes MessagePack data to structured rows
        
    Raises
    ------
    Exception
        UDF handles internal exceptions silently to prevent Spark task failures
    """
    
    # Define Spark schema structure for deserialized visibility data
    visibility_row_schema = StructType([
        # Chunk metadata fields for data provenance tracking
        StructField("chunk_id", IntegerType(), True),
        StructField("subms_id", StringType(), True), 
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
        StructField("polarization_id", IntegerType(), True),
        
        # Data range boundaries for chunk validation
        StructField("row_start", IntegerType(), True),
        StructField("row_end", IntegerType(), True),
        StructField("nrows", IntegerType(), True),
        
        # Scientific array dimensions for BDA algorithm requirements
        StructField("n_channels", IntegerType(), True),
        StructField("n_correlations", IntegerType(), True),
        
        # Baseline and scan identifiers for scientific grouping operations
        StructField("antenna1", IntegerType(), True),
        StructField("antenna2", IntegerType(), True),
        StructField("scan_number", IntegerType(), True),
        StructField("baseline_key", StringType(), True),
        
        # Temporal and spatial metadata
        StructField("time", DoubleType(), True),
        StructField("u", DoubleType(), True),
        StructField("v", DoubleType(), True),
        StructField("w", DoubleType(), True),
        
        # Integration timing parameters for BDA calculations
        StructField("exposure", DoubleType(), True),
        StructField("interval", DoubleType(), True),
        StructField("integration_time_s", DoubleType(), True),
        
        # Scientific data arrays converted to Spark-native types
        StructField("visibilities", ArrayType(DoubleType()), True),  # Complex as [real, imag] pairs
        StructField("weight", ArrayType(DoubleType()), True),        # Weight array as doubles
        StructField("flag", ArrayType(IntegerType()), True)          # Flag array as 0/1 integers
    ])
    
    rows_array_schema = ArrayType(visibility_row_schema)
    
    def deserialize_chunk_to_rows(raw_data_bytes):
        """
        Deserialize MessagePack chunk into list of visibility row structures.
        
        Processes compressed binary data from Kafka messages, extracts scientific
        arrays, and converts complex visibility data to Spark-compatible formats.
        Handles array decompression, type conversion, and baseline key normalization.
        
        Parameters
        ----------
        raw_data_bytes : bytes
            Binary MessagePack data from Kafka message
            
        Returns
        -------
        list
            List of tuples representing visibility rows with scientific data
        """
        try:
            # Deserialize MessagePack data using compatible parameters
            chunk = msgpack.unpackb(raw_data_bytes, raw=False, strict_map_key=False)
            
            # Extract chunk metadata using standard field names
            chunk_id = chunk.get('chunk_id', -1)
            subms_id = chunk.get('subms_id', 'unknown')
            field_id = chunk.get('field_id', -1)
            spw_id = chunk.get('spw_id', -1) 
            polarization_id = chunk.get('polarization_id', -1)
            nrows = chunk.get('nrows', 0)
            
            # Extract data boundaries and array dimensions
            row_start = chunk.get('row_start', 0)
            row_end = chunk.get('row_end', nrows)
            n_channels = chunk.get('n_channels', 0)
            n_correlations = chunk.get('n_correlations', 0)
            
            if nrows == 0:
                return []
            
            # Extract and decompress scientific arrays from MessagePack data
            def extract_array(key, expected_length=None):
                if key not in chunk:
                    return np.array([])
                    
                value = chunk[key]
                try:
                    if isinstance(value, dict) and value.get('type') == 'ndarray_compressed':
                        # Decompress scientific arrays maintaining data fidelity
                        compressed_size = len(value['data'])
                        if compressed_size > 50 * 1024 * 1024:  # Only warn for truly massive arrays (>50MB)
                            print(f"⚠️  Large array decompression: {compressed_size/1024/1024:.1f}MB")
                        
                        decompressed = zlib.decompress(value['data'])
                        array = np.frombuffer(decompressed, dtype=value['dtype'])
                        return array.reshape(value['shape'])
                        
                    elif isinstance(value, dict) and value.get('type') == 'ndarray':
                        array = np.frombuffer(value['data'], dtype=value['dtype'])
                        return array.reshape(value['shape'])
                        
                    elif isinstance(value, (list, np.ndarray)):
                        return np.array(value)
                        
                    else:
                        return np.array(value) if hasattr(value, '__iter__') else np.array([])
                        
                except Exception as e:
                    # Handle extraction errors without UDF failure
                    return np.array([])
            
            # Extract coordinate and metadata arrays from chunk
            antenna1 = extract_array('antenna1', nrows)
            antenna2 = extract_array('antenna2', nrows)
            scan_number = extract_array('scan_number', nrows)
            time_array = extract_array('time', nrows)
            u_array = extract_array('u', nrows)
            v_array = extract_array('v', nrows)
            w_array = extract_array('w', nrows)
            
            # Extract integration timing information
            exposure = extract_array('exposure', nrows)
            interval = extract_array('interval', nrows)
            integration_time_s = chunk.get('integration_time_s', 180.0)  # Scalar value
            
            # Extract scientific data arrays
            visibilities = extract_array('visibilities')  # Shape: [nrows, nchans, npols]
            weight = extract_array('weight')              # Shape: [nrows, npols] or [nrows, nchans, npols]
            flag = extract_array('flag')                  # Shape: [nrows, nchans, npols]
            
            # Validate that n_channels and n_correlations are consistent with actual data
            if len(visibilities) > 0 and len(visibilities.shape) >= 2:
                actual_channels = visibilities.shape[-2] if len(visibilities.shape) >= 3 else 1
                actual_correlations = visibilities.shape[-1] if len(visibilities.shape) >= 2 else visibilities.shape[-1]
                if n_channels > 0 and actual_channels != n_channels:
                    print(f"⚠️  Channel mismatch: metadata={n_channels}, data={actual_channels}")
                if n_correlations > 0 and actual_correlations != n_correlations:
                    print(f"⚠️  Correlation mismatch: metadata={n_correlations}, data={actual_correlations}")
            
            # Validate array dimensions and correct row count
            actual_rows = min(len(antenna1), len(antenna2)) if len(antenna1) > 0 and len(antenna2) > 0 else 0
            if actual_rows != nrows:
                nrows = actual_rows
            
            if nrows == 0:
                return []
            
            # Generate structured rows from extracted arrays
            rows = []
            for i in range(nrows):
                try:
                    ant1 = int(antenna1[i]) if i < len(antenna1) else -1
                    ant2 = int(antenna2[i]) if i < len(antenna2) else -1
                    
                    # Generate normalized baseline identifier for grouping
                    baseline_key = normalize_baseline_key(ant1, ant2, subms_id)
                    
                    # Extract scientific arrays for current visibility row
                    vis_row = visibilities[i] if i < len(visibilities) else np.array([], dtype=np.complex128)
                    weight_row = weight[i] if i < len(weight) else np.array([], dtype=np.float32)  
                    flag_row = flag[i] if i < len(flag) else np.array([], dtype=np.bool_)
                    
                    # Convert scientific arrays to Spark-compatible formats preserving data fidelity
                    if vis_row.size == 0:
                        vis_list = []
                        weight_list = []
                        flag_list = []
                    else:
                        # Memory-efficient conversion maintaining scientific integrity
                        # Convert complex visibilities to interleaved real/imaginary format
                        vis_flat = vis_row.ravel()
                        vis_real_imag = np.stack([vis_flat.real, vis_flat.imag], axis=0).T
                        vis_list = vis_real_imag.ravel().tolist()
                        
                        # Convert weight and flag arrays preserving full dimensions
                        weight_list = weight_row.ravel().astype(np.float32, copy=False).tolist() if weight_row.size > 0 else []
                        flag_list = flag_row.ravel().astype(np.int8, copy=False).tolist() if flag_row.size > 0 else []
                    
                    # Assemble complete row tuple for Spark schema
                    row = (
                        chunk_id,
                        subms_id,
                        field_id,
                        spw_id,
                        polarization_id,
                        row_start,
                        row_end,
                        nrows,
                        n_channels,
                        n_correlations,
                        int(antenna1[i]),
                        int(antenna2[i]),
                        int(scan_number[i]),
                        baseline_key,
                        float(time_array[i]),
                        float(u_array[i]),
                        float(v_array[i]),
                        float(w_array[i]),
                        float(exposure[i]),
                        float(interval[i]),
                        float(integration_time_s),
                        vis_list,
                        weight_list,
                        flag_list
                    )
                    rows.append(row)
                    
                except Exception as e:
                    continue
            
            # Return processed rows without logging to prevent UDF overhead
            return rows
            
        except Exception as e:
            # Handle deserialization errors without task failure
            return []
    
    return udf(deserialize_chunk_to_rows, rows_array_schema)


def process_streaming_batch_optimized(df, epoch_id, bda_config_broadcast, enable_event_time=True, watermark_duration="20 seconds"):
    """
    Process microbatch with incremental BDA using decorrelation windows.
    
    Uses memory-efficient incremental BDA that processes visibility data in small
    decorrelation-time windows rather than accumulating entire baselines in memory.
    This approach maintains constant RAM usage regardless of baseline size.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        DataFrame containing deserialized visibility rows ready for BDA processing
    epoch_id : int
        Unique identifier for the current microbatch epoch
    bda_config_broadcast : pyspark.broadcast.Broadcast
        Broadcast BDA configuration to avoid repeated loading
    enable_event_time : bool, optional
        Whether event-time watermarking is enabled for late data handling, by default True
    watermark_duration : str, optional
        Time window for late data tolerance, by default "20 seconds"
        
    Raises
    ------
    Exception
        Logs processing errors and continues execution to maintain stream stability
    """
    
    current_time = datetime.now().strftime("%H:%M:%S")
    start_time = time.time()
    
    try:
        # Display structured header for microbatch processing
        print(f"\n{'='*60}")
        print(f"🔄 MICROBATCH {epoch_id:02d} - {current_time}")
        print(f"{'='*60}")
        
        # Use broadcast BDA configuration - no loading overhead
        bda_config = bda_config_broadcast.value
        
        # CRITICAL: Removed df.isEmpty() check - it doesn't exist in PySpark and causes crashes
        # Spark automatically handles empty batches, so we proceed directly to processing
        
        # Calculate and display microbatch processing statistics
        try:
            # Quick check for empty batch without full count()
            if df.isEmpty():
                print("📭 Empty microbatch - no data received")
                print(f"{'─'*60}")
                return
                
            print(f"� Processing microbatch with data (stats via stream monitoring)")
            print(f"{'─'*60}")
            
        except Exception as e:
            print(f"⚠️  Microbatch validation: {e}")
            print(f"📦 Processing microbatch (validation bypassed)")
            print(f"{'─'*60}")
            # Proceed with BDA processing regardless
        
        # Execute distributed BDA processing on prepared visibility data
        try:
            # Apply distributed BDA pipeline to processed visibility data
            pipeline_result = apply_distributed_bda_pipeline(
                df, 
                bda_config,
                enable_event_time=enable_event_time,
                watermark_duration=watermark_duration,
                config_file_path=None  # Config already loaded and broadcast
            )
            
            # Record BDA processing execution time
            processing_time = (time.time() - start_time) * 1000
            
            # Display successful processing results
            print(f"✅ BDA processing completed successfully")
            print(f"⏱️  Processing time: {processing_time:.0f}ms")
                      
        except Exception as e:
            print(f"❌ BDA processing failed: {e}")
            traceback.print_exc()
            return
        
        # Log final microbatch execution summary
        total_time = (time.time() - start_time) * 1000
        print(f"{'─'*60}")
        print(f"🎯 Microbatch {epoch_id:02d} COMPLETED - Total time: {total_time:.0f}ms")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Microbatch {epoch_id:02d} ERROR: {e}")
        print(f"{'='*60}")
        traceback.print_exc()


def run_spark_consumer(kafka_servers: str = "localhost:9092", 
                      topic: str = "visibility-stream",
                      config_path: str = None,
                      enable_event_time: bool = True,
                      watermark_duration: str = "20 seconds",
                      max_offsets_per_trigger: int = 200) -> None:
    """
    Initialize and run Spark streaming consumer for interferometry data processing.
    
    Sets up complete streaming pipeline from Kafka consumption through distributed
    BDA processing. Configures Spark session, creates deserialization UDFs,
    establishes streaming query with appropriate partitioning and watermarking,
    and coordinates microbatch processing.
    
    Parameters
    ----------
    kafka_servers : str, optional
        Comma-separated list of Kafka bootstrap servers, by default "localhost:9092"
    topic : str, optional
        Kafka topic name to consume from, by default "visibility-stream"
    config_path : str, optional
        Path to BDA configuration file, by default None
    enable_event_time : bool, optional
        Enable event-time processing with watermarks, by default True
    watermark_duration : str, optional
        Duration for late data tolerance, by default "20 seconds"
    max_offsets_per_trigger : int, optional
        Maximum Kafka offsets processed per trigger, by default 200
        
    Raises
    ------
    Exception
        Kafka connection errors, Spark initialization failures, or streaming errors
    """
    
    start_time = datetime.now().strftime("%H:%M:%S")
    
    print("🚀 BDA Interferometry Consumer")
    print("=" * 60)
    print(f"📡 Kafka: {kafka_servers} | Topic: {topic}")
    print(f"⚡ MaxOffsets: {max_offsets_per_trigger} | Event-time: {'ON' if enable_event_time else 'OFF'}")
    print(f"🕐 Started: {start_time}")
    print("=" * 60)
    
    # Consumer operates on existing Kafka topics without administrative setup
    print("⚠️  Skipping Kafka admin checks - topic managed externally")
    print("💡 Ensure topic exists: kafka-topics --create --topic visibility-stream --partitions 4")
    
    # Create Spark session
    print("🔧 Initializing...")
    spark = create_spark_session(config_path)
    
    try:
        # Initialize memory-efficient MessagePack deserializer
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
        
        # Configure Kafka DataFrame with event-time processing
        kafka_processed = kafka_df.select(
            kafka_df.key.cast("string").alias("message_key"),
            kafka_df.value.alias("chunk_data"),
            kafka_df.timestamp.alias("kafka_timestamp")
        )
        
        # Apply MessagePack deserialization to extract visibility rows
        chunks_with_rows = kafka_processed.withColumn(
            "visibility_rows", 
            deserialize_to_structs_udf(col("chunk_data"))
        )
        
        # Expand chunk arrays into individual visibility rows
        rows_df = chunks_with_rows.select(
            col("message_key"),
            col("kafka_timestamp"),
            explode(col("visibility_rows")).alias("row")
        ).select(
            col("message_key"),
            col("kafka_timestamp"),
            # Metadata fields
            col("row.chunk_id").alias("chunk_id"),
            col("row.subms_id").alias("subms_id"),
            col("row.field_id").alias("field_id"),
            col("row.spw_id").alias("spw_id"),
            col("row.polarization_id").alias("polarization_id"),
            
            # Data boundaries and array dimensions for BDA processing
            col("row.row_start").alias("row_start"),
            col("row.row_end").alias("row_end"),
            col("row.nrows").alias("nrows"),
            col("row.n_channels").alias("n_channels"),
            col("row.n_correlations").alias("n_correlations"),
            # Baseline and scan identifiers
            col("row.antenna1").alias("antenna1"),
            col("row.antenna2").alias("antenna2"),
            col("row.scan_number").alias("scan_number"),
            col("row.baseline_key").alias("baseline_key"),
            # Temporal and spatial metadata
            col("row.time").alias("time"),
            col("row.u").alias("u"),
            col("row.v").alias("v"), 
            col("row.w").alias("w"),
            # Integration time metadata
            col("row.exposure").alias("exposure"),
            col("row.interval").alias("interval"),
            col("row.integration_time_s").alias("integration_time_s"),
            # Scientific arrays converted to Spark-native data types
            col("row.visibilities").alias("visibilities"),  # Complex as [real, imag] pairs list
            col("row.weight").alias("weight"),              # Weights as float list
            col("row.flag").alias("flag")                   # Flags as 0/1 integer list
        )
        
        # Convert visibility time to timestamp for watermarking
        if enable_event_time:
            # Convert time from double (seconds since epoch) to proper timestamp
            rows_df = rows_df.withColumn("time_timestamp", to_timestamp(from_unixtime(col("time"))))
            rows_df = rows_df.withWatermark("time_timestamp", watermark_duration)
            print(f"⏰ Event-time watermarking enabled: {watermark_duration} (using time_timestamp)")
        
        # Repartition data by baseline for efficient distributed processing (4 cores = 4 partitions)
        processed_df = rows_df.repartition(4, col("baseline_key"))
        
        # Load and broadcast BDA configuration once - avoid repeated loading
        config_path = str(project_root / "configs" / "bda_config.json")
        bda_config = load_bda_config_with_fallback(config_path)
        bda_config_broadcast = spark.sparkContext.broadcast(bda_config)
        print(f"📡 BDA configuration broadcast to all workers")
        
        # Define microbatch processing function with broadcast config
        def process_batch_with_config(df, epoch_id):
            return process_streaming_batch_optimized(
                df, epoch_id, bda_config_broadcast, enable_event_time, watermark_duration
            )
        
        # Create unique checkpoint directory for streaming state management
        import uuid
        checkpoint_path = f"/tmp/spark-bda-{uuid.uuid4().hex[:8]}-{int(time.time())}"
        
        # Initialize and start Spark streaming query - OPTIMIZED
        query = processed_df.writeStream \
            .foreachBatch(process_batch_with_config) \
            .trigger(processingTime='5 seconds') \
            .option("checkpointLocation", checkpoint_path) \
            .start()
        
        print("✅ Consumer ready - waiting for data...")
        print("� Optimized for complete BDA execution - 5s triggers | Ctrl+C to stop")
        print("=" * 60)
        
        # Monitor streaming execution with progress reporting
        try:
            monitor_count = 0
            while query.isActive:
                time.sleep(10)  # Wait between monitoring cycles
                monitor_count += 1
                
                # Display stream progress every 30 seconds (3 cycles)
                if monitor_count % 3 == 0:
                    try:
                        progress = query.lastProgress
                        if progress:
                            batch_id = progress.get('batchId', 'N/A')
                            input_rows = progress.get('inputRowsPerSecond', 0)
                            processing_time = progress.get('batchDuration', 0)
                            print(f"📊 Stream Progress - Batch: {batch_id}, Rows/sec: {input_rows:.1f}, BatchTime: {processing_time}ms")
                    except Exception as e:
                        pass  # Ignore monitoring errors
                    
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
    Generate normalized baseline identifier for consistent grouping operations.
    
    Creates standardized baseline key by ordering antenna IDs to ensure
    bidirectional baselines produce identical keys for proper scientific
    grouping in BDA processing.
    
    Parameters
    ----------
    antenna1 : int
        First antenna identifier in baseline pair
    antenna2 : int
        Second antenna identifier in baseline pair
    subms_id : str, optional
        SubMS identifier for multi-dataset discrimination, by default None
        
    Returns
    -------
    str
        Normalized baseline key in format "min_antenna-max_antenna"
    """
    # Order antennas by ID to ensure consistent baseline representation
    ant_min, ant_max = sorted([antenna1, antenna2])
    
    return f"{ant_min}-{ant_max}"


def main():
    """
    Command-line entry point for interferometry data consumer service.
    
    Parses command-line arguments and initializes Spark streaming consumer
    with specified configuration parameters for Kafka connectivity,
    BDA processing options, and stream management settings.
    
    Raises
    ------
    SystemExit
        Fatal configuration or initialization errors
    """
    
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
