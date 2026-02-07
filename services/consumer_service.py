"""
Consumer Service - Interferometry Data Processing

This service consumes visibility data chunks from a Kafka topic, processes them using a distributed BDA pipeline and gridding, and generates dirty images.
"""

import sys
import argparse
from pathlib import Path
import time
import traceback
import msgpack
import numpy as np
import zlib
import uuid
from functools import reduce
import logging

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StringType, StructType, StructField, IntegerType,
    DoubleType, ArrayType, BooleanType
)
from pyspark.sql.functions import col, decode, trim
from pyspark.sql import DataFrame

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from bda.bda_config import load_bda_config
from bda.bda_integration import apply_bda
from imaging.gridding import apply_gridding, load_grid_config
from imaging.dirty_image import generate_dirty_image
from evaluation.metrics import calculate_metrics, consolidate_metrics


def define_visibility_schema():
    """
    Define the Spark schema for visibility data rows.
    
    Returns
    -------
    StructType
        Spark schema for visibility data.
    """
    return StructType([
        StructField("subms_id", IntegerType(), True),
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
        StructField("polarization_id", IntegerType(), True),
        StructField("chunk_id", IntegerType(), True),
        
        StructField("baseline_key", StringType(), True),
        StructField("antenna1", IntegerType(), True),
        StructField("antenna2", IntegerType(), True),
        StructField("scan_number", IntegerType(), True),
        
        StructField("exposure", DoubleType(), True),
        StructField("interval", DoubleType(), True),
        StructField("time", DoubleType(), True),

        StructField("n_channels", IntegerType(), True),
        StructField("n_correlations", IntegerType(), True),

        StructField("u", DoubleType(), True),
        StructField("v", DoubleType(), True),
        StructField("w", DoubleType(), True),

        StructField("visibility", ArrayType(ArrayType(ArrayType(DoubleType()))), True),
        StructField("weight", ArrayType(ArrayType(DoubleType())), True),
        StructField("flag", ArrayType(ArrayType(IntegerType())), True)
    ])


def deserialize_array(array_data):
    if isinstance(array_data, dict):
        array_type = array_data.get('type')
        
        if array_type == 'ndarray_compressed':
            # Decompress and reconstruct
            decompressed = zlib.decompress(array_data['data'])
            dtype = np.dtype(array_data['dtype'])
            shape = tuple(array_data['shape'])
            return np.frombuffer(decompressed, dtype=dtype).reshape(shape)
            
        elif array_type == 'ndarray':
            # Uncompressed array
            dtype = np.dtype(array_data['dtype'])
            shape = tuple(array_data['shape'])
            return np.frombuffer(array_data['data'], dtype=dtype).reshape(shape)
    
    # If it's already a list/array, convert directly
    return np.array(array_data)


def process_chunk(chunk):
    try:
        # Extract metadata
        subms_id = int(chunk.get('subms_id', -1))
        field_id = int(chunk.get('field_id', -1))
        spw_id = int(chunk.get('spw_id', -1)) 
        polarization_id = int(chunk.get('polarization_id', -1))
        chunk_id = int(chunk.get('chunk_id', -1))

        # Extract measurement set info
        baseline_key = chunk.get('baseline_key', '')
        antenna1 = chunk.get('antenna1', 0)
        antenna2 = chunk.get('antenna2', 0)
        scan_number = chunk.get('scan_number', 0)
        
        exposure = chunk.get('exposure', 0)
        interval = chunk.get('interval', 0)
        time = chunk.get('time', 0)

        n_channels = chunk.get('n_channels', 0)
        n_correlations = chunk.get('n_correlations', 0)

        u = chunk.get('u', 0)
        v = chunk.get('v', 0)
        w = chunk.get('w', 0)

        # Deserialize arrays
        vs = deserialize_array(chunk.get('visibility', []))
        ws = deserialize_array(chunk.get('weight', []))
        fs = deserialize_array(chunk.get('flag', []))

        visibility = vs.tolist()
        weight = ws.tolist() if ws.size > 0 else []
        flag = fs.astype(int).tolist() if fs.size > 0 else []

        return {
            'subms_id': subms_id,
            'field_id': field_id,
            'spw_id': spw_id,
            'polarization_id': polarization_id,
            'chunk_id': chunk_id,
            'baseline_key': baseline_key,
            'antenna1': antenna1,
            'antenna2': antenna2,
            'scan_number': scan_number,
            'exposure': exposure,
            'interval': interval,
            'time': time,
            'n_channels': n_channels,
            'n_correlations': n_correlations,
            'u': u,
            'v': v,
            'w': w,
            'visibility': visibility,
            'weight': weight,
            'flag': flag
        }
    
    except Exception as e:
        print(f"[ERROR] Processing chunk: {e}")
        traceback.print_exc()
        raise


def deserialize_rows(iterator):
    for message in iterator:
        try:
            raw_data = message.chunk_data
            chunk = msgpack.unpackb(raw_data, raw=False, strict_map_key=False)

            row = process_chunk(chunk)
             
            yield row

        except Exception as e:
            print(f"[ERROR] Deserializing chunk: {e}")
            traceback.print_exc()
            raise


def process_streaming_batch(df_scientific, num_partitions, epoch_id, bda_config, grid_config):  
    """
    Process a microbatch of visibility data.

    Parameters
    ----------
    df : DataFrame
        Spark DataFrame containing visibility data for the microbatch.
    epoch_id : int
        Unique identifier for the microbatch.
    num_partitions : int
        Number of partitions for parallel processing.
    bda_config : dict
        Configuration for the BDA processing.
    grid_config : dict
        Configuration for the gridding process.

    Returns
    -------
    DataFrame
        DataFrame containing gridded visibilities.
    """  
    start_time = time.time()
    
    try:       
        row_count = df_scientific.count()
        
        if row_count == 0:
            return None
        
        chunk = df_scientific.select("chunk_id").distinct().collect()

        for c in chunk:
            print(f"[Batch {epoch_id}] Processing CHUNK ID - {c['chunk_id']}")
        
        # BDA processing
        df_averaged, df_windowed = apply_bda(df_scientific, num_partitions, bda_config)
        bda_time = (time.time() - start_time) * 1000
        print(f"[Batch {epoch_id}] BDA completed in {bda_time:.0f} ms")

        # Gridding
        df_grid = apply_gridding(df_averaged, num_partitions, grid_config, strategy="PARTIAL")
        total_time = (time.time() - start_time) * 1000
        print(f"[Batch {epoch_id}] Gridding completed in {total_time:.0f} ms")
        print(f"[Batch {epoch_id}] Total processing: {total_time:.0f} ms")

        return df_grid, df_averaged, df_windowed

    except Exception as e:
        print(f"[ERROR] Batch {epoch_id}: {e}")
        traceback.print_exc()
        return None


def create_spark_session():
    """
    Create and configure a Spark session for interferometry data processing.

    Returns
    -------
        SparkSession: Configured Spark session.
    """
    project_root = Path(__file__).parent.parent
    src_path = str(project_root / "src")

    spark = SparkSession.builder \
        .appName("BDA-Interferometry-Consumer") \
        .config("spark.executorEnv.PYTHONPATH", src_path) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    logging.getLogger("org.apache.spark.sql.kafka010").setLevel(logging.ERROR)

    print(f"[Spark] Master: {spark.sparkContext.master}")
    print(f"[Spark] App ID: {spark.sparkContext.applicationId}")

    return spark


def create_kafka_stream(spark, bootstrap_server, topic):
    """
    Create and configure Kafka streaming DataFrame.
    
    Returns:
        DataFrame: Configured Kafka stream
    """
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", bootstrap_server) \
        .option("subscribe", topic) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .option("maxOffsetsPerTrigger", "300") \
        .load()
    
    return kafka_df.select(
        kafka_df.key.alias("message_key"),
        kafka_df.value.alias("chunk_data")
    )


def check_control_messages(df_scientific, epoch_id):
    df_control = df_scientific.withColumn("key", trim(decode(col("message_key"), "UTF-8")))

    end_signal = df_control.filter(col("key") == "__CONTROL__").limit(1).count() > 0
    
    print(f"[Batch {epoch_id}] ✓ Checked control messages - END_OF_STREAM: {end_signal}")

    df_filtered = (df_control
        .filter(col("key").isNull() | (col("key") != "__CONTROL__"))
        .select("chunk_data")
    )
    
    return df_filtered, end_signal

def wait_for_stream_completion(query, max_idle_time=15, check_interval=2):
    print("[Consumer] ✓ END_OF_STREAM received, waiting for active batches...")
    
    last_activity_time = time.time()
    last_batch_id = -1
    
    while True:
        time.sleep(check_interval)
        
        try:
            last_progress = query.lastProgress
            
            if last_progress:
                current_batch_id = last_progress.get('batchId', -1)

                if current_batch_id != last_batch_id:
                    last_batch_id = current_batch_id
                    last_activity_time = time.time()
                    num_rows = last_progress.get('numInputRows', 0)
                    print(f"[Consumer] Processing batch {current_batch_id} ({num_rows} rows)...")

            idle_time = time.time() - last_activity_time
            if idle_time > max_idle_time:
                print(f"[Consumer] ✓ No new batches for {idle_time:.1f}s - all data processed")
                break
                
        except Exception as e:
            print(f"[WARN] Error checking query status: {e}")
            break


def combine_and_image(grid, num_partitions, grid_config, dirty_output, psf_output):
    if not grid:
        print("[Imaging] No data to process")
        return False
    
    print("\n[Imaging] Combining gridded visibilities...")
    df_gridded = reduce(DataFrame.unionByName, grid)
    
    print("[Imaging] Applying final gridding...")
    df_gridded = apply_gridding(df_gridded, num_partitions, grid_config, strategy="COMPLETE")
    
    print("[Imaging] Generating dirty image...")
    generate_dirty_image(df_gridded, grid_config, dirty_output, psf_output)
    
    print(f"[Imaging] ✓ Dirty image saved to: {dirty_output}")
    print(f"[Imaging] ✓ PSF image saved to: {psf_output}")
    
    return True


def run_consumer(bootstrap_server, topic, bda_config_path, grid_config_path, dirty_image_output, psf_output):
    """
    Run the consumer service to process visibility data from Kafka.

    Parameters
    ----------
    bootstrap_server : str
        Kafka bootstrap server address.
    topic : str
        Kafka topic name.
    bda_config : str
        Path to BDA configuration file.
    grid_config : str
        Path to grid configuration file.

    Returns
    -------
    None
    """
    print("=" * 80)
    print(f"[Consumer] Starting service")
    print(f"[Consumer] Kafka: {bootstrap_server}")
    print(f"[Consumer] Topic: {topic}")
    print(f"[Consumer] BDA :  {bda_config_path}")
    print(f"[Consumer] Grid:  {grid_config_path}")
    print("=" * 80)
    
    spark = create_spark_session()
    num_partitions = spark.sparkContext.defaultParallelism * 4 # 8 cores = 32 partitions

    print(f"[Consumer] ✓ Spark session created with {spark.sparkContext.defaultParallelism} cores and {num_partitions} partitions")

    try:        
        bda_config = load_bda_config(bda_config_path)
        grid_config = load_grid_config(grid_config_path)
        visibility_schema = define_visibility_schema()

        kafka_stream = create_kafka_stream(spark, bootstrap_server, topic)

        grid, averaged, windowed = [], [], []
        stream_state = {'signal_received': False}

        def process_batch(df_scientific, epoch_id):    
            df_filtered, stream_ended = check_control_messages(df_scientific, epoch_id)

            if df_filtered.isEmpty():
                print(f"[Batch {epoch_id}] Empty after filtering, skipping")
                return
            
            deserialized_rdd = df_filtered.rdd.mapPartitions(deserialize_rows)
            deserialized_df = spark.createDataFrame(deserialized_rdd, visibility_schema)

            df_grid, df_averaged, df_windowed = process_streaming_batch(deserialized_df, num_partitions, epoch_id, bda_config, grid_config)
            
            if df_grid is not None:
                grid.append(df_grid)
                averaged.append(df_averaged)
                windowed.append(df_windowed)
                print(f"[Batch {epoch_id}] ✓ Completed successfully")
            else:
                print(f"[Batch {epoch_id}] ✗ No output generated")

            print("=" * 80 + "\n")

            if stream_ended:
                stream_state['signal_received'] = True
                print(f"[Batch {epoch_id}] ✓ END signal detected")
            
        checkpoint_path = f"/tmp/spark-bda-{uuid.uuid4().hex[:8]}-{int(time.time())}"

        query = kafka_stream.writeStream \
            .foreachBatch(process_batch) \
            .trigger(processingTime="60 seconds") \
            .option("checkpointLocation", checkpoint_path) \
            .start()

        print("[Consumer] ✓ Streaming query started")
        print("[Consumer] Waiting for data...")

        while query.isActive and not stream_state['signal_received']:
            time.sleep(5)

        if query.isActive and stream_state['signal_received']:
            query.stop()

        query.awaitTermination()
        
        print("[Consumer] Stopping query...")

        # Generate final image
        if grid:
            print("[Consumer] ✓ Generating final dirty image...")
            combine_and_image(grid, num_partitions, grid_config, dirty_image_output, psf_output)
        else:
            print("[Consumer] No data processed, no image generated")

        if averaged and windowed:
            print("[Consumer] ✓ Saving processed data samples...")
            df_averaged = reduce(DataFrame.unionByName, averaged)
            df_windowed = reduce(DataFrame.unionByName, windowed)

            df_amplitude, df_rms, df_baseline_dependency, df_coverage_uv = calculate_metrics(df_windowed, df_averaged, num_partitions)
            consolidate_metrics(df_amplitude, df_rms, df_baseline_dependency, df_coverage_uv, bda_config)
        else:
            print("[Consumer] No processed data samples available for metrics")

        print("[Consumer] Consumer finished processing")
        
    except KeyboardInterrupt:
        print("[Consumer] Interrupted by user")
        
    except Exception as e:
        print(f"[ERROR] Consumer failed: {e}")
        traceback.print_exc()
        raise
        
    finally:
        spark.stop()
        print("[Consumer] ✓ Stopped successfully")


def main(): 
    """Main entry point for consumer service."""
    parser = argparse.ArgumentParser(
        description="BDA Interferometry Consumer Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--bootstrap-server", 
        required=True,
        help="Kafka bootstrap server address"
    )
    parser.add_argument(
        "--topic", 
        required=True,
        help="Kafka topic to consume from"
    )
    parser.add_argument(
        "--bda-config", 
        required=True,
        help="Path to BDA configuration JSON file"
    )
    parser.add_argument(
        "--grid-config", 
        required=True,
        help="Path to grid configuration JSON file"
    )
    parser.add_argument(
        "--dirty-image-output",
        default="./output/dirty_image.png",
        help="Path for dirty image output file"
    )
    parser.add_argument(
        "--psf-output",
        default="./output/psf.png",
        help="Path for PSF output file"
    )

    args = parser.parse_args()
    
    try:
        run_consumer(
            bootstrap_server=args.bootstrap_server,
            topic=args.topic,
            bda_config_path=args.bda_config,
            grid_config_path=args.grid_config,
            dirty_image_output=args.dirty_image_output,
            psf_output=args.psf_output
        )
    except Exception as e:
        print(f"[FATAL] Consumer failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()