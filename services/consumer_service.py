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
        StructField("chunk_id", IntegerType(), True),
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
        StructField("polarization_id", IntegerType(), True),

        StructField("row_start", IntegerType(), True),
        StructField("row_end", IntegerType(), True),
        StructField("nrows", IntegerType(), True),

        StructField("n_channels", IntegerType(), True),
        StructField("n_correlations", IntegerType(), True),

        StructField("antenna1", IntegerType(), True),
        StructField("antenna2", IntegerType(), True),
        StructField("scan_number", IntegerType(), True),
        StructField("baseline_key", StringType(), True),

        StructField("exposure", DoubleType(), True),
        StructField("interval", DoubleType(), True),
        StructField("time", DoubleType(), True),

        StructField("u", DoubleType(), True),
        StructField("v", DoubleType(), True),
        StructField("w", DoubleType(), True),

        StructField("visibilities", ArrayType(ArrayType(ArrayType(DoubleType()))), True),
        StructField("weight", ArrayType(ArrayType(DoubleType())), True),
        StructField("flag", ArrayType(ArrayType(IntegerType())), True)
    ])


def deserialize_array(field_dict, field_name, expected_shapes=None):
    """
    Deserialize a binary-encoded array from a dictionary.

    Parameters
    ----------
    field_dict : dict
        Dictionary containing 'type' and 'data' keys.
    field_name : str
        Name of the field being deserialized (for error messages).
    expected_shapes : dict, optional
        Expected shapes for different fields.
    
    Returns
    -------
    np.ndarray
        Deserialized NumPy array.
    """
    try:
        if not isinstance(field_dict, dict):
            print(f"[WARN] {field_name} is not a dict: {type(field_dict)}")
            return np.array([])
        
        if 'type' not in field_dict or 'data' not in field_dict:
            print(f"[WARN] {field_name} missing keys: {field_dict.keys()}")
            return np.array([])
        
        array_type = field_dict['type']
        if array_type not in ['ndarray', 'ndarray_compressed']:
            print(f"[WARN] {field_name} unsupported type: {array_type}")
            return np.array([])
        
        binary_data = field_dict['data']
        
        if array_type == 'ndarray_compressed':
            try:
                binary_data = zlib.decompress(binary_data)
            except zlib.error as e:
                print(f"[ERROR] Failed to decompress {field_name}: {e}")
                return np.array([])

        if 'visibilities' in field_name.lower():
            array = np.frombuffer(binary_data, dtype=np.complex128)
        elif 'weight' in field_name.lower():
            array = np.frombuffer(binary_data, dtype=np.float32)
        elif 'flag' in field_name.lower():
            array = np.frombuffer(binary_data, dtype=np.bool_)
        else:
            print(f"[ERROR] Unknown field type for {field_name}")
            return np.array([])
        
        if array.size > 0:
            return array.reshape(expected_shapes[field_name])
        
        return array
        
    except Exception as e:
        print(f"[ERROR] Deserializing {field_name}: {e}")
        traceback.print_exc()
        raise
    

def extract_data(visibilities, flag, weight, i):
    """
    Extract visibilities, weight, and flag data for a specific row index.

    Parameters
    ----------
    visibilities : np.ndarray
        Array of visibilities.
    flag : np.ndarray
        Array of flags.
    weight : np.ndarray
        Array of weights.
    i : int
        Row index to extract.
    
    Returns
    -------
    tuple
        Tuple containing:
        - visibilities as list of list of [real, imag]
        - weight as list
        - flag as list of list of int
    """
    try:
        row_visibilities = visibilities[i] if i < len(visibilities) else np.array([])
        row_weight = weight[i] if i < len(weight) else np.array([])
        row_flag = flag[i] if i < len(flag) else np.array([])

        if row_visibilities.size > 0 and np.iscomplexobj(row_visibilities):

            vis_real_imag = []
            for chan in range(row_visibilities.shape[0]):
                chan_data = []
                for corr in range(row_visibilities.shape[1]):
                    complex_val = row_visibilities[chan, corr]
                    chan_data.append([float(complex_val.real), float(complex_val.imag)])
                vis_real_imag.append(chan_data)
            row_vis_list = vis_real_imag
        else:
            row_vis_list = []

        row_weight_list = row_weight.tolist() if row_weight.size > 0 else []
        row_flag_list = row_flag.astype(int).tolist() if row_flag.size > 0 else []

        return row_vis_list, row_weight_list, row_flag_list

    except Exception as e:
        print(f"[ERROR] Extracting data for row {i}: {e}")
        traceback.print_exc()
        raise


def process_chunk(chunk):
    """
    Process a single chunk of visibility data.

    Parameters
    ----------
    chunk : dict
        Dictionary containing chunk metadata and data arrays.
    
    Returns
    -------
    list
        List of processed rows from the chunk.
    """
    try:
        # Extract metadata
        subms_id = int(chunk.get('subms_id', -1))
        chunk_id = int(chunk.get('chunk_id', -1))
        field_id = int(chunk.get('field_id', -1))
        spw_id = int(chunk.get('spw_id', -1)) 
        polarization_id = int(chunk.get('polarization_id', -1))
        nrows = int(chunk.get('nrows', 0))
    
        row_start = int(chunk.get('row_start', 0))
        row_end = int(chunk.get('row_end', nrows))
        n_channels = int(chunk.get('n_channels', 0))
        n_correlations = int(chunk.get('n_correlations', 0))

        # Get the lists
        antenna1 = chunk.get('antenna1', [])
        antenna2 = chunk.get('antenna2', [])
        scan_number = chunk.get('scan_number', [])
        exposure = chunk.get('exposure', [])
        interval = chunk.get('interval', [])
        time = chunk.get('time', [])
        u = chunk.get('u', [])
        v = chunk.get('v', [])
        w = chunk.get('w', [])

        expected_shapes = {
            'visibilities': (nrows, n_channels, n_correlations),
            'weight': (nrows, n_channels, n_correlations),
            'flag': (nrows, n_channels, n_correlations)
        }

        visibilities = deserialize_array(chunk.get('visibilities', []), 'visibilities', expected_shapes)
        weight = deserialize_array(chunk.get('weight', []), 'weight', expected_shapes)
        flag = deserialize_array(chunk.get('flag', []), 'flag', expected_shapes)

        rows = []

        for i, (a1, a2, sc, ex, it, tm, uu, vv, ww) in enumerate(
            zip(antenna1, antenna2, scan_number, exposure, interval, time, u, v, w)
        ):
            vs, wg, fg = extract_data(visibilities, flag, weight, i)

            rows.append(
                {
                    'subms_id': subms_id,
                    'chunk_id': chunk_id,
                    'field_id': field_id,
                    'spw_id': spw_id,
                    'polarization_id': polarization_id,
                    'row_start': row_start,
                    'row_end': row_end,
                    'nrows': nrows,
                    'n_channels': n_channels,
                    'n_correlations': n_correlations,
                    'antenna1': int(a1),
                    'antenna2': int(a2),
                    'scan_number': int(sc),
                    'baseline_key': normalize_baseline_key(a1, a2),
                    'exposure': float(ex),
                    'interval': float(it),
                    'time': float(tm),
                    'u': float(uu),
                    'v': float(vv),
                    'w': float(ww),
                    'visibilities': vs,
                    'weight': wg,
                    'flag': fg
                })

        return rows
    
    except Exception as e:
        print(f"[ERROR] Processing chunk: {e}")
        traceback.print_exc()
        raise


def deserialize_chunk_to_rows(iterator):
    """
    Deserialize visibility chunks from Kafka messages into rows.

    Parameters
    ----------
    iterator : iterator
        Iterator over Kafka messages.
    
    Returns
    -------
    iterator
        Iterator over deserialized visibility rows.
    """
    all_rows = []

    for message in iterator:
        try:
            raw_data = message.chunk_data
            chunk = msgpack.unpackb(raw_data, raw=False, strict_map_key=False)

            if chunk.get('message_type') == 'END_OF_STREAM':
                continue

            chunk_rows = process_chunk(chunk)

            for row in chunk_rows:
                spark_row = (
                    row.get('subms_id'),
                    row.get('chunk_id'),
                    row.get('field_id'),
                    row.get('spw_id'),
                    row.get('polarization_id'),
                    row.get('row_start'),
                    row.get('row_end'),
                    row.get('nrows'),
                    row.get('n_channels'),
                    row.get('n_correlations'),
                    row.get('antenna1'),
                    row.get('antenna2'),
                    row.get('scan_number'),
                    row.get('baseline_key'),
                    row.get('exposure'),
                    row.get('interval'),
                    row.get('time'),
                    row.get('u'),
                    row.get('v'),
                    row.get('w'),
                    row.get('visibilities', []),
                    row.get('weight', []),
                    row.get('flag', [])
                )
                all_rows.append(spark_row)
             
        except Exception as e:
            print(f"[ERROR] Deserializing chunk: {e}")
            traceback.print_exc()
            raise
    
    return iter(all_rows)


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

        # Evalutation
        df_amplitude, df_rms, df_baseline_dependency, df_cobertura_uv = calculate_metrics(df_windowed, df_averaged, num_partitions)

        return df_grid, df_amplitude, df_rms, df_baseline_dependency, df_cobertura_uv

    except Exception as e:
        print(f"[ERROR] Batch {epoch_id}: {e}")
        traceback.print_exc()
        return None


def normalize_baseline_key(antenna1, antenna2):
    """Normalize baseline key for consistent representation."""
    ant_min, ant_max = sorted([antenna1, antenna2])
    return f"{ant_min}-{ant_max}"


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

        grid, amplitude, rms, baseline_dependency, cobertura_uv = [], [], [], [], []
        stream_state = {'ended': False}

        def process_batch(df_scientific, epoch_id):    
            df_filtered, stream_ended = check_control_messages(df_scientific, epoch_id)

            if stream_ended:
                stream_state['ended'] = True
            if df_filtered.isEmpty():
                print(f"[Batch {epoch_id}] Empty after filtering, skipping")
                return
            
            deserialized_rdd = df_filtered.rdd.mapPartitions(deserialize_chunk_to_rows)
            deserialized_df = spark.createDataFrame(deserialized_rdd, visibility_schema)

            df_grid, df_amplitude, df_rms, df_baseline_dependency, df_cobertura_uv = process_streaming_batch(deserialized_df, num_partitions, epoch_id, bda_config, grid_config)
            
            if df_grid is not None:
                grid.append(df_grid)
                amplitude.append(df_amplitude)
                rms.append(df_rms)
                baseline_dependency.append(df_baseline_dependency)
                cobertura_uv.append(df_cobertura_uv)
                print(f"[Batch {epoch_id}] ✓ Completed successfully")
            else:
                print(f"[Batch {epoch_id}] ✗ No output generated")

            print("=" * 80 + "\n")

        checkpoint_path = f"/tmp/spark-bda-{uuid.uuid4().hex[:8]}-{int(time.time())}"

        query = kafka_stream.writeStream \
            .foreachBatch(process_batch) \
            .trigger(processingTime="10 seconds") \
            .option("checkpointLocation", checkpoint_path) \
            .start()

        print("[Consumer] ✓ Streaming query started")
        print("[Consumer] Waiting for data...")

        while query.isActive and not stream_state['ended']:
            time.sleep(5)

        if query.isActive and stream_state['ended']:
            query.stop()

        query.awaitTermination()
        
        print("[Consumer] Stopping query...")

        # Generate final image
        if grid:
            #combine_and_image(grid, num_partitions, grid_config, dirty_image_output, psf_output)

            print("[Consumer] ✓ Combining evaluation metrics...")
            df_amplitude = reduce(DataFrame.unionByName, amplitude)
            df_rms = reduce(DataFrame.unionByName, rms)
            df_baseline_dependency = reduce(DataFrame.unionByName, baseline_dependency)
            df_cobertura_uv = reduce(DataFrame.unionByName, cobertura_uv)
            
            print("[Consumer] ✓ Generating evaluation metrics...")

            consolidate_metrics(df_amplitude, df_rms, df_baseline_dependency, df_cobertura_uv, bda_config)
        else:
            print("[Consumer] No data processed, no image generated")

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