"""
Consumer Service - Interferometry Data Processing

This service consumes visibility data chunks from a Kafka topic, processes them using a distributed BDA pipeline and gridding, and generates dirty images.
"""

import hashlib
import io
import os
import sys
import argparse
import time
import traceback
import msgpack
import numpy as np
import uuid
import logging
from pathlib import Path
from functools import reduce

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, decode, trim
from pyspark.sql.types import (
    StringType, StructType, StructField, IntegerType,
    DoubleType, ArrayType
)

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from bda.bda_config import load_bda_config
from bda.bda_integration import apply_bda
from imaging.gridding import apply_gridding, load_grid_config, build_grid
from imaging.dirty_image import generate_dirty_image
from evaluation.metrics import calculate_metrics


def create_spark_session():
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
    kafka_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", bootstrap_server) \
        .option("subscribe", topic) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .option("maxOffsetsPerTrigger", "300") \
        .option("includeHeaders", "true") \
        .load()
    
    return kafka_df.select(
        kafka_df.key.alias("key"),
        kafka_df.value.alias("value"),
        kafka_df.headers.alias("headers")
    )


def define_visibility_schema():
    return StructType([
        StructField("message_id",       StringType(),   False),
        StructField("subms_id",         IntegerType(),  False),
        StructField("field_id",         IntegerType(),  False),
        StructField("spw_id",           IntegerType(),  False),
        StructField("polarization_id",  IntegerType(),  False),
        
        StructField("baseline_key",     StringType(),   False),
        StructField("antenna1",         IntegerType(),  False),
        StructField("antenna2",         IntegerType(),  False),
        StructField("scan_number",      IntegerType(),  False),
        
        StructField("exposure",         DoubleType(),   False),
        StructField("interval",         DoubleType(),   False),
        StructField("time",             DoubleType(),   False),

        StructField("n_channels",       IntegerType(),  False),
        StructField("n_correlations",   IntegerType(),  False),

        StructField("u",                DoubleType(),   False),
        StructField("v",                DoubleType(),   False),
        StructField("w",                DoubleType(),   False),

        StructField("visibility",       ArrayType(ArrayType(ArrayType(DoubleType()))),  False),
        StructField("weight",           ArrayType(ArrayType(DoubleType())),             False),
        StructField("flag",             ArrayType(ArrayType(IntegerType())),            False)
    ])


def check_end_signal(df_scientific, epoch_id):
    df_control = df_scientific.withColumn("key", trim(decode(col("key"), "UTF-8")))

    end_signal = df_control.filter(col("key") == "__END__").limit(1).count() > 0
    
    if end_signal:
        print(f"[Batch {epoch_id}] ✓ END_OF_STREAM signal detected in control messages")
    
    df_filtered = (df_control
        .filter(col("key").isNull() | (col("key") != "__END__"))
        .select("value", "headers")
    )
    
    return df_filtered, end_signal


def headers_to_dict(headers):
    result = {}

    for header in (headers or []):
        key = header.key if isinstance(header.key, str) else header.key.decode("utf-8")
        result[key] = bytes(header.value)
    
    return result


def parse_metadata(headers_dict):
    raw = headers_dict.get("metadata", None)
    
    if raw is None:
        return {}
    
    return msgpack.unpackb(raw, raw=False)


def deserialize_payload(value):
    with io.BytesIO(value) as buffer:
        npz = np.load(buffer, allow_pickle=True)
        return {key: npz[key] for key in npz.files}
    

def assemble_block(metadata, arrays):
    block = {
        "message_id":       metadata["message_id"],
        "subms_id":         metadata["subms_id"],
        "field_id":         metadata["field_id"],
        "spw_id":           metadata["spw_id"],
        "polarization_id":  metadata["polarization_id"],

        "n_channels":       metadata["n_channels"],
        "n_correlations":   metadata["n_correlations"],

        "antenna1":         np.asarray(arrays["antenna1"]),
        "antenna2":         np.asarray(arrays["antenna2"]),
        "scan_number":      np.asarray(arrays["scan_number"]),
        
        "time":             np.asarray(arrays["time"]),
        "exposure":         np.asarray(arrays["exposure"]),
        "interval":         np.asarray(arrays["interval"]),
        
        "u":                np.asarray(arrays["u"]),
        "v":                np.asarray(arrays["v"]),
        "w":                np.asarray(arrays["w"]),
    
        "visibilities":     np.asarray(arrays["visibilities"]),
        "weights":          np.asarray(arrays["weights"]),
        "flags":            np.asarray(arrays["flags"])
    }

    return block


def normalize_baseline(antenna1, antenna2):
    return f"{min(antenna1, antenna2)}-{max(antenna1, antenna2)}"


def process_rows(block):
    rows = []

    for i in range(len(block["antenna1"])):
        row = {
            "message_id":       block["message_id"],
            "subms_id":         int(block["subms_id"]),
            "field_id":         int(block["field_id"]),
            "spw_id":           int(block["spw_id"]),
            "polarization_id":  int(block["polarization_id"]),

            "n_channels":       int(block["n_channels"]),
            "n_correlations":   int(block["n_correlations"]),
            
            "baseline_key":     normalize_baseline(block["antenna1"][i], block["antenna2"][i]),
            "antenna1":         int(block["antenna1"][i]),
            "antenna2":         int(block["antenna2"][i]),
            "scan_number":      int(block["scan_number"][i]),

            "time":             float(block["time"][i]),
            "exposure":         float(block["exposure"][i]),
            "interval":         float(block["interval"][i]),

            "u":                float(block["u"][i]),
            "v":                float(block["v"][i]),
            "w":                float(block["w"][i]),

            "visibility":       block["visibilities"][i].tolist(),
            "weight":           block["weights"][i].tolist(),
            "flag":             block["flags"][i].tolist()
        }

        rows.append(row)

    return rows


def process_message(iterator):
    batchs = []

    for message in iterator:
        try:
            headers_dict = headers_to_dict(message.headers or [])
            metadata = parse_metadata(headers_dict)

            arrays = deserialize_payload(message.value)

            block = assemble_block(metadata, arrays)

            batchs.extend(process_rows(block))

        except Exception as e:
            print(f"[ERROR] Deserializing chunk: {e}")
            traceback.print_exc()
            raise

    return iter(batchs)


def process_streaming_batch(df_scientific, num_partitions, epoch_id, bda_config, grid_config):
    try:       
        blocks = df_scientific.select("message_id").distinct().collect()

        for block in blocks:
            print(f"[Batch {epoch_id}] Processing message - ID {block['message_id']}")
        
        df_averaged, df_windowed = None, None

        if bda_config["decorr_factor"] < 1.0:
            # BDA processing
            df_averaged, df_windowed = apply_bda(df_scientific, num_partitions, bda_config)
            print(f"[Batch {epoch_id}] ✓ BDA applied")

        # Gridding
        df_grid = apply_gridding(
            df_averaged if bda_config["decorr_factor"] < 1.0 else df_scientific, 
            num_partitions, grid_config, strategy="PARTIAL"
        )
        print(f"[Batch {epoch_id}] ✓ Gridding applied")

        return df_grid, df_averaged, df_windowed

    except Exception as e:
        print(f"[ERROR] Batch {epoch_id}: {e}")
        traceback.print_exc()
        return None


def consolidate_processing(grid, num_partitions, grid_config, slurm_job_id):
    if grid:
        df_grid = reduce(DataFrame.unionByName, grid)

        df_grid = df_grid.repartition(num_partitions * 2, "u_pix", "v_pix")
        
        print(f"[Imaging] Starting final dirty image generation")
        print(f"[Imaging] Combining gridded visibilities")

        df_gridded = apply_gridding(df_grid, num_partitions, grid_config, strategy="COMPLETE")
    
        print(f"[Imaging] Building array from gridded data")

        time_start = time.time()
        grids, weights = build_grid(df_gridded, grid_config, num_partitions)
        time_end = time.time()
        print(f"[Imaging] ✓ Grid built in {time_end - time_start:.1f} seconds")
        
        print(f"[Imaging] Generating dirty image")
        
        generate_dirty_image(grids, weights, grid_config, slurm_job_id)

        df_grid.unpersist()

    else:
        print("[Consumer] No data processed, no image generated")


def generate_metrics(windowed, averaged, num_partitions, bda_config, slurm_job_id):
    if windowed and averaged:
        df_windowed = reduce(DataFrame.unionByName, windowed)
        df_averaged = reduce(DataFrame.unionByName, averaged)

        df_windowed = df_windowed.repartition(num_partitions * 3, "baseline_key")
        df_averaged = df_averaged.repartition(num_partitions * 3, "baseline_key")

        print(f"[Evaluation] Starting metrics calculation")

        calculate_metrics(df_windowed, df_averaged, bda_config, slurm_job_id)

        print(f"[Evaluation] ✓ Metrics calculation completed")

    else:
        print("[Consumer] No processed data samples available for metrics")


def run_consumer(bootstrap_server, topic, bda_config_path, grid_config_path, slurm_job_id):
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

        processing_time = {}

        def process_batch(df_scientific, epoch_id):    
            df_filtered, stream_ended = check_end_signal(df_scientific, epoch_id)

            if df_filtered.isEmpty():
                print(f"[Batch {epoch_id}] Empty after filtering, skipping")
                return
            
            if not processing_time:
                processing_time["start"] = time.time()

            processing_time[epoch_id] = { "start": time.time() }

            deserialized_rdd = df_filtered.rdd.mapPartitions(process_message)
            deserialized_df = spark.createDataFrame(deserialized_rdd, visibility_schema)

            df_grid, df_averaged, df_windowed = process_streaming_batch(deserialized_df, num_partitions, epoch_id, bda_config, grid_config)
            
            if df_grid is not None:
                grid.append(df_grid)
                averaged.append(df_averaged)
                windowed.append(df_windowed)

                processing_time[epoch_id]["end"] = time.time()

                print(f"[Batch {epoch_id}] ✓ Completed successfully in {processing_time[epoch_id]['end'] - processing_time[epoch_id]['start']:.1f} seconds")
            
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
        processing_time["image_generation"] = { "start": time.time() }
        consolidate_processing(grid, num_partitions, grid_config, slurm_job_id)
        processing_time["image_generation"]["end"] = time.time()

        processing_time["end"] = time.time()
        total_time = processing_time["end"] - processing_time["start"]
        image_time = processing_time["image_generation"]["end"] - processing_time["image_generation"]["start"]

        print(f"[Consumer] Total time from start to image generation: {total_time/60:.1f} minutes")
        print(f"[Consumer] Time spent on image generation: {image_time/60:.1f} minutes")

        # Generate metrics
        if bda_config["decorr_factor"] < 1.0:
            generate_metrics(windowed, averaged, num_partitions, bda_config, slurm_job_id)

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
        "--slurm-job-id",
        required=True,
        help="Unique identifier for this job"
    )

    args = parser.parse_args()
    
    try:
        run_consumer(
            bootstrap_server=args.bootstrap_server,
            topic=args.topic,
            bda_config_path=args.bda_config,
            grid_config_path=args.grid_config,
            slurm_job_id=args.slurm_job_id,
        )

    except Exception as e:
        print(f"[FATAL] Consumer failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()