import sys
import argparse
from pathlib import Path
import time
import traceback
import msgpack
import numpy as np
import zlib
import uuid

from pyspark.sql.types import (
    StringType, StructType, StructField, IntegerType,
    DoubleType, ArrayType
)
from pyspark.sql import DataFrame
from functools import reduce

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from processing.spark_session import create_spark_session
from bda.bda_config import load_bda_config
from bda.bda_integration import apply_bda
from imaging.gridding import apply_gridding, load_grid_config, consolide_gridding
from imaging.dirty_image import generate_dirty_image


def define_visibility_schema():
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

        StructField("longitude", DoubleType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("lambda_ref", DoubleType(), True),
        StructField("ra", DoubleType(), True),
        StructField("dec", DoubleType(), True),

        StructField("exposure", DoubleType(), True),
        StructField("interval", DoubleType(), True),

        StructField("time", DoubleType(), True),
        StructField("u", DoubleType(), True),
        StructField("v", DoubleType(), True),
        StructField("w", DoubleType(), True),

        # visibilities: [chan][corr] con [real, imag] -> Array de Array de Array de Double
        StructField("visibilities", ArrayType(ArrayType(ArrayType(DoubleType()))), True),
        # weight: [corr] -> Array de Double
        StructField("weight", ArrayType(DoubleType()), True),
        # flag: [chan][corr] con 0/1 -> Array de Array de Integer
        StructField("flag", ArrayType(ArrayType(IntegerType())), True)
    ])

def deserialize_array(field_dict, field_name, expected_shapes):
    try:
        if not isinstance(field_dict, dict):
            print(f"Error {field_name} is not a dict: {type(field_dict)}")
            return np.array([])
        
        if 'type' not in field_dict or 'data' not in field_dict:
            print(f"Error {field_name} missing 'type' or 'data' keys: {field_dict.keys()}")
            return np.array([])
        
        array_type = field_dict['type']
        if array_type not in ['ndarray', 'ndarray_compressed']:
            print(f"Error {field_name} unsupported type: {array_type}")
            return np.array([])
        
        # Obtener datos binarios
        binary_data = field_dict['data']
        
        if array_type == 'ndarray_compressed':
            try:
                binary_data = zlib.decompress(binary_data)
            except zlib.error as e:
                print(f"Error Failed to decompress {field_name}: {e}")
                return np.array([])
        
        # Auto-detectar dtype basado en el nombre del campo
        if 'visibilities' in field_name.lower():
            array = np.frombuffer(binary_data, dtype=np.complex128)
        elif 'weight' in field_name.lower():
            array = np.frombuffer(binary_data, dtype=np.float32)
        elif 'flag' in field_name.lower():
            array = np.frombuffer(binary_data, dtype=np.bool_)
        else:
            print(f"Error unknown field type for {field_name}")
            return np.array([])
        
        if array.size > 0:
            return array.reshape(expected_shapes[field_name])
        
        return array
        
    except Exception as e:
        print(f"Error deserializing {field_name}: {e}")
        traceback.print_exc()
        raise
    

def extract_data(visibilities, flag, weight, i):
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

        # Convertir weight y flag a listas Python
        row_weight_list = row_weight.tolist() if row_weight.size > 0 else []
        row_flag_list = row_flag.astype(int).tolist() if row_flag.size > 0 else []

        return row_vis_list, row_weight_list, row_flag_list

    except Exception as e:
        print(f"Error extracting data for row {i}: {e}")
        traceback.print_exc()
        raise


def process_chunk(chunk):
    try:
        # Extract chunk metadata using standard field names
        subms_id = int(chunk.get('subms_id', -1))
        chunk_id = int(chunk.get('chunk_id', -1))
        field_id = int(chunk.get('field_id', -1))
        spw_id = int(chunk.get('spw_id', -1)) 
        polarization_id = int(chunk.get('polarization_id', -1))
        nrows = int(chunk.get('nrows', 0))
    
        # Extract data boundaries and array dimensions
        row_start = int(chunk.get('row_start', 0))
        row_end = int(chunk.get('row_end', nrows))
        n_channels = int(chunk.get('n_channels', 0))
        n_correlations = int(chunk.get('n_correlations', 0))

        longitude = float(chunk.get('longitude', 0.0))
        latitude = float(chunk.get('latitude', 0.0))
        lambda_ref = float(chunk.get('lambda_ref', 0.0))
        ra = float(chunk.get('ra', 0.0))
        dec = float(chunk.get('dec', 0.0))

        # Get the lists from the chunk
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
            'weight': (nrows, n_correlations),
            'flag': (nrows, n_channels, n_correlations)
        }

        visibilities = deserialize_array(chunk.get('visibilities', []), 'visibilities', expected_shapes)
        weight = deserialize_array(chunk.get('weight', []), 'weight', expected_shapes)
        flag = deserialize_array(chunk.get('flag', []), 'flag', expected_shapes)

        rows = []

        for i, (a1, a2, sc, ex, it, tm, uu, vv, ww) in enumerate(zip(antenna1, antenna2, scan_number, exposure, interval, time, u, v, w)):
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
                    'longitude': longitude,
                    'latitude': latitude,
                    'lambda_ref': lambda_ref,
                    'ra': ra,
                    'dec': dec,
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
        print(f"Error processing chunk: {e}")
        traceback.print_exc()
        raise


def deserialize_chunk_to_rows(iterator):
    all_rows = []

    for message in iterator:
        try:
            raw_data = message.chunk_data
            chunk = msgpack.unpackb(raw_data, raw=False, strict_map_key=False)
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
                    row.get('longitude'),
                    row.get('latitude'),
                    row.get('lambda_ref'),
                    row.get('ra'),
                    row.get('dec'),
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
            print(f"Error deserializing chunk: {e}")
            traceback.print_exc()
            raise
    
    return iter(all_rows)

def process_streaming_batch(df, epoch_id, bda_config, grid_config):    
    start_time = time.time()
    
    try:       
        print("Rows in microbatch:", df.count())

        # Apply distributed BDA pipeline to processed visibility data
        bda_result = apply_bda(df, bda_config)

        processing_time = (time.time() - start_time) * 1000
        print(f"BDA processing completed in {processing_time:.0f} ms.\n")
        
        # Apply distributed gridding to BDA-processed data
        grid_result = apply_gridding(bda_result, grid_config)

        processing_time = (time.time() - start_time) * 1000
        print(f"Gridding processing completed in {processing_time:.0f} ms.\n")

        return grid_result

    except Exception as e:
        print(f"Error in microbatch {epoch_id}: {e}")
        traceback.print_exc()


def normalize_baseline_key(antenna1: int, antenna2: int) -> str:
    # Order antennas by ID to ensure consistent baseline representation
    ant_min, ant_max = sorted([antenna1, antenna2])
    
    return f"{ant_min}-{ant_max}"


def run_consumer(kafka_servers: str = "localhost:9092", 
                      topic: str = "visibility-stream",
                      config_path: str = None,
                      max_offsets_per_trigger: int = 200) -> None:
    print(f"Running Consumer Service: Kafka={kafka_servers}, Topic={topic}.")
    spark = create_spark_session(config_path)
    
    try:        
        kafka_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", str(max_offsets_per_trigger)) \
            .load()
        
        # Configure Kafka DataFrame
        kafka_processed = kafka_df.select(kafka_df.value.alias("chunk_data"),)
        
        visibility_row_schema = define_visibility_schema()

        bda_config = load_bda_config(str(project_root / "configs" / "bda_config.json"))
        grid_config = load_grid_config(str(project_root / "configs" / "grid_config.json"))

        gridded_visibilities = []

        def process_batch(df, epoch_id):
            if df.isEmpty():
                print(f"Microbatch {epoch_id} is empty.")
                return

            print("=" * 40)
            print(f"Microbatch {epoch_id} processing.\n")

            # Usar mapPartitions para deserializar cada partición
            deserialized_rdd = df.rdd.mapPartitions(deserialize_chunk_to_rows)
            deserialized_df = spark.createDataFrame(deserialized_rdd, visibility_row_schema)

            grid_result = process_streaming_batch(deserialized_df, epoch_id, bda_config, grid_config)
            
            if grid_result is not None:
                gridded_visibilities.append(grid_result)

            print("=" * 40 + "\n")

        # Create unique checkpoint directory for streaming state management
        checkpoint_path = f"/tmp/spark-bda-{uuid.uuid4().hex[:8]}-{int(time.time())}"

        query = kafka_processed.writeStream \
            .foreachBatch(process_batch) \
            .trigger(processingTime='10 seconds') \
            .option("checkpointLocation", checkpoint_path) \
            .start()

        print("Consumer ready.")
        query.awaitTermination(timeout=100)

        print("Combining gridded visibilities...")
        gridded_visibilities_df = reduce(DataFrame.unionByName, gridded_visibilities)
        final_gridded = consolide_gridding(gridded_visibilities_df)

        print("Generating dirty image...")
        generate_dirty_image(final_gridded, grid_config)
        print("Dirty image generated.")

    except KeyboardInterrupt:
        print("\nStopping consumer.")
        
    except Exception as e:
        print(f"Error in streaming: {e}")
        traceback.print_exc()
        
    finally:
        spark.stop()
        print("Consumer stopped successfully")


def main():    
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
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
        "--max-offsets",
        type=int,
        default=200,
        help="Max offsets per trigger for throughput control (default: 200, test: 10, prod: 500+)"
    )
    
    args = parser.parse_args()
    
    try:
        run_consumer(
            kafka_servers=args.kafka_servers,
            topic=args.topic,
            config_path=args.config,
            max_offsets_per_trigger=args.max_offsets
        )
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
