"""
Spark Consumer Service - Kafka to Spark Streaming Integration

Consumes interferometry visibility data from Kafka using Spark Structured Streaming.
Deserializes chunks and provides basic distributed processing with console output.

This service replaces the traditional Kafka consumer with Spark streaming capabilities
for distributed processing of large-scale interferometry datasets.
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


def deserialize_chunk_data(raw_data_bytes: bytes) -> dict:
    """
    Deserialize MessagePack chunk data from Kafka - FASE 2 Version.
    
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
    import msgpack
    import numpy as np
    import zlib
    
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
    Descompone un chunk en filas individuales para procesamiento BDA - FASE 2.
    
    Convierte un chunk que contiene múltiples filas de visibilidades
    en una lista de registros individuales, cada uno representando
    una fila con su baseline_key y scan_number.
    
    Parameters
    ----------
    chunk : dict
        Chunk deserializado con arrays científicos
        
    Returns
    -------
    list
        Lista de filas individuales con baseline_key y metadata
    """
    import numpy as np
    
    rows = []
    
    try:
        nrows = chunk.get('nrows', 0)
        subms_id = chunk.get('subms_id', 'unknown')
        
        if nrows == 0:
            return rows
            
        # Extraer arrays principales
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
        
        # Validar dimensiones
        if len(antenna1) != nrows or len(antenna2) != nrows:
            print(f"⚠️  Dimension mismatch in chunk {chunk.get('chunk_id', 'unknown')}")
            return rows
            
        # Descomponer fila por fila
        for row_idx in range(nrows):
            ant1 = int(antenna1[row_idx]) if len(antenna1) > row_idx else -1
            ant2 = int(antenna2[row_idx]) if len(antenna2) > row_idx else -1
            scan_num = int(scan_number[row_idx]) if len(scan_number) > row_idx else -1
            
            # Crear baseline_key único
            baseline_key = f"{ant1}-{ant2}-{subms_id}"
            
            # Crear registro de fila individual
            row_record = {
                # Identificadores para agrupación
                'baseline_key': baseline_key,
                'scan_number': scan_num,
                'antenna1': ant1,
                'antenna2': ant2,
                'subms_id': subms_id,
                
                # Metadata temporal y espacial
                'time': float(time_array[row_idx]) if len(time_array) > row_idx else 0.0,
                'u': float(u_array[row_idx]) if len(u_array) > row_idx else 0.0,
                'v': float(v_array[row_idx]) if len(v_array) > row_idx else 0.0,
                'w': float(w_array[row_idx]) if len(w_array) > row_idx else 0.0,
                
                # Datos científicos para BDA (por ahora solo metadatos de forma)
                'vis_shape': list(visibilities[row_idx].shape) if len(visibilities) > row_idx else [],
                'weight_shape': list(weight[row_idx].shape) if len(weight) > row_idx else [],
                'flag_shape': list(flag[row_idx].shape) if len(flag) > row_idx else [],
                
                # Metadata original del chunk
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
    Agrupa filas individuales por baseline_key + scan_number - FASE 2.
    
    Parameters
    ----------
    rows : list
        Lista de filas individuales decomposed from chunks
        
    Returns
    -------
    dict
        Diccionario con groups key = (baseline_key, scan_number)
    """
    groups = {}
    
    for row in rows:
        baseline_key = row.get('baseline_key', 'unknown')
        scan_number = row.get('scan_number', -1)
        
        # Crear group key
        group_key = f"{baseline_key}_scan{scan_number}"
        
        if group_key not in groups:
            groups[group_key] = []
            
        groups[group_key].append(row)
        
    return groups


def log_chunk_summary(chunk_metadata: dict, rows: list) -> None:
    """
    Logging conciso con información esencial del chunk - FASE 2.
    """
    chunk_id = chunk_metadata.get('chunk_id', 'unknown')
    subms_id = chunk_metadata.get('subms_id', 'unknown')
    
    # Calcular estadísticas esenciales
    unique_baselines = set([row.get('baseline_key', 'unknown') for row in rows])
    unique_scans = set([row.get('scan_number', -1) for row in rows])
    times = [row.get('time', 0.0) for row in rows]
    avg_time = sum(times) / len(times) if times else 0.0
    
    print(f"📦 Chunk {subms_id}_{chunk_id}: {len(rows)} rows | {len(unique_baselines)} baselines | {len(unique_scans)} scans | time={avg_time:.1f}")


def log_groups_summary(groups: dict) -> None:
    """
    Logging conciso de grupos por baseline + scan - FASE 2.
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
    Process streaming batch para FASE 2: Descomposición y Agrupación BDA - VERSIÓN CONCISA.
    """
    import time
    from datetime import datetime
    from pyspark.sql.functions import spark_partition_id
    from pyspark.sql import SparkSession
    
    current_time = datetime.now().strftime("%H:%M:%S")
    start_time = time.time()
    
    try:
        # Obtener datos del DataFrame
        row_count = df.count()
        
        if row_count == 0:
            print(f"� [{current_time}] Microbatch {epoch_id}: 0 chunks (waiting...)")
            return
            
        # Recolectar y procesar mensajes de Kafka
        kafka_rows = df.collect()
        total_chunks = 0
        total_rows = 0
        all_groups = {}
        
        print(f"\n🚀 [{current_time}] Microbatch {epoch_id}: {len(kafka_rows)} chunks")
        print("-" * 50)
        
        # Procesar cada chunk
        for kafka_row in kafka_rows:
            try:
                # Deserializar y descomponer chunk
                raw_bytes = kafka_row.chunk_data
                chunk = deserialize_chunk_data(raw_bytes)
                
                if not chunk:
                    continue
                    
                total_chunks += 1
                rows = decompose_chunk_to_rows(chunk)
                total_rows += len(rows)
                
                # Logging conciso del chunk
                log_chunk_summary(chunk, rows)
                
                # Agrupar filas por baseline+scan
                chunk_groups = group_rows_by_baseline_scan(rows)
                for group_key, group_rows in chunk_groups.items():
                    if group_key not in all_groups:
                        all_groups[group_key] = []
                    all_groups[group_key].extend(group_rows)
                
            except Exception as e:
                print(f"❌ Chunk error: {e}")
                continue
        
        # Resumen final del microbatch
        processing_time = (time.time() - start_time) * 1000
        print("-" * 50)
        print(f"✅ Summary: {total_chunks} chunks | {total_rows} rows | {len(all_groups)} groups | {processing_time:.0f}ms")
        
        if all_groups:
            log_groups_summary(all_groups)
        
    except Exception as e:
        print(f"❌ Microbatch {epoch_id} error: {e}")
        
    print()  # Línea en blanco para separar microbatches


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
    
    print("🚀 BDA Interferometry Spark Consumer - FASE 2")
    print("🎯 FEATURES: Row Decomposition + Baseline+Scan Grouping")
    print("=" * 70)
    print(f"📡 Kafka Servers: {kafka_servers}")
    print(f"📻 Topic: {topic}")
    print(f"🕐 Started: {start_time}")
    print(f"⚡ Microbatch Trigger: 10 seconds")
    print(f"🔄 No Windowing (deferred to Fase 3)")
    print("=" * 70)
    
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
        
        # Simple processing without windowing (Fase 2)
        processed_df = kafka_df.select(
            kafka_df.key.cast("string").alias("message_key"),
            kafka_df.value.alias("chunk_data"),
            kafka_df.timestamp.alias("kafka_timestamp")
        )
        
        print("✅ Streaming configured for Fase 2 (chunk decomposition + baseline grouping)")
        
        # Configure Spark for optimal performance
        spark.conf.set("spark.sql.shuffle.partitions", "4")
        spark.conf.set("spark.sql.adaptive.enabled", "false")
        
        # Start streaming query with detailed monitoring
        print("\n🔍 Phase 5: Starting streaming query...")
        query = processed_df.writeStream \
            .foreachBatch(process_streaming_batch) \
            .outputMode("append") \
            .trigger(processingTime='10 seconds') \
            .option("checkpointLocation", "/tmp/spark-bda-fase2-checkpoint") \
            .start()
        
        print("✅ Consumer ready - waiting for data...")
        print(f"🔄 Microbatches every 10s | Spark UI: http://localhost:4040 | Ctrl+C to stop")
        print("=" * 60)
        
        # Wait for termination with simplified status updates
        try:
            while query.isActive:
                time.sleep(30)  # Menos heartbeats frecuentes
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"💓 [{current_time}] Listening...")
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        
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
