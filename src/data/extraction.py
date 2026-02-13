import numpy as np
import io
import json
import traceback
import dask
import dask.array as da

from dask.distributed import Client, get_worker
from kafka import KafkaProducer


def create_dask_client():
    try:
        # ====================================================================
        # DASK RESOURCE CONFIGURATION
        # ====================================================================
        # Cluster resources: 44 CPUs, 192 GB RAM
        # 
        # Current configuration:
        # - n_workers: 11 workers
        # - threads_per_worker: 4 threads
        # - Total CPUs: 11 × 4 = 44 CPUs (100% utilization)
        # - memory_limit: 16 GB per worker
        # - Total RAM: 11 × 16 = 176 GB (92% utilization, 16 GB for OS)
        #
        # Memory thresholds (% of memory_limit per worker):
        # - target (0.60): Start spilling to disk at 9.6 GB
        # - spill (0.70): Aggressively spill to disk at 11.2 GB
        # - pause (0.85): Pause accepting new tasks at 13.6 GB
        # - terminate (0.95): Kill worker at 15.2 GB
        # ====================================================================
        
        dask.config.set({
            "distributed.worker.memory.target": 0.60,
            "distributed.worker.memory.spill": 0.70,
            "distributed.worker.memory.pause": 0.85,
            "distributed.worker.memory.terminate": 0.95,
        })

        client = Client(
            n_workers=10,
            threads_per_worker=4,
            memory_limit="22GB",
            processes=True,
            local_directory="/tmp/dask-worker-space",
        )

        print(f"[Producer] Dask client created: {client}")
        print(f"[Producer] Dashboard available at: {client.dashboard_link}")
        return client
    
    except Exception as e:
        print(f"Error creating Dask client: {e}")
        traceback.print_exc()
        raise


def create_kafka_producer():
    worker = get_worker()

    producer_config = {
        "bootstrap_servers": ["localhost:9092"],
        "acks": "all",
        "retries": 10,
        "linger_ms": 50,
        "batch_size": 1_048_576,            # 1 MB
        "max_request_size": 10_485_760,     # 10 MB
        "request_timeout_ms": 120_000,
        "compression_type": "gzip",
        "max_block_ms": 120_000,
        "api_version_auto_timeout_ms": 30_000,
    }

    if not hasattr(worker, "_kafka_producer"):
        worker._kafka_producer = KafkaProducer(**producer_config)
        print(f"[Producer] Kafka producer created on worker: {worker.name}")

    return worker._kafka_producer


def select_fields(dataset):
    antenna1 = dataset.antenna1.data
    antenna2 = dataset.antenna2.data

    time = dataset.time.data
    exposure = dataset.dataset.EXPOSURE.data
    interval = dataset.dataset.INTERVAL.data

    uvw = dataset.uvw.data

    visibilities = dataset.data.data
    weights = dataset.weight.data
    flags = dataset.flag.data

    return visibilities, (
        antenna1, antenna2,
        time, exposure, interval,
        uvw,
        visibilities, weights, flags
    )


def build_payload(block):
    try:
        buffer = io.BytesIO()

        data_arrays = {
            "antenna1": np.asarray(block[0]),
            "antenna2": np.asarray(block[1]),
            "time": np.asarray(block[2]),
            "exposure": np.asarray(block[3]),
            "interval": np.asarray(block[4]),
            "uvw": np.asarray(block[5]),
            "visibilities": np.asarray(block[6]),
            "weights": np.asarray(block[7]),
            "flags": np.asarray(block[8])
        }

        np.savez_compressed(buffer, **data_arrays)

        payload = buffer.getvalue()
        buffer.close()

        return payload

    except Exception as e:
        print(f"Error building payload: {e}")
        traceback.print_exc()
        raise


def build_metadata(block_info, shape):
    try:
        info = block_info[None]
        block_id = info["chunk-location"]
        array_location = info["array-location"]

        row_start = array_location[0][0]
        row_end = array_location[0][1]

        metadata = {
            "schema": "visibilities_blocks",
            "block_id": list(block_id) if block_id is not None else None,
            "block_shape": list(info["chunk-shape"]),
            "array_location": [[int(s), int(e)] for s, e in array_location],
            "row_range": [int(row_start), int(row_end)],
            "n_rows": int(row_end - row_start),
            "n_channels": int(shape[1]) if len(shape) > 1 else 0,
            "n_correlations": int(shape[2]) if len(shape) > 2 else 0,
        }

        headers = [
            ("schema", b"visibilities_blocks"),
            ("metadata", json.dumps(metadata).encode("utf-8"))
        ]
        
        return headers, block_id, f"{row_start}_{row_end}"
    
    except Exception as e:
        print(f"Error building metadata: {e}")
        traceback.print_exc()
        raise


def make_kafka_key(block_id, message_id):
    block_str = ",".join(map(str, block_id))
    return f"{message_id}|{block_str}".encode("utf-8")


def produce_visibilities_block(*blocks, topic, block_info=None):
    try:
        if block_info is None:
            return np.empty((0,), dtype=np.uint8) 

        producer = create_kafka_producer()

        info = block_info[None]
        block_id = info["chunk-location"]

        payload = build_payload(blocks)
        headers, block_id, rows_range = build_metadata(block_info, blocks[6].shape)        
        key = make_kafka_key(block_id, message_id="visibilities_block")

        future = producer.send(topic, key=key, value=payload, headers=headers)
        future.get(timeout=60)

        block_id_str = ','.join(map(str, block_id)) if block_id else "unknown"
        print(f"[Producer] Sent Block ID: {block_id_str}, Range: {rows_range}")

        return np.ones((blocks[6].shape[0], ), dtype=np.uint8)
    
    except Exception as e:
        print(f"Error producing block: {e}")
        traceback.print_exc()
        raise


def stream_dataset(dataset, topic):
    try:
        client = create_dask_client()

        data = select_fields(dataset)

        dask.config.set(scheduler="distributed")

        out = da.map_blocks(
            produce_visibilities_block,
            *data,
            dtype=np.uint8,
            drop_axis=(1, 2),
            chunks=(data[6].chunks[0],),
            topic=topic
        )
        print("[Producer] Resulting Dask array for streaming:", out)

        out.compute()

        if client:
            client.close()

        return True

    except Exception as e:
        print(f"Error streaming dataset: {e}")
        traceback.print_exc()
        raise