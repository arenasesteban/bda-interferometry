import os
import numpy as np
import io
import traceback
import dask

import msgpack

from dask.distributed import Client, get_worker
from kafka import KafkaProducer

ROWS_PER_BLOCK = 10_000


def _encode_complex_array(array):
    return {
        "__complex_array__": True,
        "real": array.real.astype(np.float32 if array.dtype == np.complex64 else np.float64),
        "imag": array.imag.astype(np.float32 if array.dtype == np.complex64 else np.float64),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _default_encoder(data):
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.complexfloating):
        return _encode_complex_array(data)
    
    if isinstance(data, np.ndarray):
        return {
            "__ndarray__": True, 
            "data": data.tobytes(), 
            "dtype": str(data.dtype), 
            "shape": list(data.shape)
        }
    
    if isinstance(data, np.integer):
        return int(data)
    
    if isinstance(data, np.floating):
        return float(data)
    
    raise TypeError(f"Cannot serialise object of type {type(data)}")


def create_dask_client():
    try:
        dask.config.set({
            "distributed.worker.memory.target": 0.45,
            "distributed.worker.memory.spill": 0.55,
            "distributed.worker.memory.pause": 0.75,
            "distributed.worker.memory.terminate": 0.95,
            "array.slicing.split_large_chunks": True,
        })

        tmp = os.environ.get("DASK_DIR", "tmp/dask")

        client = Client(
            n_workers=1,
            threads_per_worker=4,
            memory_limit="128GB",
            processes=True,
            local_directory=tmp,
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

    if not hasattr(worker, "_kafka_producer"):
        worker._kafka_producer = KafkaProducer(
            bootstrap_servers=["localhost:9092"],
            acks="all",
            retries=10,
            linger_ms=50,
            batch_size=1_048_576,            # 1 MB
            max_request_size=10_485_760,     # 10 MB
            request_timeout_ms=120_000,
            delivery_timeout_ms=180_000,
            compression_type="lz4",
            max_block_ms=120_000,
            api_version_auto_timeout_ms=30_000,
        )

        print(f"[Producer] Kafka producer created on worker: {worker.name}")

    return worker._kafka_producer


def close_kafka_producer():
    try:
        worker = get_worker()

        if hasattr(worker, "_kafka_producer"):
            worker._kafka_producer.flush()
            worker._kafka_producer.close()

    except Exception as e:
        print(f"Error closing Kafka producer: {e}")
        traceback.print_exc()
        raise


def build_payload(block, block_id, subms, n_channels, n_correlations):
    try:
        data_arrays = {
            "subms_id":         subms.id,
            "field_id":         subms.field_id,
            "spw_id":           subms.spw_id,
            "polarization_id":  subms.polarization_id,
            "chunk_id":         block_id,   
            "n_channels":       n_channels,
            "n_correlations":   n_correlations,
            "antenna1":         np.asarray(block[0]),
            "antenna2":         np.asarray(block[1]),
            "scan_number":      np.asarray(block[2]),
            "time":             np.asarray(block[3]),
            "exposure":         np.asarray(block[4]),
            "interval":         np.asarray(block[5]),
            "uvw":              np.asarray(block[6]),
            "visibilities":     np.asarray(block[7]),
            "weights":          np.asarray(block[8]),
            "flags":            np.asarray(block[9])
        }

        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **data_arrays)
            data = buffer.getvalue()

        return msgpack.packb(data, default=_default_encoder, use_bin_type=True)

    except Exception as e:
        print(f"Error building payload: {e}")
        traceback.print_exc()
        raise


def build_metadata(block_idx, blocks):
    try:
        array_map = {
            "antenna1":    blocks[0], "antenna2":   blocks[1], "scan_number": blocks[2],
            "time":        blocks[3], "exposure":   blocks[4], "interval":    blocks[5],
            "uvw":         blocks[6],
            "visibilities": blocks[7], "weights":   blocks[8], "flags":       blocks[9],
        }

        metadata = {
            "schema":          "visibilities_blocks",
            "block_idx":       int(block_idx),
            "shapes":  {k: list(np.asarray(v).shape) for k, v in array_map.items()},
            "dtypes":  {k: str(np.asarray(v).dtype)  for k, v in array_map.items()},
        }

        metadata_bytes = msgpack.packb(metadata, use_bin_type=True)

        headers = [
            ("schema",          b"visibilities_blocks"),
            ("metadata",        metadata_bytes),
        ]

        return headers
    
    except Exception as e:
        print(f"Error building metadata: {e}")
        traceback.print_exc()
        raise


def send_visibilities(*blocks, block_idx, subms, n_channels, n_correlations, topic):
    try:
        producer = create_kafka_producer()

        payload = build_payload(blocks, block_idx, subms, n_channels, n_correlations)
        headers = build_metadata(block_idx, blocks)
        key     = f"block-{block_idx}".encode("utf-8")

        producer.send(topic, key=key, value=payload, headers=headers).get(timeout=60)
        print(f"[Producer] Sent block {block_idx}")



        return True

    except Exception as e:
        print(f"Error creating send_visibilities function: {e}")
        traceback.print_exc()
        raise


def stream_dataset(dataset, subms, topic):
    client = None

    try:
        client = create_dask_client()
        
        nrows = dataset.rows
        n_channels = dataset.data.shape[1]
        n_correlations = dataset.data.shape[2]

        for block_idx, row_start in enumerate(range(0, nrows, ROWS_PER_BLOCK)):
            row_end = min(row_start + ROWS_PER_BLOCK, nrows)

            block = (
                dataset.antenna1.data[row_start:row_end],
                dataset.antenna2.data[row_start:row_end],
                dataset.scan_number.data[row_start:row_end],
                dataset.time.data[row_start:row_end],
                dataset.dataset.EXPOSURE.data[row_start:row_end],
                dataset.dataset.INTERVAL.data[row_start:row_end],
                dataset.uvw.data[row_start:row_end],
                dataset.data.data[row_start:row_end],
                dataset.weight.data[row_start:row_end],
                dataset.flag.data[row_start:row_end]
            )

            task = dask.delayed(send_visibilities)(*block, block_idx=block_idx, subms=subms, n_channels=n_channels, n_correlations=n_correlations, topic=topic)
            client.compute(task).result()

            print(f"[Producer] Block {block_idx} sent (rows {row_start} to {row_end})")

    except Exception as e:
        print(f"Error streaming dataset: {e}")
        traceback.print_exc()
        raise

    finally:
        if client:
            client.close()
