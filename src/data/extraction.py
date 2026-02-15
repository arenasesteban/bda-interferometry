import os
import numpy as np
import io
import traceback
import dask

import msgpack

from dask.distributed import Client, get_worker
from kafka import KafkaProducer

ROWS_PER_BLOCK = 10_000


def create_dask_client():
    try:
        dask.config.set({
            "distributed.worker.memory.target":     0.45,
            "distributed.worker.memory.spill":      0.55,
            "distributed.worker.memory.pause":      0.75,
            "distributed.worker.memory.terminate":  0.95,
            "array.slicing.split_large_chunks":     True,
        })

        tmp = os.environ.get("DASK_DIR", "tmp/dask")

        client = Client(
            n_workers           = 1,
            threads_per_worker  = 4,
            memory_limit        = "128GB",
            processes           = True,
            local_directory     = tmp,
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
            bootstrap_servers           = ["localhost:9092"],
            acks                        = "all",
            retries                     = 10,
            linger_ms                   = 50,
            batch_size                  = 1_048_576,    # 1 MB
            max_request_size            = 10_485_760,   # 10 MB
            request_timeout_ms          = 120_000,
            delivery_timeout_ms         = 180_000,
            compression_type            = "lz4",
            max_block_ms                = 120_000,
            api_version_auto_timeout_ms = 30_000,
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


def build_payload(block):
    try:
        vs = np.asarray(block[7])

        data_arrays = {
            "antenna1":         np.asarray(block[0]),
            "antenna2":         np.asarray(block[1]),
            "scan_number":      np.asarray(block[2]),
            "time":             np.asarray(block[3]),
            "exposure":         np.asarray(block[4]),
            "interval":         np.asarray(block[5]),
            "u":                np.asarray(block[6])[:, 0],
            "v":                np.asarray(block[6])[:, 1],
            "w":                np.asarray(block[6])[:, 2],
            "visibilities":     np.stack([vs.real, vs.imag], axis=-1),
            "weights":          np.asarray(block[8]),
            "flags":            np.asarray(block[9]).astype(np.int8),
        }

        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **data_arrays)
            return buffer.getvalue()

    except Exception as e:
        print(f"Error building payload: {e}")
        traceback.print_exc()
        raise


def build_metadata(block_idx, data):
    try:
        metadata = {
            "schema":           "visibilities_blocks",
            "block_id":         int(block_idx),   
            "subms_id":         int(data["subms_id"]),
            "field_id":         int(data["field_id"]),
            "spw_id":           int(data["spw_id"]),
            "polarization_id":  int(data["polarization_id"]),
            "n_channels":       int(data["n_channels"]),
            "n_correlations":   int(data["n_correlations"]),
        }

        metadata_bytes = msgpack.packb(metadata, use_bin_type=True)

        headers = [
            ("schema",      b"visibilities_blocks"),
            ("metadata",    metadata_bytes),
        ]

        return headers
    
    except Exception as e:
        print(f"Error building metadata: {e}")
        traceback.print_exc()
        raise


def send_visibilities(*blocks, block_idx, data, topic):
    try:
        producer = create_kafka_producer()

        payload = build_payload(blocks)
        headers = build_metadata(block_idx, data)
        key     = f"block-{block_idx}".encode("utf-8")

        producer.send(topic, key=key, value=payload, headers=headers).get(timeout=60)

        return True

    except Exception as e:
        print(f"Error creating send_visibilities function: {e}")
        traceback.print_exc()
        raise


def send_end_signal(topic, n_blocks):
    try:
        producer = KafkaProducer(
            bootstrap_servers           = ["localhost:9092"],
            acks                        = "all",
            retries                     = 10,
            request_timeout_ms          = 120_000,
            api_version_auto_timeout_ms = 30_000,
        )
        
        metadata = {
            "schema":       "visibilities_blocks",
            "total_blocks": n_blocks,
        }
        
        headers = [
            ("schema",   b"visibilities_blocks"),
            ("metadata", msgpack.packb(metadata, use_bin_type=True)),
        ]
        
        producer.send(topic, key=b"__END__", value=None, headers=headers).get(timeout=60)
        
        print(f"[Producer] Sent end of stream signal", flush=True)

    except Exception as e:
        print(f"Error sending end signal: {e}")
        traceback.print_exc()
        raise

    finally:
        if producer:
            producer.flush()
            producer.close()


def stream_dataset(dataset, subms, topic):
    client = None

    try:
        client = create_dask_client()
        
        nrows = dataset.nrows

        data = {
            "subms_id":         subms.id,
            "field_id":         subms.field_id,
            "spw_id":           subms.spw_id,
            "polarization_id":  subms.polarization_id,
            "n_correlations":   dataset.data.shape[1],
            "n_channels":       dataset.data.shape[2],
        }

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

            task = dask.delayed(send_visibilities)(*block, block_idx=block_idx, data=data, topic=topic)
            client.compute(task).result()

            print(f"[Producer] Block {block_idx} sent (rows {row_start} to {row_end})", flush=True)

        send_end_signal(topic, len(range(0, nrows, ROWS_PER_BLOCK)))

    except Exception as e:
        print(f"Error streaming dataset: {e}")
        traceback.print_exc()
        raise

    finally:
        if client:
            client.close()
