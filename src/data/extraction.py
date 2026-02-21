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
            "distributed.worker.memory.target":     0.60,
            "distributed.worker.memory.spill":      0.80,
            "distributed.worker.memory.pause":      0.95,
            "distributed.worker.memory.terminate":  0.98,
            "array.slicing.split_large_chunks":     True,
        })

        tmp = os.environ.get("DASK_DIR", "tmp/dask")

        client = Client(
            n_workers           = 1,
            threads_per_worker  = 4,
            memory_limit        = "600GB",
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


def rechunk_dataset(dataset):
    row_chunks = dataset.data.data.chunks[0]

    def rechunk_rows(x):
        if x.chunks and x.chunks[0] == row_chunks:
            return x

        if len(x.shape) == 1:
            return x.rechunk((row_chunks,))

        if len(x.shape) == 2:
            return x.rechunk((row_chunks, -1))
        
        if len(x.shape) == 3:
            return x.rechunk((row_chunks, -1, -1))
        
        raise ValueError(f"Unsupported ndim={len(x.shape)} for array shape={x.shape}")

    dataset_chunked = (
        rechunk_rows(dataset.antenna1.data),
        rechunk_rows(dataset.antenna2.data),
        rechunk_rows(dataset.scan_number.data),
        rechunk_rows(dataset.time.data),
        rechunk_rows(dataset.dataset.EXPOSURE.data),
        rechunk_rows(dataset.dataset.INTERVAL.data),
        rechunk_rows(dataset.uvw.data),
        rechunk_rows(dataset.data.data),
        rechunk_rows(dataset.weight.data),
        rechunk_rows(dataset.flag.data),
    )

    return dataset_chunked


def create_kafka_producer():
    return KafkaProducer(
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


def iter_rows(nrows: int, rows_per_msg: int):
    for start in range(0, nrows, rows_per_msg):
        end = min(start + rows_per_msg, nrows)

        yield start, end


def compute_chunk(chunk_delayed):
    try:
        (
            antenna1,
            antenna2,
            scan_number,
            time,
            exposure,
            interval,
            uvw,
            visibility,
            weight,
            flag
        ) = dask.compute(*chunk_delayed)

        u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]

        return (
            antenna1, antenna2, scan_number,
            time, exposure, interval,
            u, v, w,
            visibility, weight, flag
        )

    except Exception as e:
        print(f"Error computing arrays: {e}")
        traceback.print_exc()
        raise


def build_payload(data, start, end):
    try:
        u = data[6][:, 0]
        v = data[6][:, 1]
        w = data[6][:, 2]

        vs = data[7][start:end]
        visibilities = np.stack((vs.real, vs.imag), axis=-1)

        fs = data[9][start:end]
        flags = fs.astype(np.int8)

        data_arrays = {
            "antenna1":         data[0][start:end],
            "antenna2":         data[1][start:end],
            "scan_number":      data[2][start:end],
            "time":             data[3][start:end],
            "exposure":         data[4][start:end],
            "interval":         data[5][start:end],
            "u":                u[start:end],
            "v":                v[start:end],
            "w":                w[start:end],
            "visibilities":     visibilities,
            "weights":          data[8][start:end],
            "flags":            flags,
        }

        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **data_arrays)
            return buffer.getvalue()

    except Exception as e:
        print(f"Error building payload: {e}")
        traceback.print_exc()
        raise


def build_metadata(message_id, data):
    try:
        metadata = {
            "schema":           "visibilities_blocks",
            "message_id":       message_id,
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


def send_visibilities(*chunk, chunk_id, data, producer, topic):
    try:
        nrows = chunk[0].shape[0]

        for block_idx, (start, end) in enumerate(iter_rows(nrows, ROWS_PER_BLOCK)):
            message_id = f"{chunk_id}-{block_idx}"
            payload = build_payload(chunk, start, end)
            headers = build_metadata(message_id, data)
            key     = f"message-{message_id}".encode("utf-8")

            producer.send(topic, key=key, value=payload, headers=headers).get(timeout=60)
            
            print(f"[Producer] Chunk {chunk_id} | Block {block_idx} sent (rows {start} to {end})", flush=True)
        
        print(f"-" * 60, flush=True)

        return True

    except Exception as e:
        print(f"Error creating send_visibilities function: {e}")
        traceback.print_exc()
        raise


def send_end_signal(producer, topic, n_blocks):
    try:        
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


def stream_dataset(dataset, subms, topic):
    producer = None

    try:
        producer = create_kafka_producer()

        data = {
            "subms_id":         subms.id,
            "field_id":         subms.field_id,
            "spw_id":           subms.spw_id,
            "polarization_id":  subms.polarization_id,
            "n_channels":       dataset.data.shape[1],
            "n_correlations":   dataset.data.shape[2],
        }

        arrays = dask.compute(*rechunk_dataset(dataset))

        nrows = dataset.nrows
        chunks = dataset.data.data.chunks[0]
        nchunks = len(chunks)

        print(f"[Extraction] Starting to stream dataset with {nrows} rows in {nchunks} chunks\n", flush=True)

        for chunk_id, (start, end) in enumerate(iter_rows(nrows, chunks[0])):
            chunk_data = tuple(a[start:end] for a in arrays)
            send_visibilities(*chunk_data, chunk_id=chunk_id, data=data, producer=producer, topic=topic)
            
        send_end_signal(producer, topic, len(range(0, nrows, ROWS_PER_BLOCK)))

    except Exception as e:
        print(f"Error streaming dataset: {e}")
        traceback.print_exc()
        raise

    finally:
        if producer:
            producer.flush()
            producer.close()
