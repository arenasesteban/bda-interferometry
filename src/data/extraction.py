import gc
import numpy as np
import traceback
import dask
from dask.distributed import Client, get_client, wait


MESSAGE_SIZE = 10000


def setup_client():
    try:
        dask.config.set({
            "distributed.worker.memory.target": 0.50,
            "distributed.worker.memory.spill": 0.60,
            "distributed.worker.memory.pause": 0.80,
            "distributed.worker.memory.terminate": 0.95,
            "distributed.worker.memory.spill-compression": "lz4",  # opcional
        })

        client_config = {
            "processes": True,
            "n_workers": 4,
            "threads_per_worker": 1,
            "memory_limit": "40GB",
            "local_directory": "/tmp/dask-worker-space",
            "silence_logs": 50,
        }

        client = Client(**client_config)

        return client
    
    except Exception as e:
        print(f"Error setting up Dask client: {e}")
        traceback.print_exc()
        raise


def baseline_key(antenna1, antenna2):
    ant_min, ant_max = sorted([antenna1, antenna2])
    return f"{ant_min}-{ant_max}"


def process_block(block):
    try:
        if block.dtype == np.complex128:
            block = block.astype(np.complex64)

        return block
    
    except Exception as e:
        print(f"Error processing block: {e}")
        traceback.print_exc()
        raise


def rechunk_dataset(dataset, chunk_size=MESSAGE_SIZE):
    try:
        dataset.antenna1.data = dataset.antenna1.data.rechunk(chunk_size)
        dataset.antenna2.data = dataset.antenna2.data.rechunk(chunk_size)
        dataset.scan_number.data = dataset.scan_number.data.rechunk(chunk_size)
        
        dataset.dataset.EXPOSURE.data = dataset.dataset.EXPOSURE.data.rechunk(chunk_size)
        dataset.dataset.INTERVAL.data = dataset.dataset.INTERVAL.data.rechunk(chunk_size)
        dataset.time.data = dataset.time.data.rechunk(chunk_size)
        
        dataset.uvw.data = dataset.uvw.data.rechunk({0: chunk_size, 1: -1})
        
        dataset.data.data = dataset.data.data.rechunk({0: chunk_size, 1: -1, 2: -1})
        dataset.weight.data = dataset.weight.data.rechunk({0: chunk_size, 1: -1, 2: -1})
        dataset.flag.data = dataset.flag.data.rechunk({0: chunk_size, 1: -1, 2: -1})

        return dataset
        
    except Exception as e:
        print(f"[Extraction] Error rechunking SubMS: {e}")
        traceback.print_exc()
        raise


def compute_chunk(dataset, start_chunk, end_chunk):
    try:
        antenna1 = dataset.antenna1.data[start_chunk:end_chunk]
        antenna2 = dataset.antenna2.data[start_chunk:end_chunk]
        scan_number = dataset.scan_number.data[start_chunk:end_chunk]

        exposure = dataset.dataset.EXPOSURE.data[start_chunk:end_chunk]
        interval = dataset.dataset.INTERVAL.data[start_chunk:end_chunk]
        time = dataset.time.data[start_chunk:end_chunk]

        uvw = dataset.uvw.data[start_chunk:end_chunk]

        weights = dataset.weight.data[start_chunk:end_chunk]
        flags = dataset.flag.data[start_chunk:end_chunk]
        
        return dask.compute(
            antenna1, antenna2, scan_number,
            exposure, interval, time,
            uvw, weights, flags
        )
    except Exception as e:
        print(f"[Extraction] Error computing chunk [{start_chunk}-{end_chunk}]: {e}")
        traceback.print_exc()
        raise


def compute_visibilities(dataset, start_chunk, end_chunk):
    try:
        client = get_client()

        arr = dataset.data.data[start_chunk:end_chunk]
        if arr.dtype == np.complex128:
            arr = arr.astype(np.complex64)

        arr_p = client.persist(arr)           # dask array, persistido en cluster
        fut = client.compute(arr_p)           # Future del resultado final (numpy)
        wait(fut)

        visibilities = fut.result()

        client.cancel(fut)
        # cancelar arr_p también ayuda a liberar refs en cluster
        client.cancel(arr_p)
        del fut, arr_p

        return visibilities

    except Exception as e:
        print(f"[Extraction] Error computing visibilities for chunk [{start_chunk}-{end_chunk}]: {e}")
        traceback.print_exc()
        raise


def create_message(subms, chunk_idx, computed_data, visibilities, start_chunk, end_chunk, n_channels, n_correlations):
    try:
        (
            antenna1, antenna2, scan_number,
            exposure, interval, time,
            uvw, weights, flags
        ) = computed_data

        u = uvw[:, 0].tolist()
        v = uvw[:, 1].tolist()
        w = uvw[:, 2].tolist()

        vs = np.asarray(visibilities)
        visibilities = np.stack([vs.real, vs.imag], axis=-1)
        
        baseline_keys = [baseline_key(int(a1), int(a2)) for a1, a2 in zip(antenna1, antenna2)]
        
        message = {
            "subms_id": subms.id,
            "field_id": subms.field_id,
            "spw_id": subms.spw_id,
            "polarization_id": subms.polarization_id,
            "chunk_id": f"chunk_{chunk_idx}[{start_chunk}-{end_chunk}]",
            
            "baseline_keys": baseline_keys,
            "antenna1": antenna1,
            "antenna2": antenna2,
            "scan_number": scan_number,
            
            "exposure": exposure,
            "interval": interval,
            "time": time,

            "n_channels": n_channels,
            "n_correlations": n_correlations,
            
            "u": u,
            "v": v,
            "w": w,
            
            "visibilities": visibilities,
            "weights": weights,
            "flags": flags,
        }

        del visibilities

        return message
    
    except Exception as e:
        print(f"Error creating message: {e}")
        traceback.print_exc()
        raise


def extract_chunks(subms):
    try:
        dataset = subms.visibilities
        # dataset = rechunk_dataset(dataset)

        n_channels = dataset.data.shape[1]
        n_correlations = dataset.data.shape[2]
        chunk_size = dataset.data.data.chunks[0]

        chunk_start = 0
        for chunk_idx, chunk_size in enumerate(chunk_size):
            chunk_end = chunk_start + chunk_size

            arr = dataset.data.data
            sl = arr[chunk_start:chunk_end]

            print("[Extraction] BASE:", arr)
            print("[Extraction] BASE chunks:", arr.chunks)

            print("[Extraction] SLICE:", sl)
            print("[Extraction] SLICE chunks:", sl.chunks)
            
            print("[Extraction] BASE tasks:", len(arr.__dask_graph__()))
            print("[Extraction] SLICE tasks:", len(sl.__dask_graph__()))

            print(f"[Extraction] Computing chunk [{chunk_start}-{chunk_end}]", flush=True)
            computed_data = compute_chunk(dataset, chunk_start, chunk_end)

            print(f"[Extraction] Computing visibilities for chunk [{chunk_start}-{chunk_end}]", flush=True)
            visibilities = compute_visibilities(dataset, chunk_start, chunk_end)

            print(f"[Extraction] Visibilities bytes:", visibilities.nbytes/1024*3, flush=True)

            print(f"[Extraction] Creating message for chunk [{chunk_start}-{chunk_end}]", flush=True)
            message = create_message(
                    subms,
                    chunk_idx,
                    computed_data,
                    visibilities,
                    chunk_start,
                    chunk_end,
                    n_channels,
                    n_correlations
                )
                
            yield message, subms.id
            
            del computed_data
            del visibilities

            gc.collect()

            chunk_start = chunk_end

    except Exception as e:
        print(f"Error extracting subms chunks: {e}")
        traceback.print_exc()
        raise


def stream_dataset(dataset):
    try:
        if not hasattr(dataset, "ms_list") or dataset.ms_list is None:
            raise ValueError("No ms_list found in dataset")

        for subms in dataset.ms_list:
            if subms is None or subms.visibilities is None:
                continue

            yield from extract_chunks(subms)

    except Exception as e:
        print(f"Error streaming dataset: {e}")
        traceback.print_exc()
        raise