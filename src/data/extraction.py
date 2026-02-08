import numpy as np
import traceback

CHUNK_SIZE = 10


def extract_field_data(xr_array, start_row, end_row, coord_idx=None):
    if xr_array is None:
        return None

    try:
        if coord_idx is not None:
            data_slice = xr_array.data[start_row:end_row, coord_idx]
        else:
            data_slice = xr_array.data[start_row:end_row]

        if hasattr(data_slice, 'compute'):
            return data_slice.compute()
        else:
            return np.array(data_slice)

    except Exception:
        print(f"Error computing slice for rows {start_row} to {end_row}")
        traceback.print_exc()
        raise


def baseline_key(antenna1, antenna2):
    ant_min, ant_max = sorted([antenna1, antenna2])
    return f"{ant_min}-{ant_max}"


def extract_data(subms, dataset, start_row, end_row, n_channels, n_correlations, visibilities):
    try:
        antenna1 = np.array(dataset.antenna1.data[start_row:end_row].compute()).tolist()
        antenna2 = np.array(dataset.antenna2.data[start_row:end_row].compute()).tolist()
        scan_number = np.array(dataset.scan_number.data[start_row:end_row].compute()).tolist()
        
        exposure = np.array(dataset.dataset.EXPOSURE[start_row:end_row].compute()).tolist()
        interval = np.array(dataset.dataset.INTERVAL[start_row:end_row].compute()).tolist()
        time = np.array(dataset.time.data[start_row:end_row].compute()).tolist()
        
        uvw = np.array(dataset.uvw.data[start_row:end_row].compute()).tolist()
        u = [uvw[i][0] for i in range(len(uvw))]
        v = [uvw[i][1] for i in range(len(uvw))]
        w = [uvw[i][2] for i in range(len(uvw))]
        
        visibilities = np.array(visibilities[start_row:end_row].compute())
        visibilities = np.stack([visibilities.real, visibilities.imag], axis=-1)
        weights = np.array(dataset.weight.data[start_row:end_row].compute())
        flags = np.array(dataset.flag.data[start_row:end_row].compute())

        baseline_keys = [baseline_key(int(a1), int(a2)) for a1, a2 in zip(antenna1, antenna2)]
        
        return {
            "subms_id": subms.id,
            "field_id": subms.field_id,
            "spw_id": subms.spw_id,
            "polarization_id": subms.polarization_id,
            "chunk_id": f"{start_row}_{end_row}",
            
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
    
    except Exception as e:
        print(f"Error extracting spectral chunk: {e}")
        traceback.print_exc()
        raise


def extract_chunks(subms):
    try:
        dataset = subms.visibilities
        visibilities = dataset.data.data.persist()

        nrows = dataset.nrows
        n_channels = dataset.data.shape[1]
        n_correlations = dataset.data.shape[2]

        for start_row in range(0, nrows, CHUNK_SIZE):
            end_row = min(start_row + CHUNK_SIZE, nrows)

            chunk = extract_data(
                subms,
                dataset,
                start_row,
                end_row,
                n_channels,
                n_correlations,
                visibilities,
            )

            yield chunk, subms.id

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