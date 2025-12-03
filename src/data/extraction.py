import numpy as np


def _get_optimal_chunk_size(vis_set):    
    chunks = vis_set.data.data.chunks[0]  # Row dimension chunks
    
    if len(chunks) > 1:
        return chunks[0]
    else:
        return min(1000, vis_set.nrows)


def _safe_compute_slice(xr_array, start_row, end_row, coord_idx):
    if xr_array is None:
        return None
        
    try:
        if coord_idx is not None:
            # For UVW coordinates (u, v, w)
            data_slice = xr_array.data[start_row:end_row, coord_idx]
        else:
            # For other arrays
            data_slice = xr_array.data[start_row:end_row]
            
        # Compute if it's a dask array
        if hasattr(data_slice, 'compute'):
            return data_slice.compute()
        else:
            return np.array(data_slice)
            
    except Exception:
        return None


def to_simple_array(np_array):
    if np_array is None:
        return []
    try:
        # Asegurar que es un array NumPy
        if not isinstance(np_array, np.ndarray):
            np_array = np.array(np_array)
        
        # Convertir a lista Python simple
        return np_array.tolist()
    except Exception as e:
        print(f"Could not convert array to list: {e}")
        return []


def _extract_chunk_data(subms, vis_set, chunk_id, start_row, end_row, n_channels, n_correlations):
    chunk_size = end_row - start_row

    antenna1 = _safe_compute_slice(vis_set.antenna1, start_row, end_row)
    antenna2 = _safe_compute_slice(vis_set.antenna2, start_row, end_row)

    return {
        # Metadata
        'subms_id': subms._id,
        'chunk_id': chunk_id,
        'field_id': subms.field_id,
        'spw_id': subms.spw_id,
        'polarization_id': subms.polarization_id,
        
        'row_start': start_row,
        'row_end': end_row,
        'nrows': chunk_size,
        
        'n_channels': n_channels,
        'n_correlations': n_correlations,

        'antenna1': to_simple_array(antenna1),
        'antenna2': to_simple_array(antenna2),
        'scan_number': to_simple_array(_safe_compute_slice(vis_set.scan_number, start_row, end_row)),

        # Timing information
        'exposure': to_simple_array(_safe_compute_slice(vis_set.dataset.EXPOSURE, start_row, end_row)),
        'interval': to_simple_array(_safe_compute_slice(vis_set.dataset.INTERVAL, start_row, end_row)),
        'time': to_simple_array(_safe_compute_slice(vis_set.time, start_row, end_row)),

        'u': to_simple_array(_safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=0)),
        'v': to_simple_array(_safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=1)),
        'w': to_simple_array(_safe_compute_slice(vis_set.uvw, start_row, end_row, coord_idx=2)),

        # Essential scientific data arrays
        'visibilities': _safe_compute_slice(vis_set.data, start_row, end_row),
        'weight': _safe_compute_slice(vis_set.weight, start_row, end_row),
        'flag': _safe_compute_slice(vis_set.flag, start_row, end_row),
    }


def _extract_subms_chunks(subms):
    vis = subms.visibilities
    
    # Get basic dimensions
    nrows = vis.nrows
    n_channels = vis.data.shape[1]
    n_correlations = vis.data.shape[2]

    # Determine chunk size based on Dask chunks or default
    chunk_size = _get_optimal_chunk_size(vis)
    
    # Generate chunks
    chunk_id = 0
    for start_row in range(0, nrows, chunk_size):
        end_row = min(start_row + chunk_size, nrows)
        
        chunk = _extract_chunk_data(
            subms, vis, chunk_id, start_row, end_row, 
            n_channels, n_correlations
        )
        
        yield chunk
        chunk_id += 1


def stream_subms_chunks(dataset):    
    if not hasattr(dataset, 'ms_list') or dataset.ms_list is None:
        raise ValueError("No ms_list found in dataset")
    
    for subms in dataset.ms_list:
        if subms is None or subms.visibilities is None:
            continue

        yield from _extract_subms_chunks(subms)