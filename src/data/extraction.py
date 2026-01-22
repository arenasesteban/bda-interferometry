import numpy as np
import traceback


def get_chunk_size(vs_set):
    """
    Get chunk size for the given visibility dataset.

    Parameters
    ----------
    vs_set : VisibilitySet
        The visibility dataset.
    
    Returns
    -------
    int
        Chunk size.
    """
    try:
        chunks = vs_set.data.data.chunks[0]

        if len(chunks) > 1:
            return chunks[0]
        else:
            return min(1000, vs_set.nrows)

    except Exception as e:
        print(f"Error getting chunk size: {e}")
        traceback.print_exc()
        raise


def extract_field_data(xr_array, start_row, end_row, coord_idx=None):
    """
    Extract data slice from an xarray DataArray.

    Parameters
    ----------
    xr_array : xarray.DataArray
        The input xarray DataArray.
    start_row : int
        The starting row index.
    end_row : int
        The ending row index.
    coord_idx : int, optional
        The index of the coordinate to extract.

    Returns
    -------
    np.ndarray
        The extracted data slice.
    """
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


def convert_to_list(np_array):
    """
    Convert a numpy array to a list.
    
    Parameters
    ----------
    np_array : np.ndarray
        The input numpy array.
    
    Returns
    -------
    list
        The converted list.
    """
    if np_array is None:
        return []

    try:
        if not isinstance(np_array, np.ndarray):
            np_array = np.array(np_array)

        return np_array.tolist()

    except Exception as e:
        print(f"Error converting array to list: {e}")
        traceback.print_exc()
        raise


def extract_chunk_data(
    subms,
    vs_set,
    chunk_id,
    start_row,
    end_row,
    n_channels,
    n_correlations,
    visibilities
):
    """
    Extract data for a specific chunk from the visibility dataset.

    Parameters
    ----------
    subms : SubMS
        The sub-measurement set.
    vs_set : VisibilitySet
        The visibility dataset.
    chunk_id : int
        The chunk identifier.
    start_row : int
        The starting row index.
    end_row : int
        The ending row index.
    n_channels : int
        Number of channels.
    n_correlations : int
        Number of correlations.
    
    Returns
    -------
    dict
        Extracted chunk data.
    """
    try:
        chunk_size = end_row - start_row

        return {
            "subms_id": subms.id,
            "chunk_id": chunk_id,
            "field_id": subms.field_id,
            "spw_id": subms.spw_id,
            "polarization_id": subms.polarization_id,

            "row_start": start_row,
            "row_end": end_row,
            "nrows": chunk_size,
            
            "n_channels": n_channels,
            "n_correlations": n_correlations,

            "antenna1": convert_to_list(extract_field_data(vs_set.antenna1, start_row, end_row)),
            "antenna2": convert_to_list(extract_field_data(vs_set.antenna2, start_row, end_row)),
            "scan_number": convert_to_list(extract_field_data(vs_set.scan_number, start_row, end_row)),
            
            "exposure": convert_to_list(extract_field_data(vs_set.dataset.EXPOSURE, start_row, end_row)),
            "interval": convert_to_list(extract_field_data(vs_set.dataset.INTERVAL, start_row, end_row)),
            "time": convert_to_list(extract_field_data(vs_set.time, start_row, end_row)),
            
            "u": convert_to_list(extract_field_data(vs_set.uvw, start_row, end_row, coord_idx=0)),
            "v": convert_to_list(extract_field_data(vs_set.uvw, start_row, end_row, coord_idx=1)),
            "w": convert_to_list(extract_field_data(vs_set.uvw, start_row, end_row, coord_idx=2)),
            
            "visibilities": visibilities[start_row:end_row].compute(),
            "weight": extract_field_data(vs_set.weight, start_row, end_row),
            "flag": extract_field_data(vs_set.flag, start_row, end_row),
        }

    except Exception as e:
        print(f"Error extracting chunk data: {e}")
        traceback.print_exc()
        raise


def extract_subms_chunks(subms):
    """
    Extract chunks of data from a sub-measurement set.

    Parameters
    ----------
    subms : SubMS
        The sub-measurement set.

    Returns
    -------
    generator
        A generator yielding extracted chunk data.
    """
    try:
        vs_set = subms.visibilities
        visibilities = vs_set.data.data.persist()

        nrows = vs_set.nrows
        n_channels = vs_set.data.shape[1]
        n_correlations = vs_set.data.shape[2]

        chunk_size = get_chunk_size(vs_set)

        chunk_id = 0
        for start_row in range(0, nrows, chunk_size):
            end_row = min(start_row + chunk_size, nrows)
            
            chunk = extract_chunk_data(
                subms, vs_set, chunk_id, start_row, end_row,
                n_channels, n_correlations,
                visibilities
            )
            
            yield chunk
            chunk_id += 1

    except Exception as e:
        print(f"Error extracting subms chunks: {e}")
        traceback.print_exc()
        raise


def stream_subms_chunks(dataset):
    """
    Stream chunks of data from a measurement set.

    Parameters
    ----------
    dataset : Dataset
        The Pyralysis dataset.
    
    Returns
    -------
    generator
        A generator yielding extracted chunk data from all sub-measurement sets.
    """
    try:
        if not hasattr(dataset, "ms_list") or dataset.ms_list is None:
            raise ValueError("No ms_list found in dataset")

        for subms in dataset.ms_list:
            if subms is None or subms.visibilities is None:
                continue

            yield from extract_subms_chunks(subms)

    except Exception as e:
        print(f"Error streaming subms chunks: {e}")
        traceback.print_exc()
        raise