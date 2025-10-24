import numpy as np
import traceback

from pyspark.sql.functions import col

grid_config = {
    "u_size": 512,
    "v_size": 512,
    "u_min": -1000,
    "u_max": 1000,
    "v_min": -1000,
    "v_max": 1000
}

def apply_gridding(df):
    try:
        scientific_df = df.repartition(4, col("baseline_key"), col("scan_number"))

        def grid_data(iterator):
            partition_grids = []

            for gridded in process_rows(iterator, grid_config):
                partition_grids.append(gridded)

            if partition_grids:
                combined_grid = combine_grids(partition_grids, grid_config)
                return iter([combined_grid])

            return iter([])
            
        gridded_rdd = scientific_df.rdd.mapPartitions(grid_data)
        return gridded_rdd

    except Exception as e:
        print(f"Error during gridding: {e}")
        traceback.print_exc()
        raise

def process_rows(iterator, grid_config):
    accumulated_grid = None

    try:
        for row in iterator:
            u, v = row.u, row.v
            u_idx, v_idx = uv_to_grid_index(u, v, grid_config)

            if u_idx < 0 or u_idx >= grid_config["u_size"] or v_idx < 0 or v_idx >= grid_config["v_size"]:
                continue

            visibilities = row.visibilities
            weights = row.weight
            flags = row.flag

            n_channels = row.n_channels
            n_correlations = row.n_correlations

            if accumulated_grid is None:
                accumulated_grid = create_grid(n_channels, n_correlations, grid_config)

            for chan in range(n_channels):
                for corr in range(n_correlations):
                    if flags[chan, corr]:
                        continue

                    vs_real = visibilities[chan, corr, 0]
                    vs_imag = visibilities[chan, corr, 1]
                    vs_complex = complex(vs_real, vs_imag)

                    ws_val = weights[corr]

                    accumulated_grid['vs_real'][chan, corr, u_idx, v_idx] += vs_complex.real * ws_val
                    accumulated_grid['vs_imag'][chan, corr, u_idx, v_idx] += vs_complex.imag * ws_val
                    accumulated_grid['weights'][chan, corr, u_idx, v_idx] += ws_val

        if accumulated_grid is not None:
            yield accumulated_grid

    except Exception as e:
        print(f"Error processing rows for gridding: {e}")
        traceback.print_exc()
        raise


def uv_to_grid_index(u, v, grid_config):
    u_min = grid_config["u_min"]
    u_max = grid_config["u_max"]
    v_min = grid_config["v_min"]
    v_max = grid_config["v_max"]

    u_size = grid_config["u_size"]
    v_size = grid_config["v_size"]
    
    # Normalizar a [0, 1]
    u_norm = (u - u_min) / (u_max - u_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    # Convertir a índices (usar redondeo al más cercano)
    u_idx = int(np.round(u_norm * (u_size - 1)))
    v_idx = int(np.round(v_norm * (v_size - 1)))
    
    return u_idx, v_idx


def create_grid(n_channels, n_correlations, grid_config):
    try:
        u_size = grid_config["u_size"]
        v_size = grid_config["v_size"]

        u_grid, v_grid = create_uv_grid(grid_config)

        return {
            'vs_real': np.zeros((n_channels, n_correlations, u_size, v_size), dtype=np.float32),
            'vs_imag': np.zeros((n_channels, n_correlations, u_size, v_size), dtype=np.float32),
            'weights': np.zeros((n_channels, n_correlations, u_size, v_size), dtype=np.float32),
            'u_grid': u_grid,
            'v_grid': v_grid
        }

    except Exception as e:
        print(f"Error creating grid: {e}")
        traceback.print_exc()
        raise

def create_uv_grid(grid_config):
    try:
        u_min = grid_config["u_min"]
        u_max = grid_config["u_max"]
        v_min = grid_config["v_min"]
        v_max = grid_config["v_max"]

        u_size = grid_config["u_size"]
        v_size = grid_config["v_size"]

        u_coords = np.linspace(u_min, u_max, num=u_size, dtype=np.float32)
        v_coords = np.linspace(v_min, v_max, num=v_size, dtype=np.float32)

        u_grid, v_grid = np.meshgrid(u_coords, v_coords, indexing='ij')

        return u_grid, v_grid

    except Exception as e:
        print(f"Error creating UV grid: {e}")
        traceback.print_exc()
        raise

def combine_grids(grid_list, grid_config):
    try:
        first = grid_list[0]
        n_channels, n_correlations = first['vs_real'].shape[0], first['vs_real'].shape[1]

        grid_result = create_grid(n_channels, n_correlations, grid_config)

        for grid in grid_list:
            grid_result['vs_real'] += grid['vs_real']
            grid_result['vs_imag'] += grid['vs_imag']
            grid_result['weights'] += grid['weights']
        
        return grid_result
    
    except Exception as e:
        print(f"Error combining grids: {e}")
        traceback.print_exc()
        raise

def process_all_batches(gridded_batches, grid_config):
    try:
        all_grids = gridded_batches.collect()
        final_grid = combine_grids(all_grids, grid_config)
        
        return normalize_grid(final_grid)


    except Exception as e:
        print(f"Error processing all batches: {e}")
        traceback.print_exc()
        raise

def normalize_grid(grid):
    mask = grid['weights'] > 0

    grid['vs_real'] = np.where(mask, grid['vs_real'] / grid['weights'], 0)
    grid['vs_imag'] = np.where(mask, grid['vs_imag'] / grid['weights'], 0)

    return grid