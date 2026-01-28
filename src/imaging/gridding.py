import math
import numpy as np
import traceback
import json
from pathlib import Path
from astropy.constants import c

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType
)

from .weighting_schemes import apply_weighting


def apply_gridding(df, grid_config):
    try:
        scientific_df = df.repartition(4, F.col("baseline_key"), F.col("scan_number"))

        def grid_data(iterator):
            return process_gridding(iterator, grid_config)

        gridded = scientific_df.rdd.mapPartitions(grid_data)
        gridded_rdd = df.sparkSession.createDataFrame(gridded, define_grid_schema())

        gridded_acc = gridded_rdd.groupBy("u_pix", "v_pix").agg(
            F.sum("real").alias("real"),
            F.sum("imag").alias("imag"),
            F.sum("weight").alias("weight")
        )
        
        return gridded_acc

    except Exception as e:
        print(f"Error during gridding: {e}")
        traceback.print_exc()
        raise


def process_gridding(iterator, grid_config):
    try:
        accumulated_grid = {}

        def accumulate_visibilities(u_pix, v_pix, u_pix_h, v_pix_h, vs, ws):

            grid_key = (u_pix, v_pix)
            grid_key_h = (u_pix_h, v_pix_h)

            accumulate_grid(accumulated_grid, grid_key, vs, ws)
            accumulate_grid(accumulated_grid, grid_key_h, np.conj(vs), ws)

        process_visibilities(iterator, grid_config, accumulate_visibilities)
        weighted_grid = apply_weighting(accumulated_grid, grid_config)

        if weighted_grid:
            for (u_pix, v_pix), values in weighted_grid.items():
                yield (
                    int(u_pix),
                    int(v_pix),
                    float(values['real']),
                    float(values['imag']),
                    float(values['weight'])
                )

    except Exception as e:
        print(f"[Gridding] Error processing gridded data: {e}")
        traceback.print_exc()
        raise


def process_visibilities(iterator, grid_config, accumulate_visibilities):
    try:
        img_size = grid_config["img_size"]
        padding_factor = grid_config["padding_factor"]
        cellsize = grid_config["cellsize"]
        chan_freq = grid_config["chan_freq"]
        corrs_string = grid_config["corrs_string"]

        grid_size = [int(img_size * padding_factor), int(img_size * padding_factor)]  # [v_size, u_size]
        du = - 1.0 / (cellsize * grid_size[1])  # u_direction (width)
        dv = 1.0 / (cellsize * grid_size[0])    # v_direction (height)
        uvcellsize = [du, dv]
        corrs_string = build_corr_map(corrs_string)

        for row in iterator:
            u, v = row.u, row.v
            visibilities = row.visibilities
            weights = row.weight
            flags = row.flag
            
            n_channels = row.n_channels
            n_correlations = row.n_correlations

            for chan in range(n_channels):
                freq = chan_freq[chan]

                u_pix, v_pix = calculate_uv_pix(u, v, freq, uvcellsize, grid_size)
                u_pix_h, v_pix_h = calculate_uv_pix(-u, -v, freq, uvcellsize, grid_size)

                if not is_valid_uv_pix(u_pix, v_pix, grid_size):
                    continue
                if not is_valid_uv_pix(u_pix_h, v_pix_h, grid_size):
                    continue

                for corr in range(n_correlations):
                    if flags[chan][corr]:
                        continue

                    corr_name = corrs_string[corr]
                    if corr_name not in {'XX', 'YY', 'RR', 'LL'}:
                        continue

                    vs = complex(visibilities[chan][corr][0], visibilities[chan][corr][1])
                    ws = weights[chan][corr] * 0.5

                    accumulate_visibilities(u_pix, v_pix, u_pix_h, v_pix_h, vs, ws)
    
    except Exception as e:
        print(f"[Gridding] Error processing visibilities: {e}")
        traceback.print_exc()
        raise


def accumulate_grid(grid, key, visibility, weight):
    try:
        if key not in grid:
            grid[key] = {
                'visibilities': [visibility],
                'weights': [weight],
            }
        else:
            grid[key]['visibilities'].append(visibility)
            grid[key]['weights'].append(weight)
        return grid[key]
    
    except Exception as e:
        print(f"[Gridding] Error accumulating grid data: {e}")
        traceback.print_exc()
        raise


def calculate_uv_pix(u, v, freq, uvcellsize, grid_size):
    try:
        u_lambda, v_lambda = u * freq / c.value, v * freq / c.value

        u_pix = math.floor((u_lambda / uvcellsize[0]) + (grid_size[1] // 2) + 0.5)
        v_pix = math.floor((v_lambda / uvcellsize[1]) + (grid_size[0] // 2) + 0.5)

        return u_pix, v_pix
    
    except Exception as e:
        print(f"[Gridding] Error calculating UV pixel coordinates: {e}")
        traceback.print_exc()
        raise


def is_valid_uv_pix(u_pix, v_pix, grid_size):
    try:
        return 0 <= u_pix < grid_size[1] and 0 <= v_pix < grid_size[0]
    
    except Exception as e:
        print(f"[Gridding] Error validating UV pixel coordinates: {e}")
        traceback.print_exc()
        raise


def build_corr_map(corrs_string):
    try:
        if isinstance(corrs_string, list):
            corrs = corrs_string[0] if isinstance(corrs_string[0], list) else corrs_string
        else:
            corrs = [c.strip() for c in corrs_string.split(',')]
        return {idx: corr for idx, corr in enumerate(corrs)}
    
    except Exception as e:
        print(f"[Gridding] Error building correlation map: {e}")
        traceback.print_exc()
        raise

def load_grid_config(config_path):
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Grid config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ["img_size", "padding_factor", "cellsize"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in grid config")

        return config

    except Exception as e:
        print(f"[Gridding] Error loading grid config: {e}")
        traceback.print_exc()
        raise


def define_grid_schema():
    return StructType([
        StructField("u_pix", IntegerType(), True),
        StructField("v_pix", IntegerType(), True),
        StructField("real", DoubleType(), True),
        StructField("imag", DoubleType(), True),
        StructField("weight", DoubleType(), True)
    ])


def consolidate_gridding(gridded_rdd):
    try:
        grid_acc = gridded_rdd.groupBy("u_pix", "v_pix").agg(
            F.sum("real").alias("real"),
            F.sum("imag").alias("imag"),
            F.sum("weight").alias("weight")
        )

        grid_avg = (grid_acc
           .withColumn("real", F.when(F.col("weight") > 0, F.col("real")/F.col("weight")).otherwise(F.lit(0.0)))
           .withColumn("imag", F.when(F.col("weight") > 0, F.col("imag")/F.col("weight")).otherwise(F.lit(0.0)))
           .select("u_pix", "v_pix", "real", "imag", "weight"))

        return grid_avg

    except Exception as e:
        print(f"[Gridding] Error during consolidation: {e}")
        traceback.print_exc()
        raise
