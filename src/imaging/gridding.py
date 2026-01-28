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


def apply_gridding(df, grid_config):
    try:
        scientific_df = df.repartition(4, F.col("baseline_key"), F.col("scan_number"))

        def grid_data(iterator):
            return process_gridding(iterator, grid_config)

        gridded = scientific_df.rdd.mapPartitions(grid_data)
        gridded_rdd = df.sparkSession.createDataFrame(gridded, define_grid_schema())

        gridded_acc = gridded_rdd.groupBy("u_pix", "v_pix").agg(
            F.sum("vs_real").alias("vs_real"),
            F.sum("vs_imag").alias("vs_imag"),
            F.sum("weights").alias("weights")
        )
        
        return gridded_acc

    except Exception as e:
        print(f"Error during gridding: {e}")
        traceback.print_exc()
        raise


def process_gridding(iterator, grid_config):
    accumulated_grid = {}

    stats = {
        'total_rows': 0,
        'out_of_bounds': 0,
        'gridded_points': 0,
        'u_pix_range': [float('inf'), float('-inf')],
        'v_pix_range': [float('inf'), float('-inf')]
    }

    try:
        img_size = grid_config["img_size"]
        padding_factor = grid_config["padding_factor"]
        cellsize = grid_config["cellsize"]

        imsize = [img_size, img_size]
        grid_size = [int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)]  # [v_size, u_size]

        du = - 1.0 / (cellsize * grid_size[1])  # u_direction (width)
        dv = 1.0 / (cellsize * grid_size[0])    # v_direction (height)
        uvcellsize = [du, dv]

        chan_freq = grid_config["chan_freq"]
        corrs_names = build_corr_map(grid_config["corrs_string"])

        for row in iterator:
            stats['total_rows'] += 1
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

                stats['u_pix_range'][0] = min(stats['u_pix_range'][0], u_pix)
                stats['u_pix_range'][1] = max(stats['u_pix_range'][1], u_pix)
                stats['v_pix_range'][0] = min(stats['v_pix_range'][0], v_pix)
                stats['v_pix_range'][1] = max(stats['v_pix_range'][1], v_pix)

                if u_pix < 0 or v_pix < 0 or u_pix >= grid_size[1] or v_pix >= grid_size[0]:
                    stats['out_of_bounds'] += 1
                    continue
                if u_pix_h < 0 or v_pix_h < 0 or u_pix_h >= grid_size[1] or v_pix_h >= grid_size[0]:
                    continue

                for corr in range(n_correlations):
                    if flags[chan][corr]:
                        continue

                    corr_name = corrs_names[corr]
                    if corr_name not in {'XX', 'YY', 'RR', 'LL'}:
                        continue

                    vs_real = visibilities[chan][corr][0]
                    vs_imag = visibilities[chan][corr][1]
                    vs_complex = complex(vs_real, vs_imag)

                    ws = weights[chan][corr] * 0.5

                    grid_key = (u_pix, v_pix)
                    grid_key_h = (u_pix_h, v_pix_h)

                    accumulate_grid(accumulated_grid, grid_key, vs_complex, ws)
                    accumulate_grid(accumulated_grid, grid_key_h, np.conj(vs_complex), ws)
    
        print(f"[Gridding] Rows: {stats['total_rows']} | Gridded: {stats['gridded_points']} | Out: {stats['out_of_bounds']}")
        print(f"[Gridding] u_pix: [{stats['u_pix_range'][0]:.0f}, {stats['u_pix_range'][1]:.0f}] (grid: 0-{grid_size[1]-1})")
        print(f"[Gridding] v_pix: [{stats['v_pix_range'][0]:.0f}, {stats['v_pix_range'][1]:.0f}] (grid: 0-{grid_size[0]-1})")
            
        if accumulated_grid:
            for (u_pix, v_pix), values in accumulated_grid.items():
                yield (
                    int(u_pix),
                    int(v_pix),
                    float(values['vs_real']),
                    float(values['vs_imag']),
                    float(values['weights'])
                )

    except Exception as e:
        print(f"Error processing rows for gridding: {e}")
        traceback.print_exc()
        raise


def accumulate_grid(grid, key, visibility, weight):
    if key not in grid:
        grid[key] = {
            'vs_real': visibility.real * weight,
            'vs_imag': visibility.imag * weight,
            'weights': weight
        }
    else:
        grid[key]['vs_real'] += visibility.real * weight
        grid[key]['vs_imag'] += visibility.imag * weight
        grid[key]['weights'] += weight
    return grid[key]


def build_corr_map(corrs_string):
    if isinstance(corrs_string, list):
        if len(corrs_string) > 0 and isinstance(corrs_string[0], list):
            corrs = corrs_string[0]
        else:
            corrs = corrs_string
    else:
        corrs = [c.strip() for c in corrs_string.split(',')]
    return {idx: corr for idx, corr in enumerate(corrs)}


def calculate_uv_pix(u, v, freq, uvcellsize, grid_size):
    u_lambda, v_lambda = u * freq / c.value, v * freq / c.value

    u_pix = math.floor((u_lambda / uvcellsize[0]) + (grid_size[1] // 2) + 0.5)
    v_pix = math.floor((v_lambda / uvcellsize[1]) + (grid_size[0] // 2) + 0.5)

    return u_pix, v_pix


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
        print(f"Error loading grid config: {e}")
        traceback.print_exc()
        raise


def define_grid_schema():
    return StructType([
        StructField("u_pix", IntegerType(), True),
        StructField("v_pix", IntegerType(), True),
        StructField("vs_real", DoubleType(), True),
        StructField("vs_imag", DoubleType(), True),
        StructField("weights", DoubleType(), True)
    ])


def consolidate_gridding(gridded_rdd):
    try:
        grid_acc = gridded_rdd.groupBy("u_pix", "v_pix").agg(
            F.sum("vs_real").alias("vs_real"),
            F.sum("vs_imag").alias("vs_imag"),
            F.sum("weights").alias("weights")
        )

        grid_avg = (grid_acc
           .withColumn("vs_real", F.when(F.col("weights") > 0, F.col("vs_real")/F.col("weights")).otherwise(F.lit(0.0)))
           .withColumn("vs_imag", F.when(F.col("weights") > 0, F.col("vs_imag")/F.col("weights")).otherwise(F.lit(0.0)))
           .select("u_pix", "v_pix", "vs_real", "vs_imag", "weights"))

        return grid_avg

    except Exception as e:
        print(f"Error consolidating gridding: {e}")
        traceback.print_exc()
        raise
