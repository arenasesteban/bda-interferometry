import numpy as np
import traceback
import json
from pathlib import Path

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

        gridded_acc = gridded_rdd.groupBy("chan", "corr", "u_pix", "v_pix").agg(
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

    try:
        img_size = grid_config["img_size"]
        padding_factor = grid_config["padding_factor"]
        cellsize = grid_config["cellsize"]

        imsize = [img_size, img_size]
        grid_size = [int(imsize[0] * padding_factor), int(imsize[1] * padding_factor)]
        uvcellsize = [1 / (cellsize * grid_size[0]), 1 / (cellsize * grid_size[1])]

        for row in iterator:
            u, v = row.u, row.v
            u_pix, v_pix = uv_to_grid_index(u, v, uvcellsize, imsize)
            u_pix_h, v_pix_h = uv_to_grid_index(-u, -v, uvcellsize, imsize)

            if not (0 <= u_pix < grid_size[0]) or not (0 <= v_pix < grid_size[1]):
                continue
            if not (0 <= u_pix_h < grid_size[0]) or not (0 <= v_pix_h < grid_size[1]):
                continue

            visibilities = row.visibilities
            weights = row.weight
            flags = row.flag
            n_channels = row.n_channels
            n_correlations = row.n_correlations

            for chan in range(n_channels):
                for corr in range(n_correlations):
                    if flags[chan][corr]:
                        continue

                    vs_real = visibilities[chan][corr][0]
                    vs_imag = visibilities[chan][corr][1]
                    vs_complex = complex(vs_real, vs_imag)

                    ws = weights[corr] * 0.5

                    grid_key = (chan, corr, u_pix, v_pix)

                    if grid_key not in accumulated_grid:
                        accumulated_grid[grid_key] = {
                            'vs_real': vs_complex.real * ws,
                            'vs_imag': vs_complex.imag * ws,
                            'weights': ws
                        }
                    else:
                        accumulated_grid[grid_key]['vs_real'] += vs_complex.real * ws
                        accumulated_grid[grid_key]['vs_imag'] += vs_complex.imag * ws
                        accumulated_grid[grid_key]['weights'] += ws

                    grid_key_h = (chan, corr, u_pix_h, v_pix_h)

                    if grid_key_h not in accumulated_grid:
                        accumulated_grid[grid_key_h] = {
                            'vs_real': vs_complex.real * ws,
                            'vs_imag': -vs_complex.imag * ws,
                            'weights': ws
                        }
                    else:
                        accumulated_grid[grid_key_h]['vs_real'] += vs_complex.real * ws
                        accumulated_grid[grid_key_h]['vs_imag'] += -vs_complex.imag * ws
                        accumulated_grid[grid_key_h]['weights'] += ws
            
        if accumulated_grid:
            for (chan, corr, u_pix, v_pix), values in accumulated_grid.items():
                yield (
                    int(chan),
                    int(corr),
                    int(u_pix),
                    int(v_pix),
                    values['vs_real'],
                    values['vs_imag'],
                    values['weights']
                )

    except Exception as e:
        print(f"Error processing rows for gridding: {e}")
        traceback.print_exc()
        raise


def uv_to_grid_index(u, v, uvcellsize, imsize):
    u_pix = int((u / uvcellsize[0]) + (imsize[0] // 2) + 0.5)
    v_pix = int((v / uvcellsize[1]) + (imsize[1] // 2) + 0.5)

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
        StructField("chan", IntegerType(), True),
        StructField("corr", IntegerType(), True),
        StructField("u_pix", IntegerType(), True),
        StructField("v_pix", IntegerType(), True),
        StructField("vs_real", DoubleType(), True),
        StructField("vs_imag", DoubleType(), True),
        StructField("weights", DoubleType(), True)
    ])


def consolidate_gridding(gridded_rdd):
    try:
        grid_acc = gridded_rdd.groupBy("chan", "corr", "u_pix", "v_pix").agg(
            F.sum("vs_real").alias("vs_real"),
            F.sum("vs_imag").alias("vs_imag"),
            F.sum("weights").alias("weights")
        )

        grid_avg = (grid_acc
           .withColumn("vs_real", F.when(F.col("weights") > 0, F.col("vs_real")/F.col("weights")).otherwise(F.lit(0.0)))
           .withColumn("vs_imag", F.when(F.col("weights") > 0, F.col("vs_imag")/F.col("weights")).otherwise(F.lit(0.0)))
           .select("chan","corr","u_pix","v_pix","vs_real","vs_imag","weights"))

        return grid_avg

    except Exception as e:
        print(f"Error consolidating gridding: {e}")
        traceback.print_exc()
        raise
