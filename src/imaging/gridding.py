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

        gridded_acc = gridded_rdd.groupBy("n_channels", "n_correlations", "u_idx", "v_idx").agg(
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

            for chan in range(n_channels):
                for corr in range(n_correlations):
                    if flags[chan][corr]:
                        continue

                    vs_real = visibilities[chan][corr][0]
                    vs_imag = visibilities[chan][corr][1]
                    vs_complex = complex(vs_real, vs_imag)

                    ws_val = weights[corr]

                    grid_key = (chan, corr, u_idx, v_idx)

                    if grid_key not in accumulated_grid:
                        accumulated_grid[grid_key] = {
                            'vs_real': vs_complex.real * ws_val,
                            'vs_imag': vs_complex.imag * ws_val,
                            'weights': ws_val
                        }
                    else:
                        accumulated_grid[grid_key]['vs_real'] += vs_complex.real * ws_val
                        accumulated_grid[grid_key]['vs_imag'] += vs_complex.imag * ws_val
                        accumulated_grid[grid_key]['weights'] += ws_val
            
        if accumulated_grid:
            for (chan, corr, u_idx, v_idx), values in accumulated_grid.items():
                yield (
                    int(chan),
                    int(corr),
                    int(u_idx),
                    int(v_idx),
                    values['vs_real'],
                    values['vs_imag'],
                    values['weights']
                )

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
    
    u_norm = (u - u_min) / (u_max - u_min)
    v_norm = (v - v_min) / (v_max - v_min)
    
    u_idx = int(np.round(u_norm * (u_size - 1)))
    v_idx = int(np.round(v_norm * (v_size - 1)))
    
    return u_idx, v_idx


def load_grid_config(config_path):
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Grid config file not found: {config_path}")

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ['u_min', 'u_max', 'v_min', 'v_max', 'u_size', 'v_size']
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
        StructField("n_channels", IntegerType(), True),
        StructField("n_correlations", IntegerType(), True),
        StructField("u_idx", IntegerType(), True),
        StructField("v_idx", IntegerType(), True),
        StructField("vs_real", DoubleType(), True),
        StructField("vs_imag", DoubleType(), True),
        StructField("weights", DoubleType(), True)
    ])


def consolide_gridding(gridded_rdd):
    try:
        grid_acc = gridded_rdd.groupBy("n_channels", "n_correlations", "u_idx", "v_idx").agg(
            F.sum("vs_real").alias("vs_real"),
            F.sum("vs_imag").alias("vs_imag"),
            F.sum("weights").alias("weights")
        )

        grid_avg = (grid_acc
           .withColumn("vs_real", F.when(F.col("weights") > 0, F.col("vs_real")/F.col("weights")).otherwise(F.lit(0.0)))
           .withColumn("vs_imag", F.when(F.col("weights") > 0, F.col("vs_imag")/F.col("weights")).otherwise(F.lit(0.0)))
           .select("n_channels","n_correlations","u_idx","v_idx","vs_real","vs_imag","weights"))

        return grid_avg

    except Exception as e:
        print(f"Error consolidating gridding: {e}")
        traceback.print_exc()
        raise
