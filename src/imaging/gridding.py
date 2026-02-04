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
import pandas as pd

from .weighting_schemes import apply_weighting


def apply_gridding(df_scientific, num_partitions, grid_config, strategy="PARTIAL"):
    try:
        if strategy == "PARTIAL":
            # Micro-batching gridding
            df_duplicated = prepare_gridding(df_scientific)
            
            df_repartitioned = df_duplicated.repartition(num_partitions * 2, "u", "v")

            df_gridded = accumulate_grid(df_repartitioned, grid_config)

            return df_gridded
        elif strategy == "COMPLETE":
            # Streaming finished, apply weighting and consolidation
            df_repartitioned = df_scientific.repartition(num_partitions * 2, "u_pix", "v_pix")
            
            df_gridded = apply_weighting(df_repartitioned, grid_config)

            df_gridded = df_gridded.coalesce(num_partitions)
            
            return df_gridded
        else:
            raise ValueError(f"Unknown gridding strategy: {strategy}")

    except Exception as e:
        print(f"Error during gridding: {e}")
        traceback.print_exc()
        raise


def prepare_gridding(scientific_df):
    df_original = scientific_df.select(
        F.col("baseline_key"),
        F.col("u"),
        F.col("v"),
        F.col("n_channels"),
        F.col("n_correlations"),
        F.col("visibilities"),
        F.col("weight"),
        F.col("flag"),
        F.lit(False).alias("is_hermitian")
    )

    df_hermitian = scientific_df.select(
        F.col("baseline_key"),
        (-F.col("u")).alias("u"),
        (-F.col("v")).alias("v"),
        F.col("n_channels"),
        F.col("n_correlations"),
        F.col("visibilities"),
        F.col("weight"),
        F.col("flag"),
        F.lit(True).alias("is_hermitian")
    )

    df_duplicated = df_original.unionByName(df_hermitian)

    return df_duplicated

def accumulate_grid(scientific_df, grid_config):
    try:
        schema = StructType([
            StructField("u_pix", IntegerType(), True),
            StructField("v_pix", IntegerType(), True),
            StructField("vs_real", DoubleType(), True),
            StructField("vs_imag", DoubleType(), True),
            StructField("weight", DoubleType(), True)
        ])

        def asign_pixels(pdf):
            img_size = grid_config["img_size"]
            padding_factor = grid_config["padding_factor"]
            cellsize = grid_config["cellsize"]
            chan_freq = grid_config["chan_freq"]
            corrs_string = grid_config["corrs_string"]

            padded_size = int(img_size * padding_factor)  # [v_size, u_size]
            du = - 1.0 / (cellsize * padded_size)  # u_direction (width)
            dv = 1.0 / (cellsize * padded_size)    # v_direction (height)
            uvcellsize = [du, dv]
            corrs_map = build_corrs_map(corrs_string)

            accumulate_pixels = []

            for index, row in pdf.iterrows():
                result = process_visibility(row, chan_freq, corrs_map, uvcellsize, padded_size)
                
                if result is not None:
                    accumulate_pixels.extend(result)

            return pd.DataFrame(accumulate_pixels, columns=["u_pix", "v_pix", "vs_real", "vs_imag", "weight"])
        
        df_accumulated = scientific_df.groupBy("baseline_key").applyInPandas(asign_pixels, schema)

        return df_accumulated

    except Exception as e:
        print(f"Error during grid accumulation: {e}")
        traceback.print_exc()
        raise


def process_visibility(row, chan_freq, corrs_map, uvcellsize, padded_size):
    try:
        u, v = row.u, row.v
        visibilities = row.visibilities
        weights = row.weight
        flags = row.flag
        
        n_channels = row.n_channels
        n_correlations = row.n_correlations
        is_hermitian = row.is_hermitian

        pixels = []

        for chan in range(n_channels):
            freq = chan_freq[chan]

            u_pix, v_pix = calculate_uv_pix(u, v, freq, uvcellsize, padded_size)

            if not is_valid_uv_pix(u_pix, v_pix, padded_size):
                continue

            for corr in range(n_correlations):
                if flags[chan][corr]:
                    continue

                corr_name = corrs_map[corr]
                if corr_name not in {'XX', 'YY', 'RR', 'LL'}:
                    continue

                vs = complex(visibilities[chan][corr][0], visibilities[chan][corr][1])
                ws = weights[chan][corr] * 0.5

                if is_hermitian:
                    vs = np.conj(vs)

                pixels.append({
                    'u_pix': int(u_pix),
                    'v_pix': int(v_pix),
                    'vs_real': float(vs.real),
                    'vs_imag': float(vs.imag),
                    'weight': float(ws)
                })

        return pixels   

    except Exception as e:
        print(f"Error processing visibility: {e}")
        traceback.print_exc()
        raise


def calculate_uv_pix(u, v, freq, uvcellsize, padded_size):
    try:
        """ u_lambda, v_lambda = u / (c.value / freq), v / (c.value / freq) """
        u_lambda, v_lambda = u * freq / c.value, v * freq / c.value

        u_pix = math.floor((u_lambda / uvcellsize[0]) + (padded_size // 2) + 0.5)
        v_pix = math.floor((v_lambda / uvcellsize[1]) + (padded_size // 2) + 0.5)

        return u_pix, v_pix
    
    except Exception as e:
        print(f"[Gridding] Error calculating UV pixel coordinates: {e}")
        traceback.print_exc()
        raise


def is_valid_uv_pix(u_pix, v_pix, padded_size):
    try:
        return 0 <= u_pix < padded_size and 0 <= v_pix < padded_size
    
    except Exception as e:
        print(f"[Gridding] Error validating UV pixel coordinates: {e}")
        traceback.print_exc()
        raise


def build_corrs_map(corrs_string):
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
