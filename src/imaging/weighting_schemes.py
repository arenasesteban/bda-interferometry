import traceback
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType
)
import pandas as pd

def natural_weighting():
    def apply_natural_weighting(weight_original, weight_grid):
        return weight_original
    
    return apply_natural_weighting


def uniform_weighting():
    def apply_uniform_weighting(weight_original, weight_grid):
        epsilon = 1e-12
        return weight_original / max(weight_grid, epsilon)

    return apply_uniform_weighting


def weight_visibilities(df_gridded, weight_fn):
    schema = StructType([
        StructField("u_pix", IntegerType(), True),
        StructField("v_pix", IntegerType(), True),
        StructField("vs_real", DoubleType(), True),
        StructField("vs_imag", DoubleType(), True),
        StructField("weight", DoubleType(), True)
    ])

    def weighting_grid(pdf):
        weight_grid = pdf['weight'].sum()

        real_weighted = 0.0
        imag_weighted = 0.0
        weight_weighted = 0.0

        for _, row in pdf.iterrows():
            weight_original = row['weight']
            weighted = weight_fn(weight_original, weight_grid)

            real_weighted += row['vs_real'] * weighted
            imag_weighted += row['vs_imag'] * weighted
            weight_weighted += weighted

        u_pix = pdf['u_pix'].iloc[0]
        v_pix = pdf['v_pix'].iloc[0]

        result = {
            "u_pix": u_pix,
            "v_pix": v_pix,
            "vs_real": real_weighted,
            "vs_imag": imag_weighted,
            "weight": weight_weighted
        }

        return pd.DataFrame([result])

    df_weighted = df_gridded.groupBy("u_pix", "v_pix").applyInPandas(weighting_grid, schema)

    df_averaged = (df_weighted
           .withColumn("vs_real", F.when(F.col("weight") > 0, F.col("vs_real")/F.col("weight")).otherwise(F.lit(0.0)))
           .withColumn("vs_imag", F.when(F.col("weight") > 0, F.col("vs_imag")/F.col("weight")).otherwise(F.lit(0.0)))
           .select("u_pix", "v_pix", "vs_real", "vs_imag", "weight"))

    return df_averaged

def apply_weighting(df_gridded, grid_config):
    try:
        weight_scheme = grid_config.get("weight_scheme", "NATURAL")

        if weight_scheme == "NATURAL":
            weight_fn = natural_weighting()
        elif weight_scheme == "UNIFORM":
            weight_fn = uniform_weighting()
        else:
            raise ValueError(f"Unknown weighting scheme: {weight_scheme}")

        df_averaged = weight_visibilities(df_gridded, weight_fn)

        return df_averaged

    except Exception as e:
        print(f"Error applying weighting: {e}")
        traceback.print_exc()
        raise
