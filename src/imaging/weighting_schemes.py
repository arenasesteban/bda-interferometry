import traceback

from pyspark.sql import functions as F


def natural_weighting(df_gridded):
    df_weighted = df_gridded.groupBy("u_pix", "v_pix").agg(
        F.sum(F.col("vs_real") * F.col("weight")).alias("vs_real_weighted"),
        F.sum(F.col("vs_imag") * F.col("weight")).alias("vs_imag_weighted"),
        F.sum("weight").alias("weight_natural")
    )

    df_weighted = (df_weighted
        .withColumn("vs_real", F.when(F.col("weight_natural") > 0, F.col("vs_real_weighted")/F.col("weight_natural")).otherwise(F.lit(0.0)))
        .withColumn("vs_imag", F.when(F.col("weight_natural") > 0, F.col("vs_imag_weighted")/F.col("weight_natural")).otherwise(F.lit(0.0)))
        .select("u_pix", "v_pix", "vs_real", "vs_imag", F.col("weight_natural").alias("weight"))
    )

    return df_weighted


def uniform_weighting(df_gridded):
    epsilon = 1e-12

    df_weights = df_gridded.groupBy("u_pix", "v_pix").agg(
        F.sum("weight").alias("weight_grid")
    ) 

    df_weighted = (df_gridded
        .join(df_weights, ["u_pix", "v_pix"], "inner")
        .withColumn("weight_uniform", 
            F.col("weight") / F.greatest(F.col("weight_grid"), F.lit(epsilon))
        )
        .withColumn("vs_real_weighted", F.col("vs_real") * F.col("weight_uniform"))
        .withColumn("vs_imag_weighted", F.col("vs_imag") * F.col("weight_uniform"))
    )

    df_weighted = df_weighted.groupBy("u_pix", "v_pix").agg(
        F.sum("vs_real_weighted").alias("vs_real_weighted"),
        F.sum("vs_imag_weighted").alias("vs_imag_weighted"),
        F.sum("weight_uniform").alias("weight_uniform")
    )

    df_weighted = (df_weighted
        .withColumn("vs_real", F.when(F.col("weight_uniform") > 0, F.col("vs_real_weighted") / F.col("weight_uniform")).otherwise(F.lit(0.0)))
        .withColumn("vs_imag", F.when(F.col("weight_uniform") > 0, F.col("vs_imag_weighted") / F.col("weight_uniform")).otherwise(F.lit(0.0)))
        .select("u_pix", "v_pix", "vs_real", "vs_imag", F.col("weight_uniform").alias("weight"))
    )

    return df_weighted


def weight_visibilities(df_gridded, weight_scheme):
    if weight_scheme == "NATURAL":
        df_weighted = natural_weighting(df_gridded)
        
    elif weight_scheme == "UNIFORM":
        df_weighted = uniform_weighting(df_gridded)
    else:
        raise ValueError(f"Unknown weighting scheme: {weight_scheme}")
    
    return df_weighted

def apply_weighting(df_gridded, grid_config):
    try:
        weight_scheme = grid_config.get("weight_scheme", "NATURAL")
        
        print(f"[Weighting] Applying {weight_scheme} weighting scheme...")
        df_weighted = weight_visibilities(df_gridded, weight_scheme)
        
        return df_weighted

    except Exception as e:
        print(f"Error applying weighting: {e}")
        traceback.print_exc()
        raise
