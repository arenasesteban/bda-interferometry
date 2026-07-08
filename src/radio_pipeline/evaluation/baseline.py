import traceback
import pyspark.sql.functions as F
from pyspark.sql.window import Window


# ------------------------------------------------------------
# 1. Compute compression per baseline
# ------------------------------------------------------------
def calculate_baseline_dependency(df_scientific, df_averaging):
    """
    Compute compression ratio per baseline.

    Output columns
    --------------
    baseline_key
    baseline_length
    n_before
    n_after
    compression_ratio
    reduction_fraction
    reduction_percent
    """

    try:
        df_before = (
            df_scientific
            .groupBy("baseline_key")
            .agg(
                F.avg("baseline_length").alias("baseline_length"),
                F.count("*").alias("n_before")
            )
        )

        df_after = (
            df_averaging
            .groupBy("baseline_key")
            .agg(
                F.count("*").alias("n_after")
            )
        )

        df_baseline = (
            df_before
            .join(df_after, "baseline_key", "left")
            .withColumn("n_after", F.coalesce(F.col("n_after"), F.lit(1)))  # fallback analítico
            .withColumn(
                "compression_ratio",
                F.col("n_before").cast("double") / F.col("n_after").cast("double")
            )
            .withColumn(
                "reduction_fraction",
                (F.col("n_before").cast("double") - F.col("n_after").cast("double"))
                / F.col("n_before").cast("double")
            )
            .withColumn(
                "reduction_percent",
                F.col("reduction_fraction") * F.lit(100.0)
            )
            .orderBy("baseline_length", "baseline_key")
        )

        return df_baseline

    except Exception as e:
        print(f"Error calculating baseline dependency: {e}")
        traceback.print_exc()
        raise


# ------------------------------------------------------------
# 3. Assign baseline quartiles
# ------------------------------------------------------------
def add_baseline_quartiles(df):

    try:
        window = Window.orderBy(F.col("baseline_length"), F.col("baseline_key"))

        df = (
            df
            .withColumn("baseline_quartile", F.ntile(4).over(window))
            .orderBy("baseline_quartile", "baseline_length", "baseline_key")
        )

        return df

    except Exception as e:
        print(f"Error assigning baseline quartiles: {e}")
        traceback.print_exc()
        raise


# ------------------------------------------------------------
# 4. Compute statistics per quartile
# ------------------------------------------------------------
def compute_quartile_stats(df):

    try:
        df_stats = (
            df
            .groupBy("baseline_quartile")
            .agg(
                F.count("*").alias("n_baselines"),

                F.mean("baseline_length").alias("mean_baseline_length"),
                F.min("baseline_length").alias("min_baseline_length"),
                F.max("baseline_length").alias("max_baseline_length"),

                F.mean("compression_ratio").alias("mean_compression_ratio"),
                F.percentile_approx("compression_ratio", 0.5).alias("median_compression_ratio"),
                F.stddev("compression_ratio").alias("std_compression_ratio"),

                F.mean("reduction_fraction").alias("mean_reduction_fraction"),
                F.mean("reduction_percent").alias("mean_reduction_percent"),
                F.percentile_approx("reduction_percent", 0.5).alias("median_reduction_percent"),

                F.sum("n_before").alias("rows_before"),
                F.sum("n_after").alias("rows_after")
            )
            .withColumn(
                "aggregated_ratio",
                F.when(F.col("rows_after") > 0,
                       F.col("rows_before").cast("double") / F.col("rows_after").cast("double"))
                 .otherwise(F.lit(None).cast("double"))
            )
            .orderBy("baseline_quartile")
        )

        return df_stats

    except Exception as e:
        print(f"Error computing quartile stats: {e}")
        traceback.print_exc()
        raise


# ------------------------------------------------------------
# 5. Export CSV safely from Spark
# ------------------------------------------------------------
def export_baseline_csv(df_baseline, df_quartiles, output):

    try:
        (
            df_baseline
            .coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(f"{output}/baseline_dependency")
        )

        (
            df_quartiles
            .coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(f"{output}/baseline_quartiles")
        )

        print(f"[Export] CSV written to prefix: {output}")

    except Exception as e:
        print(f"Error exporting CSV: {e}")
        traceback.print_exc()
        raise


# ------------------------------------------------------------
# 6. Main evaluation pipeline
# ------------------------------------------------------------
def evaluate_baseline_dependency(
    df_scientific,
    df_averaging,
    output, 
    bda_config,
    output_file
):

    try:
        print("\n[Step 1] Computing baseline compression")
        df_baseline = calculate_baseline_dependency(df_scientific, df_averaging)

        print("\n[Step 2] Assigning baseline quartiles")
        df_baseline = add_baseline_quartiles(df_baseline).cache()

        print("\n[Step 3] Computing quartile statistics")
        df_quartiles = compute_quartile_stats(df_baseline)

        print("\n[Step 4] Exporting CSV")
        export_baseline_csv(df_baseline, df_quartiles, output)

        print("\nBaseline dependency per baseline:")
        df_baseline.show(20, False)

        print("\nBaseline dependency per quartile:")
        df_quartiles.show(20, False)

        validate_baseline_dependency(df_scientific, df_averaging, bda_config, output_file)

        return df_baseline, df_quartiles

    except Exception as e:
        print(f"Error evaluating baseline dependency: {e}")
        traceback.print_exc()
        raise



def group_stats(df):
    """
    Compute summary statistics for a baseline group.

    Metrics
    -------
    - n_baselines:
        Number of baselines in the group.
    - total_before / total_after:
        Total rows before and after BDA.
    - aggregated_ratio:
        sum(n_before) / sum(n_after).
        Useful to measure total compressed volume.
    - mean_compression_ratio:
        Mean of per-baseline compression ratio.
        Best metric to compare short vs long baselines fairly.
    - median_compression_ratio:
        Robust central tendency of compression ratio.
    - std_compression_ratio:
        Dispersion of compression ratio within the group.
    """
    row = df.agg(
        F.count('*').alias('n_baselines'),
        F.sum('n_before').alias('total_before'),
        F.sum('n_after').alias('total_after'),
        F.mean('compression_ratio').alias('mean_compression_ratio'),
        F.percentile_approx('compression_ratio', 0.5).alias('median_compression_ratio'),
        F.stddev('compression_ratio').alias('std_compression_ratio'),
    ).collect()[0]

    total_before = row['total_before'] or 0
    total_after = row['total_after'] or 0
    aggregated_ratio = total_before / total_after if total_after else float('inf')

    return (
        row['n_baselines'],
        total_before,
        total_after,
        aggregated_ratio,
        row['mean_compression_ratio'] or 0.0,
        row['median_compression_ratio'] or 0.0,
        row['std_compression_ratio'] or 0.0,
    )


def validate_baseline_dependency(df_scientific, df_averaging, bda_config, output_file):
    """
    Validate baseline dependency using a precomputed threshold in meters.

    Expected config
    ---------------
    bda_config['threshold']      : baseline threshold in meters
    bda_config['decorr_factor']  : decorrelation factor
    bda_config['theta_max']      : theta_max in radians
    """
    try:
        df_baseline_dependency = calculate_baseline_dependency(df_scientific, df_averaging)

        threshold = bda_config.get('threshold')
        if threshold is None:
            raise ValueError(
                "Missing 'threshold' in bda_config. "
                "It must be precomputed as lambda_ref / theta_max and provided in meters."
            )

        decorr_factor = bda_config.get('decorr_factor', 0.95)
        theta_max = bda_config.get('theta_max', 10.0)

        short_baselines = df_baseline_dependency.filter(F.col('baseline_length') < F.lit(threshold))
        long_baselines = df_baseline_dependency.filter(F.col('baseline_length') >= F.lit(threshold))

        (
            short_n, short_before, short_after,
            short_agg_ratio, short_mean_ratio,
            short_median_ratio, short_std_ratio
        ) = group_stats(short_baselines)

        (
            long_n, long_before, long_after,
            long_agg_ratio, long_mean_ratio,
            long_median_ratio, long_std_ratio
        ) = group_stats(long_baselines)

        total_baselines = short_n + long_n
        total_before = short_before + long_before
        total_after = short_after + long_after
        total_agg_ratio = total_before / total_after if total_after else float('inf')

        write_baseline_dependency_results(
            threshold, decorr_factor, theta_max,
            total_baselines, total_before, total_after, total_agg_ratio,
            short_n, short_before, short_after,
            short_agg_ratio, short_mean_ratio, short_median_ratio, short_std_ratio,
            long_n, long_before, long_after,
            long_agg_ratio, long_mean_ratio, long_median_ratio, long_std_ratio,
            output_file,
        )

        print("[Evaluation] ✓ Baseline Dependency validation completed successfully.")

        return df_baseline_dependency

    except Exception as e:
        print(f"Error validating baseline dependency: {e}")
        traceback.print_exc()
        raise


def write_baseline_dependency_results(
    threshold, decorr_factor, theta_max,
    total_baselines, total_before, total_after, total_agg_ratio,
    short_n, short_before, short_after,
    short_agg_ratio, short_mean_ratio, short_median_ratio, short_std_ratio,
    long_n, long_before, long_after,
    long_agg_ratio, long_mean_ratio, long_median_ratio, long_std_ratio,
    output_file,
):
    """
    Write baseline dependency validation results to file.

    Notes
    -----
    - threshold is assumed to be given in meters.
    - theta_max corresponds to the maximum angle in radians.
    """
    try:
        with open(output_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write("Baseline Dependency Validation\n")
            f.write(f"{'=' * 80}\n")
            f.write("Metrics:\n")
            f.write("   Compression: rows before / rows after\n")
            f.write("\n")
            f.write("Parameters:\n")
            f.write(f"  Baseline threshold:     {threshold:.6f} m\n")
            f.write(f"  Decorrelation factor:   {decorr_factor}\n")
            f.write(f"  Theta max:              {theta_max} rad\n")
            f.write("\n")
            f.write("Global summary:\n")
            f.write(f"  Total baselines:        {total_baselines}\n")
            f.write(f"  Total rows before:      {total_before}\n")
            f.write(f"  Total rows after:       {total_after}\n")
            f.write(f"  Aggregated ratio:       {total_agg_ratio:.2f}x\n")
            f.write("\n")
            f.write(f"Short baselines (< {threshold:.6f} m):\n")
            f.write(f"  Count:                  {short_n}\n")
            f.write(f"  Rows before BDA:        {short_before}\n")
            f.write(f"  Rows after  BDA:        {short_after}\n")
            f.write(f"  Aggregated ratio:       {short_agg_ratio:.2f}x\n")
            f.write(f"  Mean   compression:     {short_mean_ratio:.2f}x\n")
            f.write(f"  Median compression:     {short_median_ratio:.2f}x\n")
            f.write(f"  Std    compression:     {short_std_ratio:.2f}x\n")
            f.write("\n")
            f.write(f"Long baselines (>= {threshold:.6f} m):\n")
            f.write(f"  Count:                  {long_n}\n")
            f.write(f"  Rows before BDA:        {long_before}\n")
            f.write(f"  Rows after  BDA:        {long_after}\n")
            f.write(f"  Aggregated ratio:       {long_agg_ratio:.2f}x\n")
            f.write(f"  Mean   compression:     {long_mean_ratio:.2f}x\n")
            f.write(f"  Median compression:     {long_median_ratio:.2f}x\n")
            f.write(f"  Std    compression:     {long_std_ratio:.2f}x\n")
            f.write(f"{'=' * 80}\n")

    except Exception as e:
        print(f"Error writing baseline dependency results: {e}")
        traceback.print_exc()
        raise