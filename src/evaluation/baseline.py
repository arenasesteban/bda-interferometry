import traceback

import pyspark.sql.functions as F


def calculate_baseline_dependency(df_scientific, df_averaging):
    try:
        df_before = (
            df_scientific
            .withColumn('baseline_length', F.sqrt(F.col('u') ** 2 + F.col('v') ** 2))
            .groupBy('baseline_key')
            .agg(
                F.first('baseline_length').alias('baseline_length'),
                F.count('*').alias('n_before')
            )
        )

        df_after = (
            df_averaging
            .groupBy('baseline_key')
            .agg(F.count('*').alias('n_after'))
        )

        df_baseline_dependecy = (
            df_before.join(df_after, 'baseline_key', 'left')
            .fillna(1, subset=['n_after'])
            .withColumn('compression_ratio', F.col('n_before') / F.col('n_after'))
            .orderBy('baseline_length')
        )

        return df_baseline_dependecy

    except Exception as e:
        print(f"Error calculating baseline dependency: {e}")
        traceback.print_exc()
        raise


def validate_baseline_dependency(df_scientific, df_averaging, bda_config, output_file):
    try:

        df_baseline_dependency = calculate_baseline_dependency(df_scientific, df_averaging)
        short_threshold = bda_config.get('threshold_baseline', 2000.0)

        short_baselines = df_baseline_dependency.filter(F.col('baseline_length') < short_threshold)
        long_baselines  = df_baseline_dependency.filter(F.col('baseline_length') >= short_threshold)

        def group_stats(df):
            row = df.agg(
                F.count('*').alias('n_baselines'),
                F.sum('n_before').alias('total_before'),
                F.sum('n_after').alias('total_after'),
            ).collect()[0]
            return row['n_baselines'], row['total_before'], row['total_after']

        short_n, short_before, short_after = group_stats(short_baselines)
        long_n,  long_before,  long_after  = group_stats(long_baselines)

        total_baselines = short_n + long_n
        total_before    = short_before + long_before
        total_after     = short_after  + long_after

        short_ratio = short_before / short_after if short_after else float('inf')
        long_ratio  = long_before  / long_after  if long_after  else float('inf')
        total_ratio = total_before / total_after  if total_after else float('inf')

        write_baseline_dependency_results(
            short_threshold,
            total_baselines, total_before, total_after, total_ratio,
            short_n, short_before, short_after, short_ratio,
            long_n,  long_before,  long_after,  long_ratio,
            output_file
        )

        print("[Metrics] ✓ Baseline Dependency validation completed successfully.")

    except Exception as e:
        print(f"Error validating baseline dependency: {e}")
        traceback.print_exc()
        raise


def write_baseline_dependency_results(
    short_threshold,
    total_baselines, total_before, total_after, total_ratio,
    short_n, short_before, short_after, short_ratio,
    long_n,  long_before,  long_after,  long_ratio,
    output_file
):
    try:
        with open(output_file, "a") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Baseline Dependency Validation\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Metric: Compression Ratio = rows_before_BDA / rows_after_BDA\n\n")

            f.write(f"GLOBAL SUMMARY:\n")
            f.write(f"  Total baselines:     {total_baselines}\n")
            f.write(f"  Total rows before:   {total_before}\n")
            f.write(f"  Total rows after:    {total_after}\n")
            f.write(f"  Compression ratio:   {total_ratio:.2f}x\n\n")

            f.write(f"SHORT BASELINES (< {short_threshold} m):\n")
            f.write(f"  Count:               {short_n}\n")
            f.write(f"  Rows before BDA:     {short_before}\n")
            f.write(f"  Rows after  BDA:     {short_after}\n")
            f.write(f"  Compression ratio:   {short_ratio:.2f}x\n\n")

            f.write(f"LONG BASELINES (>= {short_threshold} m):\n")
            f.write(f"  Count:               {long_n}\n")
            f.write(f"  Rows before BDA:     {long_before}\n")
            f.write(f"  Rows after  BDA:     {long_after}\n")
            f.write(f"  Compression ratio:   {long_ratio:.2f}x\n")
            f.write(f"{'=' * 80}\n")

    except Exception as e:
        print(f"Error writing baseline dependency results: {e}")
        traceback.print_exc()
        raise