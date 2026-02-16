import traceback

import pyspark.sql.functions as F


def baseline_dependency(df_window):
    try:
        df_baseline_dependency = calculate_baseline_dependency(df_window)

        return df_baseline_dependency
    
    except Exception as e:
        print(f"Error calculating baseline dependency: {e}")
        traceback.print_exc()
        raise

    
def calculate_baseline_dependency(df_window):
    try:
        df_baseline_lenght = df_window.withColumn(
            'baseline_lenght',
            F.sqrt(F.col('u') ** 2 + F.col('v') ** 2)
        ).select('baseline_key', 'window_id', 'baseline_lenght')

        df_baseline_window = df_baseline_lenght.groupBy('baseline_key', 'window_id').agg(
            F.first('baseline_lenght').alias('baseline_lenght'),
            F.count('*').alias('n_samples_in_window'),
        )

        df_baseline_dependency = df_baseline_window.groupBy('baseline_key').agg(
            F.first('baseline_lenght').alias('baseline_lenght'),
            F.count('*').alias('n_windows'),
            F.sum('n_samples_in_window').alias('n_original')
        ).withColumn(
            'compression_ratio',
            F.col('n_original') / F.col('n_windows')
        ).orderBy('baseline_lenght')

        return df_baseline_dependency
    
    except Exception as e:
        print(f"Error calculating baseline dependency: {e}")
        traceback.print_exc()
        raise


def write_baseline_dependency_results(
    df_baseline_dependency, short_threshold,
    short_baselines, long_baselines, 
    short_compression, long_compression, ratio_compression, 
    validation_passed, 
    output_file
):
    try:
        with open(output_file, "w") as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Baseline Dependency Validation\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Metric: Compression Ratio = N_samples_original / N_windows_BDA\n")
            f.write(f"\n")
            f.write(f"Configuration:\n")
            f.write(f"  Total Baselines:     {df_baseline_dependency.count()}\n")
            f.write(f"  Threshold:           {short_threshold} m\n")
            f.write(f"\n")
            f.write(f"SHORT BASELINES (< {short_threshold} m):\n")
            f.write(f"  Count:               {short_baselines.count()}\n")
            f.write(f"  Compression Ratio:   {short_compression:.2f}x\n")
            f.write(f"\n")
            f.write(f"LONG BASELINES (≥ {short_threshold} m):\n")
            f.write(f"  Count:               {long_baselines.count()}\n")
            f.write(f"  Compression Ratio:   {long_compression:.2f}x\n")
            f.write(f"\n")
            f.write(f"VALIDATION:\n")
            f.write(f"  Ratio (Short/Long):  {ratio_compression:.2f}\n")
            f.write(f"  Status:              {'✓ PASSED' if validation_passed else '✗ FAILED'}\n")

    except Exception as e:
        print(f"Error writing baseline dependency results: {e}")
        traceback.print_exc()
        raise


def validate_baseline_dependency(df_baseline_dependency, bda_config, output_file):
    try:
        short_threshold = bda_config.get('short_baseline_threshold', 100.0)

        short_baselines = df_baseline_dependency.filter(
            F.col('baseline_lenght') < short_threshold
        )

        long_baselines = df_baseline_dependency.filter(
            F.col('baseline_lenght') >= short_threshold
        )

        # Compression ratios
        short_compression = short_baselines.agg(F.mean('compression_ratio')).collect()[0][0]
        long_compression = long_baselines.agg(F.mean('compression_ratio')).collect()[0][0]

        # Validation: short baselines should have higher compression
        ratio_compression = short_compression / long_compression if long_compression > 0 else 0
        validation_passed = ratio_compression > 1.0

        write_baseline_dependency_results(
            df_baseline_dependency, short_threshold,
            short_baselines, long_baselines,
            short_compression, long_compression, ratio_compression,
            validation_passed,
            output_file
        )
    
    except Exception as e:
        print(f"Error validating baseline dependency: {e}")
        traceback.print_exc()
        raise
