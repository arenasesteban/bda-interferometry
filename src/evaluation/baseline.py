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


def validate_baseline_dependency(df_baseline_dependency, bda_config):
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
        
        # Additional statistics
        short_stats = short_baselines.agg(
            F.mean('compression_ratio').alias('mean_comp'),
            F.stddev('compression_ratio').alias('std_comp'),
            F.min('compression_ratio').alias('min_comp'),
            F.max('compression_ratio').alias('max_comp'),
            F.mean('baseline_lenght').alias('mean_length')
        ).collect()[0]
        
        long_stats = long_baselines.agg(
            F.mean('compression_ratio').alias('mean_comp'),
            F.stddev('compression_ratio').alias('std_comp'),
            F.min('compression_ratio').alias('min_comp'),
            F.max('compression_ratio').alias('max_comp'),
            F.mean('baseline_lenght').alias('mean_length')
        ).collect()[0]

        # Validation: short baselines should have higher compression
        ratio_compression = short_compression / long_compression if long_compression > 0 else 0
        validation_passed = ratio_compression > 1.0

        print(f"\n{'='*80}")
        print(f"Baseline Dependency Validation")
        print(f"{'='*80}")
        print(f"Metric: Compression Ratio = N_samples_original / N_windows_BDA")
        print(f"")
        print(f"Configuration:")
        print(f"  Total Baselines:     {df_baseline_dependency.count()}")
        print(f"  Threshold:  {short_threshold:.1f} λ")
        print(f"")
        print(f"SHORT BASELINES (< {short_threshold:.1f} λ):")
        print(f"  Count:              {short_baselines.count()}")
        print(f"  Average Length:     {short_stats['mean_length']:.2f} λ")
        print(f"  Compression ratio:")
        print(f"    Mean:               {short_stats['mean_comp']:.2f}x")
        print(f"    Standard Deviation:       {short_stats['std_comp']:.2f}x" if short_stats['std_comp'] else "")
        print(f"    Range:               [{short_stats['min_comp']:.2f}, {short_stats['max_comp']:.2f}]x")
        print(f"")
        print(f"LONG BASELINES (≥ {short_threshold:.1f} λ):")
        print(f"  Count:              {long_baselines.count()}")
        print(f"  Average Length:     {long_stats['mean_length']:.2f} λ")
        print(f"  Compression ratio:")
        print(f"    Mean:               {long_stats['mean_comp']:.2f}x")
        print(f"    Standard Deviation:       {long_stats['std_comp']:.2f}x" if long_stats['std_comp'] else "")
        print(f"    Range:               [{long_stats['min_comp']:.2f}, {long_stats['max_comp']:.2f}]x")
        print(f"")
        print(f"VALIDATION:")
        print(f"  Ratio (Short/Long):   {ratio_compression:.2f}")
        print(f"  Status:                {'✓ PASSED' if validation_passed else '✗ FAILED'}")
    
    except Exception as e:
        print(f"Error validating baseline dependency: {e}")
        traceback.print_exc()
        raise
