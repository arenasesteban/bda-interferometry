import traceback

from .bda_processor import process_rows


def apply_bda(df_scientific, num_partitions, bda_config):
    """
    Apply Baseline Dependent Averaging (BDA) to the input DataFrame.
    
    Parameters
    ----------
    df_scientific : pyspark.sql.DataFrame
        Input DataFrame containing visibility data.
    num_partitions : int
        Number of partitions for parallel processing.
    bda_config : dict
        Configuration parameters for BDA processing.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame after applying BDA.
    """
    try:        
        # Partition by baseline for streaming
        df_partitioned = df_scientific.repartition(num_partitions * 2, "baseline_key")

        # Sort within partitions by time
        df_sorted = df_partitioned.sortWithinPartitions("baseline_key", "scan_number", "time")
        
        df_averaged, df_windowed = process_rows(df_sorted, bda_config)

        df_averaged = df_averaged.coalesce(num_partitions)

        return df_averaged, df_windowed

    except Exception as e:
        print(f"BDA processing failed: {e}")
        traceback.print_exc()
        raise
