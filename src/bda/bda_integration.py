import traceback

from pyspark.sql.functions import col

from .bda_processor import process_rows


def apply_bda(df, bda_config):
    """
    Apply Baseline Dependent Averaging (BDA) to the input DataFrame.
    
    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame containing visibility data.
    bda_config : dict
        Configuration parameters for BDA processing.

    Returns
    -------
    pyspark.sql.DataFrame
        DataFrame after applying BDA.
    """
    try:        
        # Partition by baseline for streaming
        partitioned_df = df.repartition(col("baseline_key"), col("scan_number"))
        
        # Sort within partitions by time
        sorted_df = partitioned_df.sortWithinPartitions("baseline_key", "scan_number", "time")
        
        bda_df = process_rows(sorted_df, bda_config)

        return bda_df

    except Exception as e:
        print(f"BDA processing failed: {e}")
        traceback.print_exc()
        raise
