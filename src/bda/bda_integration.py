import traceback

from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, ArrayType
)

from .bda_processor import process_rows


def apply_bda(df, bda_config):
    try:        
        # Partition by baseline for streaming
        scientific_df = df.repartition(4, col("baseline_key"))
        
        # Sort within partitions by time
        sorted_df = scientific_df.sortWithinPartitions("baseline_key", "scan_number", "time")
        
        # Define incremental BDA processing function
        def apply_bda_to_partition(iterator):
            results = []
            
            for window in process_rows(iterator, bda_config):
                result = convert_window_to_tuple(window)
                results.append(result)

            return iter(results)
        
        bda_data = sorted_df.rdd.mapPartitions(apply_bda_to_partition)
        bda_data_df = df.sparkSession.createDataFrame(bda_data, define_bda_schema())

        return bda_data_df

    except Exception as e:
        print(f"BDA processing failed: {e}")
        traceback.print_exc()
        raise


def convert_window_to_tuple(window):
    try:
        return (
            window["subms_id"],
            window["field_id"],
            window["spw_id"],
            window["polarization_id"],
            window["nrows"],
            window["n_channels"],
            window["n_correlations"],
            window["antenna1"],
            window["antenna2"],
            window["scan_number"],
            window["baseline_key"],
            window["exposure"],
            window["interval"],
            window['time'],
            window['u'],
            window['v'],
            window['visibilities'],
            window['weight'],
            window['flag']
        )

    except Exception as e:
        print(f"Error converting window to tuple: {e}")
        traceback.print_exc()
        raise


def define_bda_schema():
    return StructType([
        StructField("subms_id", IntegerType(), True),
        StructField("field_id", IntegerType(), True),
        StructField("spw_id", IntegerType(), True),
        StructField("polarization_id", IntegerType(), True),

        StructField("nrows", IntegerType(), True),
        StructField("n_channels", IntegerType(), True),
        StructField("n_correlations", IntegerType(), True),

        StructField("antenna1", IntegerType(), True),
        StructField("antenna2", IntegerType(), True),
        StructField("scan_number", IntegerType(), True),
        StructField("baseline_key", StringType(), True),

        StructField("exposure", DoubleType(), True),
        StructField("interval", DoubleType(), True),

        StructField('time', DoubleType(), True),
        StructField('u', DoubleType(), True),
        StructField('v', DoubleType(), True),

        StructField('visibilities', ArrayType(ArrayType(ArrayType(DoubleType()))), True),
        StructField('weight', ArrayType(DoubleType()), True),
        StructField('flag', ArrayType(ArrayType(IntegerType())), True)
    ])
