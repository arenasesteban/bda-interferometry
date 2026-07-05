from services.consumer_service import (
    check_end_signal,
    define_visibility_schema,
    process_message,
)


def test_consumer_ingests_kafka_like_batch_and_builds_scientific_dataframe(
    spark,
    kafka_like_batch_df,
    consumer_nrows,
    consumer_nchan,
    consumer_ncorr,
):
    df_filtered, stream_ended = check_end_signal(
        kafka_like_batch_df,
        epoch_id=0,
    )

    assert stream_ended is True
    assert df_filtered.count() == 1

    deserialized_rdd = df_filtered.rdd.mapPartitions(process_message)

    scientific_df = spark.createDataFrame(
        deserialized_rdd,
        define_visibility_schema(),
    )

    assert scientific_df.count() == consumer_nrows

    columns = set(scientific_df.columns)

    expected_columns = {
        "message_id",
        "subms_id",
        "field_id",
        "spw_id",
        "polarization_id",
        "baseline_key",
        "antenna1",
        "antenna2",
        "scan_number",
        "exposure",
        "interval",
        "time",
        "n_channels",
        "n_correlations",
        "u",
        "v",
        "w",
        "visibility",
        "weight",
        "flag",
        "baseline_length",
    }

    assert columns == expected_columns

    rows = scientific_df.orderBy("time").collect()

    assert rows[0]["message_id"] == "0-0"
    assert rows[0]["subms_id"] == 0
    assert rows[0]["field_id"] == 1
    assert rows[0]["spw_id"] == 2
    assert rows[0]["polarization_id"] == 0

    assert rows[0]["n_channels"] == consumer_nchan
    assert rows[0]["n_correlations"] == consumer_ncorr

    assert rows[0]["baseline_key"] == "0-1"
    assert rows[1]["baseline_key"] == "0-2"
    assert rows[2]["baseline_key"] == "1-3"

    assert len(rows[0]["visibility"]) == consumer_nchan
    assert len(rows[0]["visibility"][0]) == consumer_ncorr
    assert len(rows[0]["visibility"][0][0]) == 2

    assert len(rows[0]["weight"]) == consumer_nchan
    assert len(rows[0]["weight"][0]) == consumer_ncorr

    assert len(rows[0]["flag"]) == consumer_nchan
    assert len(rows[0]["flag"][0]) == consumer_ncorr

    assert rows[0]["baseline_length"] > 0.0