import pytest

from core.bda.bda_integration import apply_bda


EXPECTED_AVERAGED_COLUMNS = {
    "subms_id",
    "field_id",
    "spw_id",
    "polarization_id",
    "n_channels",
    "n_correlations",
    "antenna1",
    "antenna2",
    "baseline_key",
    "scan_number",
    "window_id",
    "exposure",
    "interval",
    "time",
    "u",
    "v",
    "w",
    "visibility",
    "weight",
    "flag",
    "baseline_length",
}


def test_apply_bda_without_reduction_returns_original_dataframe(
    bda_scientific_df,
    integration_bda_passthrough_config,
    bda_num_partitions,
):
    """
    Valida la rama sin reducción de apply_bda.

    Con decorr_factor = 1.0, el pipeline no debe ejecutar process_rows,
    no debe crear ventanas y debe retornar el DataFrame científico original.
    """

    assert integration_bda_passthrough_config["decorr_factor"] == 1.0

    input_count = bda_scientific_df.count()

    df_averaged, df_windowed = apply_bda(
        bda_scientific_df,
        bda_num_partitions,
        integration_bda_passthrough_config,
    )

    assert df_windowed is None
    assert df_averaged is not None
    assert df_averaged.count() == input_count
    assert set(df_averaged.columns) == set(bda_scientific_df.columns)


def test_scientific_dataframe_is_reduced_by_bda_preserving_critical_fields(
    bda_scientific_df,
    integration_bda_active_config,
    bda_num_partitions,
    bda_nchan,
    bda_ncorr,
):
    """
    Valida la integración real del pipeline BDA.

    El DataFrame científico entra a apply_bda con decorr_factor < 1.0.
    La prueba verifica que se asignen ventanas temporales, que exista
    reducción de filas y que se preserven los campos científicos críticos.
    """

    assert integration_bda_active_config["decorr_factor"] == 0.95
    assert integration_bda_active_config["decorr_factor"] < 1.0

    input_count = bda_scientific_df.count()

    df_averaged, df_windowed = apply_bda(
        bda_scientific_df,
        bda_num_partitions,
        integration_bda_active_config,
    )

    assert df_averaged is not None
    assert df_windowed is not None

    windowed_count = df_windowed.count()
    averaged_count = df_averaged.count()

    assert windowed_count == input_count
    assert averaged_count < input_count

    windowed_columns = set(df_windowed.columns)

    assert "window_id" in windowed_columns
    assert "d_uv" in windowed_columns
    assert "phi_dot" in windowed_columns
    assert "sinc_value" in windowed_columns

    averaged_columns = set(df_averaged.columns)

    assert averaged_columns == EXPECTED_AVERAGED_COLUMNS

    baseline_keys = {
        row["baseline_key"]
        for row in df_averaged.select("baseline_key").distinct().collect()
    }

    assert baseline_keys == {"0-1", "2-3"}

    rows = df_averaged.orderBy("baseline_key", "window_id").collect()

    assert len(rows) < input_count

    for row in rows:
        assert row["subms_id"] == 0
        assert row["field_id"] == 1
        assert row["spw_id"] == 2
        assert row["polarization_id"] == 0

        assert row["n_channels"] == bda_nchan
        assert row["n_correlations"] == bda_ncorr

        assert row["antenna1"] in {0, 2}
        assert row["antenna2"] in {1, 3}
        assert row["scan_number"] == 1

        assert row["exposure"] > 0.0
        assert row["interval"] > 0.0
        assert row["baseline_length"] > 0.0

        assert len(row["visibility"]) == bda_nchan
        assert len(row["visibility"][0]) == bda_ncorr
        assert len(row["visibility"][0][0]) == 2

        assert len(row["weight"]) == bda_nchan
        assert len(row["weight"][0]) == bda_ncorr

        assert len(row["flag"]) == bda_nchan
        assert len(row["flag"][0]) == bda_ncorr

    windowed_by_baseline = {
        row["baseline_key"]: row["count"]
        for row in (
            df_windowed
            .groupBy("baseline_key")
            .count()
            .collect()
        )
    }

    assert windowed_by_baseline == {
        "0-1": 4,
        "2-3": 4,
    }

    averaged_by_baseline = {
        row["baseline_key"]: row["count"]
        for row in (
            df_averaged
            .groupBy("baseline_key")
            .count()
            .collect()
        )
    }

    assert set(averaged_by_baseline.keys()) == {"0-1", "2-3"}
    assert all(count >= 1 for count in averaged_by_baseline.values())
    assert sum(averaged_by_baseline.values()) == averaged_count