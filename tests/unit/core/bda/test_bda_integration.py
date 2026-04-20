import pytest

from core.bda.bda_integration import apply_bda

required = {
    "subms_id", "field_id", "spw_id", "polarization_id",
    "n_channels", "n_correlations",
    "antenna1", "antenna2", "baseline_key", "scan_number",
    "exposure", "interval", "time",
    "u", "v",
    "visibility", "weight", "flag",
}

# Test cases for apply_bda

def test_returns_tuple_of_two(df_single_row, bda_config):
    result = apply_bda(df_single_row, num_partitions=1, bda_config=bda_config)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_first_element_is_averaged_dataframe(df_single_row, bda_config):
    df_averaged, _ = apply_bda(df_single_row, num_partitions=1, bda_config=bda_config)
    columns = df_averaged.columns

    assert required.issubset(set(columns))
    assert "window_id" in columns


def test_second_element_is_windowed_dataframe(df_single_row, bda_config):
    _, df_windowed = apply_bda(df_single_row, num_partitions=1, bda_config=bda_config)
    columns = df_windowed.columns
    
    assert required.issubset(set(columns))
    assert "window_id" in columns
    assert "d_uv" in columns
    assert "phi_dot" in columns
    assert "sinc_value" in columns


def test_single_row_preserves_count(df_single_row, bda_config):
    df_averaged, _ = apply_bda(df_single_row, num_partitions=1, bda_config=bda_config)
    assert df_averaged.count() == 1


def test_same_window_rows_are_collapsed(df_two_rows_same_window, bda_config):
    df_averaged, _ = apply_bda(
        df_two_rows_same_window, num_partitions=1, bda_config=bda_config
    )
    assert df_averaged.count() == 1


def test_different_window_rows_remain_separate(df_two_rows_different_window, bda_config):
    df_averaged, _ = apply_bda(
        df_two_rows_different_window, num_partitions=1, bda_config=bda_config
    )
    assert df_averaged.count() == 2


def test_averaged_never_exceeds_input_rows(df_multi_baselines, bda_config):
    n_input = df_multi_baselines.count()
    df_averaged, _ = apply_bda(
        df_multi_baselines, num_partitions=2, bda_config=bda_config
    )
    assert df_averaged.count() <= n_input


def test_windowed_preserves_input_row_count(df_multi_baselines, bda_config):
    n_input = df_multi_baselines.count()
    _, df_windowed = apply_bda(
        df_multi_baselines, num_partitions=2, bda_config=bda_config
    )
    assert df_windowed.count() == n_input


# Test error handling

def test_apply_bda_none_dataframe_raises_error(bda_config):
    with pytest.raises(ValueError, match="Input DataFrame cannot be None"):
        apply_bda(None, num_partitions=1, bda_config=bda_config)


def test_apply_bda_invalid_partitions_raises_error(df_single_row, bda_config):
    with pytest.raises(ValueError, match="num_partitions must be a positive integer"):
        apply_bda(df_single_row, num_partitions=0, bda_config=bda_config)


def test_apply_bda_missing_columns_raises_error(df_single_row):
    with pytest.raises(ValueError, match="BDA configuration cannot be None"):
        apply_bda(df_single_row, num_partitions=1, bda_config=None)


def test_apply_bda_dataframe_missing_columns_raises_error(df_single_row, bda_config):
    df_incomplete = df_single_row.drop("baseline_key")
    with pytest.raises(Exception):
        apply_bda(df_incomplete, num_partitions=1, bda_config=bda_config)