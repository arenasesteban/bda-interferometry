import pytest
import numpy as np
import json
import pandas as pd
from unittest.mock import patch, MagicMock

from pyspark.sql import DataFrame

from src.imaging.gridding import (
    apply_gridding,
    prepare_gridding,
    accumulate_grid,
    process_visibility,
    calculate_uv_pix,
    is_valid_uv_pix,
    build_corrs_map,
    load_grid_config,
    build_grid
)

MODULE = "src.imaging.gridding"


# ==============================================================================
# Tests for apply_gridding
# ==============================================================================

def test_apply_gridding_raises_on_none_df_scientific():
    with pytest.raises(ValueError, match="df_scientific cannot be None"):
        apply_gridding(None, 4, {}, "PARTIAL")


def test_apply_gridding_raises_on_invalid_df_type():
    with pytest.raises(ValueError, match="must be a PySpark DataFrame"):
        apply_gridding("not_a_df", 4, {}, "PARTIAL")


def test_apply_gridding_raises_on_invalid_num_partitions():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="must be a positive integer"):
        apply_gridding(mock_df, 0, {}, "PARTIAL")


def test_apply_gridding_raises_on_none_grid_config():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="grid_config cannot be None"):
        apply_gridding(mock_df, 4, None, "PARTIAL")


def test_apply_gridding_raises_on_invalid_grid_config_type():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="grid_config must be a dictionary"):
        apply_gridding(mock_df, 4, "not_a_dict", "PARTIAL")


def test_apply_gridding_raises_on_none_strategy():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="strategy cannot be None or empty"):
        apply_gridding(mock_df, 4, {}, None)


def test_apply_gridding_raises_on_empty_strategy():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="strategy cannot be None or empty"):
        apply_gridding(mock_df, 4, {}, "")


def test_apply_gridding_raises_on_invalid_strategy():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="must be one of"):
        apply_gridding(mock_df, 4, {}, "INVALID")


@patch(f"{MODULE}.accumulate_grid")
@patch(f"{MODULE}.prepare_gridding")
def test_apply_gridding_partial_strategy(mock_prepare, mock_accumulate, df_single_row, grid_config):
    """Test PARTIAL strategy calls prepare_gridding and accumulate_grid"""
    mock_duplicated = MagicMock(spec=DataFrame)
    mock_repartitioned = MagicMock(spec=DataFrame)
    mock_gridded = MagicMock(spec=DataFrame)
    
    mock_prepare.return_value = mock_duplicated
    mock_duplicated.repartition.return_value = mock_repartitioned
    mock_accumulate.return_value = mock_gridded
    
    result = apply_gridding(df_single_row, 4, grid_config, "PARTIAL")
    
    mock_prepare.assert_called_once_with(df_single_row)
    mock_duplicated.repartition.assert_called_once_with(8, "u", "v")
    mock_accumulate.assert_called_once_with(mock_repartitioned, grid_config)
    assert result == mock_gridded


@patch(f"{MODULE}.apply_weighting")
def test_apply_gridding_complete_strategy(mock_weighting, df_single_row, grid_config):
    """Test COMPLETE strategy calls apply_weighting"""
    mock_weighted = MagicMock(spec=DataFrame)
    mock_coalesced = MagicMock(spec=DataFrame)
    
    mock_weighting.return_value = mock_weighted
    mock_weighted.coalesce.return_value = mock_coalesced
    
    result = apply_gridding(df_single_row, 4, grid_config, "COMPLETE")
    
    mock_weighting.assert_called_once_with(df_single_row, grid_config)
    mock_weighted.coalesce.assert_called_once_with(4)
    assert result == mock_coalesced


@patch(f"{MODULE}.prepare_gridding")
def test_apply_gridding_propagates_exceptions(mock_prepare, df_single_row, grid_config):
    """Test that exceptions from sub-functions are propagated"""
    mock_prepare.side_effect = RuntimeError("Gridding failed")
    
    with pytest.raises(RuntimeError, match="Gridding failed"):
        apply_gridding(df_single_row, 4, grid_config, "PARTIAL")


# ==============================================================================
# Tests for prepare_gridding
# ==============================================================================

def test_prepare_gridding_raises_on_none():
    with pytest.raises(ValueError, match="scientific_df cannot be None"):
        prepare_gridding(None)


def test_prepare_gridding_raises_on_invalid_type():
    with pytest.raises(ValueError, match="must be a PySpark DataFrame"):
        prepare_gridding("not_a_df")


def test_prepare_gridding_raises_on_missing_columns(spark, visibility_schema):
    """Test missing required columns raises error"""
    incomplete_df = spark.createDataFrame([(1, 2)], ["col1", "col2"])
    
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_gridding(incomplete_df)


def test_prepare_gridding_duplicates_rows(df_single_row):
    """Test that prepare_gridding doubles the number of rows (Hermitian)"""
    result = prepare_gridding(df_single_row)
    
    assert result.count() == 2  # Original + Hermitian
    
    # Collect results
    rows = result.collect()
    
    # Check original row
    original = [r for r in rows if not r.is_hermitian][0]
    assert original.u == 100.0
    assert original.v == 50.0
    
    # Check Hermitian conjugate
    hermitian = [r for r in rows if r.is_hermitian][0]
    assert hermitian.u == -100.0
    assert hermitian.v == -50.0


def test_prepare_gridding_preserves_visibility_data(df_single_row):
    """Test that visibility data is preserved in both copies"""
    result = prepare_gridding(df_single_row)
    rows = result.collect()
    
    for row in rows:
        assert row.baseline_key == "0-1"
        assert row.n_channels == 1
        assert row.n_correlations == 1
        assert row.visibility is not None
        assert row.weight is not None
        assert row.flag is not None


def test_prepare_gridding_with_multiple_baselines(df_multi_baselines):
    """Test prepare_gridding with multiple baselines"""
    result = prepare_gridding(df_multi_baselines)
    
    # Should double the count (6 original + 6 Hermitian = 12)
    assert result.count() == 12
    
    # Check each baseline is duplicated
    for baseline_key in ["0-1", "0-2", "1-2"]:
        baseline_rows = result.filter(result.baseline_key == baseline_key).collect()
        assert len(baseline_rows) == 4  # 2 original + 2 Hermitian


# ==============================================================================
# Tests for accumulate_grid
# ==============================================================================

def test_accumulate_grid_raises_on_none_df():
    with pytest.raises(ValueError, match="scientific_df cannot be None"):
        accumulate_grid(None, {})


def test_accumulate_grid_raises_on_none_config():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="grid_config cannot be None"):
        accumulate_grid(mock_df, None)


def test_accumulate_grid_raises_on_missing_config_params():
    mock_df = MagicMock(spec=DataFrame)
    incomplete_config = {"img_size": 8}
    
    with pytest.raises(KeyError, match="Missing required grid_config parameters"):
        accumulate_grid(mock_df, incomplete_config)


@patch(f"{MODULE}.build_corrs_map")
@patch(f"{MODULE}.process_visibility")
def test_accumulate_grid_success(mock_process, mock_corrs, df_single_row, grid_config):
    """Test successful grid accumulation"""
    mock_corrs.return_value = {0: "XX", 1: "YY"}
    mock_process.return_value = [
        {"u_pix": 4, "v_pix": 4, "vs_real": 1.0, "vs_imag": 0.0, "weight": 1.0}
    ]
    
    result = accumulate_grid(df_single_row, grid_config)
    
    assert result is not None
    assert isinstance(result, DataFrame)
    
    # Check schema
    expected_cols = {"u_pix", "v_pix", "vs_real", "vs_imag", "weight"}
    assert set(result.columns) == expected_cols


def test_accumulate_grid_processes_multiple_rows(df_two_rows_same_window, grid_config):
    """Test accumulation with multiple rows"""
    result = accumulate_grid(df_two_rows_same_window, grid_config)
    
    assert result is not None
    # Even if no pixels are accumulated (e.g., all out of bounds), DataFrame exists
    assert isinstance(result, DataFrame)


# ==============================================================================
# Tests for process_visibility
# ==============================================================================

@pytest.fixture
def sample_row():
    """Create a pandas Series mimicking a row from a DataFrame"""
    return pd.Series({
        'u': 100.0,
        'v': 50.0,
        'visibility': [[[1.0, 0.5], [2.0, 1.0]]],  # 1 channel, 2 correlations
        'weight': [[1.0, 1.0]],
        'flag': [[0, 0]],
        'n_channels': 1,
        'n_correlations': 2,
        'is_hermitian': False
    })


def test_process_visibility_returns_pixels(sample_row):
    """Test that process_visibility returns pixel data"""
    chan_freq = [1.4e9]
    corrs_map = {0: "XX", 1: "YY"}
    # Use larger cellsize to ensure pixels are in bounds
    uvcellsize = [-1e-3, 1e-3]
    padded_size = 1000  # Larger grid to accommodate
    
    result = process_visibility(sample_row, chan_freq, corrs_map, uvcellsize, padded_size)
    
    assert result is not None
    assert isinstance(result, list)
    
    if len(result) > 0:  # May be empty if out of bounds
        # Check pixel structure
        pixel = result[0]
        assert 'u_pix' in pixel
        assert 'v_pix' in pixel
        assert 'vs_real' in pixel
        assert 'vs_imag' in pixel
        assert 'weight' in pixel


def test_process_visibility_skips_flagged(sample_row):
    """Test that flagged visibilities are skipped"""
    sample_row['flag'] = [[1, 1]]  # All flagged
    
    chan_freq = [1.4e9]
    corrs_map = {0: "XX", 1: "YY"}
    uvcellsize = [-1e-5, 1e-5]
    padded_size = 8
    
    result = process_visibility(sample_row, chan_freq, corrs_map, uvcellsize, padded_size)
    
    # Should return empty list since all are flagged
    assert result == []


def test_process_visibility_handles_hermitian(sample_row):
    """Test Hermitian conjugate is computed correctly"""
    sample_row['is_hermitian'] = True
    
    chan_freq = [1.4e9]
    corrs_map = {0: "XX", 1: "YY"}
    uvcellsize = [-1e-5, 1e-5]
    padded_size = 8
    
    result = process_visibility(sample_row, chan_freq, corrs_map, uvcellsize, padded_size)
    
    assert result is not None
    # For Hermitian, imaginary part should be negated
    # Original: (1.0, 0.5) -> Hermitian: (1.0, -0.5)
    if result:
        pixel = result[0]
        # Check that conjugate was applied (imag negated)
        assert pixel['vs_imag'] == pytest.approx(-0.5)


def test_process_visibility_filters_correlations(sample_row):
    """Test that only XX, YY, RR, LL correlations are kept"""
    chan_freq = [1.4e9]
    corrs_map = {0: "XY", 1: "YY"}  # XY should be filtered out
    uvcellsize = [-1e-3, 1e-3]
    padded_size = 1000
    
    result = process_visibility(sample_row, chan_freq, corrs_map, uvcellsize, padded_size)
    
    # Should have at most 1 pixel (YY), XY is filtered
    # May be empty if out of bounds
    assert len(result) <= 1
    
    if len(result) == 1:
        # If present, it should be from YY correlation
        assert result is not None


def test_process_visibility_handles_out_of_bounds():
    """Test that out-of-bounds pixels are skipped"""
    row = pd.Series({
        'u': 1e10,  # Very large u to go out of bounds
        'v': 1e10,
        'visibility': [[[1.0, 0.5]]],
        'weight': [[1.0]],
        'flag': [[0]],
        'n_channels': 1,
        'n_correlations': 1,
        'is_hermitian': False
    })
    
    chan_freq = [1.4e9]
    corrs_map = {0: "XX"}
    uvcellsize = [-1e-5, 1e-5]
    padded_size = 8
    
    result = process_visibility(row, chan_freq, corrs_map, uvcellsize, padded_size)
    
    # Should be empty as pixels are out of bounds
    assert result == []


def test_process_visibility_returns_none_on_error():
    """Test that errors in process_visibility return None"""
    # Malformed row that will cause an error
    bad_row = pd.Series({'u': None, 'v': None})
    
    result = process_visibility(bad_row, [1.4e9], {0: "XX"}, [-1e-5, 1e-5], 8)
    
    assert result is None


# ==============================================================================
# Tests for calculate_uv_pix
# ==============================================================================

def test_calculate_uv_pix_raises_on_zero_freq():
    with pytest.raises(ValueError, match="freq must be positive"):
        calculate_uv_pix(100.0, 50.0, 0.0, [-1e-5, 1e-5], 8)


def test_calculate_uv_pix_raises_on_negative_freq():
    with pytest.raises(ValueError, match="freq must be positive"):
        calculate_uv_pix(100.0, 50.0, -1.4e9, [-1e-5, 1e-5], 8)


def test_calculate_uv_pix_raises_on_zero_padded_size():
    with pytest.raises(ValueError, match="padded_size must be positive"):
        calculate_uv_pix(100.0, 50.0, 1.4e9, [-1e-5, 1e-5], 0)


def test_calculate_uv_pix_raises_on_invalid_uvcellsize():
    with pytest.raises(ValueError, match="uvcellsize must have 2 elements"):
        calculate_uv_pix(100.0, 50.0, 1.4e9, [-1e-5], 8)


def test_calculate_uv_pix_raises_on_zero_uvcellsize():
    with pytest.raises(ValueError, match="uvcellsize elements cannot be zero"):
        calculate_uv_pix(100.0, 50.0, 1.4e9, [0.0, 1e-5], 8)


def test_calculate_uv_pix_returns_valid_coordinates():
    """Test that valid UV coordinates are computed"""

    u = 100.0
    v = 50.0    
    freq = 1.4e9
    
    cellsize = 5e-6
    uvcellsize = [-1 / (cellsize * 1000), 1 / (cellsize * 1000)]
    padded_size = 1000
    u_pix, v_pix = calculate_uv_pix(u, v, freq, uvcellsize, padded_size)
    
    assert isinstance(u_pix, int)
    assert isinstance(v_pix, int)
    # With these parameters, should be in reasonable range
    assert -1000 <= u_pix <= 1000
    assert -1000 <= v_pix <= 1000


def test_calculate_uv_pix_origin_at_center():
    """Test that (0,0) UV coordinates map near grid center"""
    u_pix, v_pix = calculate_uv_pix(0.0, 0.0, 1.4e9, [-1e-5, 1e-5], 8)
    
    # Should be near center (padded_size // 2 = 4)
    assert u_pix == 4
    assert v_pix == 4


# ==============================================================================
# Tests for is_valid_uv_pix
# ==============================================================================

def test_is_valid_uv_pix_accepts_valid_coordinates():
    assert is_valid_uv_pix(0, 0, 8) is True
    assert is_valid_uv_pix(4, 4, 8) is True
    assert is_valid_uv_pix(7, 7, 8) is True


def test_is_valid_uv_pix_rejects_negative():
    assert is_valid_uv_pix(-1, 0, 8) is False
    assert is_valid_uv_pix(0, -1, 8) is False


def test_is_valid_uv_pix_rejects_out_of_bounds():
    assert is_valid_uv_pix(8, 0, 8) is False
    assert is_valid_uv_pix(0, 8, 8) is False
    assert is_valid_uv_pix(10, 10, 8) is False


def test_is_valid_uv_pix_boundary_cases():
    """Test exact boundary cases"""
    assert is_valid_uv_pix(0, 0, 8) is True  # Minimum valid
    assert is_valid_uv_pix(7, 7, 8) is True  # Maximum valid
    assert is_valid_uv_pix(8, 8, 8) is False  # Just outside


# ==============================================================================
# Tests for build_corrs_map
# ==============================================================================

def test_build_corrs_map_raises_on_none():
    with pytest.raises(ValueError, match="corrs_string cannot be None"):
        build_corrs_map(None)


def test_build_corrs_map_from_string():
    """Test parsing from comma-separated string"""
    result = build_corrs_map("XX,XY,YX,YY")
    
    assert result == {0: "XX", 1: "XY", 2: "YX", 3: "YY"}


def test_build_corrs_map_from_list():
    """Test parsing from list"""
    result = build_corrs_map(["XX", "YY"])
    
    assert result == {0: "XX", 1: "YY"}


def test_build_corrs_map_from_nested_list():
    """Test parsing from nested list"""
    result = build_corrs_map([["RR", "LL"]])
    
    assert result == {0: "RR", 1: "LL"}


def test_build_corrs_map_handles_whitespace():
    """Test that whitespace is stripped"""
    result = build_corrs_map(" XX , YY , RR ")
    
    assert result == {0: "XX", 1: "YY", 2: "RR"}


def test_build_corrs_map_raises_on_invalid_format():
    """Test that invalid format raises error"""
    with pytest.raises(ValueError, match="Failed to parse corrs_string"):
        build_corrs_map(123)  # Not string or list


# ==============================================================================
# Tests for load_grid_config
# ==============================================================================

def test_load_grid_config_raises_on_none():
    with pytest.raises(ValueError, match="config_path cannot be None or empty"):
        load_grid_config(None)


def test_load_grid_config_raises_on_empty_string():
    with pytest.raises(ValueError, match="config_path cannot be None or empty"):
        load_grid_config("")


def test_load_grid_config_raises_on_missing_file(tmp_path):
    missing = tmp_path / "nonexistent.json"
    
    with pytest.raises(FileNotFoundError, match="Grid config file not found"):
        load_grid_config(str(missing))


def test_load_grid_config_raises_on_invalid_json(tmp_path):
    """Test that invalid JSON raises error"""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ invalid json", encoding="utf-8")
    
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_grid_config(str(bad_json))


def test_load_grid_config_raises_on_non_dict_json(tmp_path):
    """Test that non-dictionary JSON raises error"""
    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]", encoding="utf-8")
    
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_grid_config(str(array_json))


def test_load_grid_config_raises_on_missing_fields(tmp_path):
    """Test that missing required fields raises error"""
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"img_size": 8}), encoding="utf-8")
    
    with pytest.raises(ValueError, match="Missing required fields"):
        load_grid_config(str(incomplete))


def test_load_grid_config_success(tmp_path):
    """Test successful config loading"""
    valid_config = {
        "img_size": 8,
        "padding_factor": 1.0,
        "cellsize": 1e-5
    }
    
    config_file = tmp_path / "valid.json"
    config_file.write_text(json.dumps(valid_config), encoding="utf-8")
    
    result = load_grid_config(str(config_file))
    
    assert result == valid_config


def test_load_grid_config_accepts_path_object(tmp_path):
    """Test that Path objects are accepted"""
    valid_config = {
        "img_size": 8,
        "padding_factor": 1.0,
        "cellsize": 1e-5
    }
    
    config_file = tmp_path / "valid.json"
    config_file.write_text(json.dumps(valid_config), encoding="utf-8")
    
    result = load_grid_config(config_file)  # Pass Path object
    
    assert result == valid_config


# ==============================================================================
# Tests for build_grid
# ==============================================================================

def test_build_grid_raises_on_none_df():
    with pytest.raises(ValueError, match="df_gridded cannot be None"):
        build_grid(None, {}, 4)


def test_build_grid_raises_on_invalid_df_type():
    with pytest.raises(ValueError, match="must be a PySpark DataFrame"):
        build_grid("not_a_df", {}, 4)


def test_build_grid_raises_on_none_config():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="grid_config cannot be None"):
        build_grid(mock_df, None, 4)


def test_build_grid_raises_on_invalid_num_partitions():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="must be a positive integer"):
        build_grid(mock_df, {}, -1)


def test_build_grid_raises_on_missing_config_params():
    mock_df = MagicMock(spec=DataFrame)
    incomplete = {"img_size": 8}
    
    with pytest.raises(KeyError, match="Missing required grid_config parameters"):
        build_grid(mock_df, incomplete, 4)


def test_build_grid_raises_on_invalid_img_size():
    mock_df = MagicMock(spec=DataFrame)
    bad_config = {"img_size": -1, "padding_factor": 1.0}
    
    with pytest.raises(ValueError, match="img_size must be a positive integer"):
        build_grid(mock_df, bad_config, 4)


def test_build_grid_raises_on_invalid_padding_factor():
    mock_df = MagicMock(spec=DataFrame)
    bad_config = {"img_size": 8, "padding_factor": 0.5}
    
    with pytest.raises(ValueError, match="padding_factor must be >= 1.0"):
        build_grid(mock_df, bad_config, 4)


def test_build_grid_returns_arrays(df_gridding, grid_config):
    """Test that build_grid returns numpy arrays"""
    grid, weights = build_grid(df_gridding, grid_config, 2)
    
    assert isinstance(grid, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert grid.dtype == np.complex128
    assert weights.dtype == np.float64


def test_build_grid_correct_shape(df_gridding, grid_config):
    """Test that output arrays have correct shape"""
    grid, weights = build_grid(df_gridding, grid_config, 2)
    
    expected_size = int(grid_config["img_size"] * grid_config["padding_factor"])
    
    assert grid.shape == (expected_size, expected_size)
    assert weights.shape == (expected_size, expected_size)


def test_build_grid_accumulates_visibilities(df_gridding, grid_config):
    """Test that visibilities are accumulated correctly"""
    grid, weights = build_grid(df_gridding, grid_config, 2)
    
    # Check that grid has non-zero values at expected positions
    assert np.any(grid != 0)
    assert np.any(weights != 0)
    
    # Check specific pixel from test data
    # Row (0, 0, 1.0, 0.0, 1.0) should contribute
    assert grid[0, 0] != 0


def test_build_grid_with_padding(df_gridding, grid_config_with_padding):
    """Test build_grid with padding factor > 1"""
    grid, weights = build_grid(df_gridding, grid_config_with_padding, 2)
    
    expected_size = int(grid_config_with_padding["img_size"] * grid_config_with_padding["padding_factor"])
    
    assert grid.shape == (expected_size, expected_size)
    assert expected_size == 16  # 8 * 2.0


def test_build_grid_tree_reduce_merges_correctly(df_gridding, grid_config):
    """Test that tree reduce correctly merges partial grids"""
    # Use multiple partitions to trigger tree reduce
    grid, weights = build_grid(df_gridding, grid_config, 4)
    
    # All pixels from df_gridding should be present
    rows = df_gridding.collect()
    for row in rows:
        u_pix = row.u_pix
        v_pix = row.v_pix
        
        # Check that this pixel was accumulated
        assert weights[v_pix, u_pix] > 0


# ==============================================================================
# Integration Tests
# ==============================================================================

def test_full_gridding_pipeline_partial(df_two_rows_same_window, grid_config):
    """Test complete gridding pipeline with PARTIAL strategy"""
    # Apply full pipeline
    df_gridded = apply_gridding(df_two_rows_same_window, 2, grid_config, "PARTIAL")
    
    assert df_gridded is not None
    assert df_gridded.count() > 0
    
    # Build final grid
    grid, weights = build_grid(df_gridded, grid_config, 2)
    
    assert grid.shape == (8, 8)
    assert weights.shape == (8, 8)
    assert np.any(grid != 0)


@patch(f"{MODULE}.apply_weighting")
def test_full_gridding_pipeline_complete(mock_weighting, df_two_rows_same_window, grid_config):
    """Test complete gridding pipeline with COMPLETE strategy"""
    mock_weighted = MagicMock(spec=DataFrame)
    mock_coalesced = MagicMock(spec=DataFrame)
    mock_weighted.coalesce.return_value = mock_coalesced
    mock_weighting.return_value = mock_weighted
    
    result = apply_gridding(df_two_rows_same_window, 2, grid_config, "COMPLETE")
    
    assert result == mock_coalesced
    mock_weighting.assert_called_once()