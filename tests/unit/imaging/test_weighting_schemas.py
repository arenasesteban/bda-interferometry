import pytest
from unittest.mock import patch, MagicMock

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.imaging.weighting_schemes import (
    natural_weighting,
    uniform_weighting,
    weight_visibilities,
    apply_weighting
)

MODULE = "src.imaging.weighting_schemes"


# ==============================================================================
# Tests for natural_weighting
# ==============================================================================

def test_natural_weighting_raises_on_none():
    with pytest.raises(ValueError, match="df_gridded cannot be None"):
        natural_weighting(None)


def test_natural_weighting_raises_on_invalid_type():
    with pytest.raises(ValueError, match="must be a PySpark DataFrame"):
        natural_weighting("not_a_dataframe")


def test_natural_weighting_raises_on_missing_columns(spark):
    """Test missing required columns raises error"""
    incomplete_df = spark.createDataFrame([(1, 2)], ["col1", "col2"])
    
    with pytest.raises(ValueError, match="missing required columns"):
        natural_weighting(incomplete_df)


def test_natural_weighting_single_pixel(df_gridding):
    """Test natural weighting with single pixel (no aggregation needed)"""
    # Filter to single pixel
    single_pixel = df_gridding.filter((F.col("u_pix") == 0) & (F.col("v_pix") == 0))
    
    result = natural_weighting(single_pixel)
    
    assert result.count() == 1
    row = result.collect()[0]
    
    # Should preserve the values (vs_real * weight) / weight
    assert row.u_pix == 0
    assert row.v_pix == 0
    assert row.vs_real == pytest.approx(1.0)
    assert row.vs_imag == pytest.approx(0.0)
    assert row.weight == pytest.approx(1.0)


def test_natural_weighting_aggregates_multiple_pixels(spark, gridded_schema):
    """Test natural weighting aggregates multiple visibilities per pixel"""
    # Multiple rows for same pixel
    rows = [
        (0, 0, 1.0, 0.0, 1.0),   # weight=1.0
        (0, 0, 2.0, 1.0, 2.0),   # weight=2.0
        (0, 0, 3.0, 2.0, 3.0),   # weight=3.0
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = natural_weighting(df)
    
    assert result.count() == 1
    row = result.collect()[0]
    
    # Natural weighting: weighted average
    # vs_real = (1.0*1.0 + 2.0*2.0 + 3.0*3.0) / (1.0 + 2.0 + 3.0) = 14.0 / 6.0
    # vs_imag = (0.0*1.0 + 1.0*2.0 + 2.0*3.0) / (1.0 + 2.0 + 3.0) = 8.0 / 6.0
    # weight = 1.0 + 2.0 + 3.0 = 6.0
    
    assert row.vs_real == pytest.approx(14.0 / 6.0)
    assert row.vs_imag == pytest.approx(8.0 / 6.0)
    assert row.weight == pytest.approx(6.0)


def test_natural_weighting_handles_zero_weight(spark, gridded_schema):
    """Test natural weighting handles zero total weight gracefully"""
    rows = [
        (0, 0, 5.0, 3.0, 0.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = natural_weighting(df)
    
    row = result.collect()[0]
    
    # With zero weight, should return 0.0 for visibilities
    assert row.vs_real == pytest.approx(0.0)
    assert row.vs_imag == pytest.approx(0.0)
    assert row.weight == pytest.approx(0.0)


def test_natural_weighting_multiple_pixels(df_gridding):
    """Test natural weighting preserves multiple different pixels"""
    result = natural_weighting(df_gridding)
    
    # Should have 4 unique pixels
    assert result.count() == 4
    
    # Check schema
    expected_cols = {"u_pix", "v_pix", "vs_real", "vs_imag", "weight"}
    assert set(result.columns) == expected_cols


def test_natural_weighting_preserves_pixel_coordinates(spark, gridded_schema):
    """Test that pixel coordinates are preserved correctly"""
    rows = [
        (5, 7, 1.0, 0.0, 1.0),
        (5, 7, 2.0, 1.0, 1.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = natural_weighting(df)
    
    row = result.collect()[0]
    assert row.u_pix == 5
    assert row.v_pix == 7


# ==============================================================================
# Tests for uniform_weighting
# ==============================================================================

def test_uniform_weighting_raises_on_none():
    with pytest.raises(ValueError, match="df_gridded cannot be None"):
        uniform_weighting(None)


def test_uniform_weighting_raises_on_invalid_type():
    with pytest.raises(ValueError, match="must be a PySpark DataFrame"):
        uniform_weighting("not_a_dataframe")


def test_uniform_weighting_raises_on_missing_columns(spark):
    """Test missing required columns raises error"""
    incomplete_df = spark.createDataFrame([(1, 2)], ["col1", "col2"])
    
    with pytest.raises(ValueError, match="missing required columns"):
        uniform_weighting(incomplete_df)


def test_uniform_weighting_single_pixel(df_gridding):
    """Test uniform weighting with single pixel"""
    single_pixel = df_gridding.filter((F.col("u_pix") == 0) & (F.col("v_pix") == 0))
    
    result = uniform_weighting(single_pixel)
    
    assert result.count() == 1
    row = result.collect()[0]
    
    assert row.u_pix == 0
    assert row.v_pix == 0


def test_uniform_weighting_downweights_dense_regions(spark, gridded_schema):
    """Test uniform weighting down-weights densely sampled pixels"""
    # Pixel (0,0) has higher total weight (should be down-weighted)
    # Pixel (1,1) has lower total weight
    rows = [
        (0, 0, 1.0, 0.0, 5.0),
        (0, 0, 1.0, 0.0, 5.0),
        (1, 1, 1.0, 0.0, 1.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = uniform_weighting(df)
    
    rows_collected = result.collect()
    pixel_00 = [r for r in rows_collected if r.u_pix == 0 and r.v_pix == 0][0]
    pixel_11 = [r for r in rows_collected if r.u_pix == 1 and r.v_pix == 1][0]
    
    # Pixel (0,0) has weight_grid = 10.0, so each vis gets weight_uniform = 5.0/10.0 = 0.5
    # Total weight_uniform for (0,0) = 0.5 + 0.5 = 1.0
    
    # Pixel (1,1) has weight_grid = 1.0, so vis gets weight_uniform = 1.0/1.0 = 1.0
    
    # Uniform weighting equalizes the total weights
    assert pixel_00.weight == pytest.approx(1.0, rel=0.01)
    assert pixel_11.weight == pytest.approx(1.0, rel=0.01)


def test_uniform_weighting_handles_zero_weight(spark, gridded_schema):
    """Test uniform weighting handles zero weight gracefully"""
    rows = [
        (0, 0, 5.0, 3.0, 0.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = uniform_weighting(df)
    
    row = result.collect()[0]
    
    # With zero weight, result should be 0.0
    assert row.vs_real == pytest.approx(0.0)
    assert row.vs_imag == pytest.approx(0.0)


def test_uniform_weighting_uses_epsilon_for_division(spark, gridded_schema):
    """Test that epsilon prevents division by very small numbers"""
    # Very small weight that should trigger epsilon usage
    rows = [
        (0, 0, 5.0, 3.0, 1e-15),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = uniform_weighting(df)
    
    # Should not raise division error
    assert result.count() == 1


def test_uniform_weighting_multiple_pixels(df_gridding):
    """Test uniform weighting preserves multiple different pixels"""
    result = uniform_weighting(df_gridding)
    
    # Should have 4 unique pixels
    assert result.count() == 4
    
    # Check schema
    expected_cols = {"u_pix", "v_pix", "vs_real", "vs_imag", "weight"}
    assert set(result.columns) == expected_cols


def test_uniform_weighting_complex_aggregation(spark, gridded_schema):
    """Test uniform weighting with complex visibility aggregation"""
    rows = [
        (0, 0, 2.0, 1.0, 4.0),
        (0, 0, 4.0, 2.0, 4.0),
        (0, 0, 6.0, 3.0, 4.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = uniform_weighting(df)
    
    row = result.collect()[0]
    
    # weight_grid = 12.0
    # Each vis gets weight_uniform = 4.0 / 12.0 = 1/3
    # vs_real_weighted = 2.0*(1/3) + 4.0*(1/3) + 6.0*(1/3) = 4.0
    # total weight_uniform = 3 * (1/3) = 1.0
    # vs_real = 4.0 / 1.0 = 4.0
    
    assert row.vs_real == pytest.approx(4.0)
    assert row.vs_imag == pytest.approx(2.0)
    assert row.weight == pytest.approx(1.0)


# ==============================================================================
# Tests for weight_visibilities
# ==============================================================================

def test_weight_visibilities_raises_on_none_df():
    with pytest.raises(ValueError, match="df_gridded cannot be None"):
        weight_visibilities(None, "NATURAL")


def test_weight_visibilities_raises_on_invalid_df_type():
    with pytest.raises(ValueError, match="must be a PySpark DataFrame"):
        weight_visibilities("not_a_df", "NATURAL")


def test_weight_visibilities_raises_on_none_scheme():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="weight_scheme cannot be None or empty"):
        weight_visibilities(mock_df, None)


def test_weight_visibilities_raises_on_empty_scheme():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="weight_scheme cannot be None or empty"):
        weight_visibilities(mock_df, "")


def test_weight_visibilities_raises_on_invalid_scheme():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="must be one of"):
        weight_visibilities(mock_df, "INVALID_SCHEME")


def test_weight_visibilities_accepts_lowercase_natural():
    """Test that scheme is case-insensitive"""
    mock_df = MagicMock(spec=DataFrame)
    mock_df.columns = ["u_pix", "v_pix", "vs_real", "vs_imag", "weight"]
    mock_grouped = MagicMock()
    mock_df.groupBy.return_value = mock_grouped
    
    with patch(f"{MODULE}.natural_weighting") as mock_natural:
        mock_natural.return_value = MagicMock()
        
        weight_visibilities(mock_df, "natural")
        
        # Should call natural_weighting after converting to uppercase
        mock_natural.assert_called_once_with(mock_df)


def test_weight_visibilities_accepts_lowercase_uniform():
    """Test uniform scheme case-insensitivity"""
    mock_df = MagicMock(spec=DataFrame)
    mock_df.columns = ["u_pix", "v_pix", "vs_real", "vs_imag", "weight"]
    
    with patch(f"{MODULE}.uniform_weighting") as mock_uniform:
        mock_uniform.return_value = MagicMock()
        
        weight_visibilities(mock_df, "uniform")
        
        mock_uniform.assert_called_once_with(mock_df)


def test_weight_visibilities_calls_natural_weighting(df_gridding):
    """Test that NATURAL scheme calls natural_weighting"""
    with patch(f"{MODULE}.natural_weighting") as mock_natural:
        mock_result = MagicMock()
        mock_natural.return_value = mock_result
        
        result = weight_visibilities(df_gridding, "NATURAL")
        
        mock_natural.assert_called_once_with(df_gridding)
        assert result == mock_result


def test_weight_visibilities_calls_uniform_weighting(df_gridding):
    """Test that UNIFORM scheme calls uniform_weighting"""
    with patch(f"{MODULE}.uniform_weighting") as mock_uniform:
        mock_result = MagicMock()
        mock_uniform.return_value = mock_result
        
        result = weight_visibilities(df_gridding, "UNIFORM")
        
        mock_uniform.assert_called_once_with(df_gridding)
        assert result == mock_result


# ==============================================================================
# Tests for apply_weighting
# ==============================================================================

def test_apply_weighting_raises_on_none_df():
    with pytest.raises(ValueError, match="df_gridded cannot be None"):
        apply_weighting(None, {})


def test_apply_weighting_raises_on_invalid_df_type():
    with pytest.raises(ValueError, match="must be a PySpark DataFrame"):
        apply_weighting("not_a_df", {})


def test_apply_weighting_raises_on_none_config():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="grid_config cannot be None"):
        apply_weighting(mock_df, None)


def test_apply_weighting_raises_on_invalid_config_type():
    mock_df = MagicMock(spec=DataFrame)
    with pytest.raises(ValueError, match="grid_config must be a dictionary"):
        apply_weighting(mock_df, "not_a_dict")


def test_apply_weighting_uses_default_natural(df_gridding):
    """Test that NATURAL is used as default when weight_scheme not in config"""
    empty_config = {}
    
    with patch(f"{MODULE}.weight_visibilities") as mock_weight:
        mock_result = MagicMock()
        mock_weight.return_value = mock_result
        
        result = apply_weighting(df_gridding, empty_config)
        
        mock_weight.assert_called_once_with(df_gridding, "NATURAL")
        assert result == mock_result


def test_apply_weighting_uses_config_scheme(df_gridding, grid_config):
    """Test that weight_scheme from config is used"""
    grid_config["weight_scheme"] = "UNIFORM"
    
    with patch(f"{MODULE}.weight_visibilities") as mock_weight:
        mock_result = MagicMock()
        mock_weight.return_value = mock_result
        
        result = apply_weighting(df_gridding, grid_config)
        
        mock_weight.assert_called_once_with(df_gridding, "UNIFORM")
        assert result == mock_result


def test_apply_weighting_propagates_exceptions(df_gridding, grid_config):
    """Test that exceptions from weight_visibilities are propagated"""
    with patch(f"{MODULE}.weight_visibilities") as mock_weight:
        mock_weight.side_effect = RuntimeError("Weighting failed")
        
        with pytest.raises(RuntimeError, match="Weighting failed"):
            apply_weighting(df_gridding, grid_config)


def test_apply_weighting_handles_spark_errors(df_gridding, grid_config):
    """Test handling of PySpark-specific errors"""
    with patch(f"{MODULE}.weight_visibilities") as mock_weight:
        # Simulate a Spark execution error
        mock_weight.side_effect = Exception("Spark execution error")
        
        with pytest.raises(Exception, match="Spark execution error"):
            apply_weighting(df_gridding, grid_config)


# ==============================================================================
# Integration Tests
# ==============================================================================

def test_full_natural_weighting_pipeline(spark, gridded_schema, grid_config):
    """Test complete natural weighting pipeline"""
    # Create test data with known aggregation
    rows = [
        (0, 0, 2.0, 1.0, 1.0),
        (0, 0, 4.0, 3.0, 3.0),
        (1, 1, 5.0, 2.0, 2.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    grid_config["weight_scheme"] = "NATURAL"
    
    result = apply_weighting(df, grid_config)
    
    rows_collected = result.collect()
    
    # Pixel (0,0): vs_real = (2.0*1.0 + 4.0*3.0)/(1.0+3.0) = 14.0/4.0 = 3.5
    pixel_00 = [r for r in rows_collected if r.u_pix == 0 and r.v_pix == 0][0]
    assert pixel_00.vs_real == pytest.approx(3.5)
    assert pixel_00.vs_imag == pytest.approx(2.5)  # (1.0*1.0 + 3.0*3.0)/4.0
    assert pixel_00.weight == pytest.approx(4.0)
    
    # Pixel (1,1): vs_real = 5.0, vs_imag = 2.0, weight = 2.0
    pixel_11 = [r for r in rows_collected if r.u_pix == 1 and r.v_pix == 1][0]
    assert pixel_11.vs_real == pytest.approx(5.0)
    assert pixel_11.vs_imag == pytest.approx(2.0)
    assert pixel_11.weight == pytest.approx(2.0)


def test_full_uniform_weighting_pipeline(spark, gridded_schema, grid_config):
    """Test complete uniform weighting pipeline"""
    # Dense pixel (0,0) vs sparse pixel (1,1)
    rows = [
        (0, 0, 1.0, 0.0, 3.0),
        (0, 0, 1.0, 0.0, 3.0),
        (0, 0, 1.0, 0.0, 3.0),
        (1, 1, 1.0, 0.0, 1.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    grid_config["weight_scheme"] = "UNIFORM"
    
    result = apply_weighting(df, grid_config)
    
    rows_collected = result.collect()
    pixel_00 = [r for r in rows_collected if r.u_pix == 0 and r.v_pix == 0][0]
    pixel_11 = [r for r in rows_collected if r.u_pix == 1 and r.v_pix == 1][0]
    
    # Both should have similar effective weights after uniform weighting
    assert pixel_00.weight == pytest.approx(1.0, rel=0.01)
    assert pixel_11.weight == pytest.approx(1.0, rel=0.01)


def test_weighting_preserves_data_integrity(df_gridding, grid_config):
    """Test that weighting doesn't corrupt data"""
    original_count = df_gridding.count()
    
    result = apply_weighting(df_gridding, grid_config)
    
    # Should not lose pixels during weighting
    assert result.count() <= original_count
    
    # All weights should be non-negative
    weights = [r.weight for r in result.collect()]
    assert all(w >= 0 for w in weights)


def test_natural_vs_uniform_difference(spark, gridded_schema, grid_config):
    """Test that natural and uniform produce different results"""
    # Create unequally sampled data
    rows = [
        (0, 0, 1.0, 0.0, 5.0),
        (0, 0, 1.0, 0.0, 5.0),
        (1, 1, 1.0, 0.0, 1.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    # Natural weighting
    grid_config["weight_scheme"] = "NATURAL"
    natural_result = apply_weighting(df, grid_config)
    natural_weights = {(r.u_pix, r.v_pix): r.weight for r in natural_result.collect()}
    
    # Uniform weighting
    grid_config["weight_scheme"] = "UNIFORM"
    uniform_result = apply_weighting(df, grid_config)
    uniform_weights = {(r.u_pix, r.v_pix): r.weight for r in uniform_result.collect()}
    
    # Weights should differ (natural favors dense regions, uniform equalizes)
    assert natural_weights[(0, 0)] != pytest.approx(uniform_weights[(0, 0)])


def test_weighting_with_complex_visibilities(spark, gridded_schema, grid_config):
    """Test weighting preserves complex visibility structure"""
    rows = [
        (0, 0, 3.0, 4.0, 1.0),  # Complex visibility
        (0, 0, -2.0, 1.0, 1.0),
    ]
    df = spark.createDataFrame(rows, schema=gridded_schema)
    
    result = apply_weighting(df, grid_config)
    
    row = result.collect()[0]
    
    # Should compute weighted average: ((3.0 - 2.0)/2, (4.0 + 1.0)/2) = (0.5, 2.5)
    assert row.vs_real == pytest.approx(0.5)
    assert row.vs_imag == pytest.approx(2.5)