import math
import pytest
import numpy as np

from core.bda.bda_processor import (
    process_rows,
    assign_temporal_window,
    assign_windows,
    average_by_window,
    average_visibilities
)

# Constants for testing
DECORR_FACTOR = 0.95
LAMBDA_REF = 0.01
FOV = 0.01


def _prepare(df, decorr_factor=DECORR_FACTOR, fov=FOV, lambda_ref=LAMBDA_REF):
    df_windowed = assign_temporal_window(df, decorr_factor, fov, lambda_ref)
    return average_by_window(df_windowed)


# Test cases for shape and type invariants

def test_visibility_shape_preserved_multi_channel(df_multi_channel):
    result = _prepare(df_multi_channel).toPandas()
    
    assert len(result) == 1
    vis = result.loc[0, "visibility"]

    assert len(vis) == 4  # 4 channels
    assert len(vis[0]) == 1  # 1 correlation
    assert len(vis[0][0]) == 2  # [real, imag]


def test_visibility_shape_preserved_multi_correlation(df_multi_correlation):
    result = _prepare(df_multi_correlation).toPandas()
    
    assert len(result) == 1
    vis = result.loc[0, "visibility"]
    
    assert len(vis) == 1  # 1 channel
    assert len(vis[0]) == 4  # 4 correlations
    assert len(vis[0][0]) == 2  # [real, imag]


def test_visibility_shape_multi_channel_correlation(df_multi_channel_correlation):
    result = _prepare(df_multi_channel_correlation).toPandas()
    
    assert len(result) == 1
    vis = result.loc[0, "visibility"]
    
    assert len(vis) == 4  # 4 channels
    assert len(vis[0]) == 4  # 4 correlations
    assert len(vis[0][0]) == 2  # [real, imag]


def test_flag_shape_matches_visibility(df_multi_channel_correlation):
    result = _prepare(df_multi_channel_correlation).toPandas()
    
    vis = result.loc[0, "visibility"]
    flag = result.loc[0, "flag"]
    
    assert len(flag) == len(vis)
    assert len(flag[0]) == len(vis[0])


def test_visibility_dtype_preserved(df_multi_channel_correlation):
    pdf_windowed = assign_temporal_window(
        df_multi_channel_correlation, DECORR_FACTOR, FOV, LAMBDA_REF
    ).toPandas()
    result = average_visibilities(pdf_windowed)
    
    vis = result.loc[0, "visibility"]
    
    # Extract a single value and check type
    real_value = vis[0][0][0]
    imag_value = vis[0][0][1]
    
    assert isinstance(real_value, (float, np.floating))
    assert isinstance(imag_value, (float, np.floating))


def test_flag_dtype_is_integer(df_multi_channel):
    result = _prepare(df_multi_channel).toPandas()
    
    flag = result.loc[0, "flag"]
    
    for chan_flags in flag:
        for f in chan_flags:
            assert isinstance(f, (int, np.integer))
            assert f in [0, 1]


def test_weight_dtype_is_float(df_multi_correlation):
    result = _prepare(df_multi_correlation).toPandas()
    
    weight = result.loc[0, "weight"]
    
    for chan_weights in weight:
        for w in chan_weights:
            assert isinstance(w, (float, np.floating))


# Test cases for identity (no averaging, factor = 1.0)

def test_decorr_factor_one_creates_individual_windows(df_identity_case):
    pdf = df_identity_case.toPandas()
    result = assign_windows(pdf, decorr_factor=1.0, fov=FOV, lambda_ref=LAMBDA_REF)
    
    assert len(result) == 3
    assert result['window_id'].tolist() == [1, 2, 3]


def test_decorr_factor_one_preserves_all_data(df_identity_case):
    pdf_input = df_identity_case.toPandas()
    result = _prepare(df_identity_case, decorr_factor=1.0).toPandas()
    assert len(result) == 3
    
    for i in range(len(result)):
        # Visibility, weights, flags identical
        assert result.loc[i, 'visibility'] == pdf_input.loc[i, 'visibility']
        assert result.loc[i, 'weight'] == pdf_input.loc[i, 'weight']
        assert result.loc[i, 'flag'] == pdf_input.loc[i, 'flag']
        
        # Scalar values identical
        assert math.isclose(result.loc[i, 'exposure'], pdf_input.loc[i, 'exposure'], rel_tol=1e-9)
        assert math.isclose(result.loc[i, 'interval'], pdf_input.loc[i, 'interval'], rel_tol=1e-9)
        assert math.isclose(result.loc[i, 'u'], pdf_input.loc[i, 'u'], rel_tol=1e-9)
        assert math.isclose(result.loc[i, 'v'], pdf_input.loc[i, 'v'], rel_tol=1e-9)


def test_single_row_window_is_identity(df_single_row):
    pdf_input = df_single_row.toPandas()
    result = _prepare(df_single_row).toPandas()
    
    assert result.loc[0, 'antenna1'] == pdf_input.loc[0, 'antenna1']
    assert result.loc[0, 'antenna2'] == pdf_input.loc[0, 'antenna2']
    assert result.loc[0, 'baseline_key'] == pdf_input.loc[0, 'baseline_key']
    assert math.isclose(result.loc[0, 'time'], pdf_input.loc[0, 'time'], rel_tol=1e-9)
    assert math.isclose(result.loc[0, 'u'], pdf_input.loc[0, 'u'], rel_tol=1e-9)
    assert math.isclose(result.loc[0, 'v'], pdf_input.loc[0, 'v'], rel_tol=1e-9)
    
    assert result.loc[0, 'visibility'] == pdf_input.loc[0, 'visibility']
    assert result.loc[0, 'weight'] == pdf_input.loc[0, 'weight']
    assert result.loc[0, 'flag'] == pdf_input.loc[0, 'flag']


def test_natural_single_window_with_multiple_rows(df_single_window_multi_rows):
    result = _prepare(df_single_window_multi_rows).toPandas()
    
    assert len(result) == 1
    
    assert math.isclose(result.loc[0, 'exposure'], 3.0, rel_tol=1e-9)
    assert math.isclose(result.loc[0, 'interval'], 3.0, rel_tol=1e-9)


# Test cases for constant data

def test_constant_complex_visibility_preserved(df_constant_visibility):
    result = _prepare(df_constant_visibility).toPandas()
    
    assert len(result) == 1
    vis = result.loc[0, 'visibility']
 
    for chan in vis:
        for corr in chan:
            real, imag = corr
            assert math.isclose(real, 3.0, abs_tol=1e-9)
            assert math.isclose(imag, 4.0, abs_tol=1e-9)


def test_constant_visibility_unequal_weights(df_constant_unequal_weights):
    result = _prepare(df_constant_unequal_weights).toPandas()
    
    assert len(result) == 1
    vis = result.loc[0, 'visibility']
    
    real, imag = vis[0][0]
    assert math.isclose(real, 2.0, abs_tol=1e-9)
    assert math.isclose(imag, 1.0, abs_tol=1e-9)


def test_constant_per_channel_preserved(df_constant_multi_channel):
    result = _prepare(df_constant_multi_channel).toPandas()
    
    assert len(result) == 1
    vis = result.loc[0, 'visibility']
    
    assert math.isclose(vis[0][0][0], 1.0, abs_tol=1e-9)
    assert math.isclose(vis[0][0][1], 1.0, abs_tol=1e-9)
    assert math.isclose(vis[1][0][0], 2.0, abs_tol=1e-9)
    assert math.isclose(vis[1][0][1], 2.0, abs_tol=1e-9)
    assert math.isclose(vis[2][0][0], 3.0, abs_tol=1e-9)
    assert math.isclose(vis[2][0][1], 3.0, abs_tol=1e-9)


# Test cases for weights

def test_zero_weight_excluded_from_average(df_zero_weight_excluded):
    result = _prepare(df_zero_weight_excluded).toPandas()
    
    vis = result.loc[0, 'visibility']
    real, imag = vis[0][0]
    
    expected_real = (2.0 * 1.0 + 4.0 * 2.0) / (1.0 + 2.0)
    expected_imag = (1.0 * 1.0 + 2.0 * 2.0) / (1.0 + 2.0)
    
    assert math.isclose(real, expected_real, abs_tol=1e-9)
    assert math.isclose(imag, expected_imag, abs_tol=1e-9)


def test_all_zero_weights_convention(df_all_zero_weights):
    result = _prepare(df_all_zero_weights).toPandas()
    
    vis = result.loc[0, 'visibility']
    flag = result.loc[0, 'flag']
    weight = result.loc[0, 'weight']
    real, imag = vis[0][0]
    
    assert real == 0.0
    assert imag == 0.0
    assert not math.isnan(real)
    assert not math.isnan(imag)
    assert flag[0][0] == 1
    assert weight[0][0] == 0.0


# Test cases for flags

def test_flagged_samples_excluded_from_average(df_partially_flagged):
    result = _prepare(df_partially_flagged).toPandas()
    
    vis = result.loc[0, 'visibility']
    real, imag = vis[0][0]
    
    expected_real = (2.0 + 4.0) / 2.0
    expected_imag = (1.0 + 2.0) / 2.0
    
    assert math.isclose(real, expected_real, abs_tol=1e-9)
    assert math.isclose(imag, expected_imag, abs_tol=1e-9)


def test_all_flagged_produces_flagged_output(df_all_flagged):
    result = _prepare(df_all_flagged).toPandas()
    
    vis = result.loc[0, 'visibility']
    flag = result.loc[0, 'flag']    
    real, imag = vis[0][0]

    assert real == 0.0
    assert imag == 0.0
    assert not math.isnan(real)
    assert not math.isnan(imag)
    assert flag[0][0] == 1


def test_mixed_flags_and_weights(df_mixed_flags):
    result = _prepare(df_mixed_flags).toPandas()

    vis = result.loc[0, 'visibility']
    flag = result.loc[0, 'flag']
    real, imag = vis[0][0]
    
    expected_real = (6.0 * 2.0 + 8.0 * 3.0) / (2.0 + 3.0)
    expected_imag = (3.0 * 2.0 + 4.0 * 3.0) / (2.0 + 3.0)
    
    assert math.isclose(real, expected_real, abs_tol=1e-9)
    assert math.isclose(imag, expected_imag, abs_tol=1e-9)
    assert flag[0][0] == 0


def test_flag_propagation_rule(df_partially_flagged):
    result = _prepare(df_partially_flagged).toPandas()
    flag = result.loc[0, 'flag']
    assert flag[0][0] == 0


# Test cases for determinism

# ...existing code...

# ==============================================================================
# Test cases for determinism
# ==============================================================================

def test_same_input_produces_same_output(df_multi_baselines):
    result1 = _prepare(df_multi_baselines).toPandas()
    result2 = _prepare(df_multi_baselines).toPandas()

    assert len(result1) == len(result2)
    
    for i in range(len(result1)):
        assert result1.loc[i, 'visibility'] == result2.loc[i, 'visibility']
        assert result1.loc[i, 'weight'] == result2.loc[i, 'weight']
        assert result1.loc[i, 'flag'] == result2.loc[i, 'flag']
        assert result1.loc[i, 'window_id'] == result2.loc[i, 'window_id']


def test_row_order_with_explicit_sort(df_unordered_rows):
    result_unordered = _prepare(df_unordered_rows).toPandas()
    
    pdf_ordered = df_unordered_rows.toPandas().sort_values('time').reset_index(drop=True)
    result_ordered = assign_windows(
        pdf_ordered, decorr_factor=DECORR_FACTOR, fov=FOV, lambda_ref=LAMBDA_REF
    )
    result_ordered = average_visibilities(result_ordered)
    
    vis_unordered = result_unordered.loc[0, 'visibility']
    vis_ordered = result_ordered.loc[0, 'visibility']
    
    assert vis_unordered == vis_ordered


def test_window_assignment_deterministic(df_multi_baselines):
    df_windowed1 = assign_temporal_window(
        df_multi_baselines, DECORR_FACTOR, FOV, LAMBDA_REF
    ).toPandas()
    
    df_windowed2 = assign_temporal_window(
        df_multi_baselines, DECORR_FACTOR, FOV, LAMBDA_REF
    ).toPandas()
    
    assert df_windowed1['window_id'].tolist() == df_windowed2['window_id'].tolist()


def test_baseline_processing_order_independence(df_multi_baselines):
    result1 = _prepare(df_multi_baselines).toPandas()
    
    pdf_reversed = df_multi_baselines.toPandas().iloc[::-1].reset_index(drop=True)
    df_reversed = df_multi_baselines.sql_ctx.createDataFrame(
        pdf_reversed, schema=df_multi_baselines.schema
    )
    result2 = _prepare(df_reversed).toPandas()
    
    result1_sorted = result1.sort_values(['baseline_key', 'time']).reset_index(drop=True)
    result2_sorted = result2.sort_values(['baseline_key', 'time']).reset_index(drop=True)

    assert len(result1_sorted) == len(result2_sorted)
    for i in range(len(result1_sorted)):
        assert result1_sorted.loc[i, 'visibility'] == result2_sorted.loc[i, 'visibility']
        assert result1_sorted.loc[i, 'baseline_key'] == result2_sorted.loc[i, 'baseline_key']


# Test error handling

def test_assign_windows_missing_column_raises_error(df_single_row):
    pdf = df_single_row.toPandas()
    pdf_incomplete = pdf.drop(columns=['u'])
    
    with pytest.raises(ValueError, match="Missing required columns for window assignment"):
        assign_windows(pdf_incomplete, DECORR_FACTOR, FOV, LAMBDA_REF)


def test_assign_windows_empty_dataframe_handling(spark, visibility_schema):
    df_empty = spark.createDataFrame([], schema=visibility_schema)
    empty_pdf = df_empty.toPandas()
    result = assign_windows(empty_pdf, DECORR_FACTOR, FOV, LAMBDA_REF)
    
    assert len(result) == 0


def test_assign_temporal_window_empty_dataframe_handling(spark, visibility_schema):
    df_empty = spark.createDataFrame([], schema=visibility_schema)
    
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty or None."):
        assign_temporal_window(df_empty, DECORR_FACTOR, FOV, LAMBDA_REF)


def test_average_temporal_window_missing_column_raises_error(df_single_row):
    df_incomplete = df_single_row.drop("baseline_key", "scan_number")
    with pytest.raises(Exception):
        average_by_window(df_incomplete)


def test_average_visibilities_empty_group(spark, visibility_schema):
    df_empty = spark.createDataFrame([], schema=visibility_schema)
    empty_pdf = df_empty.toPandas()
    
    with pytest.raises(ValueError, match="Cannot average empty group"):
        average_visibilities(empty_pdf)


def test_average_by_window_missing_window(df_single_row):
    with pytest.raises(ValueError, match="Input DataFrame must contain 'window_id' column for averaging."):
        average_by_window(df_single_row)


def test_average_by_window_missing_required_columns(df_single_row):
    df_incomplete = df_single_row.drop("baseline_key", "scan_number", "window_id")
    with pytest.raises(Exception):
        average_by_window(df_incomplete)
