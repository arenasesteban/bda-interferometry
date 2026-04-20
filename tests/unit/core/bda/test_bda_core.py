import math
import numpy as np

from core.bda.bda_core import (
    calculate_uv_distance,
    calculate_phase_difference, 
    sinc
)

# Test cases for calculate_uv_distance function

def test_same_coordinates_returns_zero():
    result = calculate_uv_distance(10.0, 20.0, 10.0, 20.0, lambda_ref=0.1)
    assert result == 0.0


def test_known_value_axis_u():
    result = calculate_uv_distance(0.0, 0.0, 3.0, 0.0, lambda_ref=1.0)
    assert math.isclose(result, 3.0, rel_tol=1e-9)


def test_known_value_axis_v():
    result = calculate_uv_distance(0.0, 0.0, 0.0, 4.0, lambda_ref=1.0)
    assert math.isclose(result, 4.0, rel_tol=1e-9)


def test_negative_coordinates_returns_positive():
    result = calculate_uv_distance(-3.0, -4.0, 0.0, 0.0, lambda_ref=0.1)
    assert result > 0.0


def test_symetryc_displacement():
    d_forward = calculate_uv_distance(1.0, 2.0, 4.0, 6.0, lambda_ref=0.1)
    d_backward = calculate_uv_distance(4.0, 6.0, 1.0, 2.0, lambda_ref=0.1)
    assert math.isclose(d_forward, d_backward, rel_tol=1e-9)


def test_return_type_float():
    result = calculate_uv_distance(1.0, 2.0, 3.0, 4.0, lambda_ref=0.1)
    assert isinstance(result, float)


# Test cases for calculate_phase_difference function

def test_zero_uv_distance_returns_zero():
    result = calculate_phase_difference(0.0, fov=1.0)
    assert result == 0.0

def test_zero_fov_returns_zero():
    result = calculate_phase_difference(1.0, fov=0.0)
    assert result == 0.0

def test_known_value():
    expected = 2.0 * math.pi * 10.0 * 0.0001
    result = calculate_phase_difference(10.0, fov=0.0001)
    assert math.isclose(result, expected, rel_tol=1e-9)

def test_linear_in_duv_and_fov():
    phi = calculate_phase_difference(5.0, fov=0.01)
    phi_scaled_duv = calculate_phase_difference(10.0, fov=0.01)
    phi_scaled_fov = calculate_phase_difference(5.0, fov=0.02)

    assert math.isclose(phi_scaled_duv, 2 * phi, rel_tol=1e-9)
    assert math.isclose(phi_scaled_fov, 2 * phi, rel_tol=1e-9)

def test_return_type_float():
    result = calculate_phase_difference(1.0, fov=0.1)
    assert isinstance(result, (float, np.floating))


# Test cases for sinc function

def test_sinc_zero_returns_one():
    result = sinc(0.0)
    assert result == 1.0


def test_sinc_returns_one_for_very_small_values():
    result = sinc(1e-9)
    assert result == 1.0


def test_sinc_known_value():
    x = 0.5
    expected = math.sin(x) / x
    result = sinc(x)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_sinc_symmetry():
    x = 0.5
    result_positive = sinc(x)
    result_negative = sinc(-x)
    assert math.isclose(result_positive, result_negative, rel_tol=1e-9)