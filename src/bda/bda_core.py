import numpy as np
import traceback
import math


def calculate_uv_distance(u1, v1, u2, v2):
    """
    Calculate Euclidean distance in the UV plane.

    Parameters
    ----------
    u1, v1 : float
        Reference coordinates (meters).
    u2, v2 : float
        Current coordinates (meters).

    Returns
    -------
    float
        Calculated UV distance.
    """
    try:
        # Calculate UV distance
        # Δuv = √[(u - u_ref)² + (v - v_ref)²]
        d_uv = math.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)

        return d_uv

    except Exception as e:
        print(f"Error calculating UV distance: {e}")
        traceback.print_exc()
        raise


def calculate_phase_difference(d_uv, fov):
    """
    Calculate the phase difference.

    Parameters
    ----------
    d_uv : float
        UV distance in wavelengths.
    fov : float
        Field of view (radians).
    
    Returns
    -------
    float
        Calculated phase difference.
    """
    try:
        # Calculate phase difference
        # ΔΦ = 2π × Δuv × FOV
        phi_dot = 2 * np.pi * d_uv * fov

        return phi_dot

    except Exception as e:
        print(f"Error calculating phase difference: {e}")
        traceback.print_exc()
        raise


def sinc(x):
    """
    Calculate the sinc function value.

    Parameters
    ----------
    x : float
        Input value.

    Returns
    -------
    float
        Sinc function value.
    """
    try:
        if x < 1e-8:
            return 1.0
        else:
            return np.sin(x) / x

    except Exception as e:
        print(f"Error calculating sinc function: {e}")
        traceback.print_exc()
        raise