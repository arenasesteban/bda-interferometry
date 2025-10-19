"""
BDA Core Scientific Algorithms

Implementation of fundamental scientific algorithms for baseline-dependent averaging
in radio interferometry data processing. Provides mathematical computations for
decorrelation time calculations, fringe rate analysis, temporal windowing, and
weighted visibility averaging based on Wijnholds et al. 2018 methodology.

Functions handle interferometry-specific calculations including baseline length
determination, optimal averaging time computation, and vectorized visibility
processing for distributed scientific computing environments.
"""

import numpy as np
from typing import Tuple
import traceback
from astropy.time import Time
from astropy.coordinates import EarthLocation


def calculate_decorrelation_time(u_dot, v_dot, decorr_factor, field_offset_deg):
    try:
        # Convertimos el ángulo del borde del FoV a radianes
        theta = np.radians(field_offset_deg)

        # Definimos la dirección de la fuente en coordenadas de imagen
        l = np.sin(theta)        # dirección horizontal (borde)
        m = 0.0                  # dirección vertical

        # Calculamos la tasa de fase efectiva para esta fila
        phi_dot = np.pi * (u_dot * l + v_dot * m)

        # Aproximación de pequeña pérdida (Taylor de sinc) para resolver T
        T_eff = np.sqrt(6 * (1.0 - decorr_factor)) / abs(phi_dot)

        return T_eff

    except Exception as e:
        print(f"Error calculating decorrelation time: {e}")
        traceback.print_exc()
        raise


def calculate_uv_rate(time, u, v, lambda_ref, declination_deg, longitude, ascension) -> Tuple[float, float]:
    try:
        angular_velocity_earth = 7.2921150e-5

        time_utc = Time(time, format='mjd', scale='utc')
        location = EarthLocation(lon=longitude * u.deg)
        lst = time_utc.sidereal_time('apparent', longitude=location.lon)

        lst_rad = lst * np.pi / 12
        hour_angle = lst_rad - ascension

        u_dot = ((u * np.cos(hour_angle)) - (v * np.sin(hour_angle))) * angular_velocity_earth / lambda_ref
        v_dot = ((u * np.sin(declination_deg) * np.sin(hour_angle)) + (v * np.sin(declination_deg)) * np.cos(hour_angle)) * angular_velocity_earth / lambda_ref

        return u_dot, v_dot

    except Exception as e:
        print(f"Error calculating uv rates: {e}")
        traceback.print_exc()
        raise


def average_visibilities(visibilities, weights, flags):
    try:
        vs = np.stack(visibilities, axis=0)  # Shape: (N, C, P, 2)
        ws = np.stack(weights, axis=0)       # Shape: (N, P)
        fs = np.stack(flags, axis=0)         # Shape: (N, C, P)
        
        masked_fs = ~fs
        count = masked_fs.sum(axis=0)

        vs_real = (vs[..., 0] * masked_fs).sum(axis=0)
        vs_imag = (vs[..., 1] * masked_fs).sum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            real_avg = np.where(count > 0, vs_real / count, 0)
            imag_avg = np.where(count > 0, vs_imag / count, 0)

        N, C, P = fs.shape
        ws_bc = np.broadcast_to(ws[:, None, :], (N, C, P))
        ws_valid = np.where(masked_fs, ws_bc, 0.0)

        vs_avg = np.stack([real_avg, imag_avg], axis=-1)
        ws_avg = ws_valid.sum(axis=0)
        fs_avg = (count == 0).astype(np.int8)

        return vs_avg, ws_avg, fs_avg

    except Exception as e:
        print(f"Error averaging visibilities: {e}")
        traceback.print_exc()
        raise


def average_fields(u, v, w, time, flags):
    try:
        us = np.array(u, dtype=np.float64)
        vs = np.array(v, dtype=np.float64)
        ws = np.array(w, dtype=np.float64)
        times = np.array(time, dtype=np.float64)

        fs = np.array(flags)
        masked_fs = ~fs

        rows = (masked_fs).any(axis=(1, 2))

        if np.any(rows):
            u_avg = np.mean(us[rows])
            v_avg = np.mean(vs[rows])
            w_avg = np.mean(ws[rows])
            time_avg = np.mean(times[rows])

        return u_avg, v_avg, w_avg, time_avg

    except Exception as e:
        print(f"Error averaging fields: {e}")
        traceback.print_exc()
        raise
