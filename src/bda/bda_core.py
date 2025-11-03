import numpy as np
import traceback
import astropy.units as units
from astropy.time import Time
from astropy.coordinates import EarthLocation


def calculate_decorrelation_time(u_dot, v_dot, decorr_factor, field_offset_deg):
    try:
        theta = np.radians(field_offset_deg)

        l = np.sin(theta)
        m = 0.0

        phi_dot = np.pi * (u_dot * l + v_dot * m)

        T_eff = np.sqrt(6 * (1.0 - decorr_factor)) / abs(phi_dot)

        return float(T_eff)

    except Exception as e:
        print(f"Error calculating decorrelation time: {e}")
        traceback.print_exc()
        raise


def calculate_uv_rate(time, Lx, Ly, dec, ra, lambda_, longitude, latitude):
    try:
        angular_velocity_earth = 7.2921150e-5

        time_utc = Time(time, format='mjd', scale='utc')
        location = EarthLocation(lon=longitude * units.deg, lat=latitude * units.deg)
        lst = time_utc.sidereal_time('apparent', longitude=location.lon)

        lst_rad = lst.to(units.rad).value
        hour_angle = lst_rad - ra

        u_dot = ((Lx * np.cos(hour_angle)) - (Ly * np.sin(hour_angle))) * angular_velocity_earth / lambda_
        v_dot = ((Lx * np.sin(dec) * np.sin(hour_angle)) + (Ly * np.sin(dec)) * np.cos(hour_angle)) * angular_velocity_earth / lambda_

        return float(u_dot), float(v_dot)

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

        N, C, P = fs.shape
        ws_bc = np.broadcast_to(ws[:, None, :], (N, C, P))
        ws_valid = np.where(masked_fs, ws_bc, 0.0)
        ws_sum = ws_valid.sum(axis=0)

        vs_real = (vs[..., 0] * ws_valid).sum(axis=0)
        vs_imag = (vs[..., 1] * ws_valid).sum(axis=0)
        
        eps = 1e-12
        with np.errstate(divide='ignore', invalid='ignore'):
            real_avg = np.where(ws_sum > eps, vs_real / ws_sum, 0)
            imag_avg = np.where(ws_sum > eps, vs_imag / ws_sum, 0)
        
        fs_valid = ~(fs.astype(bool)).any(axis=1)

        vs_avg = np.stack([real_avg, imag_avg], axis=-1)
        ws_avg = (ws * fs_valid).sum(axis=0)
        fs_avg = (ws_sum < eps).astype(int)

        return vs_avg.tolist(), ws_avg.tolist(), fs_avg.tolist()

    except Exception as e:
        print(f"Error averaging visibilities: {e}")
        traceback.print_exc()
        raise


def average_uv(u, v):
    try:
        us = np.array(u, dtype=np.float64)
        vs = np.array(v, dtype=np.float64)

        u_avg = np.mean(us)
        v_avg = np.mean(vs)

        return float(u_avg), float(v_avg)

    except Exception as e:
        print(f"Error averaging fields: {e}")
        traceback.print_exc()
        raise
