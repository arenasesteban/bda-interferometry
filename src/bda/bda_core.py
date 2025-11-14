import numpy as np
import traceback
import astropy.units as units
from astropy.time import Time
from astropy.coordinates import EarthLocation


def calculate_amplitude_loss(x):
    try:
        x = float(np.float64(x))
        sinc_value = np.sinc(x / np.pi)

        return float(1.0 - sinc_value)

    except Exception as e:
        print(f"Error calculating amplitude loss: {e}")
        traceback.print_exc()
        raise


def calculate_loss_exact(decorr_limit):
    try:
        x = float(np.float64(decorr_limit))

        x_min, x_max = 0.0, np.pi
        f_min = calculate_amplitude_loss(x_min) - x
        f_max = calculate_amplitude_loss(x_max) - x

        if not (f_min * f_max < 0):
            return None
        
        while True:
            x_mid = (x_min + x_max) / 2.0
            f_mid = calculate_amplitude_loss(x_mid) - x

            if f_mid == 0.0:
                return x_mid
            
            ulp_space = np.spacing(x_mid)
            if (x_max - x_min) <= ulp_space:
                return x_mid

            if f_min * f_mid < 0:
                x_max = x_mid
                f_max = f_mid
            else:
                x_min = x_mid
                f_min = f_mid

    except Exception as e:
        print(f"Error calculating loss exact: {e}")
        traceback.print_exc()
        raise


def calculate_threshold_loss(x_exact):
    try:
        x_arr = np.array(x_exact, dtype=np.float64)
        
        return np.nextafter(x_arr, np.inf)

    except Exception as e:
        print(f"Error calculating threshold loss: {e}")
        traceback.print_exc()
        raise


def calculate_phase_rate(u_dot, v_dot, lambda_, min_diameter):
    try:
        fov = 1.02 * (lambda_ / min_diameter)
        phi_dot = (2.0 * np.pi) * (np.sqrt(u_dot ** 2 + v_dot ** 2) * fov)

        return phi_dot

    except Exception as e:
        print(f"Error calculating phase velocity: {e}")
        traceback.print_exc()
        raise


def calculate_uv_rate(time, Lx, Ly, dec, ra, lambda_, longitude, latitude):
    try:
        angular_velocity_earth = 7.2921150e-5

        time_utc = Time(time, format='mjd', scale='utc')
        location = EarthLocation(lon=longitude * units.deg, lat=latitude * units.deg)
        lst = time_utc.sidereal_time('apparent', longitude=location.lon)

        lst_rad = lst.to(units.rad).value
        HA = lst_rad - ra

        u_dot = ((Lx * np.cos(HA)) - (Ly * np.sin(HA))) * angular_velocity_earth / lambda_
        v_dot = ((Lx * np.sin(dec) * np.sin(HA)) + (Ly * np.sin(dec) * np.cos(HA))) * angular_velocity_earth / lambda_

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

        N, C, P, _ = vs.shape
        valid_mask = ~fs.astype(bool)

        ws_broadcast = ws[:, None, :]
        ws_broadcast = np.broadcast_to(ws_broadcast, (N, C, P))
        ws_valid = np.where(valid_mask, ws_broadcast, 0.0)
        ws_sum = ws_valid.sum(axis=0)

        vs_real = (vs[..., 0] * ws_valid).sum(axis=0)
        vs_imag = (vs[..., 1] * ws_valid).sum(axis=0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            real_avg = np.where(ws_sum > 0, vs_real / ws_sum, 0.0)
            imag_avg = np.where(ws_sum > 0, vs_imag / ws_sum, 0.0)
        
        valid_count = (ws_valid > 0).sum(axis=0)

        vs_avg = np.stack([real_avg, imag_avg], axis=-1)
        ws_avg = np.where(valid_count > 0, ws_sum / valid_count, 0.0).mean(axis=0)
        fs_avg = (ws_sum == 0).astype(np.int32)

        return vs_avg.tolist(), ws_avg.tolist(), fs_avg.tolist()

    except Exception as e:
        print(f"Error averaging visibilities: {e}")
        traceback.print_exc()
        raise


def average_uv(u, v, exposure):
    try:
        us = np.array(u, dtype=np.float64)
        vs = np.array(v, dtype=np.float64)

        if exposure is None:
            u_avg = np.mean(us)
            v_avg = np.mean(vs)
        else:
            u_avg = np.sum(us * exposure) / np.sum(exposure)
            v_avg = np.sum(vs * exposure) / np.sum(exposure)

        return float(u_avg), float(v_avg)

    except Exception as e:
        print(f"Error averaging fields: {e}")
        traceback.print_exc()
        raise
