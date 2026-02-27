import os
import numpy as np
import dask.array as da
import datetime
import traceback
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.constants import c

from pyralysis.io.antenna_config_io import AntennaConfigurationIo
from pyralysis.simulation.antenna_configs import get_ska_station_list
from pyralysis.models.sky import PointSource, CompositeSource, NonParametricSource
from pyralysis.simulation.core import Simulator
from pyralysis.io.fits import FITS

da.random.seed(42)
np.random.seed(42)


def load_antenna_configuration(antenna_config_path):
    """
    Load antenna configuration from a JSON file.

    Parameters
    ----------
    antenna_config_path : str
        Path to the antenna configuration JSON file.

    Returns
    -------
    dict
        Antenna configuration data.
    """
    try:
        if not os.path.exists(antenna_config_path):
            raise FileNotFoundError(f"Antenna config not found: {antenna_config_path}")
        
        io_instance = AntennaConfigurationIo(input_name=antenna_config_path)
        return io_instance.read()
    
    except Exception as e:
        print(f"Error in load_antenna_configuration: {e}")
        traceback.print_exc()
        raise


def filter_antenna_configuration(interferometer, array_type, assembly):
    try:
        if assembly != "AA4":
            array_ids = get_ska_station_list(array_type, assembly)
            interferometer.antenna_array.filter_by_ids(array_ids, inplace=True)
    
    except Exception as e:
        print(f"Error in filter_antenna_configuration: {e}")
        traceback.print_exc()
        raise


def configure_observation(interferometer, sim_config):
    try:
        if interferometer is None:
            raise ValueError("Interferometer cannot be None")

        # Create frequency array
        freq_min = sim_config["freq_min"]
        freq_max = sim_config["freq_max"]
        n_chans = sim_config["n_chans"]

        freq = np.linspace(freq_min, freq_max, n_chans)

        if sim_config["interferometer"] == "SKA":
            freq = freq * u.MHz
            date_string = datetime.datetime.now().strftime("%Y-%m-%d")

            # Configure interferometer
            interferometer.configure_observation(
                frequencies=freq,
                reference_time=Time(date_string, format='iso'),
                observation_time=sim_config["observation_time"],
                declination=Angle(sim_config["declination"]),
                frequency_step=None,
                integration_time=sim_config["integration_time"] * u.s,
            )
        
        else:
            freq = freq * u.GHz
            step = (freq[1] - freq[0]).to(u.Hz).value
            
            # Configure interferometer
            interferometer.configure_observation(
                hour_angle="transit",
                observation_time=sim_config["observation_time"],
                declination=Angle(sim_config["declination"]),
                min_frequency=freq_min * 1e9,
                max_frequency=freq_max * 1e9,
                frequency_step=step,
                integration_time=sim_config["integration_time"] * u.s,
            )
        
        ref_freq = np.median(freq)

        return freq, ref_freq
    
    except Exception as e:
        print(f"Error in configure_observation: {e}")
        traceback.print_exc()
        raise


def generate_point_sources(ref_freq, freq, sim_config, interferometer):
    try:
        if ref_freq is None:
            raise ValueError("Reference frequency must be configured first")

        sources = []

        if sim_config["interferometer"] == "SKA":
            io_fits = FITS(sim_config["source_path"])
            image = io_fits.read()

            cellsize_rad = image.cellsize[1].to(u.rad).value
            n_pixels_fov = image.shape[-1]
            
            non_parametric_source = NonParametricSource(image=image, direction_cosines=(0, 0))
            sources.append(non_parametric_source)

            n_sources = np.random.randint(8, high=15, size=1, dtype=int)[0]
            s_0 = 0.15 * u.Jy

            pixel_coords_l = np.random.randint(-n_pixels_fov//2, n_pixels_fov//2, size=n_sources)
            pixel_coords_m = np.random.randint(-n_pixels_fov//2, n_pixels_fov//2, size=n_sources)

            l_0 = pixel_coords_l * cellsize_rad
            m_0 = pixel_coords_m * cellsize_rad
            
            for i in range(n_sources):
                source = PointSource(
                    reference_intensity=s_0,
                    spectral_index=sim_config["spectral_index"],
                    reference_frequency=ref_freq,
                    direction_cosines=(l_0[i], m_0[i])
                )
                sources.append(source)
        else:
            nodim_freq = freq.to(u.Hz).value
            diameter = interferometer.antenna_array.diameters.compute()

            fov = (c.value / nodim_freq[0]) / diameter[0]

            n_sources = np.random.randint(1, high=20, size=1, dtype=int)[0]

            s_0 = sim_config["flux_density"] * u.mJy
            l_0 = np.random.uniform(low=-fov, high=fov, size=n_sources)
            m_0 = np.random.uniform(low=-fov, high=fov, size=n_sources)

            for i in range(n_sources):
                source = PointSource(
                    reference_intensity=s_0,
                    spectral_index=sim_config["spectral_index"],
                    reference_frequency=ref_freq,
                    direction_cosines=([l_0[i]], [m_0[i]])
                )
                sources.append(source)
            
        return sources

    except Exception as e:
        print(f"Error in generate_point_sources: {e}")
        traceback.print_exc()
        raise


def simulate_dataset(interferometer, sources):
    """
    Simulate a dataset using the given interferometer and sources.

    Parameters
    ----------
    interferometer : Interferometer
        The interferometer instance.
    sources : list
        List of point sources to simulate.

    Returns
    -------
    Dataset
        The simulated dataset.
    """
    if interferometer is None:
        raise ValueError("Interferometer cannot be None")
    
    if sources is None:
        raise ValueError("Sources cannot be None")

    sim = Simulator(interferometer=interferometer, sources=sources)
    dataset = sim.simulate()

    return dataset


def generate_dataset(antenna_config_path, sim_config):
    # Load antenna configuration
    interferometer = load_antenna_configuration(antenna_config_path)


    if sim_config["interferometer"] == "SKA":
        filter_antenna_configuration(interferometer, sim_config["array_type"], sim_config["assembly"])

    interferometer_name = f"{sim_config['interferometer']} {sim_config['array_type'] + ' ' + sim_config['assembly'] if sim_config['interferometer'] == 'SKA' else ''}"
    n_antennas = interferometer.antenna_array.positions.shape[0]
    n_baselines = n_antennas * (n_antennas - 1) // 2

    print(f"[Simulation] Interferometer - {interferometer_name}", flush=True)
    print(f"[Simulation] Antenna positions: {n_antennas}", flush=True)
    print(f"[Simulation] Number of baselines: {n_baselines}", flush=True)

    # Configure observation parameters
    freq, ref_freq = configure_observation(
        interferometer,
        sim_config
    )

    # Generate point sources
    sources = generate_point_sources(
        interferometer=interferometer,
        sim_config=sim_config,
        ref_freq=ref_freq, 
        freq=freq,
    )

    # Create composite source model
    composite_source = CompositeSource(sources=sources)

    # Run simulation
    dataset = simulate_dataset(interferometer, sources=composite_source)

    return dataset, interferometer