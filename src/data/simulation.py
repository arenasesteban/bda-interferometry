import os
import numpy as np
import dask.array as da
import datetime
import traceback

from pyralysis.io.antenna_config_io import AntennaConfigurationIo
from pyralysis.models.sky import PointSource, CompositeSource, NonParametricSource
from pyralysis.simulation.core import Simulator
from pyralysis.io.fits import FITS

from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
from astropy.constants import c

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


def configure_observation(interferometer, freq_min, freq_max, n_chans, observation_time, declination, integration_time, date_string=None):
    """
    Configure observation parameters for the interferometer.

    Parameters
    ----------
    interferometer : Interferometer
        The interferometer instance.
    freq_min : float
        Minimum frequency in GHz.
    freq_max : float
        Maximum frequency in GHz.
    n_chans : int
        Number of frequency channels.
    observation_time : float
        Total observation time in seconds.
    declination : float
        Declination of the observation in degrees.
    integration_time : float
        Integration time per sample in seconds.
    date_string : str, optional
        Date string for the observation.

    Returns
    -------
    float
        Reference frequency in GHz.
    """
    try:
        if interferometer is None:
            raise ValueError("Interferometer cannot be None")

        # Create frequency array
        freq = np.linspace(freq_min, freq_max, n_chans)

        if date_string is None:
            freq = freq * u.MHz
            date_string = datetime.datetime.now().strftime("%Y-%m-%d")

            # Configure interferometer
            interferometer.configure_observation(
                frequencies=freq,
                reference_time=Time(date_string, format='iso'),
                observation_time=observation_time,
                declination=Angle(declination),
                frequency_step=None,
                integration_time=integration_time * u.s,
            )
        
        else:
            freq = freq * u.GHz
            step = (freq[1] - freq[0]).to(u.Hz).value
            
            # Configure interferometer
            interferometer.configure_observation(
                hour_angle="transit",
                observation_time=observation_time,
                declination=Angle(declination),
                min_frequency=freq_min*1e9,
                max_frequency=freq_max*1e9,
                frequency_step=step,
                integration_time=integration_time * u.s,
            )
        
        ref_freq = np.median(freq)

        return freq, ref_freq
    
    except Exception as e:
        print(f"Error in configure_observation: {e}")
        traceback.print_exc()
        raise


def generate_point_sources(ref_freq, source_path=None, freq=None, interferometer=None, flux_density=None, spectral_index=None):
    """
    Generate point sources for the simulation.

    Parameters
    ----------
    ref_freq : float
        Reference frequency in GHz.
    source_path : str
        Path to the source FITS file.

    Returns
    -------
    list
        List of generated point sources.
    """
    try:
        if ref_freq is None:
            raise ValueError("Reference frequency must be configured first")

        sources = []

        if source_path is not None:
            io_fits = FITS(source_path)
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
                    spectral_index=spectral_index,
                    reference_frequency=ref_freq,
                    direction_cosines=(l_0[i], m_0[i])
                )
                sources.append(source)
        else:
            nodim_freq = freq.to(u.Hz).value
            diameter = interferometer.antenna_array.diameters.compute()

            fov = (c.value / nodim_freq[0]) / diameter[0]

            n_sources = np.random.randint(1, high=20, size=1, dtype=int)[0]

            s_0 = flux_density * u.mJy
            l_0 = np.random.uniform(low=-fov, high=fov, size=n_sources)
            m_0 = np.random.uniform(low=-fov, high=fov, size=n_sources)

            for i in range(n_sources):
                source = PointSource(
                    reference_intensity=s_0,
                    spectral_index=spectral_index,
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


def generate_dataset(antenna_config_path,
                     freq_min, freq_max, n_chans, 
                     observation_time, declination, integration_time,
                     source_path=None, date_string=None,
                     flux_density=None, spectral_index=None):
    """
    Generate a simulated dataset based on the provided parameters.

    Parameters
    ----------
    antenna_config_path : str
        Path to the antenna configuration JSON file.
    freq_min : float
        Minimum frequency in GHz.
    freq_max : float
        Maximum frequency in GHz.
    n_chans : int
        Number of frequency channels.
    observation_time : float
        Total observation time in seconds.
    declination : float
        Declination of the observation in degrees.
    integration_time : float
        Integration time per sample in seconds.
    source_path : str
        Path to the source FITS file.
    
    Returns
    -------
    Dataset
        The generated simulated dataset.
    """
    # Load antenna configuration
    interferometer = load_antenna_configuration(antenna_config_path)
    
    # Configure observation parameters
    freq, ref_freq = configure_observation(
        interferometer,
        freq_min,
        freq_max,
        n_chans,
        observation_time,
        declination,
        integration_time,
        date_string=date_string
    )

    # Generate point sources
    if source_path is not None:
        sources = generate_point_sources(ref_freq, source_path, spectral_index)
    
    else:
        sources = generate_point_sources(
            ref_freq,
            source_path,
            freq,
            interferometer,
            flux_density,
            spectral_index,
        )

    # Create composite source model
    composite_source = CompositeSource(sources=sources)

    # Run simulation
    dataset = simulate_dataset(interferometer, sources=composite_source)

    return dataset