"""
Simulation - Radio Interferometry Dataset Generation

Generates simulated radio interferometry datasets using the Pyralysis framework.
Creates synthetic visibility data with configurable antenna arrays, source models,
and observation parameters for testing and development of data processing pipelines.

This module provides a complete end-to-end simulation pipeline from antenna
configuration loading through source model generation to final dataset creation.
"""

import os
import numpy as np
import dask.array as da
from typing import List, Tuple, Optional

from pyralysis.io.antenna_config_io import AntennaConfigurationIo
from pyralysis.models.sky import PointSource, GaussianSource, CompositeSource
from pyralysis.simulation.core import Simulator

from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
from astropy.constants import c

da.random.seed(777)
np.random.seed(777)


def load_antenna_configuration(antenna_config_path: str) -> object:
    """
    Load antenna configuration from file.
    
    Reads antenna array configuration using Pyralysis AntennaConfigurationIo
    and returns the configured interferometer object.
    
    Parameters
    ----------
    antenna_config_path : str
        Path to the antenna configuration file
        
    Returns
    -------
    object
        Configured interferometer object from Pyralysis
        
    Raises
    ------
    FileNotFoundError
        If the antenna configuration file does not exist
    """
    if not os.path.exists(antenna_config_path):
        raise FileNotFoundError(f"Antenna config not found: {antenna_config_path}")
    
    io_instance = AntennaConfigurationIo(input_name=antenna_config_path)
    return io_instance.read()


def configure_observation(interferometer: object,
                          freq_start: float,
                          freq_end: float,
                          n_frequencies: int,
                          date_string: str,
                          observation_time: str,
                          declination: str,
                          integration_time: float) -> Tuple[object, object, object]:
    """
    Configure observation parameters for interferometer.
    
    Sets up the observation configuration including frequency ranges,
    time parameters, and pointing direction. Returns the configured
    interferometer along with frequency information.
    
    Parameters
    ----------
    interferometer : object
        Pyralysis interferometer object to configure
    freq_start : float
        Start frequency in GHz
    freq_end : float
        End frequency in GHz
    n_frequencies : int
        Number of frequency channels
    date_string : str
        Observation date in ISO format
    observation_time : str
        Total observation duration
    declination : str
        Declination angle string
    integration_time : float
        Integration time in seconds
        
    Returns
    -------
    tuple
        Tuple containing (configured_interferometer, freq_array, reference_frequency)
        
    Raises
    ------
    ValueError
        If interferometer is None
    """
    if interferometer is None:
        raise ValueError("Interferometer cannot be None")
        
    # Create frequency array
    freq = np.linspace(freq_start, freq_end, n_frequencies) * u.GHz
    ref_freq = np.median(freq)
    
    # Configure interferometer
    interferometer.configure_observation(
        frequencies=freq,
        reference_time=Time(date_string, format='iso'),
        observation_time=observation_time,
        declination=Angle(declination),
        frequency_step_hz=None,
        integration_time=integration_time * u.s,
    )
    
    return interferometer, freq, ref_freq


def calculate_field_of_view(interferometer: object, freq: object) -> float:
    """
    Calculate field of view based on frequency and antenna diameter.
    
    Computes the primary beam field of view using the standard formula
    FOV = lambda / D, where lambda is the wavelength and D is the antenna diameter.
    
    Parameters
    ----------
    interferometer : object
        Configured interferometer object
    freq : object
        Frequency array with astropy units
        
    Returns
    -------
    float
        Field of view in radians
        
    Raises
    ------
    ValueError
        If interferometer or freq is None
    """
    if interferometer is None or freq is None:
        raise ValueError("Interferometer and frequency must be configured first")
        
    nodim_freq = freq.to(u.Hz).value
    diameter = interferometer.antenna_array.diameters.compute()
    
    # FOV = lambda / D (in radians)
    fov = (c.value / nodim_freq[0]) / diameter[0]

    return fov


def generate_point_sources(ref_freq: object,
                          interferometer: object,
                          freq: object,
                          n_sources: int,
                          flux_density: float,
                          spectral_index: float) -> List[PointSource]:
    """
    Generate random point sources within the field of view.
    
    Creates a specified number of point sources with random positions
    distributed uniformly within the primary beam field of view.
    
    Parameters
    ----------
    ref_freq : object
        Reference frequency with astropy units
    interferometer : object
        Configured interferometer object
    freq : object
        Frequency array with astropy units
    n_sources : int
        Number of point sources to generate
    flux_density : float
        Flux density in mJy
    spectral_index : float
        Spectral index for frequency dependence
        
    Returns
    -------
    List[PointSource]
        List of generated PointSource objects
        
    Raises
    ------
    ValueError
        If reference frequency is None
    """
    if ref_freq is None:
        raise ValueError("Reference frequency must be configured first")
        
    fov = calculate_field_of_view(interferometer, freq)
    
    # Random positions within FOV
    l_coords = np.random.uniform(low=-fov, high=fov, size=n_sources)
    m_coords = np.random.uniform(low=-fov, high=fov, size=n_sources)
    
    sources = []
    for i in range(n_sources):
        source = PointSource(
            reference_intensity=flux_density * u.mJy,
            spectral_index=spectral_index,
            reference_frequency=ref_freq,
            direction_cosines=(l_coords[i], m_coords[i])
        )
        sources.append(source)
        
    return sources


def generate_gaussian_source(flux_density: float,
                            position: Tuple[float, float],
                            minor_radius: float,
                            major_radius: float,
                            theta_angle: float) -> GaussianSource:
    """
    Generate a Gaussian source with specified parameters.
    
    Creates a single extended source with Gaussian morphology at the
    specified position with given size and orientation parameters.
    
    Parameters
    ----------
    flux_density : float
        Flux density in Jy
    position : tuple of float
        (l, m) position in direction cosines
    minor_radius : float
        Minor axis radius in arcsec
    major_radius : float
        Major axis radius in arcsec
    theta_angle : float
        Position angle in degrees
        
    Returns
    -------
    GaussianSource
        Configured GaussianSource object
    """
    source = GaussianSource(
        reference_intensity=flux_density * u.Jy,
        direction_cosines=position,
        minor_radius=Angle(minor_radius * u.arcsec),
        major_radius=Angle(major_radius * u.arcsec),
        theta_angle=Angle(theta_angle * u.deg)
    )
    
    return source


def create_composite_source(point_sources: List[PointSource],
                           gaussian_source: Optional[GaussianSource] = None) -> CompositeSource:
    """
    Create a composite source with multiple components.
    
    Combines multiple point sources and optionally one Gaussian source
    into a single CompositeSource object for simulation.
    
    Parameters
    ----------
    point_sources : List[PointSource]
        List of point source objects
    gaussian_source : GaussianSource, optional
        Optional Gaussian source to include
        
    Returns
    -------
    CompositeSource
        Combined source object containing all components
    """
    sources = []
    
    # Add point sources
    sources.extend(point_sources)
    
    # Add Gaussian source if provided
    if gaussian_source is not None:
        sources.append(gaussian_source)
    
    return CompositeSource(sources=sources)


def simulate_dataset(interferometer: object, sources: CompositeSource) -> object:
    """
    Generate the simulated interferometry dataset.
    
    Creates a Pyralysis Simulator instance with the configured interferometer
    and source model, then runs the simulation to generate visibility data.
    
    Parameters
    ----------
    interferometer : object
        Configured interferometer object
    sources : CompositeSource
        Combined source model for simulation
        
    Returns
    -------
    object
        Simulated dataset from Pyralysis
        
    Raises
    ------
    ValueError
        If interferometer or sources is None
    """
    if interferometer is None:
        raise ValueError("Interferometer cannot be None")
    if sources is None:
        raise ValueError("Sources cannot be None")

    simulator = Simulator(interferometer=interferometer, sources=sources)
    dataset = simulator.simulate()

    return dataset


def generate_dataset(antenna_config_path: str,
                     freq_start: float,
                     freq_end: float,
                     n_frequencies: int,
                     date_string: str,
                     observation_time: str,
                     declination: str,
                     integration_time: float,
                     n_point_sources: int,
                     point_flux_density: float,
                     point_spectral_index: float,
                     include_gaussian: bool,
                     gaussian_flux_density: Optional[float] = None,
                     gaussian_position: Optional[Tuple[float, float]] = None,
                     gaussian_minor_radius: Optional[float] = None,
                     gaussian_major_radius: Optional[float] = None,
                     gaussian_theta_angle: Optional[float] = None) -> object:
    """
    Complete pipeline to generate a simulated interferometry dataset.
    
    This is the main entry point for generating simulated visibility data.
    It orchestrates the entire pipeline from antenna configuration loading
    through source generation to final dataset simulation.
    
    Parameters
    ----------
    antenna_config_path : str
        Path to antenna configuration file
    freq_start : float
        Start frequency in GHz
    freq_end : float
        End frequency in GHz
    n_frequencies : int
        Number of frequency channels
    date_string : str
        Observation date in ISO format
    observation_time : str
        Total observation duration
    declination : str
        Declination angle string
    integration_time : float
        Integration time in seconds
    n_point_sources : int
        Number of point sources to generate
    point_flux_density : float
        Point source flux density in mJy
    point_spectral_index : float
        Point source spectral index
    include_gaussian : bool
        Whether to include a Gaussian source
    gaussian_flux_density : float, optional
        Gaussian source flux density in Jy
    gaussian_position : tuple, optional
        Gaussian source (l, m) position
    gaussian_minor_radius : float, optional
        Gaussian minor radius in arcsec
    gaussian_major_radius : float, optional
        Gaussian major radius in arcsec
    gaussian_theta_angle : float, optional
        Gaussian position angle in degrees
        
    Returns
    -------
    object
        Complete simulated dataset from Pyralysis
    """
    
    # Load antenna configuration
    interferometer = load_antenna_configuration(antenna_config_path)
    
    # Configure observation parameters
    interferometer, freq, ref_freq = configure_observation(
        interferometer=interferometer,
        freq_start=freq_start,
        freq_end=freq_end,
        n_frequencies=n_frequencies,
        date_string=date_string,
        observation_time=observation_time,
        declination=declination,
        integration_time=integration_time
    )
    
    # Generate point sources
    point_sources = generate_point_sources(
        ref_freq=ref_freq,
        interferometer=interferometer,
        freq=freq,
        n_sources=n_point_sources,
        flux_density=point_flux_density,
        spectral_index=point_spectral_index
    )
    
    # Generate Gaussian source if requested
    gaussian_source = None
    if include_gaussian:
        gaussian_source = generate_gaussian_source(
            flux_density=gaussian_flux_density,
            position=gaussian_position,
            minor_radius=gaussian_minor_radius,
            major_radius=gaussian_major_radius,
            theta_angle=gaussian_theta_angle
        )
    
    # Create composite source model
    sources = create_composite_source(
        point_sources=point_sources,
        gaussian_source=gaussian_source
    )
    
    # Run simulation
    dataset = simulate_dataset(interferometer=interferometer, sources=sources)

    return dataset