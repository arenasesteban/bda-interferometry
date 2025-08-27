"""
Dataset Generator for BDA Interferometry Pipeline

This module generates simulated interferometry datasets using Pyralysis.
It encapsulates the dataset generation logic from the notebook into a
reusable pipeline component.
"""

import os
import numpy as np
import dask.array as da
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

from pyralysis.io.antenna_config_io import AntennaConfigurationIo
from pyralysis.models.sky import PointSource, GaussianSource, CompositeSource
from pyralysis.simulation.core import Simulator

from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
from astropy.constants import c


@dataclass
class DatasetGenerator:
    """
    Generates simulated interferometry datasets using Pyralysis.
    
    This class encapsulates the configuration and generation of simulated
    visibility data for radio interferometry observations.
    """
    antenna_config_path: str
    interferometer: Optional[object] = field(default=None, init=False)
    sources: Optional[CompositeSource] = field(default=None, init=False)
    freq: Optional[object] = field(default=None, init=False)
    ref_freq: Optional[object] = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize random seeds and validate antenna config path."""
        # Set random seeds for reproducible simulations
        seed = 777
        da.random.seed(seed)
        np.random.seed(seed)
        
        if not os.path.exists(self.antenna_config_path):
            raise FileNotFoundError(f"Antenna config not found: {self.antenna_config_path}")
        
    def load_antenna_configuration(self) -> None:
        """Load antenna configuration from file."""
        io_instance = AntennaConfigurationIo(input_name=self.antenna_config_path)
        self.interferometer = io_instance.read()

    def configure_observation(self,
                              freq_start: float,
                              freq_end: float,
                              n_frequencies: int,
                              date_string: str,
                              observation_time: str,
                              declination: str,
                              integration_time: float) -> None:
        """
        Configure observation parameters.
        
        Args:
            freq_start: Start frequency in GHz
            freq_end: End frequency in GHz
            n_frequencies: Number of frequency channels
            date_string: Observation date (ISO format)
            observation_time: Total observation duration
            declination: Declination angle string
            integration_time: Integration time in seconds
        """
        if self.interferometer is None:
            raise ValueError("Must load antenna configuration first")
            
        # Create frequency array
        freq = np.linspace(freq_start, freq_end, n_frequencies) * u.GHz
        ref_freq = np.median(freq)
        
        # Configure interferometer
        self.interferometer.configure_observation(
            frequencies=freq,
            reference_time=Time(date_string, format='iso'),
            observation_time=observation_time,
            declination=Angle(declination),
            frequency_step_hz=None,
            integration_time=integration_time * u.s,
        )
        
        # Store frequency info for FOV calculation
        self.freq = freq
        self.ref_freq = ref_freq
        
    def calculate_fov(self) -> float:
        """Calculate field of view based on frequency and antenna diameter."""
        if self.interferometer is None or self.freq is None:
            raise ValueError("Must configure observation first")
            
        nodim_freq = self.freq.to(u.Hz).value
        diameter = self.interferometer.antenna_array.diameters.compute()
        
        # FOV = lambda / D (in radians)
        fov = (c.value / nodim_freq[0]) / diameter[0]

        return fov
        
    def generate_point_sources(self, 
                               n_sources: int,
                               flux_density: float,
                               spectral_index: float) -> List[PointSource]:
        """
        Generate random point sources within the field of view.
        
        Args:
            n_sources: Number of sources
            flux_density: Flux density in mJy
            spectral_index: Spectral index
            
        Returns:
            List of PointSource objects
        """
        if self.ref_freq is None:
            raise ValueError("Must configure observation first")
            
        fov = self.calculate_fov()
        
        # Random positions within FOV
        l_coords = np.random.uniform(low=-fov, high=fov, size=n_sources)
        m_coords = np.random.uniform(low=-fov, high=fov, size=n_sources)
        
        sources = []
        for i in range(n_sources):
            source = PointSource(
                reference_intensity=flux_density * u.mJy,
                spectral_index=spectral_index,
                reference_frequency=self.ref_freq,
                direction_cosines=(l_coords[i], m_coords[i])
            )
            sources.append(source)
            
        return sources
        
    def generate_gaussian_source(self,
                                 flux_density: float,
                                 position: Tuple[float, float],
                                 minor_radius: float,
                                 major_radius: float,
                                 theta_angle: float) -> GaussianSource:
        """
        Generate a Gaussian source.
        
        Args:
            flux_density: Flux density in Jy
            position: (l, m) position in direction cosines
            minor_radius: Minor axis radius in arcsec
            major_radius: Major axis radius in arcsec
            theta_angle: Position angle in degrees
            
        Returns:
            GaussianSource object
        """
        source = GaussianSource(
            reference_intensity=flux_density * u.Jy,
            direction_cosines=position,
            minor_radius=Angle(minor_radius * u.arcsec),
            major_radius=Angle(major_radius * u.arcsec),
            theta_angle=Angle(theta_angle * u.deg)
        )
        
        return source
        
    def create_composite_source(self,
                                point_sources: List[PointSource],
                                gaussian_source: Optional[GaussianSource] = None) -> CompositeSource:
        """
        Create a composite source with multiple components.
        
        Args:
            point_sources: List of point sources
            gaussian_source: Optional Gaussian source
            
        Returns:
            CompositeSource object
        """
        sources = []
        
        # Add point sources
        sources.extend(point_sources)
        
        # Add Gaussian source if provided
        if gaussian_source is not None:
            sources.append(gaussian_source)
        
        self.sources = CompositeSource(sources=sources)
        
        return self.sources
        
    def simulate_dataset(self):
        """
        Generate the simulated dataset.
        
        Returns:
            Simulated dataset from Pyralysis
        """
        if self.interferometer is None:
            raise ValueError("Must load antenna configuration first")
        if self.sources is None:
            raise ValueError("Must create sources first")

        simulator = Simulator(interferometer=self.interferometer, sources=self.sources)
        dataset = simulator.simulate()

        return dataset
        
    def generate_dataset(self,
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
        Complete pipeline to generate a simulated dataset.
        
        Args:
            freq_start: Start frequency in GHz
            freq_end: End frequency in GHz
            n_frequencies: Number of frequency channels
            date_string: Observation date (ISO format)
            observation_time: Total observation duration
            declination: Declination angle string
            integration_time: Integration time in seconds
            n_point_sources: Number of point sources
            point_flux_density: Point source flux density in mJy
            point_spectral_index: Point source spectral index
            include_gaussian: Whether to include Gaussian source
            gaussian_flux_density: Gaussian source flux density in Jy
            gaussian_position: Gaussian source position
            gaussian_minor_radius: Gaussian minor radius in arcsec
            gaussian_major_radius: Gaussian major radius in arcsec
            gaussian_theta_angle: Gaussian position angle in degrees
            
        Returns:
            Complete simulated dataset
        """
        
        # Load configuration
        self.load_antenna_configuration()
        
        # Configure observation
        self.configure_observation(
            freq_start=freq_start,
            freq_end=freq_end,
            n_frequencies=n_frequencies,
            date_string=date_string,
            observation_time=observation_time,
            declination=declination,
            integration_time=integration_time
        )
        
        # Generate point sources
        point_sources = self.generate_point_sources(
            n_sources=n_point_sources,
            flux_density=point_flux_density,
            spectral_index=point_spectral_index
        )
        
        # Generate Gaussian source if requested
        gaussian_source = None
        if include_gaussian:
            gaussian_source = self.generate_gaussian_source(
                flux_density=gaussian_flux_density,
                position=gaussian_position,
                minor_radius=gaussian_minor_radius,
                major_radius=gaussian_major_radius,
                theta_angle=gaussian_theta_angle
            )
        
        # Create composite source
        self.create_composite_source(
            point_sources=point_sources,
            gaussian_source=gaussian_source
        )
        
        # Simulate
        dataset = self.simulate_dataset()

        return dataset