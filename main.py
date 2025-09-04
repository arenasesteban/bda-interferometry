"""
Main Entry Point - BDA Interferometry Dataset Generation

Primary script for generating simulated radio interferometry datasets using
the Pyralysis framework. Provides a simple command-line interface for
creating test datasets with predefined observation parameters.

This script serves as the main entry point for dataset generation and can be
used for testing the complete simulation pipeline before streaming operations.
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from data.simulation import generate_dataset


def main():
    """
    Generate a standard interferometry dataset for testing.
    
    Creates a simulated radio interferometry dataset using predefined
    observation parameters suitable for development and testing of the
    streaming pipeline infrastructure.
    
    Returns
    -------
    object or bool
        Generated Pyralysis dataset object if successful, False if failed
        
    Raises
    ------
    SystemExit
        If antenna configuration file is not found
    """
    
    antenna_config_path = "./antenna_configs/alma.cycle10.1.cfg"
    
    if not Path(antenna_config_path).exists():
        print(f"Antenna configuration file not found: {antenna_config_path}")
        return False

    try:
        dataset = generate_dataset(
            antenna_config_path=antenna_config_path,
            freq_start=35.0,
            freq_end=50.0,
            n_frequencies=50,
            date_string="2002-05-10",
            observation_time="4h",
            declination="-45d00m00s",
            integration_time=180.0,
            n_point_sources=15,
            point_flux_density=1.0,
            point_spectral_index=3.0,
            include_gaussian=True,
            gaussian_flux_density=10.0,
            gaussian_position=(0, 0),
            gaussian_minor_radius=20.0,
            gaussian_major_radius=30.0,
            gaussian_theta_angle=60.0
        )
        
        return dataset
        
    except Exception as e:
        print(f"Dataset generation failed: {e}")
        return False


if __name__ == "__main__":
    result = main()
    if result:
        print("Dataset generation successful")
    else:
        print("Dataset generation failed")
        sys.exit(1)
