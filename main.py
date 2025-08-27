"""
Main script for BDA Interferometry Dataset Generation

Simple script that creates and executes a DatasetGenerator
based on the notebook implementation.

Author: Pipeline Team
Date: 2025
"""

import os
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from pipeline.dataset_generator import DatasetGenerator


def main():
    """Main function to generate interferometry dataset."""
    
    print("BDA Interferometry Dataset Generation")
    print("=" * 50)
    
    # Configuration paths
    antenna_config_path = "./antenna_configs/alma.cycle10.1.cfg"
    
    # Check if antenna config exists
    if not Path(antenna_config_path).exists():
        print(f"Error: Antenna configuration file not found: {antenna_config_path}")
        return False
    
    print(f"Using antenna configuration: {antenna_config_path}")
    
    # Create DatasetGenerator instance
    generator = DatasetGenerator(antenna_config_path=antenna_config_path)
    
    # Generate dataset with configuration
    print("Generating dataset...")
    dataset = generator.generate_dataset(
        # Frequency configuration
        freq_start=35.0,           # GHz - ALMA Band 1 start
        freq_end=50.0,             # GHz - ALMA Band 1 end
        n_frequencies=50,          # Number of frequency channels
        
        # Observation configuration
        date_string="2002-05-10",  # Observation date
        observation_time="4h",     # Total observation time
        declination="-45d00m00s",  # Declination angle
        integration_time=180.0,    # Integration time in seconds
        
        # Point sources configuration
        n_point_sources=15,        # Number of point sources
        point_flux_density=1.0,    # Flux density in mJy
        point_spectral_index=3.0,  # Spectral index
        
        # Gaussian source configuration
        include_gaussian=True,     # Include Gaussian source
        gaussian_flux_density=10.0,   # Flux density in Jy
        gaussian_position=(0, 0),     # Position (l, m)
        gaussian_minor_radius=20.0,   # Minor radius in arcsec
        gaussian_major_radius=30.0,   # Major radius in arcsec
        gaussian_theta_angle=60.0     # Position angle in degrees
    )
    
    print("Dataset generation completed successfully!")
    print(f"Generated {len(dataset.ms_list)} MS file(s)")
    
    return dataset


if __name__ == "__main__":
    try:
        dataset = main()
        if dataset:
            print("✅ Success: Dataset ready for processing")
        else:
            print("❌ Error: Dataset generation failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
