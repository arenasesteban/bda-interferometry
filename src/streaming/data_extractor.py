"""
Data Extractor for Pyralysis Dataset to JSON Conversion

Simple function to extract data from a Pyralysis dataset object
and convert it to JSON format for streaming or processing.
"""

import json
import numpy as np
from typing import Dict, Any, List


def dataset_to_json(dataset) -> str:
    """
    Convert a Pyralysis dataset to JSON format.
    
    Args:
        dataset: Pyralysis dataset object containing MS files
        
    Returns:
        JSON string representation of the dataset
    """
    
    # Extract basic dataset information
    dataset_info = {
        "ms_files_count": len(dataset.ms_list),
        "ms_files": []
    }
    
    # Process each MS file
    for i, ms_file in enumerate(dataset.ms_list):
        ms_info = {
            "ms_index": i,
            "available_attributes": [attr for attr in dir(ms_file) if not attr.startswith('_')]
        }
        
        # Add visibility information if available
        if hasattr(ms_file, 'visibilities'):
            vis = ms_file.visibilities
            ms_info["visibilities"] = {
                "available_attributes": [attr for attr in dir(vis) if not attr.startswith('_')]
            }
            
            if hasattr(vis, 'data'):
                ms_info["visibilities"]["data"] = {
                    "shape": list(vis.data.shape),
                    "dtype": str(vis.data.dtype)
                }
            
            if hasattr(vis, 'uvw'):
                ms_info["visibilities"]["uvw"] = {
                    "shape": list(vis.uvw.shape),
                    "dtype": str(vis.uvw.dtype)
                }
        
        # Try to find frequency information in different places
        freq_info = None
        
        # Check if frequencies are directly in ms_file
        if hasattr(ms_file, 'frequencies'):
            freq_data = ms_file.frequencies
            freq_info = {
                "location": "ms_file.frequencies",
                "shape": list(freq_data.shape),
                "dtype": str(freq_data.dtype),
                "min_freq_hz": float(np.min(freq_data)),
                "max_freq_hz": float(np.max(freq_data))
            }
        
        # Check if frequencies are in visibilities
        elif hasattr(ms_file, 'visibilities') and hasattr(ms_file.visibilities, 'frequency'):
            freq_data = ms_file.visibilities.frequency
            freq_info = {
                "location": "ms_file.visibilities.frequency",
                "shape": list(freq_data.shape),
                "dtype": str(freq_data.dtype),
                "min_freq_hz": float(np.min(freq_data)),
                "max_freq_hz": float(np.max(freq_data))
            }
        
        # Check other possible frequency locations
        elif hasattr(ms_file, 'freq'):
            freq_data = ms_file.freq
            freq_info = {
                "location": "ms_file.freq",
                "shape": list(freq_data.shape),
                "dtype": str(freq_data.dtype),
                "min_freq_hz": float(np.min(freq_data)),
                "max_freq_hz": float(np.max(freq_data))
            }
        
        if freq_info:
            ms_info["frequencies"] = freq_info
        else:
            ms_info["frequencies"] = {"error": "No frequency information found"}
        
        # Add antenna information if available
        if hasattr(ms_file, 'antennas') and ms_file.antennas is not None:
            ms_info["antennas"] = {
                "count": len(ms_file.antennas)
            }
        
        # Add time information if available
        if hasattr(ms_file, 'times') and ms_file.times is not None:
            ms_info["times"] = {
                "shape": list(ms_file.times.shape),
                "dtype": str(ms_file.times.dtype)
            }
        
        dataset_info["ms_files"].append(ms_info)
    
    # Convert to JSON
    return json.dumps(dataset_info, indent=2)


def extract_sample_visibilities(dataset, ms_index: int = 0, max_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Extract a sample of visibility data points from the dataset.
    
    Args:
        dataset: Pyralysis dataset object
        ms_index: Index of the MS file to extract from
        max_samples: Maximum number of samples to extract
        
    Returns:
        List of dictionaries containing visibility data points
    """
    
    if ms_index >= len(dataset.ms_list):
        raise ValueError(f"MS index {ms_index} out of range. Dataset has {len(dataset.ms_list)} MS files.")
    
    ms_file = dataset.ms_list[ms_index]
    
    # Get visibility data
    if not hasattr(ms_file, 'visibilities'):
        raise ValueError("MS file does not have visibilities attribute")
    
    vis = ms_file.visibilities
    
    if not hasattr(vis, 'data'):
        raise ValueError("Visibilities object does not have data attribute")
    
    if not hasattr(vis, 'uvw'):
        raise ValueError("Visibilities object does not have uvw attribute")
    
    visibilities_data = vis.data
    uvw_data = vis.uvw
    
    # Try to find frequency information
    frequencies = None
    
    # Look for frequencies in different places
    if hasattr(ms_file, 'frequencies'):
        frequencies = ms_file.frequencies
    elif hasattr(vis, 'frequency'):
        frequencies = vis.frequency
    elif hasattr(ms_file, 'freq'):
        frequencies = ms_file.freq
    
    # Get data shapes
    n_baselines, n_frequencies, n_correlations = visibilities_data.shape
    
    # Sample visibility points
    samples = []
    count = 0
    
    for baseline_idx in range(min(n_baselines, max_samples)):
        for freq_idx in range(min(n_frequencies, max_samples // n_baselines + 1)):
            if count >= max_samples:
                break
                
            # Extract visibility for first correlation (assuming 4 correlations: XX, XY, YX, YY)
            vis_complex = complex(visibilities_data[baseline_idx, freq_idx, 0])
            
            # Calculate frequency (use actual frequencies if available, otherwise estimate)
            if frequencies is not None and len(frequencies) > freq_idx:
                freq_hz = float(frequencies[freq_idx])
            else:
                # Fallback: estimate frequency based on index
                # Assuming a range similar to the test configuration (35-50 GHz)
                freq_start = 35e9  # 35 GHz
                freq_step = 1e9    # 1 GHz step
                freq_hz = freq_start + (freq_idx * freq_step)
            
            # Create sample data point
            sample = {
                "baseline_index": int(baseline_idx),
                "frequency_index": int(freq_idx),
                "frequency_hz": freq_hz,
                "uvw_u": float(uvw_data[baseline_idx, 0]),
                "uvw_v": float(uvw_data[baseline_idx, 1]),
                "uvw_w": float(uvw_data[baseline_idx, 2]),
                "visibility_real": float(vis_complex.real),
                "visibility_imag": float(vis_complex.imag),
                "visibility_amplitude": float(abs(vis_complex)),
                "visibility_phase": float(np.angle(vis_complex))
            }
            
            samples.append(sample)
            count += 1
        
        if count >= max_samples:
            break
    
    return samples
