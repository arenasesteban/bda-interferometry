"""
Data Extractor for Pyralysis Dataset to JSON Conversion

Clean extractor that converts Pyralysis dataset objects to JSON format,
preserving essential structure while eliminating unnecessary fields.
"""

import json
import numpy as np
from typing import Dict, Any, List


def _safe_quantity_value(quantity):
    """Safely extract value from astropy Quantity, handling None cases."""
    if quantity is None:
        return None
    try:
        return float(quantity.value) if hasattr(quantity, 'value') else float(quantity)
    except:
        return None


def dataset_to_json(dataset) -> str:
    """
    Convert a complete Pyralysis dataset to JSON format.
    
    Args:
        dataset: Pyralysis dataset object containing all MS components
        
    Returns:
        JSON string representation of the complete dataset
    """
    
    dataset_info = {}
    
    # Core components - only include if they exist and have meaningful data
    if hasattr(dataset, 'antenna'):
        antenna_info = _extract_antenna_info(dataset.antenna)
        if antenna_info:
            dataset_info["antenna"] = antenna_info
            
    if hasattr(dataset, 'baseline'):
        baseline_info = _extract_baseline_info(dataset.baseline)
        if baseline_info:
            dataset_info["baseline"] = baseline_info
            
    if hasattr(dataset, 'field'):
        field_info = _extract_field_info(dataset.field)
        if field_info:
            dataset_info["field"] = field_info
            
    if hasattr(dataset, 'spws'):
        spw_info = _extract_spw_info(dataset.spws)
        if spw_info:
            dataset_info["spws"] = spw_info
            
    if hasattr(dataset, 'polarization'):
        pol_info = _extract_polarization_info(dataset.polarization)
        if pol_info:
            dataset_info["polarization"] = pol_info
            
    if hasattr(dataset, 'observation'):
        obs_info = _extract_observation_info(dataset.observation)
        if obs_info:
            dataset_info["observation"] = obs_info
            
    if hasattr(dataset, 'psf'):
        psf_info = _extract_psf_info(dataset.psf)
        if psf_info:
            dataset_info["psf"] = psf_info
    
    # Dataset-level properties - only include if they exist and are meaningful
    if hasattr(dataset, 'do_psf_calculation') and dataset.do_psf_calculation is not None:
        dataset_info["do_psf_calculation"] = dataset.do_psf_calculation
        
    if hasattr(dataset, 'corrected_column_present') and dataset.corrected_column_present is not None:
        dataset_info["corrected_column_present"] = dataset.corrected_column_present
        
    if hasattr(dataset, 'ndatasets'):
        dataset_info["ndatasets"] = dataset.ndatasets
    
    # Physical properties
    if hasattr(dataset, 'max_baseline'):
        max_bl = _safe_quantity_value(dataset.max_baseline)
        if max_bl is not None:
            dataset_info["max_baseline"] = max_bl
            
    if hasattr(dataset, 'min_baseline'):
        min_bl = _safe_quantity_value(dataset.min_baseline)
        if min_bl is not None:
            dataset_info["min_baseline"] = min_bl
            
    if hasattr(dataset, 'max_antenna_diameter'):
        max_ant = _safe_quantity_value(dataset.max_antenna_diameter)
        if max_ant is not None:
            dataset_info["max_antenna_diameter"] = max_ant
            
    if hasattr(dataset, 'min_antenna_diameter'):
        min_ant = _safe_quantity_value(dataset.min_antenna_diameter)
        if min_ant is not None:
            dataset_info["min_antenna_diameter"] = min_ant
            
    if hasattr(dataset, 'theo_resolution'):
        theo_res = _safe_quantity_value(dataset.theo_resolution)
        if theo_res is not None:
            dataset_info["theo_resolution"] = theo_res
            
    if hasattr(dataset, 'fov'):
        fov_val = _safe_quantity_value(dataset.fov)
        if fov_val is not None:
            dataset_info["fov"] = fov_val
    
    # Process MS files
    if hasattr(dataset, 'ms_list') and dataset.ms_list:
        ms_list = []
        for i, ms_file in enumerate(dataset.ms_list):
            ms_info = _extract_ms_info(ms_file, i)
            if ms_info:
                ms_list.append(ms_info)
        if ms_list:
            dataset_info["ms_list"] = ms_list
    
    # Additional properties that may exist
    if hasattr(dataset, 'corr_weight_sum') and dataset.corr_weight_sum is not None:
        dataset_info["corr_weight_sum"] = dataset.corr_weight_sum
    
    # Convert to JSON
    return json.dumps(dataset_info, indent=2, default=_numpy_json_serializer)


def _extract_antenna_info(antenna) -> Dict[str, Any]:
    """Extract antenna information preserving dataset structure."""
    if antenna is None:
        return None
        
    info = {}
    
    if hasattr(antenna, 'max_diameter') and antenna.max_diameter is not None:
        info["max_diameter_m"] = _safe_quantity_value(antenna.max_diameter)
        
    if hasattr(antenna, 'min_diameter') and antenna.min_diameter is not None:
        info["min_diameter_m"] = _safe_quantity_value(antenna.min_diameter)
    
    # Preserve the dataset as a structured object if present
    if hasattr(antenna, 'dataset') and antenna.dataset is not None:
        dataset = antenna.dataset
        # Keep the dataset structure intact, only extract essential metadata
        ds_info = {
            "antenna_count": len(dataset.ROWID) if hasattr(dataset, 'ROWID') else 0,
            "has_names": hasattr(dataset, 'NAME'),
            "has_positions": hasattr(dataset, 'POSITION'),
            "has_dish_diameters": hasattr(dataset, 'DISH_DIAMETER')
        }
        
        # Only include actual data if explicitly needed for validation
        if hasattr(dataset, 'NAME'):
            try:
                ds_info["names"] = dataset.NAME.data.compute().tolist()
            except:
                ds_info["names"] = "Could not extract"
                
        if hasattr(dataset, 'POSITION'):
            try:
                ds_info["positions"] = dataset.POSITION.data.compute().tolist()
            except:
                ds_info["positions"] = "Could not extract"
                
        if hasattr(dataset, 'DISH_DIAMETER'):
            try:
                ds_info["dish_diameters"] = dataset.DISH_DIAMETER.data.compute().tolist()
            except:
                ds_info["dish_diameters"] = "Could not extract"
                
        info["dataset"] = ds_info
    
    return info if info else None


def _extract_baseline_info(baseline) -> Dict[str, Any]:
    """Extract baseline information preserving dataset structure."""
    if baseline is None:
        return None
        
    info = {}
    
    if hasattr(baseline, 'max_baseline') and baseline.max_baseline is not None:
        info["max_baseline_m"] = _safe_quantity_value(baseline.max_baseline)
        
    if hasattr(baseline, 'min_baseline') and baseline.min_baseline is not None:
        info["min_baseline_m"] = _safe_quantity_value(baseline.min_baseline)
    
    # Preserve the dataset as a structured object if present
    if hasattr(baseline, 'dataset') and baseline.dataset is not None:
        dataset = baseline.dataset
        # Keep the dataset structure intact, only extract essential metadata
        ds_info = {
            "baseline_count": len(dataset.ROWID) if hasattr(dataset, 'ROWID') else 0,
            "has_baseline_lengths": hasattr(dataset, 'BASELINE_LENGTH'),
            "has_antenna_pairs": hasattr(dataset, 'ANTENNA1') and hasattr(dataset, 'ANTENNA2')
        }
        
        # Only include actual data if explicitly needed for validation
        if hasattr(dataset, 'BASELINE_LENGTH'):
            try:
                ds_info["baseline_lengths"] = dataset.BASELINE_LENGTH.data.compute().tolist()
            except:
                ds_info["baseline_lengths"] = "Could not extract"
                
        if hasattr(dataset, 'ANTENNA1'):
            try:
                ds_info["antenna1"] = dataset.ANTENNA1.data.compute().tolist()
            except:
                ds_info["antenna1"] = "Could not extract"
                
        if hasattr(dataset, 'ANTENNA2'):
            try:
                ds_info["antenna2"] = dataset.ANTENNA2.data.compute().tolist()
            except:
                ds_info["antenna2"] = "Could not extract"
                
        info["dataset"] = ds_info
    
    return info if info else None


def _extract_field_info(field) -> Dict[str, Any]:
    """Extract field information without unnecessary details."""
    if field is None:
        return None
        
    info = {}
    
    # Only include MS version if it's meaningful
    if hasattr(field, 'is_ms_v3'):
        info["is_ms_v3"] = field.is_ms_v3
    
    # Extract coordinate information in a clean format
    if hasattr(field, 'ref_dirs') and field.ref_dirs is not None:
        ref_dirs = field.ref_dirs
        info["reference_directions"] = {
            "ra_deg": ref_dirs.ra.deg.tolist() if hasattr(ref_dirs, 'ra') else None,
            "dec_deg": ref_dirs.dec.deg.tolist() if hasattr(ref_dirs, 'dec') else None,
            "frame": str(ref_dirs.frame) if hasattr(ref_dirs, 'frame') else None
        }
    
    if hasattr(field, 'phase_dirs') and field.phase_dirs is not None:
        phase_dirs = field.phase_dirs
        info["phase_directions"] = {
            "ra_deg": phase_dirs.ra.deg.tolist() if hasattr(phase_dirs, 'ra') else None,
            "dec_deg": phase_dirs.dec.deg.tolist() if hasattr(phase_dirs, 'dec') else None,
            "frame": str(phase_dirs.frame) if hasattr(phase_dirs, 'frame') else None
        }
    
    if hasattr(field, 'center_ref_dir') and field.center_ref_dir is not None:
        center = field.center_ref_dir
        info["center_reference_direction"] = {
            "ra_deg": float(center.ra.deg) if hasattr(center, 'ra') else None,
            "dec_deg": float(center.dec.deg) if hasattr(center, 'dec') else None
        }
    
    return info if info else None


def _extract_spw_info(spws) -> Dict[str, Any]:
    """Extract spectral window information focusing on essential data."""
    if spws is None:
        return None
        
    info = {}
    
    # Core spectral window information
    if hasattr(spws, 'ndatasets'):
        info["ndatasets"] = spws.ndatasets
        
    # Frequency information
    if hasattr(spws, 'max_nu') and spws.max_nu is not None:
        info["max_freq_hz"] = _safe_quantity_value(spws.max_nu)
    if hasattr(spws, 'min_nu') and spws.min_nu is not None:
        info["min_freq_hz"] = _safe_quantity_value(spws.min_nu)
    if hasattr(spws, 'ref_nu') and spws.ref_nu is not None:
        info["ref_freq_hz"] = _safe_quantity_value(spws.ref_nu)
        
    # Wavelength information
    if hasattr(spws, 'lambda_max') and spws.lambda_max is not None:
        info["lambda_max_m"] = _safe_quantity_value(spws.lambda_max)
    if hasattr(spws, 'lambda_min') and spws.lambda_min is not None:
        info["lambda_min_m"] = _safe_quantity_value(spws.lambda_min)
    if hasattr(spws, 'lambda_ref') and spws.lambda_ref is not None:
        info["lambda_ref_m"] = _safe_quantity_value(spws.lambda_ref)
    
    # Channel information
    if hasattr(spws, 'nchans') and spws.nchans is not None:
        info["nchans"] = spws.nchans.tolist()
    
    if hasattr(spws, 'ref_freqs') and spws.ref_freqs is not None:
        info["ref_frequencies_hz"] = [_safe_quantity_value(freq) for freq in spws.ref_freqs]
    
    return info if info else None


def _extract_polarization_info(polarization) -> Dict[str, Any]:
    """Extract polarization information without redundant data."""
    if polarization is None:
        return None
        
    info = {}
    
    if hasattr(polarization, 'ndatasets'):
        info["ndatasets"] = polarization.ndatasets
        
    if hasattr(polarization, 'feed_kind') and polarization.feed_kind is not None:
        info["feed_kind"] = polarization.feed_kind
        
    if hasattr(polarization, 'ncorrs') and polarization.ncorrs is not None:
        info["ncorrs"] = polarization.ncorrs.tolist()
        
    if hasattr(polarization, 'corrs_string') and polarization.corrs_string is not None:
        info["correlations_string"] = polarization.corrs_string

    return info if info else None


def _extract_observation_info(observation) -> Dict[str, Any]:
    """Extract observation information without unnecessary complexity."""
    if observation is None:
        return None
        
    info = {}
    
    if hasattr(observation, 'ntelescope'):
        info["ntelescope"] = observation.ntelescope
    
    if hasattr(observation, 'telescope') and observation.telescope is not None:
        try:
            telescope_data = observation.telescope.compute()
            info["telescopes"] = telescope_data.tolist() if hasattr(telescope_data, 'tolist') else [str(telescope_data)]
        except:
            info["telescopes"] = "Could not extract telescope names"
    
    return info if info else None


def _extract_psf_info(psf) -> Dict[str, Any]:
    """Extract PSF information in a cleaner format."""
    if psf is None:
        return None
    
    # Handle single PSF or list of PSFs
    if isinstance(psf, list):
        info = {
            "type": "list",
            "psf_count": len(psf),
            "psf_list": []
        }
        for i, single_psf in enumerate(psf):
            psf_data = _extract_single_psf_info(single_psf, i)
            if psf_data:
                info["psf_list"].append(psf_data)
    else:
        info = _extract_single_psf_info(psf)
        if info:
            info["type"] = "single"
    
    return info if info else None


def _extract_single_psf_info(psf, index=None) -> Dict[str, Any]:
    """Extract information from a single PSF object."""
    psf_info = {}
    
    if index is not None:
        psf_info["index"] = index
    
    # Extract beam parameters if available
    if hasattr(psf, 'major') and psf.major is not None:
        psf_info["major_axis_rad"] = _safe_quantity_value(psf.major)
    
    if hasattr(psf, 'minor') and psf.minor is not None:
        psf_info["minor_axis_rad"] = _safe_quantity_value(psf.minor)
    
    if hasattr(psf, 'pa') and psf.pa is not None:
        psf_info["position_angle_rad"] = _safe_quantity_value(psf.pa)
    
    return psf_info if psf_info else None


def _extract_ms_info(ms_file, index: int) -> Dict[str, Any]:
    """Extract information from a single MS file."""
    if ms_file is None:
        return None
        
    ms_info = {
        "ms_index": index,
    }
    
    if hasattr(ms_file, '_id') and ms_file._id is not None:
        ms_info["id"] = ms_file._id
        
    if hasattr(ms_file, 'field_id') and ms_file.field_id is not None:
        ms_info["field_id"] = ms_file.field_id
        
    if hasattr(ms_file, 'polarization_id') and ms_file.polarization_id is not None:
        ms_info["polarization_id"] = ms_file.polarization_id
        
    if hasattr(ms_file, 'spw_id') and ms_file.spw_id is not None:
        ms_info["spw_id"] = ms_file.spw_id
    
    # Extract visibility information
    if hasattr(ms_file, 'visibilities') and ms_file.visibilities is not None:
        vis_info = _extract_visibility_info(ms_file.visibilities)
        if vis_info:
            ms_info["visibilities"] = vis_info
    
    return ms_info if len(ms_info) > 1 else None  # Only return if has more than just index


def _extract_visibility_info(vis) -> Dict[str, Any]:
    """Extract visibility information focusing on essential metadata."""
    if vis is None:
        return None
        
    vis_info = {}
    
    if hasattr(vis, 'nrows'):
        vis_info["nrows"] = vis.nrows
    
    # Extract shape information for key data arrays
    key_arrays = ['data', 'uvw', 'weight', 'sigma', 'flag', 'antenna1', 'antenna2', 'time', 'baseline']
    
    for array_name in key_arrays:
        if hasattr(vis, array_name):
            array_obj = getattr(vis, array_name)
            if array_obj is not None and hasattr(array_obj, 'shape'):
                vis_info[f"{array_name}_shape"] = list(array_obj.shape)
                if hasattr(array_obj, 'dtype'):
                    vis_info[f"{array_name}_dtype"] = str(array_obj.dtype)
    
    # Extract unflagged visibility count if available
    if hasattr(vis, 'visibility_number') and vis.visibility_number is not None:
        try:
            vis_info["unflagged_visibility_count"] = float(vis.visibility_number.compute())
        except:
            vis_info["unflagged_visibility_count"] = "Could not compute"
    
    return vis_info if vis_info else None


def _numpy_json_serializer(obj):
    """JSON serializer for numpy data types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def extract_sample_visibilities(dataset, ms_index: int = 0, max_samples: int = 100) -> List[Dict[str, Any]]:
    """
    Extract a sample of visibility data points for inspection.
    
    Args:
        dataset: Pyralysis dataset object
        ms_index: Index of MS file to sample from
        max_samples: Maximum number of samples to extract
        
    Returns:
        List of dictionaries containing sample visibility data
    """
    if not hasattr(dataset, 'ms_list') or not dataset.ms_list or ms_index >= len(dataset.ms_list):
        return []
    
    ms_file = dataset.ms_list[ms_index]
    if not hasattr(ms_file, 'visibilities') or ms_file.visibilities is None:
        return []
    
    vis = ms_file.visibilities
    samples = []
    
    try:
        # Get the actual number of samples to extract
        nrows = min(vis.nrows, max_samples) if hasattr(vis, 'nrows') else max_samples
        
        for i in range(nrows):
            sample = {"row_index": i}
            
            # Extract basic visibility data
            if hasattr(vis, 'data') and vis.data is not None:
                try:
                    data_sample = vis.data[i].compute()
                    sample["visibility_data"] = {
                        "shape": list(data_sample.shape),
                        "dtype": str(data_sample.dtype),
                        "real_mean": float(np.mean(np.real(data_sample))),
                        "imag_mean": float(np.mean(np.imag(data_sample)))
                    }
                except:
                    sample["visibility_data"] = "Could not extract"
            
            # Extract UVW coordinates
            if hasattr(vis, 'uvw') and vis.uvw is not None:
                try:
                    uvw_sample = vis.uvw[i].compute()
                    sample["uvw"] = uvw_sample.tolist()
                except:
                    sample["uvw"] = "Could not extract"
            
            # Extract antenna information
            if hasattr(vis, 'antenna1') and vis.antenna1 is not None:
                try:
                    sample["antenna1"] = int(vis.antenna1[i].compute())
                except:
                    sample["antenna1"] = "Could not extract"
            
            if hasattr(vis, 'antenna2') and vis.antenna2 is not None:
                try:
                    sample["antenna2"] = int(vis.antenna2[i].compute())
                except:
                    sample["antenna2"] = "Could not extract"
            
            # Extract time information
            if hasattr(vis, 'time') and vis.time is not None:
                try:
                    sample["time"] = float(vis.time[i].compute())
                except:
                    sample["time"] = "Could not extract"
            
            samples.append(sample)
            
    except Exception as e:
        return [{"error": f"Failed to extract samples: {str(e)}"}]
    
    return samples
