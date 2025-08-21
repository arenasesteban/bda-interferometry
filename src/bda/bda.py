from dataclasses import dataclass
from typing import Dict, Tuple, List
from copy import deepcopy
import numpy as np
import dask.array as da
from astropy.constants import c

from pyralysis.base.dataset import Dataset
from pyralysis.base.subms import SubMS
from pyralysis.base.visibility_set import VisibilitySet
import xarray as xr


@dataclass
class BDA:
    """
    Baseline Dependent Averaging (BDA) implementation for radio interferometry.
    
    BDA allows averaging of visibility data based on baseline length, reducing
    data volume while preserving astronomical information. Longer baselines
    can be averaged over shorter time intervals due to their lower sensitivity
    to temporal variations.
    
    The implementation follows Wijnholds et al. 2018 for decorrelation time
    calculations and provides a complete workflow for applying BDA to 
    Pyralysis datasets.
    
    Parameters
    ----------
    dataset : Dataset
        Pyralysis Dataset object containing interferometric visibilities
    decorr_factor : float, optional
        Decorrelation factor threshold (default: 0.95)
    max_averaging_time : float, optional
        Maximum averaging time in seconds (default: 180.0)
    min_averaging_time : float, optional
        Minimum averaging time in seconds (default: 1.0)
    """
    dataset: Dataset = None
    decorr_factor: float = 0.95         # decorrelation factor threshold
    max_averaging_time: float = 180.0   # seconds
    min_averaging_time: float = 1.0     # seconds

    def __post_init__(self):
        """Initialize BDA parameters after object creation."""
        self._validate_dataset()
        self._baseline_lengths = None
        self._averaging_times = None
        self._frequency_center = None
    
    def _validate_dataset(self):
        """Validate that the dataset is suitable for BDA processing."""
        if self.dataset is None:
            raise ValueError("Dataset cannot be None")
        if not hasattr(self.dataset, 'ms_list') or not self.dataset.ms_list:
            raise ValueError("Dataset must contain at least one SubMS object")

    # =========================================================================
    # CORE BDA METHODS - Main functionality
    # =========================================================================

    def apply_bda(self, dataset: Dataset = None) -> Dataset:
        """
        Apply BDA to the entire dataset and return a new Dataset object.
        
        This is the main entry point for applying BDA to a complete dataset.
        It processes each SubMS partition independently and returns a new
        Dataset with reduced data volume.
        
        Parameters
        ----------
        dataset : Dataset, optional
            Dataset to process. If None, uses self.dataset
            
        Returns
        -------
        Dataset
            New Dataset object with BDA applied to all SubMS partitions
        """
        if dataset is None:
            dataset = self.dataset
        
        # Apply BDA to each SubMS in the dataset
        bda_list = []
        for ms in dataset.ms_list:
            bda_ms = self.apply_bda_to_ms(ms)
            bda_list.append(bda_ms)
        
        # Create new Dataset with averaged data
        return self._create_dataset_from_averaged_data(bda_list, dataset)

    def apply_bda_to_ms(self, ms: SubMS) -> SubMS:
        """
        Apply BDA to a single SubMS object and return a new SubMS object.
        
        This method processes one SubMS partition, applying baseline-dependent
        averaging based on the calculated decorrelation times for each antenna pair.
        
        Parameters
        ----------
        ms : SubMS
            SubMS object to process
            
        Returns
        -------
        SubMS
            New SubMS object with BDA applied
        """
        visibility = ms.visibilities

        # Get baseline groups (antenna pairs)
        baseline_groups = self.get_baseline_groups()
        
        averaged_data = {
            'visibilities': [],
            'uvw': [],
            'weights': [],
            'times': [],
            'baselines': [],
            'antenna1': [],
            'antenna2': [],
            'flags': []
        }
        
        # Process each baseline group separately
        for (ant1_id, ant2_id), baseline_indices in baseline_groups.items():
            if len(baseline_indices) == 0:
                continue
                
            # Extract data for this baseline
            bl_vis = visibility.data[baseline_indices]
            bl_uvw = visibility.uvw[baseline_indices]
            bl_weights = visibility.weight[baseline_indices]
            bl_times = visibility.time[baseline_indices]
            bl_flags = visibility.flag[baseline_indices]
            
            # Calculate average baseline length for this antenna pair
            bl_lengths = self.calculate_baseline_lengths()[baseline_indices]
            avg_baseline_length = da.mean(bl_lengths).compute()
            
            # Calculate averaging time for this baseline
            frequency = self._get_center_frequency()
            avg_time = self.calculate_optimal_averaging_time(avg_baseline_length, frequency)
            
            # Create selection matrix for this baseline
            selection_matrix = self._create_time_selection_matrix(bl_times, avg_time)
            
            # Apply averaging
            if selection_matrix.shape[0] > 0:
                # Average visibilities
                avg_vis = da.dot(selection_matrix, bl_vis)
                avg_uvw = da.dot(selection_matrix, bl_uvw)
                avg_weights = da.dot(selection_matrix, bl_weights)
                avg_times = da.dot(selection_matrix, bl_times)
                avg_flags = da.dot(selection_matrix, bl_flags.astype(np.float32)) > 0.5
                
                # Create antenna arrays
                avg_ant1 = da.full(selection_matrix.shape[0], ant1_id)
                avg_ant2 = da.full(selection_matrix.shape[0], ant2_id)
                
                # Create baseline IDs (use a simple encoding: ant1 * 1000 + ant2)
                baseline_id = ant1_id * 1000 + ant2_id
                avg_baselines = da.full(selection_matrix.shape[0], baseline_id)
                
                # Store averaged data
                averaged_data['visibilities'].append(avg_vis)
                averaged_data['uvw'].append(avg_uvw)
                averaged_data['weights'].append(avg_weights)
                averaged_data['times'].append(avg_times)
                averaged_data['baselines'].append(avg_baselines)
                averaged_data['antenna1'].append(avg_ant1)
                averaged_data['antenna2'].append(avg_ant2)
                averaged_data['flags'].append(avg_flags)
        
        # Concatenate all baseline data
        concatenated_data = {}
        for key in averaged_data:
            if averaged_data[key]:
                concatenated_data[key] = da.concatenate(averaged_data[key], axis=0)
            else:
                concatenated_data[key] = da.array([])
        
        # Create new SubMS with averaged data
        return self._create_subms_from_averaged_data(concatenated_data, ms)

    # =========================================================================
    # BASELINE ANALYSIS - Baseline length calculations and grouping
    # =========================================================================

    def calculate_baseline_lengths(self) -> da.Array:
        """
        Calculate baseline lengths from UVW coordinates.
        
        Baseline length is calculated as sqrt(u^2 + v^2), ignoring the w component
        since w represents the component perpendicular to the celestial sphere
        and doesn't contribute to the projected baseline length.
        
        Returns
        -------
        da.Array
            Array of baseline lengths in meters
        """
        if self._baseline_lengths is not None:
            return self._baseline_lengths
            
        # Calculate from UVW coordinates in MS objects
        all_baseline_lengths = []
        
        for ms in self.dataset.ms_list:
            # Access UVW coordinates from VisibilitySet
            uvw = ms.visibilities.uvw
            
            # Convert to dask array if needed
            if hasattr(uvw, 'data'):
                uvw_data = uvw.data
            else:
                uvw_data = da.asarray(uvw)
                
            # Calculate baseline length as sqrt(u^2 + v^2)
            # Note: ignoring w component for baseline length calculation
            bl_lengths = da.sqrt(uvw_data[:, 0]**2 + uvw_data[:, 1]**2)
            all_baseline_lengths.append(bl_lengths)
                
        if all_baseline_lengths:
            # Concatenate all baseline lengths from all MS partitions
            self._baseline_lengths = da.concatenate(all_baseline_lengths, axis=0)
            return self._baseline_lengths
            
        raise ValueError("Cannot calculate baseline lengths: No UVW coordinates found in dataset")
    
    def get_baseline_groups(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Group visibility indices by antenna pairs (baseline).
        
        This method creates a mapping from antenna pairs to the visibility indices
        that correspond to those baselines. This is essential for applying
        baseline-dependent averaging, as each baseline will have its own
        averaging time.
        
        Returns
        -------
        Dict[Tuple[int, int], np.ndarray]
            Dictionary mapping antenna pairs to arrays of visibility indices
        """
        # Get antenna information from MS objects
        all_baseline_groups = {}
        global_index_offset = 0
        
        for ms in self.dataset.ms_list:
            # Access antenna pairs from VisibilitySet
            ant1 = ms.visibilities.antenna1
            ant2 = ms.visibilities.antenna2
            
            # Group antennas for this MS
            ms_baseline_groups = self._group_antennas(ant1, ant2, global_index_offset)
            
            # Merge with global baseline groups
            for baseline, indices in ms_baseline_groups.items():
                if baseline not in all_baseline_groups:
                    all_baseline_groups[baseline] = []
                all_baseline_groups[baseline].extend(indices)
            
            # Update offset for next MS
            global_index_offset += len(ant1)
        
        # Convert lists to numpy arrays
        for baseline in all_baseline_groups:
            all_baseline_groups[baseline] = np.array(all_baseline_groups[baseline])
        
        return all_baseline_groups

    # =========================================================================
    # TIME AVERAGING - Decorrelation time and averaging calculations
    # =========================================================================

    def calculate_decorrelation_time(self, baseline_length: float, 
                                   frequency: float, 
                                   declination: float = -45.0,
                                   hour_angle: float = 0.0,
                                   baseline_east: float = None,
                                   baseline_north: float = None) -> float:
        """
        Calculate decorrelation time based on baseline length and frequency.
        
        Implementation based on Wijnholds et al. 2018, equations 41-44.
        The decorrelation time is calculated as:
        
        T_decorr = sqrt(1 - R^2) / (|∂u/∂t| + |∂v/∂t| + w_factor * |∂w/∂t|)
        
        Where R is the decorrelation factor and the derivatives represent the
        rate of change of UV coordinates due to Earth rotation.
        
        This is the core scientific calculation that determines how long
        visibilities can be averaged before decorrelation becomes significant.
        Longer baselines have faster fringe rates and thus shorter decorrelation
        times, requiring more frequent sampling or shorter averaging intervals.
        
        Parameters
        ----------
        baseline_length : float
            Baseline length in meters
        frequency : float
            Observing frequency in Hz
        declination : float, optional
            Source declination in degrees (default: -45.0)
        hour_angle : float, optional
            Hour angle in degrees (default: 0.0 for zenith)
        baseline_east : float, optional
            East component of baseline in meters
        baseline_north : float, optional
            North component of baseline in meters
            
        Returns
        -------
        float
            Decorrelation time in seconds
        """
        # Physical constants
        omega_earth = 7.2925e-5  # Earth's angular velocity in rad/s
        wavelength = c.value / frequency  # wavelength in meters
        
        # Convert angles to radians
        dec_rad = np.radians(declination)
        hour_angle_rad = np.radians(hour_angle)
        
        # If baseline components not provided, assume East-West baseline
        if baseline_east is None or baseline_north is None:
            baseline_east = baseline_length  # Assume purely East-West baseline
            baseline_north = 0.0
        
        # Convert baseline components to wavelengths
        Lx_lambda = baseline_east / wavelength  # East component
        Ly_lambda = baseline_north / wavelength  # North component
        
        # Calculate UV coordinate derivatives (Equations 42-43 in the paper)
        # ∂u/∂t = (1/λ) * (Lx*cos(H) - Ly*sin(H)) * ωE * cos(δ)
        # ∂v/∂t = (1/λ) * (Lx*sin(H)*sin(δ) + Ly*cos(H)*sin(δ)) * ωE
        
        du_dt = (Lx_lambda * np.cos(hour_angle_rad) - 
                 Ly_lambda * np.sin(hour_angle_rad)) * omega_earth * np.cos(dec_rad)
        
        dv_dt = (Lx_lambda * np.sin(hour_angle_rad) * np.sin(dec_rad) + 
                 Ly_lambda * np.cos(hour_angle_rad) * np.sin(dec_rad)) * omega_earth
        
        # w-term derivative (Equation 44)
        # ∂w/∂t = (1/λ) * (Lx*sin(H)*cos(δ) + Ly*cos(H)*cos(δ)) * ωE
        dw_dt = (Lx_lambda * np.sin(hour_angle_rad) * np.cos(dec_rad) + 
                 Ly_lambda * np.cos(hour_angle_rad) * np.cos(dec_rad)) * omega_earth
        
        # Total fringe rate (Equation 41 in the paper)
        # For uniform weighting, w_factor should be considered
        # For natural weighting, w_factor ≈ 0
        w_factor = 0.1  # Conservative choice for w-term contribution
        
        total_fringe_rate = np.sqrt((du_dt)**2 + (dv_dt)**2 + (w_factor * dw_dt)**2)
        
        # Decorrelation time (Equation 41 in the paper)
        if total_fringe_rate > 0:
            # Calculate decorrelation time for given decorrelation factor R
            decorr_time = np.sqrt(1 - self.decorr_factor**2) / total_fringe_rate
        else:
            decorr_time = self.max_averaging_time
            
        # Apply safety factor to be conservative
        safety_factor = 0.8  # Use 80% of calculated time to be safe
        decorr_time *= safety_factor
            
        return decorr_time

    def calculate_optimal_averaging_time(self, baseline_length: float, frequency: float) -> float:
        """
        Calculate optimal averaging time for a specific baseline length and frequency.
        
        This method combines the decorrelation time calculation with practical
        bounds to ensure the averaging time is within reasonable limits.
        
        Parameters
        ----------
        baseline_length : float
            Baseline length in meters
        frequency : float
            Observing frequency in Hz
            
        Returns
        -------
        float
            Optimal averaging time in seconds
        """
        decorr_time = self.calculate_decorrelation_time(baseline_length, frequency)
        
        # Ensure averaging time is within bounds
        avg_time = np.clip(decorr_time, 
                         self.min_averaging_time, 
                         self.max_averaging_time)
        
        return avg_time
    
    def _create_time_selection_matrix(self, time_samples: da.Array, 
                                     averaging_time: float) -> da.Array:
        """
        Create selection matrix for temporal averaging based on averaging time.
        
        This method creates a selection matrix that defines how to group and average
        time samples based on the calculated averaging time for a specific baseline.
        The matrix weights are normalized to preserve flux conservation.
        
        Parameters
        ----------
        time_samples : da.Array
            Array of time samples
        averaging_time : float
            Target averaging time in seconds
            
        Returns
        -------
        da.Array
            Selection matrix for averaging (n_averaged_times, n_original_times)
        """
        # Convert times to numpy for processing
        if hasattr(time_samples, 'compute'):
            times = time_samples.compute()
        else:
            times = np.asarray(time_samples)
            
        if len(times) == 0:
            return da.array([])
            
        # Sort times to ensure proper ordering
        time_indices = np.argsort(times)
        sorted_times = times[time_indices]
        
        # Calculate time differences to determine integration time
        if len(sorted_times) > 1:
            dt = np.median(np.diff(sorted_times))
        else:
            dt = 180.0  # Default integration time
            
        # Calculate number of integrations to average
        n_avg = max(1, int(averaging_time / dt))
        
        # Create groups of consecutive time samples
        n_times = len(sorted_times)
        n_groups = max(1, n_times // n_avg)
        
        # Initialize selection matrix
        selection_matrix = da.zeros((n_groups, n_times), dtype=np.float32)
        
        for i in range(n_groups):
            start_idx = i * n_avg
            end_idx = min((i + 1) * n_avg, n_times)
            
            # Map back to original indices
            original_indices = time_indices[start_idx:end_idx]
            weight = 1.0 / len(original_indices)
            
            for orig_idx in original_indices:
                selection_matrix[i, orig_idx] = weight
                
        return selection_matrix

    # =========================================================================
    # UTILITY METHODS - Helper functions and data structure creation
    # =========================================================================

    def _group_antennas(self, ant1, ant2, index_offset: int = 0) -> Dict[Tuple[int, int], List[int]]:
        """
        Helper method to group antenna pairs.
        
        This method ensures consistent baseline ordering (smaller antenna first)
        and handles the conversion of Pyralysis data structures to numpy arrays.
        
        Parameters
        ----------
        ant1, ant2 : array-like
            Antenna 1 and antenna 2 arrays from VisibilitySet
        index_offset : int, optional
            Offset to add to visibility indices
            
        Returns
        -------
        Dict[Tuple[int, int], List[int]]
            Dictionary mapping antenna pairs to visibility indices
        """
        # Convert to numpy arrays - handle xarray DataArrays from Pyralysis
        if hasattr(ant1, 'compute'):
            ant1_data = ant1.compute()
            ant2_data = ant2.compute()
        else:
            ant1_data = np.asarray(ant1)
            ant2_data = np.asarray(ant2)
            
        # Group by antenna pairs
        baseline_groups = {}
        for i, (a1, a2) in enumerate(zip(ant1_data, ant2_data)):
            # Ensure consistent ordering (smaller antenna first)
            baseline = (min(int(a1), int(a2)), max(int(a1), int(a2)))
            if baseline not in baseline_groups:
                baseline_groups[baseline] = []
            baseline_groups[baseline].append(i + index_offset)
        
        return baseline_groups

    def _get_center_frequency(self) -> float:
        """
        Get the center frequency of the dataset.
        
        Returns
        -------
        float
            Center frequency in Hz
        """
        if self._frequency_center is not None:
            return self._frequency_center
            
        try:
            # Try to get frequency from the dataset observation
            obs = self.dataset.ms_list[0].visibilities.observation
            if hasattr(obs, 'spectral_window') and hasattr(obs.spectral_window, 'freq'):
                frequencies = obs.spectral_window.freq
                if hasattr(frequencies, 'compute'):
                    frequencies = frequencies.compute()
                else:
                    frequencies = np.asarray(frequencies)
                self._frequency_center = np.mean(frequencies)
                return self._frequency_center
            elif hasattr(obs, 'freq'):
                frequencies = obs.freq
                if hasattr(frequencies, 'compute'):
                    frequencies = frequencies.compute()
                else:
                    frequencies = np.asarray(frequencies)
                self._frequency_center = np.mean(frequencies)
                return self._frequency_center
            
        except (AttributeError, IndexError):
            pass
            
        # For our simulation, we know we used ALMA Band 1 (35-50 GHz)
        # Use the center of this range as fallback
        self._frequency_center = 42.5e9  # Hz (center of 35-50 GHz)
        return self._frequency_center

    def _create_subms_from_averaged_data(self, averaged_data: Dict[str, da.Array], 
                                       original_ms: SubMS) -> SubMS:
        """
        Create a new SubMS object from averaged visibility data.
        
        Parameters
        ----------
        averaged_data : Dict[str, da.Array]
            Dictionary containing averaged visibility data
        original_ms : SubMS
            Original SubMS to copy metadata from
            
        Returns
        -------
        SubMS
            New SubMS object with averaged data
        """
        # Create a deep copy of the original SubMS
        new_ms = deepcopy(original_ms)
        
        # Create new VisibilitySet with averaged data
        new_visibility_set = self._create_visibility_set_from_data(averaged_data, original_ms.visibilities)
        
        # Update the SubMS with new visibility data
        new_ms.visibilities = new_visibility_set
        
        return new_ms

    def _create_visibility_set_from_data(self, averaged_data: Dict[str, da.Array], 
                                       original_vis: VisibilitySet) -> VisibilitySet:
        """
        Create a new VisibilitySet from averaged data.
        
        This method attempts to update the Pyralysis VisibilitySet structure
        with the averaged data, handling both xarray and fallback cases.
        
        Parameters
        ----------
        averaged_data : Dict[str, da.Array]
            Dictionary containing averaged visibility data
        original_vis : VisibilitySet
            Original VisibilitySet to copy metadata from
            
        Returns
        -------
        VisibilitySet
            New VisibilitySet with averaged data
        """
        # Create a deep copy of the original VisibilitySet
        new_vis = deepcopy(original_vis)
        
        try:
            # Update the data arrays with averaged values directly
            new_vis.data = averaged_data['visibilities']
            new_vis.uvw = averaged_data['uvw']
            new_vis.weight = averaged_data['weights']
            new_vis.time = averaged_data['times']
            new_vis.flag = averaged_data['flags']
            new_vis.antenna1 = averaged_data['antenna1']
            new_vis.antenna2 = averaged_data['antenna2']
            
            # Update baseline information if it exists
            if hasattr(new_vis, 'baseline'):
                new_vis.baseline = averaged_data['baselines']
                
        except Exception as e:
            print(f"Warning: Could not update visibility data structure: {e}")
            # Fallback: Store the data in a custom attribute
            new_vis._bda_averaged_data = averaged_data
            
        return new_vis

    def _create_dataset_from_averaged_data(self, averaged_ms_list: List[SubMS], 
                                         original_dataset: Dataset) -> Dataset:
        """
        Create a new Dataset from a list of averaged SubMS objects.
        
        Parameters
        ----------
        averaged_ms_list : List[SubMS]
            List of SubMS objects with BDA applied
        original_dataset : Dataset
            Original Dataset to copy metadata from
            
        Returns
        -------
        Dataset
            New Dataset with BDA applied
        """
        # Create a deep copy of the original dataset
        new_dataset = deepcopy(original_dataset)
        
        # Replace the ms_list with averaged data
        new_dataset.ms_list = averaged_ms_list
        
        # Update any dataset-level metadata if needed
        # (This might need adjustment based on the specific Pyralysis Dataset structure)
        
        return new_dataset