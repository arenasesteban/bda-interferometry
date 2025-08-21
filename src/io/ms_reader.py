"""
MS (Measurement Set) file reader main module.

This module provides the MSReader class to read and analyze MS files,
extracting all available information including visibilities, metadata,
antennas and file structure.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
import casacore.tables as ct

from ...models.ms_file import MSFile
from ...models.antenna import Antenna
from ...models.spectral_info import SpectralInfo
from ...models.column_info import ColumnInfo
from ...models.visibility import Visibility

@dataclass
class MSReader:
    """
    Main reader for MS (Measurement Set) files.
    
    This class provides a complete interface to read and analyze MS files
    with a single method call that extracts all available information.
    """
    
    ms_path: str
    table: Optional[ct.table] = None
    antenna_table: Optional[ct.table] = None
    spectral_window_table: Optional[ct.table] = None
    polarization_table: Optional[ct.table] = None
    _column_info_cache: Optional[List[ColumnInfo]] = None
    _antenna_info_cache: Optional[List[Antenna]] = None
    _table_info_cache: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize MSReader after object creation."""
        self.ms_path = os.path.abspath(self.ms_path)
        self._validate_ms_path()
    
    def _validate_ms_path(self):
        """Validate that the MS path exists and is accessible."""
        if not os.path.exists(self.ms_path):
            raise FileNotFoundError(f"MS file not found: {self.ms_path}")
        
        # Verify it's a valid MS directory
        if os.path.isdir(self.ms_path):
            # Must contain at least the main table
            main_table_path = os.path.join(self.ms_path, "table.dat")
            if not os.path.exists(main_table_path):
                raise ValueError(f"Directory does not contain a valid MS table: {self.ms_path}")
        else:
            raise ValueError(f"Path must be an MS directory: {self.ms_path}")
    
    @contextmanager
    def _open_tables(self):
        """Context manager for opening and closing MS tables."""
        try:
            # Open main table
            self.table = ct.table(self.ms_path, readonly=True)
            
            # Open auxiliary tables if they exist
            try:
                self.antenna_table = ct.table(os.path.join(self.ms_path, "ANTENNA"), readonly=True)
            except:
                self.antenna_table = None
                
            try:
                self.spectral_window_table = ct.table(os.path.join(self.ms_path, "SPECTRAL_WINDOW"), readonly=True)
            except:
                self.spectral_window_table = None
                
            try:
                self.polarization_table = ct.table(os.path.join(self.ms_path, "POLARIZATION"), readonly=True)
            except:
                self.polarization_table = None
                
            yield self
            
        finally:
            # Close all tables
            if self.table:
                self.table.close()
            if self.antenna_table:
                self.antenna_table.close()
            if self.spectral_window_table:
                self.spectral_window_table.close()
            if self.polarization_table:
                self.polarization_table.close()
    
    def read_ms_file(self) -> MSFile:
        """Read and analyze the complete MS file."""
        with self._open_tables():
            # 1. Get table information
            table_info = self._get_table_info()
            
            # 2. Get column information
            columns = self._get_column_info()
            
            # 3. Get antenna information
            antennas = self._get_antenna_info()
            
            # 4. Get spectral information
            spectral_info = self._get_spectral_info()
            
            # 5. Validate data integrity
            validation = self._validate_data_integrity()
            
            # 6. Get sample visibility data
            sample_visibilities = self._get_sample_visibilities()
            
            # Create and return MSFile object
            return MSFile(
                file_path=self.ms_path,
                table_info=table_info,
                columns=columns,
                antennas=antennas,
                spectral_info=spectral_info,
                validation=validation,
                sample_visibilities=sample_visibilities
            )
    
    def _get_table_info(self) -> Dict[str, Any]:
        """Get general information about the main table."""
        if self._table_info_cache is not None:
            return self._table_info_cache
            
        info = {
            'n_rows': self.table.nrows(),
            'n_columns': len(self.table.colnames()),
            'column_names': self.table.colnames(),
            'table_type': 'MAIN' if self.table.nrows() > 0 else 'EMPTY'
        }
        
        # Additional information if available
        if hasattr(self.table, 'getdesc'):
            desc = self.table.getdesc()
            info.update({
                'table_desc': desc.get('table_desc', ''),
                'table_type': desc.get('table_type', 'MAIN')
            })
        
        self._table_info_cache = info
        return info
    
    def _get_column_info(self) -> List[ColumnInfo]:
        """Get detailed information about all columns."""
        if self._column_info_cache is not None:
            return self._column_info_cache
            
        columns = []
        
        for colname in self.table.colnames():
            try:
                coldesc = self.table.getcoldesc(colname)
                colinfo = self.table.getcol(colname, 0, 1)
                
                if isinstance(colinfo, np.ndarray):
                    data_type = str(colinfo.dtype)
                    shape = colinfo.shape
                else:
                    data_type = str(type(colinfo))
                    shape = (1,)
                
                unit = coldesc.get('unit', None)
                description = coldesc.get('comment', None)
                
                column_info = ColumnInfo(
                    name=colname,
                    data_type=data_type,
                    shape=shape,
                    unit=unit,
                    description=description
                )
                columns.append(column_info)
                
            except Exception as e:
                column_info = ColumnInfo(
                    name=colname,
                    data_type="unknown",
                    shape=(1,),
                    unit=None,
                    description=f"Error reading: {str(e)}"
                )
                columns.append(column_info)
        
        self._column_info_cache = columns
        return columns
    
    def _get_antenna_info(self) -> List[Antenna]:
        """Get information about all antennas in the interferometer."""
        if self._antenna_info_cache is not None:
            return self._antenna_info_cache
            
        if not self.antenna_table:
            return []
            
        antennas = []
        antenna_ids = self.antenna_table.getcol('ANTENNA_ID')
        names = self.antenna_table.getcol('NAME')
        positions = self.antenna_table.getcol('POSITION')
        
        try:
            diameters = self.antenna_table.getcol('DISH_DIAMETER')
        except:
            diameters = np.full(len(antenna_ids), 12.0)
            
        try:
            mounts = self.antenna_table.getcol('MOUNT')
        except:
            mounts = np.full(len(antenna_ids), 'alt-az')
            
        try:
            types = self.antenna_table.getcol('TYPE')
        except:
            types = np.full(len(antenna_ids), 'GROUND-BASED')
        
        for i in range(len(antenna_ids)):
            antenna = Antenna(
                id=int(antenna_ids[i]),
                name=str(names[i]),
                position=tuple(positions[i]),
                diameter=float(diameters[i]),
                mount=str(mounts[i]),
                type=str(types[i])
            )
            antennas.append(antenna)
        
        self._antenna_info_cache = antennas
        return antennas
    
    def _get_spectral_info(self) -> Optional[SpectralInfo]:
        """Get spectral information from the MS file."""
        if not self.spectral_window_table:
            return None
            
        try:
            n_channels = self.spectral_window_table.getcol('NUM_CHAN')[0]
            n_polarizations = self.spectral_window_table.getcol('NUM_CORR')[0]
            frequencies = self.spectral_window_table.getcol('CHAN_FREQ')[0]
            channel_widths = self.spectral_window_table.getcol('CHAN_WIDTH')[0]
            
            spectral_info = SpectralInfo(
                n_channels=int(n_channels),
                n_polarizations=int(n_polarizations),
                frequencies=frequencies.tolist(),
                channel_width=channel_widths.tolist(),
                total_bandwidth=float(np.sum(channel_widths))
            )
            
            return spectral_info
            
        except Exception as e:
            return None
    
    def _validate_data_integrity(self) -> Dict[str, Any]:
        """Validate the integrity of MS file data."""
        issues = []
        statistics = {}
        
        try:
            # Verify that main columns exist
            required_columns = ['TIME', 'ANTENNA1', 'ANTENNA2', 'UVW', 'DATA', 'WEIGHT', 'FLAG']
            missing_columns = [col for col in required_columns if col not in self.table.colnames()]
            
            if missing_columns:
                issues.append(f"Missing columns: {missing_columns}")
            
            # Verify number of rows
            n_rows = self.table.nrows()
            if n_rows == 0:
                issues.append("MS file contains no data")
            
            # Verify data consistency if there are rows
            if n_rows > 0:
                # Read a sample of data to validate
                sample_data = self._read_sample_visibilities(0, min(100, n_rows))
                
                # Verify no NaN values in critical data
                for vis in sample_data:
                    if np.any(np.isnan(vis.data)):
                        issues.append("Visibility data contains NaN values")
                        break
                    
                    if np.any(np.isnan(vis.uvw)):
                        issues.append("UVW coordinates contain NaN values")
                        break
                
                # Basic statistics
                times = [vis.time for vis in sample_data]
                statistics['time_range'] = (min(times), max(times))
                statistics['n_antennas'] = max(max(vis.antenna1 for vis in sample_data), 
                                             max(vis.antenna2 for vis in sample_data)) + 1
                statistics['data_shape'] = sample_data[0].data.shape if sample_data else None
            
            # Verify antenna information
            try:
                antenna_info = self._get_antenna_info()
                statistics['n_antennas_actual'] = len(antenna_info)
            except:
                issues.append("Cannot read antenna information")
            
            # Verify spectral information
            try:
                spectral_info = self._get_spectral_info()
                if spectral_info:
                    statistics['spectral_info'] = spectral_info
            except:
                issues.append("Cannot read spectral information")
            
        except Exception as e:
            issues.append(f"Error during validation: {str(e)}")
        
        # Determine if data is valid
        is_valid = len(issues) == 0
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'statistics': statistics
        }
    
    def _get_sample_visibilities(self, n_samples: int = 5) -> List[Visibility]:
        """Get a sample of visibility data for inspection."""
        try:
            n_rows = self.table.nrows()
            if n_rows == 0:
                return []
            
            sample_size = min(n_samples, n_rows)
            return self._read_sample_visibilities(0, sample_size)
            
        except Exception as e:
            return []
    
    def _read_sample_visibilities(self, start_row: int, n_rows: int) -> List[Visibility]:
        """Read a sample of visibility data for validation purposes."""
        total_rows = self.table.nrows()
        if start_row >= total_rows:
            return []
        
        end_row = min(start_row + n_rows, total_rows)
        actual_n_rows = end_row - start_row
        
        times = self.table.getcol('TIME', start_row, actual_n_rows)
        antenna1 = self.table.getcol('ANTENNA1', start_row, actual_n_rows)
        antenna2 = self.table.getcol('ANTENNA2', start_row, actual_n_rows)
        uvw = self.table.getcol('UVW', start_row, actual_n_rows)
        data = self.table.getcol('DATA', start_row, actual_n_rows)
        weight = self.table.getcol('WEIGHT', start_row, actual_n_rows)
        flag = self.table.getcol('FLAG', start_row, actual_n_rows)
        
        try:
            sigma = self.table.getcol('SIGMA', start_row, actual_n_rows)
        except:
            sigma = None
            
        try:
            baseline = self.table.getcol('BASELINE', start_row, actual_n_rows)
        except:
            baseline = antenna1 * 1000 + antenna2
        
        visibilities = []
        for i in range(actual_n_rows):
            vis = Visibility(
                row_id=start_row + i,
                time=float(times[i]),
                antenna1=int(antenna1[i]),
                antenna2=int(antenna2[i]),
                baseline=int(baseline[i]),
                uvw=tuple(uvw[i]),
                data=data[i],
                weight=weight[i],
                flag=flag[i],
                sigma=sigma[i] if sigma is not None else None,
                data_shape=data[i].shape
            )
            visibilities.append(vis)
        
        return visibilities
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Tables are automatically closed in _open_tables
        pass
