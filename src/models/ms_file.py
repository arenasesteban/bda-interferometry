"""
MS (Measurement Set) file information container.

This module contains the MSFile class that holds all extracted information
from an MS file including structure, metadata, and visibility data.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .antenna import Antenna
from .spectral_info import SpectralInfo
from .column_info import ColumnInfo
from .visibility import Visibility

@dataclass
class MSFile:
    """
    Complete MS file information container.
    
    This class holds all the information extracted from an MS file,
    including table structure, antenna configuration, spectral information,
    and visibility data.
    """
    
    file_path: str
    table_info: Dict[str, Any]
    columns: List[ColumnInfo]
    antennas: List[Antenna]
    spectral_info: Optional[SpectralInfo] = None
    validation: Optional[Dict[str, Any]] = None
    sample_visibilities: Optional[List[Visibility]] = None
    
    def get_summary(self) -> str:
        """Get a formatted summary of the MS file."""
        summary = f"""
=== MS FILE SUMMARY ===
Path: {self.file_path}

TABLE INFORMATION:
- Total rows: {self.table_info.get('n_rows', 'N/A'):,}
- Columns: {self.table_info.get('n_columns', 'N/A')}
- Type: {self.table_info.get('table_type', 'N/A')}

MAIN COLUMNS:
"""
        # Show only the most important columns
        important_columns = ['TIME', 'ANTENNA1', 'ANTENNA2', 'UVW', 'DATA', 'WEIGHT', 'FLAG']
        for col in self.columns:
            if col.name in important_columns:
                summary += f"- {col.name}: {col.data_type} {col.shape}\n"
        
        summary += f"""
ANTENNA INFORMATION:
- Number of antennas: {len(self.antennas)}
"""
        
        if self.spectral_info:
            summary += f"""
SPECTRAL INFORMATION:
- Channels: {self.spectral_info.n_channels}
- Polarizations: {self.spectral_info.n_polarizations}
- Total bandwidth: {self.spectral_info.total_bandwidth/1e6:.1f} MHz
"""
        
        if self.validation:
            summary += f"""
VALIDATION:
- Valid: {'YES' if self.validation.get('is_valid', False) else 'NO'}
- Issues: {len(self.validation.get('issues', []))}
"""
            if self.validation.get('issues'):
                for issue in self.validation['issues']:
                    summary += f"  * {issue}\n"
        
        if self.sample_visibilities:
            summary += f"""
SAMPLE DATA:
- Sample rows: {len(self.sample_visibilities)}
- Data shape: {self.sample_visibilities[0].data_shape if self.sample_visibilities else 'N/A'}
"""
        
        return summary
