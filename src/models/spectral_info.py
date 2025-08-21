from dataclasses import dataclass
from typing import List

@dataclass
class SpectralInfo:
    n_channels: int
    n_polarizations: int
    frequencies: List[float]
    channel_width: List[float]
    total_bandwidth: float
