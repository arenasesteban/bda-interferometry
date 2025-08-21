from dataclasses import dataclass
from typing import Any, Optional, Tuple

@dataclass
class Visibility:
    row_id: int
    time: float
    antenna1: int
    antenna2: int
    baseline: int
    uvw: Tuple[float, float, float]
    data: Any
    weight: Any
    flag: Any
    sigma: Optional[Any]
    data_shape: Any
