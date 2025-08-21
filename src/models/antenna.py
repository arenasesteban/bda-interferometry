from dataclasses import dataclass
from typing import Tuple

@dataclass
class Antenna:
    id: int
    name: str
    position: Tuple[float, float, float]
    diameter: float
    mount: str
    type: str
