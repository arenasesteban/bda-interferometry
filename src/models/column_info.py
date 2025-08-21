from dataclasses import dataclass
from typing import Optional, Union, Tuple, List

@dataclass
class ColumnInfo:
    name: str
    data_type: str
    shape: Union[Tuple[int, ...], List[int]]
    unit: Optional[str]
    description: Optional[str]
