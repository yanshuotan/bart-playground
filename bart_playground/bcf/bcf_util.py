
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class EnsembleName(Enum):
    MU = "mu"
    TAU = "tau"

@dataclass
class BCFEnsembleIndex:
    ensemble_name: EnsembleName
    _index: int = -1

    def __post_init__(self):
        # If ensemble_id is TAU, index must be provided.
        if self.ensemble_name == EnsembleName.TAU and self._index == -1:
            raise ValueError("Index must be provided when ensemble_id is 'tau'.")
        
    @property
    def index(self):
        if self.ensemble_name == EnsembleName.MU:
            return -1
        return self._index
    
BCFTreeIndices = tuple[BCFEnsembleIndex, list[int]]
        