
from ..util import Dataset
import numpy as np

class BCFDataset(Dataset):
    def __init__(self, X, y, z, thresholds):
        self.z = z
        super().__init__(X, y, thresholds)
        
    @property
    def treated(self):
        return (self.z == 1)
    