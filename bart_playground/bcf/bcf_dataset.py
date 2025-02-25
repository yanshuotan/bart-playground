import numpy as np
from ..util import Dataset, DefaultPreprocessor
class BCFDataset(Dataset):
    def __init__(self, X, y, z):
        if z.ndim == 1:
            z = z.reshape(-1, 1)
        self.z = z
        for i in range(z.shape[0]):
            assert np.sum(z[i, :]) <= 1 # At most one treatment arm shall be True
        super().__init__(X, y)
        
    @property
    def treated(self):
        assert self.z.shape[1] == 1, "Only one treatment arm is supported when using property treated."
        return self.treated_by(0)
    
    def treated_by(self, arm = 0):
        return self.z[:, arm] == 1
    
class BCFPreprocessor(DefaultPreprocessor):
    def fit_transform(self, X, y, z):
        dataset = super().fit_transform(X, y)
        return BCFDataset(dataset.X, dataset.y, z)
    