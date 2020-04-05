from BaseModel import BaseModel
from time import sleep
import numpy as np


class SimpleModel(BaseModel):
    """Simple model used as stub for testing
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        # Sleeping to simulate training model...
        # sleep(0.0001)
        pass

    def predict(self, X):
        # Should return list with same number of samples as in X
        return np.array([1 if (ele+np.random.randn()) > 0.8 else 0 for ele in X[:, 0]])
