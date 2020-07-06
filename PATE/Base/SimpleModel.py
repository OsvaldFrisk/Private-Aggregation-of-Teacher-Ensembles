from pate.base.AbstractBaseModel import ABM
from time import sleep
import numpy as np


class SimpleModel(ABM):
    """Simple model used as stub for testing
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.ones(len(X))
