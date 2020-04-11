"""Interface to use for models to be used in the PATE framework
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError
