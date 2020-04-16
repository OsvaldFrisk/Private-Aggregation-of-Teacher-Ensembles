from abc import ABC, abstractmethod
import numpy as np


class ABM(ABC):
    """Abstract Base Model, used for inheritance by the models used in the PATE framework
    """
    @abstractmethod
    def fit(self, X, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        raise NotImplementedError
