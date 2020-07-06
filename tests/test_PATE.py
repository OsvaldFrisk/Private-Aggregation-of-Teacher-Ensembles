"""
Script for testing the Teacher class of the PATE module

Test naming convention:
    test_[the name of the tested method]_[expected input / tested state]_[expected behavior]
"""

from pate import PATE
from pate.base.AbstractBaseModel import ABM
from pate.base.SimpleModel import SimpleModel

import unittest
import numpy as np
import numpy.testing as npt


class fakeModel(ABM):
    """
    Fake model class used for testing behavior of PATE class
    """
    def __init__(self, *args, **kwargs):
        self._n_fit = 0
        self._n_predict = 0

    def fit(self, X, y) -> None:
        self._n_fit += 1

    def predict(self, X) -> np.ndarray:
        self._n_predict += 1
        return np.ones((X.shape[0]))


class testPATE(unittest.TestCase):
    def setUp(self):
        self.X_private = np.zeros((100, 10))
        self.y_private = np.ones((100))
        self.X_public = np.zeros((10,10))
        self.X_validation = np.zeros((10, 10))

    def test_smoke(self):
        """Smoke test, making sure model can be instanciated, fitted and used
        to predict.
        """
        model = PATE.PATE(model=fakeModel, n_classes=10)
        model.fit(self.X_private, self.y_private, self.X_public)
        prediction = model.predict(self.X_validation)

    def test_fit_simpleXy_Teachers100FitsStudent10Fits(self):
        """Testing behavior of fit"""
        # arrange 
        model = PATE.PATE(model=fakeModel, n_teachers=20, teacher_epochs=5,
        student_epochs=10, n_classes=10)
        
        # act
        model.fit(self.X_private, self.y_private, self.X_public)

        # assert
        n_fits_teachers = sum([model._n_fit for model in model.teachers._models])
        n_fits_student = model.student._model._n_fit
        assert n_fits_teachers == 100
        assert n_fits_student == 10

