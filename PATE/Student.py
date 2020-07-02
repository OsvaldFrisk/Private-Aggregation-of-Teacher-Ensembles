import tqdm
import numpy as np
from PATE.Base.AbstractBaseModel import ABM

from typing import List, Union, Tuple


class Student(ABM):
    """A model to be trained on public data labeled by the teacher ensemble network.
    """

    def __init__(
            self,
            model: ABM,
            epochs: int = 1,
            verbose: bool = False,
            n_classes: Union[int, None] = None):
        """
        Args:
            model: Model that implements the abstract base model, used in every teacher
            epochs: Integer of epochs the model should be trained
            verbose: Boolean describing whether to print training information
            n_classes: The number of output classes to used in the classification
        """
        self._n_classes = n_classes
        self._model = model
        self._n_teachers = n_teachers
        self._epochs = epochs
        self._epsilon = epsilon
        self._verbose = verbose
        self._fit = False

    def fit(
        self,
        X,
        y,
    ) -> None:
        """
        Function for fitting the ensemble of teacher models to the training data.

        Args:
            X: Feature columns used as inputs for the model
            y: Label column used to calculate loss
        """
        assert len(X) == len(
            y), f"Length of Columns ({len(X)}) and Lables ({len(y)}) are not equal"

        if self._verbose:
            epoch_range = tqdm.tqdm(
                range(1, self._epochs + 1), desc="Teachers", unit="epoch")
        else:
            epoch_range = range(1, self._epochs + 1)

        # Fitting model
        for epoch in epoch_range:
            self._model.fit(X, y)

        self._fit = True

    def predict(
        self,
        X,
    ) -> np.ndarray:
        """Returns the predictions by the model.

        Args:
            X: Feature columns used for prediction

        Returns:
            predictions: numpy array of  predictions
        """
        assert self._n_classes, "The number of classes is not defined, add in parameter list or assign Teacher.n_classes"
        assert isinstance(
            self._n_classes, int), "the number of classes (n_classes) should be an integer"

        return self._model.predict(X)
