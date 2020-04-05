import tqdm
import numpy as np
from BaseModel import BaseModel

from typing import List, Union, Tuple


class Teacher(BaseModel):
    """A set of models to be trained on private data and predict using differential
    privacy as an aggregated ensemble of teachers.
    """

    def __init__(
            self,
            model,
            n_teachers: int = 50,
            epochs: int = 1,
            learning_rate: float = 0.1,
            momentum: float = 0.5,
            epsilon: float = 1,
            verbose: bool = False,
            n_classes: Union[int, None] = None):
        """
        Args:
            model: Model that implements the BaseModel interface, used in every teacher
            n_teachers: Integer number of teachers used in the ensemble
            epsilon: The amount of Laplacian noise to inject into each aggregated teacher predition
            epochs: Integer of epochs the model should be trained
            learning_rate: Float for learning rate used by the stochastic gradient descent optimizer
            momentum: Float for momentum used by the stochastic gradient descent optimizer
            verbose: Boolean describing whether to print training information
        """
        self._n_classes = n_classes
        self._model = model
        self._n_teachers = n_teachers
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._epsilon = epsilon
        self._verbose = verbose
        self._fit = False

        # List filled with n_teachers lists
        self._models = [self._model() for _ in range(self._n_teachers)]

    def _partition(
        self,
        X,
        y,
    ) -> Tuple[List, int]:
        """Function to partition training data into real partitions, meaning
        there are no overlaps of data between any of the partitions.

        Args:
            X: Feature columns used as inputs for the model
            y: Label column used to calculate loss

        Returns:
            partitions: List of partitioned data, into equal sized real
            partitions
            remainder: Integer equal to the number of rows that was
            not included in partitions
        """
        partitions = list()
        n_samples = len(X)
        samples_per_teacher = n_samples // self._n_teachers

        # The number of remaining samples not used for training
        remainder = n_samples % self._n_teachers
        idx = 0

        # Creating real partitions for each teacher
        for _ in range(self._n_teachers):
            X_partition = list()
            y_partition = list()

            for _ in range(samples_per_teacher):
                X_partition.append(X[idx])
                y_partition.append(y[idx])
                idx += 1

            partitions.append([X_partition, y_partition])

        return partitions, remainder

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

        # Data partitioned for teacher training
        partitions, remainder = self._partition(X, y)

        # Setting verbosity
        if self._verbose:
            epoch_range = tqdm.tqdm(
                range(1, self._epochs + 1), desc="Teachers", unit="epoch")
        else:
            epoch_range = range(1, self._epochs + 1)

        # Fitting model
        for epoch in epoch_range:
            for teacher_idx in range(self._n_teachers):

                (X, y) = partitions[teacher_idx]
                _model = self._models[teacher_idx]
                _model.fit(X, y)

        self._fit = True

    def _one_hot(
        self,
        predictions: np.ndarray,
    ) -> np.ndarray:
        """https://stackoverflow.com/questions/38592324/one-hot-encoding-using-numpy
        """
        predictions = predictions.reshape(-1)
        one_hot = np.eye(self._n_classes)[predictions]
        return one_hot

    def predict(
        self,
        X,
    ):
        assert self._n_classes, "The number of classes is not defined, add in parameter list or assign Teacher.n_classes"
        assert isinstance(
            self._n_classes, int), "the number of classes (n_classes) should be an integer"

        predictions = np.zeros((self._n_teachers, len(X), self._n_classes))

        # Each teacher tries to predicts the labels of the data, and produce a
        # onehot encoded prediction
        for i, model in enumerate(self._models):
            model_predictions = model.predict(X)
            one_hot_predictions = self._one_hot(model_predictions)
            predictions[i] = one_hot_predictions

        # Because the class predictions are onehot-encoded they can just be
        # summed to get the histogram and therefore also the aggregated prediction
        histograms = predictions.sum(axis=0)

        # Laplacian noise is added shaped as histogram
        # TODO:: Consider to use sensitivity as epsilon
        noise = np.random.laplace(
            0, self._epsilon, histograms.shape)
        noisy_histograms = histograms + noise

        # The aggregated noisy prediction
        noisy_predictions = noisy_histograms.argmax(axis=1)
        print(histograms.argmax(axis=1))
        print(noisy_histograms.argmax(axis=1))

        # TODO:: model_counts
        self._model_counts = list()

        return noisy_predictions
