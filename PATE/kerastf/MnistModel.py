from tensorflow import keras

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

from ..BaseModel import BaseModel


def load_mnist():
    img_rows, img_cols = 28, 28
    num_classes = 10

    (X_train, y_train), (X_validation, y_validation) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_validation = X_validation.reshape(
        X_validation.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_validation = X_validation.astype('float32')
    X_train /= 255
    X_validation /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)

    y_validation = keras.utils.to_categorical(y_validation, num_classes)

    return X_train, X_validation, y_train, y_validation


class CNN(BaseModel):
    def __init__(self):
        self._batch_size = 64
        self._verbose = False
        self._input_shape = (28, 28, 1)
        self._num_classes = 10

        self._model = self._create_model()

    def fit(self, X, y):
        self._model.fit(X, y,
                        batch_size=self._batch_size,
                        verbose=self._verbose)

    def predict(self, X):
        # TODO: remake network to output class
        one_hot_predictions = self._model.predict(X)
        class_prediction = one_hot_predictions.argmax(axis=1)
        return class_prediction

    def _create_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self._input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self._num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model
