from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..BaseModel import BaseModel


def load_mnist(validation_size=0.1, shuffle=False):
    name = 'mnist_784'
    dir_path = './cache/'
    cache_path_X = dir_path + name + '_X.npy'
    cache_path_y = dir_path + name + '_y.npy'
    try:
        X = np.load(cache_path_X)
        y = np.load(cache_path_y)
    except:
        X, y = fetch_openml(name=name, version=1, cache=True, return_X_y=True)
        y = y.astype(int)

        Path(dir_path).mkdir(parents=True, exist_ok=True)
        np.save(cache_path_X, X)
        np.save(cache_path_y, y)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=validation_size, shuffle=shuffle)
    return X_train, X_validation, y_train, y_validation


class RFC(BaseModel):
    def __init__(self):
        self._model = RandomForestClassifier(max_depth=10)

    def fit(self, X, y):
        return self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)
