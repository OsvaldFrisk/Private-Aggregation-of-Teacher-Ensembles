from Teacher import Teacher
from SimpleModel import SimpleModel
import numpy as np


X_train = np.random.random((20, 12))
y_train = np.random.random((20))

X_validation = np.random.random((10, 12))

teachers = Teacher(SimpleModel, epochs=100, verbose=True, n_classes=2)
teachers.fit(X_train, y_train)
teachers.predict(X_validation)
