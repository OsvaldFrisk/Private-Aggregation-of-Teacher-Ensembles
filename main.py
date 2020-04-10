from Teacher import Teacher
from SklearnMnistModel import load_mnist, RFC
from sklearn.metrics import accuracy_score
import numpy as np

# from SimpleModel import SimpleModel
# X_train = np.random.random((20, 12))
# y_train = np.random.random((20))
# X_validation = np.random.random((10, 12))

X_train, X_validation, y_train, y_validation = load_mnist()

teachers = Teacher(RFC, epochs=1, verbose=True, n_classes=10)
teachers.fit(X_train, y_train)
predicted = teachers.predict(X_validation)

print(f"Accuracy: {accuracy_score(y_validation, predicted)}")
