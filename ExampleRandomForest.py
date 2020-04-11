from PATE.Teacher import Teacher
from PATE.sklearn.MnistModel import load_mnist, RFC
from sklearn.metrics import accuracy_score
import numpy as np

X_train, X_validation, y_train, y_validation = load_mnist()

teachers = Teacher(RFC, epochs=2, n_teachers=10, verbose=True, n_classes=10)
teachers.fit(X_train, y_train)
predicted = teachers.predict(X_validation)

print(f"Accuracy: {round(accuracy_score(y_validation, predicted), 3)}")
