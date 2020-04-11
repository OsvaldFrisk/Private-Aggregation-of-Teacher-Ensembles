from PATE.Teacher import Teacher
from PATE.kerastf.MnistModel import load_mnist, CNN
from sklearn.metrics import accuracy_score
import numpy as np

X_train, X_validation, y_train, y_validation = load_mnist()

teachers = Teacher(CNN, epochs=1, n_teachers=10, verbose=True, n_classes=10)
teachers.fit(X_train, y_train)
predicted = teachers.predict(X_validation)

print(f"Accuracy: {round(accuracy_score(y_validation, predicted), 3)}")
