from PATE.Teacher import Teacher
from PATE.Base.SimpleModel import SimpleModel
from sklearn.metrics import accuracy_score
import numpy as np

X_train = np.random.random((20, 12))
y_train = np.random.random((20))
X_validation = np.random.random((10, 12))

teachers = Teacher(SimpleModel, epochs=10, n_teachers=10,
                   verbose=True, n_classes=10, epsilon=1)
teachers.fit(X_train, y_train)
predicted = teachers.predict(X_validation)
print(predicted)
