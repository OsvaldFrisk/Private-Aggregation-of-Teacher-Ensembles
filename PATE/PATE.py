from pate.base.AbstractBaseModel import ABM
from pate.Teacher import Teacher
from pate.Student import Student
from typing import Union

class PATE:
    def __init__(
            self,
            model: ABM,
            n_teachers: int = 50,
            teacher_epochs: int = 1,
            student_epochs: int = 1,
            epsilon: float = 1,
            verbose: bool = False,
            n_classes: Union[int, None] = None
        ):
        """TODO
        """
        self.teachers = Teacher(
            model=model,
            n_teachers=n_teachers,
            epochs=teacher_epochs,
            epsilon=epsilon,
            verbose=verbose,
            n_classes=n_classes)

        self.student = Student(
            model=model,
            epochs=student_epochs,
            verbose=verbose,
            n_classes=n_classes)

    def fit(self, X_private, y_private, X_public):
        """TODO
        """
        self.teachers.fit(X_private, y_private)
        y_public_predictions = self.teachers.predict(X_public)
        self.student.fit(X_public, y_public_predictions)

    def predict(self, X):
        """TODO
        """
        return self.student.predict(X)