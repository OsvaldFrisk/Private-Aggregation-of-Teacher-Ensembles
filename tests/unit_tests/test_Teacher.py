"""
Script for testing the Teacher class of the PATE module

Test naming convention:
    test_[the name of the tested method]_[expected input / tested state]_[expected behavior]
"""

import numpy as np
import numpy.testing as npt
from pate.Teacher import Teacher as Teacher
from pate.base.SimpleModel import SimpleModel


def setup():
    setup_dict = dict()

    setup_dict['uut'] = Teacher(
        model=SimpleModel,
        n_teachers=2
    )

    setup_dict['X'] = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9],
                                [10, 11, 12]])

    setup_dict['y'] = np.array([[1],
                                [2],
                                [3],
                                [4]])

    return setup_dict


class TestTeacher:
    def test_InitTeacher_SimpleModel_ReturnTeacher(self):
        test = setup()
        uut = test['uut']
        npt.assert_(uut,
                    'Cannot create a simple Teacher object')

    def test_Partition_InputDivisibleByNumberofTeachers_ZeroRemainder(self):
        test = setup()
        uut = test['uut']
        X = test['X']
        y = test['y']
        (X, y), remainder = uut._partition(X, y)

        npt.assert_(X.shape == (2, 2, 3))
        npt.assert_(y.shape == (2, 2, 1))
        npt.assert_(remainder == 0)

    def test_Partition_InputDivisibleByNumberofTeachers_TenRemainder(self):
        uut = Teacher(
            model=SimpleModel,
            n_teachers=20
        )
        X = np.random.random((230, 12))
        y = np.random.random((230, 1))
        (X, y), remainder = uut._partition(X, y)

        npt.assert_(X.shape == (20, 11, 12))
        npt.assert_(y.shape == (20, 11, 1))
        npt.assert_(remainder == 10)

    def test_Partition_InputDivisibleByNumberofTeachers_CorrectOrder(self):
        test = setup()
        uut = test['uut']
        X = test['X']
        y = test['y']

        (X, y), remainder = uut._partition(X, y)
        npt.assert_array_equal(X, [[[1,  2,  3], [4,  5,  6]],
                                   [[7, 8, 9], [10, 11, 12]]])
