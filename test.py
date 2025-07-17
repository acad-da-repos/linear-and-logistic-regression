
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from assignment import train_linear_regression, train_logistic_regression

class TestRegressionModels(unittest.TestCase):
    def test_train_linear_regression(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 5, 4, 5])
        model = train_linear_regression(X, y)
        self.assertIsInstance(model, LinearRegression)

    def test_train_logistic_regression(self):
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([0, 0, 1, 1, 1])
        model = train_logistic_regression(X, y)
        self.assertIsInstance(model, LogisticRegression)

if __name__ == '__main__':
    unittest.main()
