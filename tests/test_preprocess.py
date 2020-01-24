import os
import unittest

import numpy as np
from models.preprocess import preprocessCsv


class TestPreprocessCSV(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(__file__), "..", "HR-Employee.csv")

    def test_data_is_numpy_array(self):
        (X, y) = preprocessCsv(self.path)
        self.assertIsInstance(X, np.ndarray)

    def test_zero_var_removed(self):
        (X, y) = preprocessCsv(self.path)
        mask = X.var() == 0
        self.assertTrue(not mask.all())

    def test_correct_shape(self):
        (X, y) = preprocessCsv(self.path)
        self.assertEqual(X.shape, (1470, 45))


if __name__ == "__main__":
    unittest.main()
