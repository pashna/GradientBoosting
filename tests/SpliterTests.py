__author__ = 'popka'

import unittest
from Splitter import Splitter
from Impurity.Gini import Gini
import numpy as np

class SpliterTests(unittest.TestCase):

    def test_split_categorial(self):
        gini = Gini()
        splitter = Splitter()

        x = np.asarray([1,2,3,3,3,1,1,2])
        y = np.asarray([1,2,2,2,2,1,1,2])

        best_c, max_impurity = splitter.split_categorial(x=x, y=y, impurity=gini)
        self.assertTrue(np.array_equal((best_c), np.asarray([2,3])) or np.array_equal((best_c), np.asarray([1])))


    def test_split_quantitative(self):
        gini = Gini()
        splitter = Splitter()
        x = np.asarray([5., 3., 10, -1, -10, -0.5])
        y = np.asarray([2, 2, 2, 1, 1, 1])

        best_value, max_impurity = splitter.split_quantitative(x=x, y=y, impurity=gini)

        self.assertTrue(best_value == -0.5)