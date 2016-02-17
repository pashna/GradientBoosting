__author__ = 'popka'

import numpy as np
import unittest
from Impurity.Gini import Gini

class DecisionTreeTests(unittest.TestCase):

    def test_gini(self):
        gini = Gini()

        y = np.asarray([1,1,1,1,1,1,1])
        gini_value = gini.calculate_node(y)
        self.assertTrue(gini_value == 0)

        y = np.asarray([1,1,1,1,2,2,2,2])
        gini_value = gini.calculate_node(y)
        self.assertTrue(gini_value == 0.5)

        y = np.asarray([1,2,3,1,2,3])
        gini_value = gini.calculate_node(y)
        self.assertTrue(round(gini_value, 3) == 0.667)