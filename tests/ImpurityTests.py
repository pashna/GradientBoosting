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

        y = np.asarray([1])
        gini_value = gini.calculate_node(y)
        self.assertTrue(gini_value == 0)


    def test_gini_split(self):
        gini = Gini()

        y_left = np.array([1,1,1,1,1])
        y_right = np.array([0,0,0,0,0])
        delta_gini = gini.calculate_split(y_left, y_right)
        self.assertTrue(delta_gini == 0.5)

        y_left = np.array([1,1,1,1,1])
        y_right = np.array([1,1,1,1,1])
        delta_gini = gini.calculate_split(y_left, y_right)
        self.assertTrue(delta_gini == 0.0)

        y_left_1 = np.array([1,1,1,1,0])
        y_right_1 = np.array([0,0,0,0,0])
        delta_gini_1 = gini.calculate_split(y_left_1, y_right_1)

        y_left_2 = np.array([1,1,1,1])
        y_right_2 = np.array([0,0,0,0,0,0])
        delta_gini_2 = gini.calculate_split(y_left_2, y_right_2)
        self.assertTrue(delta_gini_2 > delta_gini_1)
