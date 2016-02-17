__author__ = 'popka'

import unittest
from DecisionTree import DecisionTree
import numpy as np

class DecisionTreeTests(unittest.TestCase):

    """
    def test_gini(self):
        tree = DecisionTree()

        y = np.asarray([1,1,1,1,1,1,1])
        gini = tree._gini(y)
        self.assertTrue(gini == 0)

        y = np.asarray([1,1,1,1,2,2,2,2])
        gini = tree._gini(y)
        self.assertTrue(gini == 0.5)

        y = np.asarray([1,2,3,1,2,3])
        gini = tree._gini(y)
        self.assertTrue(round(gini, 3) == 0.667)
    """

    def test_is_categorial(self):
        tree = DecisionTree()

        y = np.asarray([1,1,1,1,0,0,0])
        self.assertTrue(tree._is_categorical(y))

        y = np.asarray([1,1,2,4,1,2,4,4,4,4,4])
        self.assertTrue(tree._is_categorical(y))

        y = np.asarray([1.1,0.8,2.1,4,1,2.5,4,4,4,4.8,4])
        self.assertFalse(tree._is_categorical(y))

        y = np.asarray([100000002131, 12, 12])
        self.assertTrue(tree._is_categorical(y))