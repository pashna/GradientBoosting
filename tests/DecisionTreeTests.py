__author__ = 'popka'

import unittest
from DecisionTree import DecisionTree
import numpy as np

class DecisionTreeTests(unittest.TestCase):

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



    def test_decision_tree(self):
        tree = DecisionTree()
        X = np.asarray([[1, 1],[0, 2], [3, 2]])
        y = np.asarray([0, 1, 1])

        tree.fit(X_=X, y_=y)

        self.assertTrue(tree.predict(np.asarray([[1,1]]))[0] == 0)


    def test_is_stop_criterion(self):
        tree = DecisionTree()

        self.assertTrue(tree._is_stop_criterion(np.asarray([1])))
        self.assertTrue(tree._is_stop_criterion(np.asarray([1, 1, 1, 1, 1])))
        self.assertFalse(tree._is_stop_criterion(np.asarray([1, 1, 0, 0, 1])))