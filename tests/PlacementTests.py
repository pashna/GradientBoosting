__author__ = 'popka'

import unittest
from utils.Placement import Placement
import numpy as np


class PlacementTests(unittest.TestCase):

    def test_int_to_bin_array(self):
        lst = np.asarray([1,2,3])

        pl = Placement(lst)

        self.assertTrue(np.array_equal(pl._int_to_bin_array(2), np.asarray([False,True,False])))
        self.assertTrue(np.array_equal(pl._int_to_bin_array(0), np.asarray([False,False,False])))
        self.assertTrue(np.array_equal(pl._int_to_bin_array(7), np.asarray([True,True,True])))


    def test_iterator(self):

        lst = np.asarray([1,4,7])

        placement = Placement(lst)

        right_answers = [np.asarray([7]),
                         np.asarray([4]),
                         np.asarray([4, 7]),
                         np.asarray([1]),
                         np.asarray([1,7]),
                         np.asarray([1,4])]

        i = 0

        for pl in placement:
            self.assertTrue(np.array_equal(pl, right_answers[i]))
            i+=1