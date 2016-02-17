# -*- coding: utf-8 -*-

__author__ = 'popka'
from utils.Placement import Placement
import numpy as np
from Node import Predicate


class Splitter():
    """
    Сплиттер находит оптимальное разделение "параметр предиката"
    """

    @staticmethod
    def split_categorial(x, y, impurity):
        combiner = Placement(np.unique(x))
        max_impurity = -1
        best_c = []
        for c in combiner:

            mask = np.in1d(x, c)
            y_left = y[mask]
            y_right = y[np.invert(mask)]
            imp = impurity.calculate_split(y_left, y_right)

            if imp > max_impurity:
                max_impurity = imp
                best_c = c

        return best_c


    @staticmethod
    def split_quantitative(x, y, impurity):

        max_impurity = -1
        best_value = 0

        #x = np.copy(x) ??? надо ???
        argsort = x.argsort()
        x = x[argsort]
        y = y[argsort]

        for value in x:
            y_left = y[x<=value]
            y_right = y[x>value]
            imp = impurity.calculate_split(y_left, y_right)

            if imp > max_impurity:
                max_impurity = imp
                best_value = value

        return best_value






