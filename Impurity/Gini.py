# -*- coding: utf-8 -*-

__author__ = 'popka'

from Impurity import Impurity
import numpy as np

class Gini(Impurity):

    def calculate_split(self, y_left, y_right):
        """
        Уменьшение gini.
        Критерий разделения узла. Для разделения выбирается предикат, с максимальным _delta_gini
        :param y_left: np.array левого узла дерева
        :param y_right: np.array правого узла дерева
        :return:
        """
        y = np.append(y_left, y_right)
        return self.calculate_node(y) - ( float(len(y_left))*self.calculate_node(y_left) - (float(len(y_right)))*self.calculate_node(y_right) ) / len(y)


    def calculate_node(self, y):
        """
        считает gini
        :param y:
        :return:
        """

        gini = 1
        classes = np.unique(y)

        for c in classes:
            p = float(len(y[y==c]))/len(y)
            gini -= p**2

        return gini