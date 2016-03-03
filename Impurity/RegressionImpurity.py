# -*- coding: utf-8 -*-

__author__ = 'popka'

from Impurity import Impurity
import numpy as np

class RegressionImpurity(Impurity):

    def calculate_split(self, y_left, y_right):
        """
        Уменьшение gini.
        Критерий разделения узла. Для разделения выбирается предикат, с максимальным _delta_gini
        :param y_left: np.array левого узла дерева
        :param y_right: np.array правого узла дерева
        :return:
        """
        y = np.append(y_left, y_right)
        left_imp = float(len(y_left))*self.calculate_node(y_left)/len(y)
        right_imp = (float(len(y_right))*self.calculate_node(y_right))/len(y)
        return self.calculate_node(y) - right_imp - left_imp


    def calculate_node(self, y):
        """
        считает gini
        :param y:
        :return:
        """
        """
        mean = np.mean(y)
        diff = y - mean
        diff = diff**2
        impurity = sum(diff)/len(y)
        """

        impurity = np.var(y)

        return impurity