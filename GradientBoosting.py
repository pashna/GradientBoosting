# -*- coding: utf-8 -*-

__author__ = 'popka'

import numpy as np
from DecisionTree import DecisionTree

class GradientBoosting():


    def __init__(self, n_estimators=10, shrinkage=0.05, impurity=None, min_samples_leaf=5, min_impurity=0.1, max_features=15, max_steps=100):
        # Boosting Parameters
        self._n_estimators = n_estimators#-1 ???
        self._estimators = []
        self._shrinkage = shrinkage

        # Tree Parameters
        self._impurity = impurity
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features
        self._min_impurity = min_impurity
        self._max_steps = max_steps


    def fit(self, X, y):

        self._initial_approximation(X, y)

        for i in range(self._n_estimators):
            anti_grad = self.calculate_antigradient(X, y)
            estimator = DecisionTree(is_classification=False, impurity=self._impurity, min_impurity=self._min_impurity, min_samples_leaf=self._min_samples_leaf, max_features=self._max_features, max_steps=self._max_steps)
            estimator.fit(X, anti_grad)
            self._estimators.append(estimator)


    def calculate_antigradient(self, X, y):
        """
        Вычисляет антиградиент.
        Для квадратичной функции потерь антиградиент равен y - h(x)
        :param X:
        :param y:
        :return:
        """
        return y - self.predict(X)


    def predict(self, X):
        y_predicted = self._shrinkage*self._get_h_0(X)

        for estimator in self._estimators:
            y_predicted += self._shrinkage*estimator.predict(X)

        return y_predicted


    def _initial_approximation(self, X, y):
        """
        Реализует начальное приближение.
        В самом простом случае - константа - среднее значение.
        :param X:
        :param y:
        """
        self._y_mean = np.mean(y)


    def _get_h_0(self, X):
        return np.asarray([self._y_mean]*len(X))