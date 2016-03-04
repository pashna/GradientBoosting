# -*- coding: utf-8 -*-

__author__ = 'popka'

import numpy as np
from DecisionTree import DecisionTree
import time
from sklearn.tree import DecisionTreeRegressor as DecisionTreeSK

class GradientBoosting():


    def __init__(self, n_estimators=10, shrinkage=0.05, max_depth=10, impurity=None, min_samples_leaf=1, max_features=25, min_features=5, max_steps=100, rsm=True):
        # Boosting Parameters
        self._n_estimators = n_estimators#-1 ???
        self._estimators = []
        self._b = []
        self._shrinkage = shrinkage

        # Tree Parameters
        self._max_depth = max_depth
        self._impurity = impurity
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features
        self._min_features = min_features
        self._max_steps = max_steps
        self._rsm = rsm

        self._current_predicted = None


    def fit(self, X, y):

        self._initial_approximation(X, y)

        for i in range(self._n_estimators):
            anti_grad = self.calculate_antigradient(X, y)
            estimator = DecisionTree(max_depth=self._max_depth, is_classification=False, impurity=self._impurity, min_samples_leaf=self._min_samples_leaf, max_features=self._max_features, min_features=self._min_features, max_steps=self._max_steps, rsm=self._rsm)
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
        if self._current_predicted is not None:
            self._current_predicted += self._shrinkage * self._estimators[len(self._estimators) - 1].predict(X)

        else:
            self._current_predicted = self.predict(X)

        q = y - self._current_predicted
        #q = q/np.absolute(q)
        return q


    def predict(self, X):
        y_predicted = self._get_h_0(X)

        for estimator in self._estimators:
            y_predicted += self._shrinkage*estimator.predict(X)

        return y_predicted


    def predict_n(self, X, n):
        """
        Функция считает predict по первым n-деревьям
        :param X:
        :param n:
        """
        y_predicted = self._get_h_0(X)

        for i in range(n):
            y_predicted += self._shrinkage*self._estimators[i].predict(X)

        return y_predicted

    def _initial_approximation(self, X, y):
        """
        Реализует начальное приближение.
        В самом простом случае - константа - среднее значение.
        :param X:
        :param y:
        """
        self._y_mean = np.mean(y)
        #self._first_estimator = DecisionTree(max_depth=3)#, is_classification=False, rsm=False)
        #self._first_estimator.fit(X, y)
        #self._b.append(1)


    def _get_h_0(self, X):
        return np.asarray([self._y_mean]*len(X))
        #return self._first_estimator.predict(X)