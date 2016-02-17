# -*- coding: utf-8 -*-
__author__ = 'popka'


import numpy as np
from random import randint
from Impurity.Gini import Gini
from Node import Node, Predicate

class DecisionTree():

    def __init__(self, max_depth = 10, is_classification = True, impurity=None):
        self._max_depth = max_depth

        self._is_classification = is_classification
        if impurity is None:
            self._impurity = Gini()
        else:
            self._impurity = impurity

        self._root = None
        self._depth = 0


    def fit(self, X_, y_):

        y = np.copy(np.asarray(y_))
        X = np.copy(np.asmatrix(X_))

        if (self._is_classification):
            self._classes = np.unique(y) # Возможно, не нужно делать свойством объекта
            self._build_tree(X, y)


    def _build_tree(self, X, y):
        """
        Построение дерева
        :param X:
        :param y:
        """
        pass


    def select_predicate(self, X, y):

        feature_indexes = DecisionTree.rsm(len(X[0])) # Массив индексов фичей (какие столбцы будем просматривать)

        for feature_index in feature_indexes:
            feature_values = X[:,feature_index] # Столбец значений фичи (значения фичи для всех объектов)

            if DecisionTree._is_categorical(feature_values):
                pass

            else:
                pass


    #TODO:
    """
        Чтобы получить число размещений, нужно перебрать поочередно все варианты. Например, для 4-х классов будет
        2^4-2 вариантов, бинарно закадированными. [0001, 0010, 0011, 0100..., 1110]
        Считаем все варианты и выбираем разбиение

        Здесь напрашивается фича max_feature, наконец стало понятно, что это за параметр.
    """

    @staticmethod
    def _is_categorical(x, max_categorial_feature_length=7):
        """
        Предполагаем, что категориальная фича должна быть:
        целой,
        меньше длины max_categorial_feature_length,

        :param x: столбец фичи
        :param max_categorial_feature_length: максимально возможная длина категориальный фичи. Чтобы не офигеть от количества вариантов
        :return:
        """

        if 'int' in str(x.dtype):
            if len(np.unique(x)) < max_categorial_feature_length:
                return True

        return False


    @staticmethod
    def rsm(feature_count):
        """
        Функция релизует random subspace method - возвращает массив случайной длины со случайными значениями, не превосходящими feature_count
        :param feature_count:
        """
        # TODO: Возможно имеет смысл ввести min_feature_count,
        rand_array = np.random.randint(feature_count, size=randint(1, feature_count))
        rand_array = np.unique(rand_array)

        return rand_array