# -*- coding: utf-8 -*-
from Predicate import Predicate

__author__ = 'popka'


import numpy as np
from random import randint
from Impurity.Gini import Gini
from Impurity.RegressionImpurity import RegressionImpurity
from Node import Node
from Splitter import Splitter


class DecisionTree():

    def __init__(self, is_classification=True, impurity=None, min_samples_leaf=5, min_impurity=0.1, max_features=15):

        self._impurity = impurity
        self._is_classification = is_classification
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features
        self._min_impurity = min_impurity

        if self._is_classification:
            if impurity is None:
                self._impurity = Gini()

        else:
            if impurity is None:
                self._impurity = RegressionImpurity()

        self._root = None
        self._splitter = Splitter()


    def fit(self, X_, y_):

        y = np.copy(np.asarray(y_))
        X = np.copy(np.matrix(X_))
        self._build_tree(X, y)


    def _build_tree(self, X, y):

        if not self._is_stop_criterion(y) > 0:
            predicate = self.select_predicate(X, y)
            self._root = Node(predicate=predicate)

            X_left, y_left, X_right, y_right = self._root.predicate.split_by_predicate(X, y)
            self._root.left_node = self._create_node(X_left, y_left)
            self._root.right_node = self._create_node(X_right, y_right)
        else:
            self._root = Node(is_leaf=True, value=y[0])


    def _create_node(self, X, y):
        """
        Построение дерева
        :param X:
        :param y:
        """

        if not self._is_stop_criterion(y):
            #print "select_predicate started"
            predicate = self.select_predicate(X, y)
            #print "select_predicate stoped"
            node = Node(predicate=predicate)
            X_left, y_left, X_right, y_right = node.predicate.split_by_predicate(X, y)

            if len(y_left) == 0 or len(y_right) == 0:
                """
                Если оптимальным считается разделить узел так, что в одной части будут значения, а в другой - нет
                (это может случится, когда все значения признака одинаковые), считаем, что это лист
                """
                value = self._select_leaf_value(y)
                return Node(predicate=None, is_leaf=True, value=value)

            node.left_node = self._create_node(X_left, y_left)
            node.right_node = self._create_node(X_right, y_right)

            return node
        else:

            value = self._select_leaf_value(y)
            return Node(predicate=None, is_leaf=True, value=value)


    def _select_leaf_value(self, y):
        """
        Функция вычисляет и возвращает значение, которое будет находиться в листе
        :param y:
        """
        if self._is_classification:
            counts = np.bincount(y)
            value = np.argmax(counts)
        else:
            value = np.mean(y)

        return value

    def _is_stop_criterion(self, y):
        """
        Критерий останова.
        Пока прос
        :param y:
        :return:
        """
        return self._impurity.calculate_node(y) < self._min_impurity or \
                len(y) < self._min_samples_leaf


    def select_predicate(self, X, y):
        """
        Функция выбирает оптимальный предикат и возвращает его
        :param X:
        :param y:
        :return:
        """

        feature_indexes = DecisionTree.rsm(min(len(X[0]), self._max_features)) # Массив индексов фичей (какие столбцы будем просматривать)
        max_delta_impurity = None

        delta_imp_debug = []
        for feature_index in feature_indexes:
            x = X[:,feature_index] # Столбец значений фичи (значения фичи для всех объектов)

            if DecisionTree._is_categorical(x):
                type = Predicate.CAT
                value, delta_impurity = self._splitter.split_categorial(x=x, y=y, impurity=self._impurity)

            else:
                type = Predicate.QUAN
                value, delta_impurity = self._splitter.split_quantitative(x=x, y=y, impurity=self._impurity)

            delta_imp_debug.append(delta_impurity)

            if max_delta_impurity < delta_impurity:
                max_delta_impurity = delta_impurity
                best_feature_index = feature_index
                best_value = value
                print best_value

            return Predicate(type=type, feature_id=best_feature_index, value=best_value)



    def predict(self, X):
        y = np.zeros(len(X))

        for i in range(len(X)):
            x = X[i]
            node = self._root
            while(not node.is_leaf):
                node = node.get_next_node(x)
            y[i] = node.value

        if self._is_classification:
            return y.astype(int)
        else:
            return y

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