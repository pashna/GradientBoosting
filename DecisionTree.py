# -*- coding: utf-8 -*-
from Predicate import Predicate

__author__ = 'popka'


import numpy as np
from random import randint
from Impurity.Gini import Gini
from Node import Node
from Splitter import Splitter
import traceback



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
        self._splitter = Splitter()


    def fit(self, X_, y_):

        y = np.copy(np.asarray(y_))
        X = np.copy(np.matrix(X_))

        if (self._is_classification):
            #self._classes = np.unique(y) # Возможно, не нужно делать свойством объекта
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

        if not self._is_stop_criterion(y) > 0:
            predicate = self.select_predicate(X, y)
            node = Node(predicate=predicate)
            X_left, y_left, X_right, y_right = node.predicate.split_by_predicate(X, y)

            if (len(y_left) == 0) or (len(y_right) == 0):
                """
                Если данные больше нельзя разделить. (все оставшиеся имеют одинаковый признак),
                то считаем, что нашли узел, и выбираем значение узла как наиболее встречающееся
                ПОКА РАНДОМ, ПОПРАВИТЬ СРАЗУ ЖЕ!
                """
                counts = np.bincount(y)
                value = np.argmax(counts)
                return Node(predicate=None, is_leaf=True, value=value)

            node.left_node = self._create_node(X_left, y_left)
            node.right_node = self._create_node(X_right, y_right)

            return node
        else:
            return Node(predicate=None, is_leaf=True, value=y[0])




    def _is_stop_criterion(self, y):
        """
        Критерий останова.
        Пока прос
        :param y:
        :return:
        """
        return self._impurity.calculate_node(y) == 0


    def select_predicate(self, X, y):
        """
        Функция выбирает оптимальный предикат и возвращает его
        :param X:
        :param y:
        :return:
        """
        try:
            feature_indexes = DecisionTree.rsm(len(X[0])) # Массив индексов фичей (какие столбцы будем просматривать)
            max_delta_impurity = -1

            for feature_index in feature_indexes:
                x = X[:,feature_index] # Столбец значений фичи (значения фичи для всех объектов)

                if DecisionTree._is_categorical(x):
                    type = Predicate.CAT
                    value, delta_impurity = self._splitter.split_categorial(x=x, y=y, impurity=self._impurity)

                else:
                    type = Predicate.QUAN
                    value, delta_impurity = self._splitter.split_quantitative(x=x, y=y, impurity=self._impurity)

                if max_delta_impurity < delta_impurity:
                    max_delta_impurity = delta_impurity
                    best_feature_index = feature_index
                    best_value = value


            return Predicate(type=type, feature_id=best_feature_index, value=best_value)

        except Exception:
            print X
            traceback.print_exc()
            exit()


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