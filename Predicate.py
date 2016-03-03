# -*- coding: utf-8 -*-
__author__ = 'popka'
import numpy as np

class Predicate():
    CAT = "categorial"
    QUAN = "quantitative"
    TYPES = [QUAN, CAT]

    def __init__(self, type, feature_id, value, gain):
        """

        :param type: тип признака - количественый(вещественное число) или категориальный(конечное множество целых)
        :param feature_id: index фичи
        :param value: если type=quantitative, то value - float.
                      если type=categorial, то value - [int, ]
        :raise Exception:
        """
        if type not in Predicate.TYPES:
            raise Exception('Type Error. Type should be {}'.format(Predicate.TYPES))

        self.type = type
        self.feature_id = feature_id
        self.value = value
        self.gain = gain


    def operate(self, x):
        """
        Функция возвращает результат работы предиката на объект x.
        true если x<=value или x in [value_1, value_2, ...]
        :param x: объект (строка в таблице)
        :rtype : bool
        """
        feature_value = x[self.feature_id]
        if self.type == Predicate.QUAN:
            #если признак количественный
            return feature_value <= self.value
        else:
            #если признак категориальный
            return feature_value in self.value


    def split_by_predicate(self, X, y):
        """
        Функция делит Матрицу X и массив Y по предикату self
        :param X:
        :param y:
        :return:
        """

        x = X[:,self.feature_id]
        if self.type == Predicate.QUAN:
            #если признак количественный
            split_mask = x <= self.value

        else:
            #если признак категориальный
            split_mask = np.in1d(x, self.value)

        X_left = X[split_mask]
        y_left = y[split_mask]

        X_right = X[np.invert(split_mask)]
        y_right = y[np.invert(split_mask)]

        return X_left, y_left, X_right, y_right


    def print_predicate(self):
        return "{} predicate. if feature[{}] <= {} -> left".format(self.type, self.feature_id, self.value)