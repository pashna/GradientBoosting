# -*- coding: utf-8 -*-

__author__ = 'popka'

class Node():

    def __init__(self, predicate):
        """
        :param predicate: если предикат соотвествует значению объекта, то он пойдет в левый узел, иначе в правый
        """
        self.predicate = predicate

        self.is_leaf = False
        self._class = None

        self.left_node = None
        self.right_node = None


    def get_next_node(self, x):

        if self.predicate.operate(x):
            return self.left_node

        else:
            return self.right_node



class Predicate():

    TYPES = ["quantitative", "categorial"]

    def __init__(self, type, feature_id, value):
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


    def operate(self, x):
        """
        Функция возвращает результат работы предиката на объект x.
        true если x<=value или x in [value_1, value_2, ...]
        :param x: объект (строка в таблице)
        :rtype : bool
        """
        feature_value = x[self.feature_id]
        if self.type == Predicate.TYPES[0]:
            #если признак количественный
            return feature_value <= self.value
        else:
            #если признак категориальный
            return feature_value in self.value
