# -*- coding: utf-8 -*-

__author__ = 'popka'

from abc import ABCMeta, abstractmethod

class Impurity():
    __metaclass__=ABCMeta

    @abstractmethod
    def calculate_split(self, y_left, y_right):
        """
        Вычисляет изменение Impurity для разделения узла на y_left и y_right
        ЧЕМ ВЫШЕ ЭТО ЗНАЧЕНИЕ, ТЕМ ЛУЧШЕ РАЗДЕЛЕНИЕ
        :param y_left: массив значений классов(или значений регрессии), который пойдет влево
        :param y_right: массив значений классов(или значений регрессии), который пойдет вправо
        """
        pass

    @abstractmethod
    def calculate_node(self, y):
        """
        Функция считает impurity в самом узле.
        Чем меньше значение, тем лучше
        :param y:
        """
        pass
