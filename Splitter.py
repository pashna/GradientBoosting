# -*- coding: utf-8 -*-
from Predicate import Predicate

__author__ = 'popka'
from utils.Placement import Placement
import numpy as np
from math import fabs


class Splitter():
    """
    Сплиттер находит оптимальное разделение "параметр предиката"
    """

    @staticmethod
    def split_categorial(x, y, impurity):

        combiner = Placement(np.unique(x))
        max_impurity = -1
        best_c = []
        for c in combiner:

            mask = np.in1d(x, c)
            y_left = y[mask]
            y_right = y[np.invert(mask)]
            imp = impurity.calculate_split(y_left, y_right)

            if imp > max_impurity:
                max_impurity = imp
                best_c = c

        return best_c, max_impurity


    @staticmethod
    def split_quantitative(x, y, impurity, steps=100):
        """
        Функция делит количественную переменную так, чтобы разделение приводило к максимальному уменьшению импюрити
        :param x: столбец признака
        :param y: столбец целевой функции
        :param impurity: объект-наследний класса Impurity
        :param steps: количество шагов. Если None, то просмотреть все объекты в отдельности
        :return: возвращает лучшее разделение и знаение импюрити
        """
        max_impurity = -1
        best_value = 0

        if steps is None:
            argsort = x.argsort()
            x = x[argsort]
            y = y[argsort]


            for value in x[:-1]:

                y_left = y[x<=value]
                y_right = y[x>value]
                if len(y_right) == 0:
                    continue
                imp = impurity.calculate_split(y_left, y_right)

                if imp > max_impurity:
                    max_impurity = imp
                    best_value = value
        else:
            current = min(x)
            end = max(x)
            step = float(end-current)/steps
            current += step
            # 1.1 только, чтобы оставался в конце хотя бы один элемент. Т.е. Значение было больше одного шага и меньше двух шагов
            while (current < end-step*1.1):
                y_left = y[x<=current]
                y_right = y[x>current]
                imp = impurity.calculate_split(y_left, y_right)

                if imp > max_impurity:
                    max_impurity = imp
                    best_value = current

                current += step

        return best_value, max_impurity


    @staticmethod
    def split_quick_quantitative(x, y, impurity):

        max_gain = None
        best_value = None

        argsort = x.argsort()
        x = x[argsort]
        y = y[argsort]

        imp_full = impurity.calculate_node(y)
        length = len(y)

        #rs - right_split
        rs = Split(np.mean(y[1:]), imp_full, impurity.calculate_node(y[1:]), len(y)-1)

        #ls - left_split
        ls = Split(y[0], 0, 0, 1)


        for i in range(1, len(y)-1):

            rs.prev_mean = rs.mean
            rs.mean = (rs.length*rs.mean - y[i])/(rs.length-1)
            rs.var = fabs((rs.var*rs.length-(y[i]-rs.mean)*(y[i]-rs.prev_mean))/(rs.length-1))
            rs.length -= 1.

            ls.length += 1.
            ls.prev_mean = ls.mean
            ls.mean = ls.prev_mean+(y[i]-ls.prev_mean)/ls.length
            ls.var = fabs(((ls.length-1)*ls.var + (y[i]-ls.prev_mean)*(y[i]-ls.mean))/ls.length)

            if (x[i+1]!=x[i]):
                gain = imp_full - rs.var*rs.length/length - ls.var*ls.length/length

                if gain > max_gain:
                    max_gain = gain
                    best_value = x[ls.length-1]

        # если не нашли максимального (по сути, когда все значения столбца одинаковы)
        # возвращаем None, None
        return best_value, max_gain



class Split():

    def __init__(self, mean, prev_mean, var, length):
        self.mean = float(mean)
        self.prev_mean = float(prev_mean)
        self.var = float(var)
        self.length = float(length)