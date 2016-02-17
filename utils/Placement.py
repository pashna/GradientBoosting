# -*- coding: utf-8 -*-
__author__ = 'popka'
import numpy as np

class Placement(object):
    """
    Класс-итератор. В конструктор получает целочисленный массив.
    Затем каждая итерация возвращает следующее значение размещения без повторений, кроме всех единичек и всех

    Работает, перебирая 0,1,2,3,... и представляя эти числа в бинарном виде. Затем, накладывает эту маску на исходный массив
    """
    def __init__(self, unique_list):
        self.unique_list = unique_list

        self._max_length = len(unique_list) # Длина списка маски
        self._max_value = 2**self._max_length # Максимальное значение маски

        self._i = 0



    def _int_to_bin_array(self, value):
        """
        Функция возвращает двоичный массив по числу. Например, self._max_number = 4, value = 3, на выходе массив [0,0,1,1]
        :param value:
        """
        bin_array = np.zeros(self._max_length, dtype=bool)
        bin_str = list(bin(value)[2:])

        for i in range(1, len(bin_str)+1):
            bin_array[-i] = bin_str[-i] == '1'

        return bin_array


    def __iter__(self):
        return self


    def next(self):
        self._i += 1

        if self._i < self._max_value-1: # -1, потому что [1,1,1] нам не нужен
            mask = self._int_to_bin_array(self._i)
            return self.unique_list[ mask ]
        else:
            raise StopIteration
