# -*- coding: utf-8 -*-

__author__ = 'popka'

class Node():

    def __init__(self, predicate=None, is_leaf=False, value=None):
        """
        :param predicate: если предикат соотвествует значению объекта, то он пойдет в левый узел, иначе в правый
        """
        self.predicate = predicate

        self.is_leaf = is_leaf
        self.value = value

        self.left_node = None
        self.right_node = None


    def get_next_node(self, x):

        if self.predicate.operate(x):
            return self.left_node

        else:
            return self.right_node