# -*- coding: utf-8 -*-
__author__ = 'popka'


import numpy as np
import pandas as pd
import matplotlib.pylab as pl
from DecisionTree import DecisionTree
import sklearn.cross_validation as cv


FOLDER = "data/"
FILE = "iris.txt"
df = pd.read_csv(FOLDER+FILE, sep=",", header=None)#, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)


# Подготавливаем признаки и целевую функцию
if (FILE == "iris.txt"):
    df[4] = pd.factorize(df[4])[0]
    X = df[[0,1,2,3]].as_matrix()
    y = df[4].as_matrix()



x_train, x_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.25)

tree = DecisionTree()
tree.fit(x_train, y_train)
print tree.predict(x_test)
