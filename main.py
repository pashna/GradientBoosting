# -*- coding: utf-8 -*-
__author__ = 'popka'


import numpy as np

import pandas as pd
import matplotlib.pylab as pl
import sklearn.cross_validation as cv
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from DecisionTree import DecisionTree
from sklearn.metrics import mean_squared_error as mse
from GradientBoosting import GradientBoosting

FOLDER = "data/"
FILES = [
        "iris.txt", "bezdekIris.txt", "wine.txt", "bupa.txt", "housing.txt", "auto-mpg.txt", "spam"
        ]
FILE = "auto-mpg.txt"

# Подготавливаем признаки и целевую функцию
if FILE in FILES[:6]:

    df = pd.read_csv(FOLDER+FILE, sep=",", header=None)#, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)

    if FILE in FILES[:2]:
        # ИРИСЫ
        df[4] = pd.factorize(df[4])[0]
        X = df[[0,1,2,3]].as_matrix()
        y = df[4].as_matrix()

    if FILE == FILES[2]:
        x_indexes = [x for x in range(1,14)]
        X = df[x_indexes].as_matrix()
        y = df[0].as_matrix()

    if FILE == FILES[3]:
        X = df[[0,1,2,3,4,5]].as_matrix()
        y = df[6]

    if FILE == FILES[4]:
        df = pd.read_csv(FOLDER+FILE, sep=" ", header=None)#, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
        X = df[df.columns[1:]].as_matrix()
        y = df[df.columns[0]].as_matrix()

    if FILE == FILES[5]:
        df = pd.read_csv(FOLDER+FILE, sep=" ", header=None)#, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
        X = df[df.columns[1:-1]].as_matrix()
        y = df[df.columns[0]].as_matrix()

    x_train, x_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.25)

else:

    df_train = pd.read_csv(FOLDER+FILE+".train.txt", sep=" ", header=None)#, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    df_test = pd.read_csv(FOLDER+FILE+".test.txt", sep=" ", header=None)#, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)
    x_train = df_train[df_train.columns[1:]].as_matrix()
    y_train = df_train[df_train.columns[0]].as_matrix()
    x_test = df_test[df_test.columns[1:]].as_matrix()
    y_test = df_test[df_test.columns[0]].as_matrix()

"""
my_gb = GradientBoosting(n_estimators=0, max_depth=4, shrinkage=0, max_steps=None, rsm=False)
my_gb.fit(x_train, y_train)
y_predicted = my_gb.predict(x_test)
print mse(y_test, y_predicted)
"""
my_tree = DecisionTree(is_classification=False, max_features=len(x_train[0]), max_steps=None)
my_tree.fit(x_train, y_train)
y_predicted = my_tree.predict(x_test)
print mse(y_test, y_predicted)
