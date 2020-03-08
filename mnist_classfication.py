# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:35:13 2020

@author: CVPR

for Kaggle
"""


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import os


# load csv
csv_path = list()
csv_path.append(os.path.join("", "train.csv"))
csv_path.append(os.path.join("", "test.csv"))
csv_path.append(os.path.join("", "sample_submission.csv"))


# data Part
mnist = pd.read_csv(csv_path[0])
mnistData = mnist.drop("label", axis=1)
mnistLabels = mnist["label"].copy()


'''
# csv -> numpy Array
mnistNumpy = mnist.values
'''


# training & K-fold 검증
leaf = 8192
estimators = 512
forestCla = RandomForestClassifier(n_estimators=estimators, max_leaf_nodes=leaf, n_jobs=-1)
forestCla.fit(mnistData, mnistLabels)
scores = cross_val_score(forestCla, mnistData, mnistLabels, cv=5)
print(scores)


# predict
mnistTest = pd.read_csv(csv_path[1])
predict = forestCla.predict(mnistTest)
mnistOutput = pd.read_csv(csv_path[2])
mnistOutput = mnistOutput.drop("Label", axis=1)
mnistOutput.insert(1, 'Label', predict)
mnistOutput.to_csv("sample_submission.csv", mode="w", index=False)