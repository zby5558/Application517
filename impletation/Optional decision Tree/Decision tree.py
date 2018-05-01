# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 01:11:59 2018

@author: Administrator
"""
import math
from sklearn import svm
import time
import numpy as np
import xlrd
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
fname = 'cancer1.xlsx'

def getData(fname): # getData from the xlsx
    
    df = pd.read_excel(fname, sheet_name='Sheet1')
    df = df.values
    yTr = df[:,0:1]
    xTr = df[:,1:31]
    return xTr, yTr
xTr,yTr = getData(fname)
clf = DecisionTreeClassifier(max_depth = 5)
cross = cross_val_score(clf, xTr, yTr, cv=10)