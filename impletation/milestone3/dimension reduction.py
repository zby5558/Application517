# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 05:24:28 2018

@author: Administrator
"""

import scipy.io
import numpy as np
from sklearn.decomposition import PCA
import math
from sklearn import svm
import time
import numpy as np
import xlrd
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

import pandas as pd
fname = 'cancer1.xlsx'
def getData(fname): # getData from the xlsx
    
    df = pd.read_excel(fname, sheet_name='Sheet1')
    df = df.values
    yTr = df[:,0:1]
    xTr = df[:,1:31]
    return xTr, yTr
xTr,yTr = getData(fname)
def normalize(x):
    m,n = x.shape
    x_norm = np.zeros((m,n))
    for i in range(n):
        x_norm[:,i] = 2*(x[:,i]-np.min(x[:,i]))/(np.max(x[:,i])-np.min(x[:,i]))-1
    return x_norm
x_norm = normalize(xTr)
pca = PCA(n_components=0.8)
pca.fit(x_norm)
xn = pca.transform(x_norm)
mdict = {}
mdict['pca'] = xn
mdict['y'] = yTr
scipy.io.savemat('f:/517/impletation/pcadata.mat', mdict)
lin = LogisticRegression()
lin1 = LogisticRegression()
def crossValidation(xTr,yTr): # 10-fold cross validation. Every time use 10% to test and other 90% to learn
    m,n = xTr.shape 
    t = round(m/10)
    res = 0;
    for i in range(10):
        yTe = yTr[i*t:(i+1)*t,0:1]
        xTe = xTr[i*t:(i+1)*t,0:n]
        xTrain = xTr
        yTrain = yTr
        for j in range(int(i*t),int((i+1)*t)):           
            xTrain = np.delete(xTrain,[i*t],axis = 0)
            yTrain = np.delete(yTrain,[i*t],axis = 0)
        lin.fit(xTrain,yTrain)
        
        loss = lin.score(xTe,yTe)
        res = res+loss;
    return res/10
res = crossValidation(xn,yTr) 

