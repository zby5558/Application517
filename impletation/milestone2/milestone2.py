# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 01:29:55 2018

@author: Administrator
"""
import math
from sklearn import svm
import time
import numpy as np
import xlrd
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
fname = 'cancer1.xlsx'
def getData(fname): # getData from the xlsx
    
    df = pd.read_excel(fname, sheet_name='Sheet1')
    df = df.values
    yTr = df[:,0:1]
    xTr = df[:,1:31]
    return xTr, yTr
xTr,yTr = getData(fname)
'''m = yTr.shape
yTr1 = yTr.reshape(m[0],1)
yTr = yTr1
yTr = yTr[0:m[0]-2,0:1]
yTr = yTr.reshape(m[0]-2,1)
yTr.astype(int)
xTr = xTr[0:m[0]-2,:]
'''
#xTr_norm = preprocessing.normalize(xTr, norm='l2',axis = 0)
#xTr_norm = np.zeros(xTr.shape)
#for i in range(14):
#   xTr_norm[i] = (xTr[i]-np.mean(xTr[i]))/math.sqrt(np.var(xTr[i]))
#y = yTr[0:1000]
#X = xTr[0:1000,:]
gp_fix = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                   optimizer=None)

def crossValidation(xTr,yTr): # 10-fold cross validation. Every time use 10% to test and other 90% to learn
    m,n = xTr.shape
    t = round(m/10)
    scores = []
    for i in range(10):
        yTe = yTr[i*t:(i+1)*t,0:1]
        xTe = xTr[i*t:(i+1)*t,0:n]
        xTrain = xTr
        yTrain = yTr
        for j in range(i*t,(i+1)*t):           
            xTrain = np.delete(xTrain,[i*t],axis = 0)
            yTrain = np.delete(yTrain,[i*t],axis = 0)
        gp_fix = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0),
                                   optimizer=None)
        gp_fix.fit(xTrain,yTrain)    
        ypre = gp_fix.predict(xTe)
        error = 0
        for k in range(len(ypre)):
            if ypre[k] != yTe[k]:
                error = error+1
        scores.append(1-error/len(ypre))
    return scores
scores = crossValidation(xTr,yTr)
