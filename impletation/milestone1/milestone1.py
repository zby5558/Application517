import time
from numpy import *
import xlrd
import pandas as pd
import tensorflow as tf
fname = 'cancer1.xlsx'
def getData(fname): # getData from the xlsx
    
    df = pd.read_excel(fname, sheet_name='Sheet1')
    df = df.values
    yTr = df[:,0:1]
    xTr = df[:,1:31]
    return xTr, yTr
xTr,yTr = getData(fname)
def test(w, xTe,yTe): #test funstion is used to calculate the accuracy after giving test set and weight vectors
    m,n = xTe.shape 
    x = ones((m,1))
    xTe = c_[x,xTe]
    ytrain = dot(xTe,w)
    ytrain = sign(ytrain)        
    ytrain[where(ytrain == 0)] = 1
    b = ytrain*yTe
    t = 0
    for i in range(m):
            if(b[i] == -1):
                t = t+1
    m,n = yTe.shape
    return t/m     
def pla(xTr,yTr): # pla is used to get the weight vector w by moving the line to the right direction 
    m,n = xTr.shape
    x = ones((m,1))
    xTr = c_[x,xTr]
    w = ones((n+1,1))     
    print(w.shape)
    prevloss = 1
    loss = 0.99
    k = 0
    while(loss>0.08):
        t = 0
        ytrain = dot(xTr,w)
        ytrain = sign(ytrain)        
        ytrain[where(ytrain == 0)] = 1
        b = ytrain*yTr
        for i in range(m):
            if(b[i] == -1):
                t = t+1
                wp = yTr[i]*xTr[i,:].T   
                for j in range(n):
                    w[j] = w[j]+wp[j]                   
        prevloss = loss        
        loss = t/m;     
        print(loss)
        
    return w, loss
def crossValidation(xTr,yTr): # 10-fold cross validation. Every time use 10% to test and other 90% to learn
    m,n = xTr.shape
    
    t = round(m/10)
    res = 0;
    for i in range(10):
        yTe = yTr[i*t:(i+1)*t,0:1]
        xTe = xTr[i*t:(i+1)*t,0:n]
        xTrain = xTr
        yTrain = yTr
        for j in range(i*t,(i+1)*t):           
            xTrain = delete(xTrain,[i*t],axis = 0)
            yTrain = delete(yTrain,[i*t],axis = 0)
        w,k = pla(xTrain,yTrain)
        print(w.shape)
        loss = test(w,xTe,yTe)
        res = res+loss;
    return res/10
res = crossValidation(xTr,yTr)     
print(res)
