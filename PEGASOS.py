# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:57:14 2016

@author: Ruggiero
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.kernel_approximation import RBFSampler

lam = 0.000001
m=1000
np.random.seed(13)
#mean=np.zeros([400],dtype='float')
#covariance=np.zeros([400,400], dtype='float')
#np.fill_diagonal(covariance,1)
w1=np.zeros([m,400],dtype='float')/np.sqrt(m)
b=np.zeros([m],dtype='float')

for i in range(m):
    w1[i]=np.random.normal(0,1,[400])
    b[i]=np.random.uniform(0,2*np.pi)
#eta = 0.001
def transform(X):
    if np.ndim(X)==2:
        nData=len(X[:,1])
    else:
        nData=len(X)
    
    if np.ndim(X)==2:
        Xtrans=np.zeros([nData,m],dtype='float')
        print X.shape
        for i in range(len(X[:,1])):#prima dimensione
            for j in range(len(X[1,:])): # seconda dimensione
                Xtrans[i][j]=np.sqrt(2)*np.cos(np.inner(w1[j],X[i])+b[j])
    else:    
        Xtrans=np.zeros([m],dtype='float')
        for i in range(X.size):
            Xtrans[i]=np.sqrt(2)*np.cos(np.inner(w1[i],X)+b[i])
    return Xtrans
    
    '''
    rbf_feature = RBFSampler(gamma=30,  n_components=400, random_state=1)#5300
    if np.ndim(X)==2:
        for i in range(len(X[:,1])):
            rbf_feature.fit(X[i])
            x_trans = rbf_feature.transform(X[i])
            X[i]=x_trans 

    else:
        X=X.reshape(1,-1)
        rbf_feature.fit(X)
        x_trans = rbf_feature.transform(X)
        X=x_trans
    #print x_trans[0].shape


    return X
    '''
  
def mapper(key, value):
    # key: None
    # value: m line of input file
    global lam
    w = np.zeros([m], dtype='float')
    cnt = 0.0
    a=0
    cont=0
    for i in value:
        a+=1

    '''sampling'''
    for i in value:
        cont+=1
        '''
        if np.random.uniform()>=0.001:
            continue
        '''

        #print(i)
        tokens = i.split()
        y = float(tokens[0])
        x = np.asarray(tokens[1:]).astype(np.float)
        x=transform(x)
        cnt = cnt + 1.0
        eta = 1/(lam*cnt)
        #print w.size,x.size
        if y*np.inner(w,x)<1:
            w=(1-eta*lam)*w+eta*y*x
        else:
            w=(1-eta*lam)*w
        w=min(1,1/np.sqrt(lam)/np.linalg.norm(w,1))*w

        if cont%5000==0:
            print cont,a,cnt
    yield 1,w

def reducer(key, values):
    wfin = 0.0
    count = 0.0
    for i in values:
        wfin = wfin + i
        count = count + 1.0
    yield wfin/count

'''
def mapper(key,value):
    # key: None
    # value: m line of input file
    global lam
    w = np.zeros([400], dtype='float')
    cnt = 0.0
    s=np.zeros([400],dtype='float')
    for i in value:
 
        #print(i)
        tokens = i.split()
        y = float(tokens[0])
        x = np.asarray(tokens[1:]).astype(np.float)
        cnt = cnt + 1.0
        eta = 1/(lam*cnt)
        for j in range(400):
            s[j]=s[j]+np.power((y*x[j]),2)
            if (s[j]==0):
                if s[j]==0:
                    s[j]=1
            w[j]=w[j]+eta/np.sqrt(s[j])*(y*x[j])

    yield 1,w

'''
