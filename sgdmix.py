# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 00:38:59 2016

@author: Siggi
"""

import numpy as np

lam = 0.00000002
#lam lam = 0.00000001
gamm = 100
c = 0
m = 2000
n = 400
k = 10
gamma = 8.75
#eta = 0.001
def transform(x_original):
    global w1
    global b
    global gamma
    global mDim
    Xtrans=gamma*np.dot(x_original,np.reshape(w1, (400, mDim)))
    Xtrans = 0.25*(np.concatenate((np.cos(Xtrans+np.reshape(b, (1, mDim))),np.cos(-Xtrans+np.reshape(b, (1, mDim)))),axis = 1)+1)
    Xtrans = np.concatenate((x_original,Xtrans),axis =1)
    return Xtrans

def mapper(key, value):
    global lam
    global gamm
    dat = np.zeros([len(value),401], dtype='f')
    count = 0
    for i in value:
        tokens = i.split()
        x = np.asarray(tokens[0:]).astype(np.float)
        dat[count,:] = x
        count = count + 1

    randindex = np.random.permutation(np.shape(dat)[0])
    for i in range(k):
        randindex = np.concatenate((randindex,np.random.permutation(np.shape(dat)[0])),axis=0)
    cnt = 0.0
    ratio = 0.0
    A = transform(dat[:,1:])
    w = np.zeros([np.shape(A)[1]], dtype='float')

    for i in range(np.shape(randindex)[0]):
        y = dat[randindex[i],0]
        x = A[randindex[i],:]
        cnt = cnt + 1.0
        eta = 1/(lam*cnt)
        L = np.inner(w,x)
        if y*L<=0:
            w=(1-eta*lam)*w+eta*y*x
        elif (y*L > 0) and (y*L < 1):
           w=(1-eta*lam)*w+eta*y*(1-L)*x
        else:
            w=(1-eta*lam)*w
        w=min(1,1/np.sqrt(lam)/np.linalg.norm(x,1))*w
    yield 1,w

def reducer(key, values):
    wfin = 0.0
    count = 0.0
    for i in values:
        wfin = wfin + i
        print(wfin)
        count = count + 1.0
    print(wfin/count)
    yield wfin/count
