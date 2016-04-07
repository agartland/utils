#! /usr/bin/env python
"""
Python wrapper to execute c++ tSNE implementation
for more information on tSNE, go to :
http://ticc.uvt.nl/~lvdrmaaten/Laurens_van_der_Maaten/t-SNE.html

HOW TO USE
Just call the method calc_tsne(dataMatrix)

Created by Philippe Hamel
hamelphi@iro.umontreal.ca
October 24th 2008
"""

from struct import *
import sys
import os
from numpy import *
import subprocess

TSNE_DIR = 'C:/Program Files (x86)/tSNE/'

def run_tsne(X = array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
    """
    This is the main function.
    dataMatrix is a 2D numpy array containing your data (each row is a data point)
    Remark : LANDMARKS is a ratio (0<LANDMARKS<=1)
    If LANDMARKS == 1 , it returns the list of points in the same order as the input
    """
    LANDMARKS = 1
    
    X=PCA(X,initial_dims)
    writeDat(X,no_dims,perplexity,LANDMARKS)
    tSNE()
    Xmat,LM,costs=readResult()
    clearData()
    if LANDMARKS==1:
        return reOrder(Xmat,LM)
    return Xmat,LM

def PCA(dataMatrix, INITIAL_DIMS) :
    """
    Performs PCA on data.
    Reduces the dimensionality to INITIAL_DIMS
    """
    print 'Performing PCA'

    dataMatrix= dataMatrix-dataMatrix.mean(axis=0)

    if dataMatrix.shape[1]<INITIAL_DIMS:
        INITIAL_DIMS=dataMatrix.shape[1]

    (eigValues,eigVectors)=linalg.eig(cov(dataMatrix.T))
    perm=argsort(-eigValues)
    eigVectors=eigVectors[:,perm[0:INITIAL_DIMS]]
    dataMatrix=dot(dataMatrix,eigVectors)
    return dataMatrix

def readbin(type,file) :
    """
    used to read binary data from a file
    """
    return unpack(type,file.read(calcsize(type)))

def writeDat(dataMatrix,NO_DIMS,PERPLEX,LANDMARKS):
    """
    Generates data.dat
    """
    print 'Writing data.dat'
    print 'Dimension of projection : %i \nPerplexity : %i \nLandmarks(ratio) : %f'%(NO_DIMS,PERPLEX,LANDMARKS)
    n,d = dataMatrix.shape
    f = open(TSNE_DIR+'data.dat', 'wb')
    f.write(pack('=iiid',n,d,NO_DIMS,PERPLEX))
    f.write(pack('=d',LANDMARKS))
    for inst in dataMatrix :
        for el in inst :
            f.write(pack('=d',el))
    f.close()


def tSNE():
    """
    Calls the tsne c++ implementation depending on the platform
    """
    cmd=[TSNE_DIR+'tSNE.exe']
    print 'Calling executable "%s"' % cmd[0]

    tsneProcess=subprocess.Popen(cmd,stdout=subprocess.PIPE,cwd=TSNE_DIR)
    print tsneProcess.communicate()[0]

def readResult():
    """
    Reads result from result.dat
    """
    print 'Reading result.dat'
    f=open(TSNE_DIR+'result.dat','rb')
    n,ND=readbin('ii',f)
    Xmat=empty((n,ND))
    for i in range(n):
        for j in range(ND):
            Xmat[i,j]=readbin('d',f)[0]
    LM=readbin('%ii'%n,f)
    costs=readbin('%id'%n,f)
    f.close()
    return (Xmat,LM,costs)

def reOrder(Xmat, LM):
    """
    Re-order the data in the original order
    Call only if LANDMARKS==1
    """
    print 'Reordering results'
    X=zeros(Xmat.shape)
    for i,lm in enumerate(LM):
        X[lm]=Xmat[i]
    return X

def clearData():
    """
    Clears files data.dat and result.dat
    """
    print 'Clearing data.dat and result.dat'
    os.system('rm %sdata.dat' % TSNE_DIR)
    os.system('rm %sresult.dat' % TSNE_DIR)
