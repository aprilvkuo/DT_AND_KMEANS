#!/usr/bin/env python
# encoding: utf-8

"""
@author: Aprilvkuo
@file: km.py
@time: 16-11-23 下午11:30
"""



from numpy import *
import numpy
import time
import matplotlib.pyplot as plt
from math import sqrt
import pandas as pd
import numpy as np

def readmod(name):
    data = (pd.read_csv(name))
    #print data
    return np.array(data.values)

# calculate Euclidean distance
def euclDistance(vector1, vector2):
    return sqrt(sum(numpy.power(vector2 - vector1, 2)))


# init centroids with random samples
'''
def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    for i in range(k):
        index = int(random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids
'''


def initCentroids(dataSet, k):
    centroindex=[]
    numSamples, dim = dataSet.shape
    centroids = zeros((k, dim))
    index = int(random.uniform(0, numSamples))
    centroindex.append(index)
    for centroid_num in range(k-1):
        max = -1
        max_index= -1
        for Sample_num in range (numSamples):
            if Sample_num in centroindex:
                continue
            sum= 0
            for index in  centroindex:
                sum += euclDistance(dataSet[index, :],dataSet[Sample_num,:])
            if sum >max:
                max ,max_index= sum,Sample_num
        centroindex.append(max_index)
    for i in range (len(centroids)):
        centroids[i, :] = dataSet[centroindex[i], :]

    return centroids

def cacu_sse(pointsInCluster):
    sum = 0
    for i in range (len(pointsInCluster)):
        sum += pointsInCluster[i,1]
    return sum

def distance(dataSet,point,index_in_set):
    dis_in_set = 0.0
    for set_index in index_in_set:
        dis_in_set += euclDistance(point, dataSet[set_index, :])
    return dis_in_set

def cacu_sc(dataSet,clusterAssment,N):
    len_data=len(dataSet)
    for index in range (len_data):
        data = dataSet[index,:]
        set = clusterAssment[index][0]
        index_in_set = [i for i in range(len_data) if clusterAssment[i][0]==set]
        dis_in_set=distance(dataSet,data,index_in_set)
        dis_in_set /=float(len(index_in_set)-1)#类内平均距离
        otherset = [i for i in range(N) if i!=set]
        best_b = -1

# k-means cluster
def kmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    # n*2的矩阵,第一列表示数据所属的类别,第二列为数据到样本中心的距离的平方
    clusterAssment = mat(zeros((numSamples, 2)))
    clusterChanged = True

    ## step 1: init centroids
    centroids = initCentroids(dataSet, k)

    while clusterChanged:
        clusterChanged = False
        ## for each sample
        for i in xrange(numSamples):
            minDist = 100000.0
            minIndex = 0
            ## for each centroid
            ## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j

                    ## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2

                ## step 4: update centroids
        for j in range(k):
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
            centroids[j, :] = mean(pointsInCluster, axis=0)
            #中心点为全部点的平均值

    #print 'Congratulations, cluster complete!'
    return centroids, clusterAssment



def biKmeans(dataSet, k):
    numSamples = dataSet.shape[0]
    # first column stores which cluster this sample belongs to,
    # second column stores the error between this sample and its centroid
    clusterAssment = mat(zeros((numSamples, 2)))

    # step 1: the init cluster is the whole data set
    centroid = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid]
    for i in xrange(numSamples):
        clusterAssment[i, 1] = euclDistance(mat(centroid), dataSet[i, :]) ** 2

    while len(centList) < k:
        # min sum of square error
        minSSE = 100000.0
        numCurrCluster = len(centList)
        # for each cluster
        for i in range(numCurrCluster):
            # step 2: get samples in cluster i
            pointsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            if len(pointsInCurrCluster)<=1:
                continue
            # step 3: cluster it to 2 sub-clusters using k-means
            centroids, splitClusterAssment = kmeans(pointsInCurrCluster, 2)

            # step 4: calculate the sum of square error after split this cluster
            splitSSE = sum(splitClusterAssment[:, 1])
            notSplitSSE = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            currSplitSSE = splitSSE + notSplitSSE
            if i==0: return kmeans(dataSet, k)
            # step 5: find the best split cluster which has the min sum of square error
            if currSplitSSE < minSSE:
                minSSE = currSplitSSE
                bestCentroidToSplit = i
                bestNewCentroids = centroids.copy()
                bestClusterAssment = splitClusterAssment.copy()

                # step 6: modify the cluster index for adding new cluster
        bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 1)[0], 0] = numCurrCluster
        bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 0)[0], 0] = bestCentroidToSplit

        # step 7: update and append the centroids of the new 2 sub-cluster
        centList[bestCentroidToSplit] = bestNewCentroids[0, :]
        centList.append(bestNewCentroids[1, :])

        # step 8: update the index and error of the samples whose cluster have been changed
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentroidToSplit), :] = bestClusterAssment

    return mat(centList), clusterAssment




if __name__ == '__main__':
    data = readmod('2_knowledge.csv')
    X=[]
    y=[]
    for i in range (5,15):
        dis = -1
        for j in range (10):
            centroids, clusterAssment= kmeans(data,i)
            if dis == -1:
                dis = cacu_sse(clusterAssment)
            else:
                tmp = cacu_sse(clusterAssment)
                if tmp <dis:
                    dis = tmp
        X.append(i)
        y.append(dis)
        print "聚类中心:",i,dis
    import numpy as np
    import pylab as pl
    pl.plot(X, y)  # use pylab to plot x and y
    pl.show()  # show the plot on the screen
    #print centroids, clusterAssment