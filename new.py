#!/usr/bin/env python
#encoding=utf8
__author__ = 'ZGD'


import  numpy as np
import matplotlib.pyplot  as plt
class KMean:
    def __init__(self):
        pass

    def calEuDistance(self,vecA,vecB):
        """

        :param vecA:
        :param vecB:
        :return:
        """
        return np.sqrt(np.sum(np.power(vecA-vecB,2)))

    def initCentroids(self,dataSet,k):
        """

        :param dataSet:
        :param k:
        :return:
        """
        #获取数据集的数据个数以及维数
        numSamples,dim = dataSet.shape
        centroids = np.zeros((k,dim))
        #随机初始化k个起始中心点
        for i in range(k):
            index = int(np.random.uniform(0,numSamples))
            centroids[i,:] = dataSet[index,:]
        return centroids

    def calSSE(self,clusterDict,centroidList):
        """

        :param clusterDict:
        :param centroidList:
        :return:
        """
        sum = 0.0
        for key in clusterDict.keys():
            vec1 = np.array(centroidList[key])
            distance = 0.0
            for item in clusterDict[key]:
                vec2 = np.array(item)
                distance += self.calEuDistance(vec1,vec2)
            sum += distance
        return sum


    def kmeans(self,dataSet,centroids):
        """

        :param dataSet:
        :param k:
        :return:
        """
        numSamples = dataSet.shape[0]
        clusterDict = dict()
        #选择k个点作为初始质心
        # centroids = self.initCentroids(dataSet,k)

        for item in dataset:
            vec1 = np.array(item)
            flag = 0
            minDist = float("inf")
            for i in range(len(centroids)):
                vec2 = np.array(centroids[i])
                distance = self.calEuDistance(vec1,vec2)
                if distance < minDist:
                    minDist = distance
                    flag = i
            if flag not in clusterDict.keys():
                clusterDict[flag] = list()
            clusterDict[flag].append(item)
        return clusterDict

    def getCentroids(self,clusterDict):
        # 得到k个质心
        centroidList = list()
        for key in clusterDict.keys():
            centroid = np.mean(np.array(clusterDict[key]), axis=0)  # 计算每列的均值，即找到质心
            # print key, centroid
            centroidList.append(centroid)

        return np.array(centroidList).tolist()

    def showCluster(self,centroidList, clusterDict):
        # 展示聚类结果

        colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow','oc', '^r', '+r', 'sr', 'dr', '<r', 'pr']  # 不同簇类的标记 'or' --> 'o'代表圆，'r'代表red，'b':blue
        centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw','dc','^b', '+b', 'sb', 'db', '<b', 'pb']  # 质心标记 同上'd'代表棱形
        for key in clusterDict.keys():
            plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize=12)  # 画质心点
            for item in clusterDict[key]:
                plt.plot(item[0], item[1], colorMark[key])  # 画簇类下的点

        plt.show()
import pandas as pd
def readmod(name):
    data = (pd.read_csv(name))
    #print data
    return np.array(data.values)
def Main():
    kme = KMean()
    global dataset
    dataset = readmod("seeds.csv")
    stroreSSE = np.zeros(16)
    # print stroreSSE
    for kSize in range(1,16):
        centroidList = kme.initCentroids(dataset, kSize)
        clusterDict = kme.kmeans(dataset, centroidList)  # 第一次聚类迭代
        newVar = kme.calSSE(clusterDict, centroidList)  # 获得均方误差值，通过新旧均方误差来获得迭代终止条件
        oldVar = -0.0001  # 旧均方误差值初始化为-1
        # kme.showCluster(centroidList, clusterDict)  # 展示聚类结果
        k = 2
        while abs(newVar - oldVar) >= 0.0001:  # 当连续两次聚类结果小于0.0001时，迭代结束
            centroidList = kme.getCentroids(clusterDict)  # 获得新的质心
            clusterDict = kme.kmeans(dataset, centroidList)  # 新的聚类结果
            oldVar = newVar
            newVar = kme.calSSE(clusterDict, centroidList)
            k += 1
        stroreSSE[kSize] = newVar
    x=[i for i in range(1,16)]
    print x
    y = [j for j in stroreSSE[1:]]
    print y
    plt.plot(x,y,'b')
    plt.show()

def Print():
    Main()
