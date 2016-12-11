#!/usr/bin/env python
# encoding: utf-8

"""
@author: Aprilvkuo
@file: kmeans.py
@time: 16-11-15 上午11:06
"""
import pandas as pd
import xlrd
def readmod(name):
    data = (pd.read_csv(name))
    print data
    return np.array(data.values)
        #data = (pd.read_csv(name))
    #return np.array(data.values)



if __name__ == '__main__':
    from sklearn.cluster import KMeans
    import numpy as np
    import random
    data = readmod('2_seeds.csv')
    x=[]
    y=[]
    for i in range (5,10):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
        print '随机产生初始聚类中心数',i
        #print  '评分:',kmeans.score(X)
        #print kmeans.predict([[0, 0], [4, 4]])
        #print kmeans.cluster_centers_
        print 'sse=',kmeans.inertia_
        x.append(i)
        y.append(kmeans.inertia_)
    import numpy as np
    import pylab as pl
    pl.plot(x, y)  # use pylab to plot x and y
    pl.show()
    '''
    for i in range (5):
        matrix=  np.random.randint(0, 6,(len(X),len(X[0])))
        print matrix
        kmeans = KMeans(n_clusters=5,init=matrix, random_state=0).fit(X)
        print '随机产生初始聚类中心数', 5
        print  '评分:', kmeans.score(X)
        # print kmeans.predict([[0, 0], [4, 4]])
        print kmeans.cluster_centers_
        print 'sse=', kmeans.inertia_

        '''