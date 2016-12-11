#!/usr/bin/env python
# encoding: utf-8

"""

@author: aprilvkuo
@file: Adaboost.py
@time: 16-11-12 下午10:34
"""

from __future__ import division
import numpy as np
import scipy as sp
import dt
import random
import math
def holdout(X, y, scoref='entropy',M=10):
    # assume that we begin with two inputs:
    #    features - a matrix of input features
    #    target - an array of target variables
    #       corresponding to those features

    N = X.shape[0]
    N_train = int(math.floor(0.7 * N))
    #print N_train
    # randomly select indices for the training subset
    idx_train = random.sample(np.arange(N), N_train)

    # break your data into training and testing subsets
    X_train = X[idx_train, :]
    y_train = y[idx_train, :]
    # print len(data_train)
    idx_test = []
    for i in range(N):
        if i not in idx_train:
            idx_test.append(i)
    X_test = X[idx_test, :]
    y_test = y[idx_test, :]
    # build a model on the training set
    adboost = Adaboost(X_train, y_train, scoref)

    count = 0

    adboost.train(M)
    y_pred=adboost.pred(X_test)
    for index in range(len(y_test)):

        if y_pred[index] == y_test[index]:
            count += 1
    accuracy = (float(count) / float(len(y_test)))
    print 'holdout:', accuracy


def kfold(base, X, y, scoref='entropy',M=10):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=base)
    kf.get_n_splits(data)
    # print X,Y
    accuracy = []

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # print len(data_train), len(data_test)
        #print y_train
        adboost = Adaboost(X_train, y_train, scoref)
        count = 0
        adboost.train(M)
        y_pred = adboost.pred(X_test)
        for index in range(len(y_test)):
            # print X_test[index], y_test[index], d_tree.classify(X_test[index], d_tree.root).keys()[0]
            # print classify(X_test[index],d_tree).keys()[0],y_test[index]
            if y_pred[index] == y_test[index]:
                count += 1
        # print float(count) / float(len(y_test))
        accuracy.append(float(count) / float(len(y_test)))
    print 'kfold:', np.mean(np.asarray(accuracy))


def bootstrap(X, y, scoref='entropy',M=10):
    N = X.shape[0]
    idx_train = np.random.randint(N, size=N)

    # break your data into training and testing subsets
    X_train = X[idx_train, :]
    y_train = y[idx_train, :]
    # print len(data_train)
    idx_test = []
    for i in range(N):
        if i not in idx_train:
            idx_test.append(i)

    data_test = data[idx_test, :]
    # print float(len(data_test)) / N

    # build a model on the training set
    adboost = Adaboost(X_train, y_train, scoref)
    X_test = X[idx_test, :]
    y_test = y[idx_test, :]
    count = 0
    accuracy = 0.0
    adboost.train(M)
    y_pred = adboost.pred(X_test)
    for index in range(len(y_test)):
        # print X_test[index], y_test[index], d_tree.classify(X_test[index], d_tree.root).keys()[0]
        # print classify(X_test[index],d_tree).keys()[0],y_test[index]
        if y_pred[index] == y_test[index]:
            count += 1
    accuracy = (float(count) / float(len(y_test)))
    print 'bootstrap:', accuracy



class Adaboost:
    def __init__(self, X, y,scoref='entropy',Weaker=dt.decisiontree):
        '''
            W是训练数据的权重
            X,y分别为训练的观察属性和预测属性
        '''
        self.scoref=scoref
        self.X = np.array(X)
        self.y = np.array(y)
        assert self.X.shape[0] == self.y.shape[0]
        self.Weaker = Weaker
        self.sums = np.zeros(self.y.shape)
        self.W = np.ones((self.X.shape[0], 1)).flatten() / self.X.shape[0]
        self.Q = 0

    # print self.W
    def train(self, M=20):
        '''
            M is the maximal Weaker classification
        '''
        self.G = {}
        self.alpha = {}
        #初始化
        for i in range(M):
            self.G.setdefault(i)
            self.alpha.setdefault(i)
        for i in range(M):
            self.Q += 1  # 分类器的数量
            #弱分类器训练
            train_data=self.bootstrp(self.W,100)
            # train_data为一个n*1的数组,每一位代表数据项被采样的次数.
            # 按概率bootstrp,生成训练数据
            train_X=[]
            train_y=[]
            for k in range (len(train_data)):
                for j in range (int(train_data[k])):
                    train_X.append(self.X[k])
                    train_y.append(self.y[k])
            train_X=np.array(train_X)
            train_Y=np.array(train_y)
            #生成训练数据集


            self.G[i] = self.Weaker(train_X,train_Y,scoref=self.scoref)
            e = self.G[i].error(self.X,self.y,self.W)
            #在训练集上的分类误差率
            #print 'e=',e
            if e < 0.0001:
                #print i + 1, " weak classifier is enough to  make the error to 0"
                self.alpha[i] = 10
                break

            # print self.G[i].t_val,self.G[i].t_b,e
            self.alpha[i] = 1.0 / 2.0 * np.log((1 - e) / e)
            #print self.alpha[i]

            # print self.alpha[i]
            y_pred = self.G[i].result(self.X)
            #print self.W
            Z = self.W
            for k in range (len(self.W)):
                if self.y[k][0]==y_pred[k]:
                    tmp=-1
                else:
                    tmp=1
                #print self.W[k]
                Z[k]=self.W[k] * np.exp(-self.alpha[i] * tmp)
            self.W = (Z / Z.sum()).flatten(1)

            if self.finalclassifer(i):
                #print "True"
                return
            # print self.finalclassifer(i),'==========='

    def finalclassifer(self, t):
        '''
            the 1 to t weak classifer come together
        '''
        y_pred=self.G[t].result(self.X).flatten(1)
        y=np.zeros(self.y.shape)
        for i in range (len(y_pred)):
            #print y_pred[i]
            if y_pred[i] !=self.y[i][0]:
                y[i]=-1
            else:
                y[i]=1
            #预测结果
            self.sums[i] = self.sums[i] + y[i] * self.alpha[t]

        return self.caculate_error_num(self.sums)


    def caculate_error_num(self,sums):
        for i in sums:
            #print i
            if i <= 0:
                return False
        return True

    def pred(self, test_set):
        test_set = np.array(test_set)
        sums = []

        #sums 统计了每一行到每个结果的概率字典
        for i in range (test_set.shape[0]):
            sums.append({})
        #print 'len(sums)=',len(sums)
        #print test_set
        #print self.Q
        for i in range(self.Q):
            y=self.G[i].result(test_set)

            #print y
            #y为预测结果
            for j in range (test_set.shape[0]):#test_set.shape[0]为行数
                tmp=self.alpha[i]
                if sums[j].has_key(y[j]):
                    sums[j][y[j]]+=tmp
                else:
                    sums[j][y[j]]=tmp
        y_final=[]

        for item in sums:
            max_y = sorted(item, cmp=lambda x, y: cmp(y, x), key=lambda k: item[k])
            y_final.append(max_y[0])
        return y_final

    def bootstrp(self,W,count):
        train = np.zeros(len(W))
        for i in range (count):
            tmp=random.random()
            #生成0-1之间的小数
            sum=0.0
            for j in range (len(W)):
                sum+=W[j]
                #计算数据集权重之和,数据集的权重区间覆盖到tmp,则采样一样数据集
                if sum>=tmp:
                    train[j]+=1
                    break
        return train



if __name__ == '__main__':
    data=dt.readmod('1_iris.data')
    X=np.array(data[:,:-1])
    y=np.array([data[:,-1]]).T

    boost=Adaboost(X,y)
    boost.train(1000)
    y_pred=boost.pred(X)
    #print "pred"
    #print y_pred
    count = 0
    for i in range (X.shape[0]):
        if y_pred[i]==y[i][0]:
            count+=1
    print float(count)/float(X.shape[0])
    holdout(X,y)
    kfold(10,X,y)
    bootstrap(X,y)