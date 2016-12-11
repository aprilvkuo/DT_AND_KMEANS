#!/usr/bin/env python
# encoding: utf-8

"""
@version:
@author:
@license: Apache Licence 
@file: decision_tree.py
@time: 16-10-28 下午4:40
"""
import math
from sklearn.externals.six import StringIO
import pydot
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import cross_val_predict

def entropy(featureVec):
    #featureVec 为class的特征向量
    dic={}
    count=0
    entropy=0
    for i in featureVec:
        count+=1
        if dic.has_key(i):
            dic[i] += 1
        else:
            dic[i] = 1
    for key,value in dic.items():
        pro=float(value)/count
        entropy-= (pro * math.log(pro,2))
    return entropy

def value_en(feature_num,featureVec,value):
    classlist=[]
    for item in featureVec:
        if item[feature_num]==value:
            classlist.append(item[-1])
    return entropy(classlist),len(classlist)



def feature_en(feature_num,featureVec):
    value_list=set([item for item in featureVec[feature_num]])
    en=0
    fea_len=len(featureVec)
    for item in value_list:
        den,count=value_en(feature_num,featureVec,item)
        en-=(float(count)/fea_len*den)
    return en

def get_best_feature(featureVec):
    numFeature=len(featureVec[0])-1
    en = 0
    index = -1
    for i in range (numFeature):
        tmp=feature_en(i,featureVec)
        if tmp<en:
            print tmp,i
            en,index=tmp,i
    return i

def tree_build(featureVec):
    select_list=[]
    feature_len=len(featureVec[0])-1
    while len(select_list)<feature_len:
        axis=get_best_feature(featureVec)
        select_list.append(axis)
        retDataSet=[]
        for featVec in featureVec:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
        featureVec=retDataSet
    return select_list

def readmod(name):
    import pandas as pd
    data = (pd.read_csv(name))
    return np.array(data.values)

def question1(type):
    iris=load_iris()
    print iris.target
    data=readmod('iris.data')
    X = np.array(data[:, :-1])
    y = np.array(data[:, -1])


    print type+':'


    clf = tree.DecisionTreeClassifier(criterion=type)
    clf = clf.fit(X, y)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png('./iris_' + type + '.png')
    print"hold-out:", np.mean(cross_val_score(clf, X, y)),'\n'
    print"2-fold-cross:", (cross_val_score(clf, X, y, cv=2)),'\n'
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    kf.get_n_splits(X)
    #print (iris.data)
    scores=[]
    for train_index, test_index in kf.split(X):
       # print("TRAIN:", train_index, "TEST:", test_index)
        clf1 =tree.DecisionTreeClassifier(criterion=type)
        clf1 = clf1.fit(X[train_index], y[train_index])
        scores.append(clf1.score(X[test_index], y[test_index]))
    print np.mean(np.array(scores))

    '''
    clf = clf.fit(iris.data, iris.target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_png('./iris_'+type+'.png')
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10)
    accuracy=0
    for train_index, test_index in kf.split(iris.data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = iris.data[train_index], iris.data[test_index]
        y_train, y_test = iris.target[train_index], iris.target[test_index]
        clf = clf.fit(X_train, y_train)
        print score
           dot_data = StringIO()
            tree.export_graphviz(clf, out_file=dot_data)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            graph[0].write_png('./iris_' + type + '.png')
    '''


    #bagging = BaggingClassifier(KNeighborsClassifier(),max_samples = 0.5, max_features = 0.5)









if __name__ == '__main__':
    question1('gini')
    question1('entropy')






    '''
    en=  -float(4)/9*math.log(float(4)/9,2)-float(5)/9*math.log(float(5)/9,2)
    vec=[]


    dataSet = [[1, '', 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [1, 1, 'no'],
               [1, 1, 'no'],
               [1, '', 'yes'],
               ]
    list=tree_build(dataSet)
    ac_list=[]
    for i in range (len(list)):
        tmp=list[i]
        for j in range(i):
            if ac_list[j]<=list[i]:
                list[i]+=1
        ac_list.append(tmp)
    print ac_list
    '''


