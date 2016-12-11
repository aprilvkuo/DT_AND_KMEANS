#!/usr/bin/env python
# encoding: utf-8

"""
@author: aprilvkuo
@license: Apache Licence 
@file: dt.py.py
@time: 16-11-8 上午10:21
"""
import numpy as np
import pandas as pd
import math
import random





def readmod(name):
    data = (pd.read_csv(name))
    return np.array(data.values)


def holdout(X, y, scoref='entropy',prune_flag=False):
    N = X.shape[0]
    N_train = int(math.floor(0.7 * N))

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
    d_tree = decisiontree(X_train, y_train, scoref,prune_flag=prune_flag)

    count = 0
    accuracy = 0.0
    for index in range(len(y_test)):
        # print X_test[index], y_test[index], d_tree.classify(X_test[index], d_tree.root).keys()[0]
        # print classify(X_test[index],d_tree).keys()[0],y_test[index]
        if d_tree.predict(X_test[index]) == y_test[index]:
            count += 1
    accuracy = (float(count) / float(len(y_test)))
    print 'holdout:', accuracy


def kfold(base, X, y, scoref='entropy',prune_flag=False):
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
        d_tree = decisiontree(X_train, y_train, scoref,prune_flag=prune_flag)
        count = 0
        for index in range(len(y_test)):
            # print X_test[index], y_test[index], d_tree.classify(X_test[index], d_tree.root).keys()[0]
            # print classify(X_test[index],d_tree).keys()[0],y_test[index]
            if d_tree.predict(X_test[index]) == y_test[index]:
                count += 1
        # print float(count) / float(len(y_test))
        accuracy.append(float(count) / float(len(y_test)))
    print 'kfold:', np.mean(np.asarray(accuracy))


def bootstrap(X, y, scoref='entropy',prune_flag=False):
    N = X.shape[0]
    idx_train = np.random.randint(N, size=N)

    X_train = X[idx_train, :]
    y_train = y[idx_train, :]
    # print len(data_train)
    idx_test = []
    for i in range(N):
        if i not in idx_train:
            idx_test.append(i)

    # print float(len(data_test)) / N

    # build a model on the training set
    d_tree = decisiontree(X_train, y_train, scoref)
    X_test = X[idx_test, :]
    y_test = y[idx_test, :]
    count = 0

    for index in range(len(y_test)):
        # print X_test[index], y_test[index], d_tree.classify(X_test[index], d_tree.root).keys()[0]
        # print classify(X_test[index],d_tree).keys()[0],y_test[index]
        if d_tree.predict(X_test[index]) == y_test[index]:
            count += 1
    accuracy = (float(count) / float(len(y_test)))
    print 'bootstrap:', accuracy


class decisionnode:
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col = col
        self.value = value
        self.results = results
        self.tb = tb
        self.fb = fb


class decisiontree:
    def __init__(self, X=None, y=None, scoref='entropy',prune_flag=False):
        self.X = np.array(X)
        self.y = np.array(y)
        # print X.shape ,y.shape
        assert X.shape[0]==y.shape[0]
        self.scoref = scoref
        self.prune_flag=prune_flag
        try:
            self.rows = np.concatenate((self.X, self.y), axis=1)
        except:
            print X,y
            print X.shape(),y.shape()
        self.root = self.buildtree(self.rows)
        if prune_flag:
            self.prune(self.root, 0.01)

    # Divides a set on a specific column. Can handle numeric
    # or nominal values
    def divideset(self, rows, column, value):
        # Make a function that tells us if a row is in
        # the first group (true) or the second group (false)
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[column] >= value
        else:
            split_function = lambda row: row[column] == value

        # Divide the rows into two sets and return them
        set1 = [row for row in rows if split_function(row)]
        set2 = [row for row in rows if not split_function(row)]
        return (set1, set2)

    # Create counts of possible results (the last column of
    # each row is the result)
    def uniquecounts(self, rows):
        results = {}
        for row in rows:
            # The result is the last column
            r = row[len(row) - 1]
            if r not in results: results[r] = 0
            results[r] += 1
        return results

    # Probability that a randomly placed item will
    # be in the wrong category
    def giniimpurity(self, rows):
        total = len(rows)
        counts = self.uniquecounts(rows)
        imp = 0
        for k1 in counts:
            p1 = float(counts[k1]) / total
            for k2 in counts:
                if k1 == k2: continue
                p2 = float(counts[k2]) / total
                imp += p1 * p2
        return imp

    # Entropy is the sum of p(x)log(p(x)) across all
    # the different possible results
    def entropy(self, rows):
        from math import log
        log2 = lambda x: log(x) / log(2)
        results = self.uniquecounts(rows)
        # Now calculate the entropy
        ent = 0.0
        for r in results.keys():
            p = float(results[r]) / len(rows)
            ent = ent - p * log2(p)
        return ent

    # classification_error is the result of 1-max(P(c|t))
    # the different possible results
    def classification_error(self, rows):
        results = self.uniquecounts(rows)

        ce = 1.0
        for r in results.keys():
            pro = float(results[r]) / float(len(rows))
            if 1 - pro < ce:
                ce = 1 - pro

        return ce

    def printtree(self, tree, indent=''):
        # Is this a leaf node?
        if tree.results != None:
            print str(tree.results)
        else:
            # Print the criteria
            print str(tree.col) + ':' + str(tree.value) + '? '

            # Print the branches
            print indent + 'T->',
            self.printtree(tree.tb, indent + '  ')
            print indent + 'F->',
            self.printtree(tree.fb, indent + '  ')

    def getwidth(self, tree):
        if tree.tb == None and tree.fb == None: return 1
        return self.getwidth(tree.tb) + self.getwidth(tree.fb)

    def getdepth(self, tree):
        if tree.tb == None and tree.fb == None: return 0
        return max(self.getdepth(tree.tb), self.getdepth(tree.fb)) + 1

    from PIL import Image, ImageDraw

    def drawtree(self, jpeg='_tree.jpg'):
        jpeg=self.scoref+jpeg
        w = self.getwidth(self.root) * 100
        h = self.getdepth(self.root) * 100 + 120

        img = self.Image.new('RGB', (w, h), (255, 255, 255))
        draw = self.ImageDraw.Draw(img)

        self.drawnode(draw, self.root, w / 2, 20)
        img.save(jpeg, 'JPEG')

    def drawnode(self, draw, tree, x, y):
        if tree.results == None:
            # Get the width of each branch
            w1 = self.getwidth(tree.fb) * 100
            w2 = self.getwidth(tree.tb) * 100

            # Determine the total space required by this node
            left = x - (w1 + w2) / 2
            right = x + (w1 + w2) / 2

            # Draw the condition string
            draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))

            # Draw links to the branches
            draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
            draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))

            # Draw the branch nodes
            self.drawnode(draw, tree.fb, left + w1 / 2, y + 100)
            self.drawnode(draw, tree.tb, right - w2 / 2, y + 100)
        else:
            txt = ' \n'.join(['%s:%d' % v for v in tree.results.items()])
            draw.text((x - 20, y), txt, (0, 0, 0))


    def classify(self, observation, tree=None):
        if tree == None:
        #初始化
            tree = self.root
        if tree.results != None:
            #print tree.results
            #如果是叶子节点,返回分类准确率最高的预测结果
            return tree.results
        else:
            v = observation[tree.col]
            #待观察的属性.即判断的属性
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classify(observation, branch)


    def prune(self, tree, mingain):
        # If the branches aren't leaves, then prune them
        if tree.tb==None:
            return
        if tree.tb.results == None:
            self.prune(tree.tb, mingain)
        if tree.fb.results == None:
            self.prune(tree.fb, mingain)

        # If both the subbranches are now leaves, see if they
        # should merged
        if tree.tb.results != None and tree.fb.results != None:
            # Build a combined dataset
            tb, fb = [], []
            for v, c in tree.tb.results.items():
                tb += [[v]] * c
            for v, c in tree.fb.results.items():
                fb += [[v]] * c

            # Test the reduction in entropy
            delta = 0
            if self.scoref=='entropy':
                delta = self.entropy(tb + fb) - (self.entropy(tb) + self.entropy(fb) / 2)
            elif self.scoref =='gini':
                delta = self.giniimpurity(tb + fb) - (self.giniimpurity(tb) + self.giniimpurity(fb) / 2)
            if delta < mingain:
                # Merge the branches
                tree.tb, tree.fb = None, None
                tree.results = self.uniquecounts(tb + fb)

    def mdclassify(self, observation, tree=None):
        """
        返回一个list
        :param observation:
        :param tree:
        :return:
        """
        if tree == None:
        #初始化
            tree = self.root
        if tree.results != None:
            return tree.results
        else:
            v = observation[tree.col]
            if v == None:
            #如果属性缺失
                tr, fr = self.mdclassify(observation, tree.tb), self.mdclassify(observation, tree.fb)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = float(tcount) / (tcount + fcount)
                fw = float(fcount) / (tcount + fcount)
                result = {}
                for k, v in tr.items(): result[k] = v * tw
                for k, v in fr.items(): result[k] = v * fw
                return result
            else:
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value:
                        branch = tree.tb
                    else:
                        branch = tree.fb
                else:
                    if v == tree.value:
                        branch = tree.tb
                    else:
                        branch = tree.fb
                return self.mdclassify(observation, branch)

    def predict(self,observation):
        results =self.classify(observation)
        return sorted(results, cmp=lambda x, y: cmp(y, x), key=lambda k: results[k])[0]

    def buildtree(self, rows):
        if len(rows) == 0:
            return decisionnode()#数据为空
        current_score = 100
        if self.scoref == 'entropy':
            current_score = self.entropy(rows)
        elif self.scoref == 'gini':
            current_score = self.giniimpurity(rows)
        elif self.scoref == 'ce':
            current_score = self.classification_error(rows)

        # Set up some variables to track the best criteria
        best_gain = 0.0
        best_criteria = None
        best_sets = None


        column_count = len(rows[0]) - 1
        #观察的属性值数目
        for col in range(0, column_count):
            #对于每一个属性,记录它的所有不同值
            column_values = []
            for row in rows:
                column_values.append(row[col])
            column_values = set(column_values)
            # 对每个属性对应特定的值,将训练集进行划分(set1>=(==)value,set2<(!=)value)
            for value in column_values:
                (set1, set2) = self.divideset(rows, col, value)
                # 不纯性度量的增益
                p = float(len(set1)) / len(rows)
                if self.scoref == 'entropy':
                    gain = current_score - p * self.entropy(set1) - (1 - p) * self.entropy(set2)
                elif self.scoref == 'gini':
                    gain = current_score-p * self.giniimpurity(set1) - (1 - p) * self.giniimpurity(set2)
                elif self.scoref == 'ce':
                    gain =current_score - p * self.classification_error(set1) - (1 - p) * self.classification_error(set2)
                if gain > best_gain and len(set1) > 0 and len(set2) > 0:
                    best_gain = gain
                    best_criteria = (col, value)
                    best_sets = (set1, set2)
        # 创建分支,best_gain为预剪枝的阈值,设为0
        if self.prune_flag:
            tmp=0.01
        else:
            tmp=0
        if best_gain >tmp :
            trueBranch = self.buildtree(best_sets[0])
            falseBranch = self.buildtree(best_sets[1])
            return decisionnode(col=best_criteria[0], value=best_criteria[1],
                                tb=trueBranch, fb=falseBranch)
        else:
            return decisionnode(results=self.uniquecounts(rows))

    def error(self,X=None,y=None,W=None):
        '''
        对于一组数据X,y,用决策树进行预测,正确预测的样例数目
        :param X:
        :param y:
        :param W: 样例的权重
        :return:
        '''
        count = 0.0
        for i in range(len(X)):
            y_tmp = self.predict(X[i])
            if y_tmp != y[i][0]:
                count += W[i]
        #print count
        return count

    def result(self, X=None):
        '''
        用样例X预测得到的预测的结果y
        :param X:
        :return:
        '''
        X = X
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.predict(X[i]))
        return np.array(y_pred)


if __name__ == '__main__':
    data = readmod('1_iris.data')
    # print data
    X = np.array(data[:, :-1])
    y = np.array([data[:, -1]]).T
    # print np.concatenate((X,y), axis=1)
    tree = decisiontree(X, y,scoref='ce')
    tree.printtree(tree.root)
    #tree.drawtree()
    holdout(X, y,scoref='gini',prune_flag=False)
    kfold(10, X, y,scoref='gini',prune_flag=False)
    bootstrap(X, y,scoref='gini',prune_flag=False)
 #   d_tree.drawtree()
