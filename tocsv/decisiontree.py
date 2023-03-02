# -*- coding: UTF-8 -*-
import time

import joblib
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
#from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import random
def GetData(dir='./ddos.log'):
    data0 = []
    data1 = []
    label0 = []
    label1 = []
    with open(dir,'r')as f:
        d = f.readline().strip()
        while d:
            # 时间 时间 五特征 是否攻击
            array_data = d.split()[1:]
            # print(array_data)
            line = [float(i) for i in array_data]
            label = line[-1]
            dd = line[:5]
            if label == 0:
                data0.append(dd)
                label0.append(label)
            else:
                data1.append(dd)
                label1.append(label)
            d=f.readline().strip()
    # print(len(data1),len(data0))
    random.shuffle(data1)
    random.shuffle(data0)

    c0=int(len(data0)*4/5)
    c1=int(len(data1)*4/5)
    train_data=data0[:c0]+data1[:c1]
    test_data=data0[c0:]+data1[c1:]
    train_label=label0[:c0]+label1[:c1]
    test_label=label0[c0:]+label1[c1:]

    train=list(zip(train_data,train_label))
    random.shuffle(train)
    train_data,train_label=zip(*train)
    test=list(zip(test_data,test_label))
    random.shuffle(test)
    test_data,test_label=zip(*test)
    print('the number of train\'s data is:', len(train_data))
    print('the number of test\'s data is:', len(test_data))
    return train_data,train_label,test_data, test_label

def classification(train_data,train_label,test_data,test_label):
    # PCA降维
    pca = PCA(n_components=3, whiten=True)
    pca.fit(train_data)
    pca.fit(test_data)
    start_time = time.time()
    #model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
    # model = SVC(C=0.5, kernel='linear')
    # model = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=1)
    # model = LogisticRegression()
    model = RandomForestClassifier(n_jobs=-1, n_estimators=20, max_features=3, max_depth=5)
    # clf = tree.DecisionTreeClassifier(criterion="gini", random_state=30, splitter="random")
    model.fit(train_data, train_label)
    elapse_time = time.time() - start_time
    print("tree训练耗时为:" + str(float(elapse_time * 1000)) + "ms")
    # 为了python2版本能够识别
    pre_y0 = model.predict(train_data)
    pre_y1 = model.predict(test_data)
    # output accuracy precious recall f-score
    perf_measure(test_label, pre_y1)

    joblib.dump(model, '/Users/winwin/PycharmProjects/pythonProject/tocsv/decisiontree.m', protocol=2)

    '''
    cm = confusion_matrix(test_label, pre_y1)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        plt.text(x_val, y_val, "%0.8f" % (c,), color='red', fontsize=7, va='center', ha='center')

    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm, title='SVM-confusion matrix')
    '''
def perf_measure(test_data, pred_data):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(test_data)):
        if test_data[i] == 1.0 and pred_data[i] == 1.0:
            TP += 1
        if test_data[i] == 0.0 and pred_data[i] == 1.0:
            FP += 1
        if test_data[i] == 0.0 and pred_data[i] == 0.0:
            TN += 1
        if test_data[i] == 1.0 and pred_data[i] == 0.0:
            FN += 1

    print('accuracy is:', (TP+TN)/(TP+FP+FN+TN))
    print('precision is:',(TP/(TP+FP)))
    print('recall is:',TP/(TP+FN))
    print('F-measure:',2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+(TP/(TP+FN))))
    return TP, FP, TN, FN

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    labels = [0, 1]
    tick_marks = np.array(range(len(labels))) + 0.5
    train_data, train_label,test_data,test_label=GetData()
    classification(train_data,train_label,test_data,test_label)