# -*- coding: UTF-8 -*-
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
#from sklearn.externals import joblib
import time

import random
def GetData(dir='./ddos.log'):
    data0 = []
    data1 = []
    label0 = []
    label1 = []
    with open(dir,'r')as f:
        d = f.readline().strip()
        while d:
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
            d = f.readline().strip()
    print(len(data1), len(data0))
    random.shuffle(data1)
    random.shuffle(data0)

    c0=int(len(data0)*2/3)
    c1=int(len(data1)*2/3)
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

def classification(train_data, train_label, test_data, test_label):
    # PCA降维
    pca = PCA(n_components=3, whiten=True)
    pca.fit(train_data)
    pca.fit(test_data)
    start_time = time.time()
    model = SVC(C=0.5, kernel='linear')
    model.fit(train_data, train_label)
    elapse_time = time.time() - start_time
    print("svm训练耗时为:" + str(float(elapse_time * 1000)) + "ms")
    joblib.dump(model, '/Users/winwin/PycharmProjects/pythonProject/tocsv/svm.m', protocol=2)

    pre_y1 = model.predict(test_data)
    #output accuracy precious recall f-score
    #perf_measure(test_label, pre_y1)


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

if __name__ == '__main__':
    train_data, train_label,test_data,test_label=GetData()
    classification(train_data,train_label,test_data,test_label)
