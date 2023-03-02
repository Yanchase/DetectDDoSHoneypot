import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import bentoml
from sklearn.preprocessing import LabelEncoder

from svmclassifier import SVMClassifier

from sklearn import datasets
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC


ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
ddos.head()
column_names = ddos.columns
# print(ddos.describe())

# 处理空值
pd.options.mode.use_inf_as_na = True
nan_list = ddos.isnull().sum().tolist()
# print(nan_list)
# print(sum(nan_list))
# print(pd.isna(ddos).any())
ddos.dropna(inplace=True)


# 分离特征和标签
X = ddos.iloc[:, :-1]
Y = ddos.iloc[:, -1]

# 分离训练集和测试集，测试集占20%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#print(type(x_train.iloc[0, 0]))
# int
for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])
'''print(y_train)
print(y_test)'''
encoder = LabelEncoder().fit(y_train)
y_train = pd.DataFrame(encoder.transform(y_train))
y_test = pd.DataFrame(encoder.transform(y_test))

rm = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
rm.fit(x_train, y_train.values.ravel())


importances = rm.feature_importances_
indices = np.argsort(importances)[::-1]
#print(importances)
#print(indices)

# np.argsort()返回待排序集合从下到大的索引值，[::-1]实现倒序，即最终imp_result内保存的是从大到小的索引值
imp_result = np.argsort(importances)[::-1][:]

# 按重要性从高到低输出属性列名和其重要性
for i in range(len(imp_result)):
    print("%2d. %-*s %f" % (i + 1, 30, x_train.columns[imp_result[i]], importances[imp_result[i]]))

x_train = x_train.iloc[:, [6,8,53,68,3,63,66]]
x_test = x_test.iloc[:, [6,8,53,68,3,63,66]]
#print(x_train)


# 实例化svc对象
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
clf.fit(x_train, y_train.values.ravel())
print(clf.score(x_train, y_train))

svmclf = SVMClassifier()
svmclf.pack('svmclf', clf)
saved_path = svmclf.save()


'''print(clf.score(x_train, y_train))  # 精度

tra_label=clf.predict(y_train) #训练集的预测标签
tes_label=clf.predict(y_test) #测试集的预测标签
print("训练集：", accuracy_score(y_train,tra_label) )
print("测试集：", accuracy_score(y_test,tes_label) )'''
