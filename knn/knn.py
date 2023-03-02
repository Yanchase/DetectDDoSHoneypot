from datetime import time, datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors
import bentoml
from sklearn.preprocessing import LabelEncoder

from sklearn import datasets
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix
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
# print(type(x_train.iloc[0, 0]))
# int
for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])

encoder = LabelEncoder().fit(y_train)
y_train = pd.DataFrame(encoder.transform(y_train))
y_test = pd.DataFrame(encoder.transform(y_test))



x_train = x_train.iloc[:, [6, 8, 53, 68, 3, 63, 66]]
x_test = x_test.iloc[:, [6, 8, 53, 68, 3, 63, 66]]

# 混淆矩阵函数
labels = [0, 1]
tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


clf = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
clf.fit(x_train, y_train.values.ravel())

print(clf.score(x_train, y_train))
result = clf.predict(x_test).tolist()
cm = confusion_matrix(y_test.values.ravel().tolist(), result)
tn, fp, fn, tp = confusion_matrix(y_test.values.ravel().tolist(), result).ravel()
print(tn, fp, fn, tp)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    plt.text(x_val, y_val, "%0.8f" % (c,), color='red', fontsize=13, va='center', ha='center')

# offset the tick

plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('Predicted Label',fontsize=14)
plt.ylabel('True Label',fontsize=14)
plot_confusion_matrix(cm, title='KNN-confusion matrix')

# show confusion matrix
plt.show()