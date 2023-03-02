import csv
from itertools import islice

import pandas as pd
from sklearn.preprocessing import LabelEncoder


ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

column_names = ddos.columns
# print(ddos.describe())
# 处理空值
pd.options.mode.use_inf_as_na = True
nan_list = ddos.isnull().sum().tolist()
ddos.dropna(inplace=True)
ddos=ddos.iloc[:, [6, 8, 53, 68, 3, 78]]
print(ddos)
encoder = LabelEncoder().fit(ddos.iloc[:,5])
ddos.iloc[:,5] = pd.DataFrame(encoder.transform(ddos.iloc[:,5]))

ddos.to_csv("/Users/winwin/PycharmProjects/pythonProject/tocsv/ddos.csv",header=None)

'''
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
'''
'''
rm = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
rm.fit(x_train, y_train.values.ravel())

importances = rm.feature_importances_
indices = np.argsort(importances)[::-1]
# print(importances)
# print(indices)

# np.argsort()返回待排序集合从下到大的索引值，[::-1]实现倒序，即最终imp_result内保存的是从大到小的索引值
imp_result = np.argsort(importances)[::-1][:]

# 按重要性从高到低输出属性列名和其重要性
for i in range(len(imp_result)):
    print("%2d. %-*s %f" % (i + 1, 30, x_train.columns[imp_result[i]], importances[imp_result[i]]))

'''
'''
x_train = x_train.iloc[:, [6, 8, 53, 68, 3, 63, 66]]
x_test = x_test.iloc[:, [6, 8, 53, 68, 3, 63, 66]]
'''

'''
times = time()
for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=kernel,
              gamma=20,
              degree=1,
              cache_size=5000
              ).fit(x_train, y_train.values.ravel())
    result = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    recall = recall_score(y_train, result)
    auc = roc_auc_score(y_test, clf.decision_function(x_train))
    print("%s 's testing accuracy %f, recall is %f, auc is %f" % kernel, score, recall, auc)
    print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))
'''
# 混淆矩阵函数
'''
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


# 实例化svc对象
times = time()
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
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
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
plt.xlabel('Predicted Label',fontsize=14)
plt.ylabel('True Label',fontsize=14)

plot_confusion_matrix(cm, title='SVM-confusion matrix')

# show confusion matrix
plt.show()


def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    return TP, FP, TN, FN

'''
'''
result = clf.predict(x_test)
score = clf.score(x_test, y_test)
recall = recall_score(y_train, result)
auc = roc_auc_score(y_test, clf.decision_function(x_train))
print("testing accuracy %f, recall is %f, auc is %f" % score, recall, auc)
print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))
'''
'''svmclf = SVMClassifier()
svmclf.pack('svmclf', clf)
saved_path = svmclf.save()'''

'''print(clf.score(x_train, y_train))  # 精度

tra_label=clf.predict(y_train) #训练集的预测标签
tes_label=clf.predict(y_test) #测试集的预测标签
print("训练集：", accuracy_score(y_train,tra_label) )
print("测试集：", accuracy_score(y_test,tes_label) )'''


