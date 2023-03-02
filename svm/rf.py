import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

rm = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
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


plt.barh(x_train.columns[imp_result[1:7]], importances[imp_result[1:7]])
plt.xlabel("Random Forest Feature Importance")
plt.show()