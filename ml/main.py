import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

print(ddos.info())
#print(ddos.isnull().mean())

column_names = ddos.columns
# print(ddos.describe())

# 处理空值
pd.options.mode.use_inf_as_na = True
nan_list = ddos.isnull().sum().tolist()
# print(nan_list)
# print(sum(nan_list))
# print(pd.isna(ddos).any())
ddos.dropna(inplace=True)


X = ddos.iloc[:, :-1]
Y = ddos.iloc[:, -1]

print(np.unique(Y))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(x_train)

for i in [x_train, x_test, y_train, y_test]:
    i.index = range(i.shape[0])
print(y_train.head())
print(y_train.value_counts())
print(y_test.value_counts())

encoder = LabelEncoder().fit(y_train)
y_train = pd.DataFrame(encoder.transform(y_train))
y_test = pd.DataFrame(encoder.transform(y_test))
print(y_train.head())
# DDOS = 1; Benign = 0
df = pd.read_json(parsed_json, orient='table')