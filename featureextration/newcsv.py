#   coding: UTF-8
#   @author: winwin
#   Created by yww on 2022-03-28
#   IDE PyCharm

import pandas as pd
from sklearn.preprocessing import LabelEncoder

ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

column_names = ddos.columns
# print(ddos.describe())
# 处理空值
pd.options.mode.use_inf_as_na = True
nan_list = ddos.isnull().sum().tolist()
ddos.dropna(inplace=True)
# 数据集按随机森林重要性排序切分
ddos=ddos.iloc[:, [6, 8, 53, 68, 3, 78]]
print(ddos)
encoder = LabelEncoder().fit(ddos.iloc[:,5])
ddos.iloc[:,5] = pd.DataFrame(encoder.transform(ddos.iloc[:,5]))
# 保存为新csv文件
ddos.to_csv("/Users/winwin/PycharmProjects/pythonProject/featureextration/ddos.csv",header=None)
