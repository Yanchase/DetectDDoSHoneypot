import pandas as pd

ddos = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
# 处理空值
pd.options.mode.use_inf_as_na = True
nan_list = ddos.isnull().sum().tolist()
# print(nan_list)
# print(sum(nan_list))
# print(pd.isna(ddos).any())
ddos.dropna(inplace=True)

ddos=ddos.iloc[:, [6, 8, 53, 68, 3, 63, 66]]
print(ddos)
