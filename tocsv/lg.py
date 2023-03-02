import random

from sklearn.linear_model import LogisticRegression

data0 = []
data1 = []
label0 = []
label1 = []
with open('./ddos.log', 'r') as f:
    d = f.readline().strip()
    while d:
        # 特征 是否攻击
        array_data = d.split()[2:]
        # print(array_data)
        line = [float(i) for i in array_data]
        label = line[-1]
        # print(label)
        dd = line[:-1]
        # print(dd)
        if label == 0:
            data0.append(dd)
            label0.append(label)
        else:
            data1.append(dd)
            label1.append(label)
        d = f.readline().strip()
# print(len(data1),len(data0))
#random.shuffle(data1)
#random.shuffle(data0)

c0 = int(len(data0) * 4 / 5)
c1 = int(len(data1) * 4 / 5)
train_data = data0[:c0] + data1[:c1]
test_data = data0[c0:] + data1[c1:]
train_label = label0[:c0] + label1[:c1]
test_label = label0[c0:] + label1[c1:]

train = list(zip(train_data, train_label))
random.shuffle(train)
train_data, train_label = zip(*train)
test = list(zip(test_data, test_label))
random.shuffle(test)
test_data, test_label = zip(*test)
print('the number of train\'s data is:', len(train_data))
print('the number of test\'s data is:', len(test_data))
# print(train_data)
# print(train_label)

model = LogisticRegression()
#print(type(train_data[1][4]))
model.fit(train_data, train_label)