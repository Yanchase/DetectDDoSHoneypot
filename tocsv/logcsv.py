import csv

a = open('ddos.csv', 'r')
reader = csv.reader(a)

with open('ddos.log', 'w') as f:
    for i in reader:
        for x in i:
            f.write(x)
            f.write(' ')
        f.write('\n')
a.close()





'''import pandas as pd
import os

data = pd.read_csv('ddos.csv', encoding='utf-8')
with open('ddos.log', 'a+', encoding='utf-8') as f:
    for line in data.values:
        f.write((str(line[0]) + ' ' + str(line[1]) + '\n'))'''